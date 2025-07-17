"""
src/analysis/class_performance_analyzer.py

클래스별 성능 분석 및 취약점 식별 도구
Class performance analyzer to identify vulnerable classes and corruption sensitivities
"""
import sys
import os
from pathlib import Path

# 🔧 프로젝트 루트 경로 설정 (항상 첫 번째로)
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import fire
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="NanumGothic", font_scale=1.1, rc={'axes.unicode_minus': False}
)
from collections import defaultdict, Counter

from src.utils.config_utils import load_config, safe_classification_report_access


class ClassPerformanceAnalyzer:
    """클래스별 성능 및 corruption 취약점 분석기"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.setup_paths()
        self.load_class_mappings()
        ic("클래스 성능 분석기 초기화 완료")
        
    def setup_paths(self):
        """경로 설정"""
        self.output_dir = Path('outputs/class_performance_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Corruption 분석 결과 경로
        self.corruption_dir = Path('outputs/corruption_analysis')
        
        ic(f"결과 저장: {self.output_dir}")
        ic(f"Corruption 분석 경로: {self.corruption_dir}")
    
    def load_class_mappings(self):
        """클래스 매핑 정보 로드"""
        data_config = self.config.get('data', {})
        meta_file = data_config.get('meta_file', 'data/dataset/meta.csv')
        
        if Path(meta_file).exists():
            self.meta_df = pd.read_csv(meta_file)
            self.class_names = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
            self.num_classes = len(self.class_names)
            ic(f"클래스 정보 로드: {self.num_classes}개 클래스")
        else:
            ic(f"메타 파일 없음, 기본 클래스명 사용: {meta_file}")
            self.num_classes = 17
            self.class_names = {i: f"class_{i}" for i in range(self.num_classes)}
    
    def load_prediction_results(self, predictions_csv: str, ground_truth_csv: Optional[str] = None) -> pd.DataFrame:
        """예측 결과 로드 및 전처리"""
        ic(f"예측 결과 로드: {predictions_csv}")
        
        # 예측 결과 로드
        df_pred = pd.read_csv(predictions_csv)
        
        # 정답 데이터가 제공된 경우
        if ground_truth_csv and Path(ground_truth_csv).exists():
            ic(f"정답 데이터 로드: {ground_truth_csv}")
            df_true = pd.read_csv(ground_truth_csv)
            
            # 파일명으로 조인
            df_pred['join_key'] = df_pred['filename'].str.replace(r'\.(jpg|jpeg|png)$', '', regex=True)
            df_true['join_key'] = df_true['ID'].str.replace(r'\.(jpg|jpeg|png)$', '', regex=True)
            
            df_merged = pd.merge(df_pred, df_true, on='join_key', how='inner')
            
            if df_merged.empty:
                raise ValueError("예측 결과와 정답 데이터 간 매칭 실패")
            
            ic(f"매칭된 샘플: {len(df_merged)}개")
            return df_merged
        else:
            # 정답 데이터 없이 예측 결과만 분석
            ic("정답 데이터 없음 - 예측 분포만 분석")
            return df_pred
    
    def analyze_class_predictions(self, df: pd.DataFrame) -> Dict:
        """클래스별 예측 분석"""
        ic("클래스별 예측 분석 시작")
        
        analysis = {}
        
        # 1. 클래스별 예측 분포
        pred_distribution = df['predicted_target'].value_counts().sort_index()
        analysis['prediction_distribution'] = {
            str(k): int(v) for k, v in pred_distribution.items()
        }
        # Handle true class distribution if available
        if 'target' in df.columns:
            true_distribution = df['target'].value_counts().sort_index()
            analysis['true_distribution'] = {
            str(k): int(v) for k, v in true_distribution.items()
            }        
        # 2. 클래스별 평균 신뢰도
        confidence_by_class = df.groupby('predicted_target')['confidence'].agg(['mean', 'std', 'count'])
        analysis['confidence_by_predicted_class'] = {
            int(str(k)): {
                'mean_confidence': float(v['mean']),
                'std_confidence': float(v['std']) if not pd.isna(v['std']) else 0.0,
                'count': int(v['count'])
            }
            for k, v in confidence_by_class.iterrows()
        }
        
        # 3. 정답이 있는 경우 클래스별 성능 분석
        if 'target' in df.columns:
            analysis.update(self._analyze_class_performance_with_ground_truth(df))
        
        return analysis
    
    def _analyze_class_performance_with_ground_truth(self, df: pd.DataFrame) -> Dict:
        """정답 데이터가 있는 경우의 클래스별 성능 분석"""
        ic("정답 기반 클래스별 성능 분석")
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        analysis = {}
        
        # 1. 전체 분류 보고서
        y_true = df['target']
        y_pred = df['predicted_target']
        
        class_names_list = [self.class_names.get(i, f"class_{i}") for i in range(self.num_classes)]
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names_list, 
                                     output_dict=True,
                                     zero_division=0)
        
        # Ensure report is a dictionary
        if not isinstance(report, dict):
            ic("Warning: classification_report did not return a dictionary")
            report = {}
        
        # 2. 클래스별 성능 메트릭 추출
        class_performance = {}
        for class_id in range(self.num_classes):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # classification_report에서 해당 클래스 메트릭 추출
            class_metrics = safe_classification_report_access(report, class_name)
            
            # 추가 분석: 해당 클래스의 예측 분포
            true_class_samples = df[df['target'] == class_id]
            if len(true_class_samples) > 0:
                # 정확도
                correct_predictions = len(true_class_samples[true_class_samples['predicted_target'] == class_id])
                accuracy = correct_predictions / len(true_class_samples)
                
                # 평균 신뢰도 (올바른 예측과 틀린 예측 분리)
                correct_samples = true_class_samples[true_class_samples['predicted_target'] == class_id]
                wrong_samples = true_class_samples[true_class_samples['predicted_target'] != class_id]
                
                avg_confidence_correct = correct_samples['confidence'].mean() if len(correct_samples) > 0 else 0
                avg_confidence_wrong = wrong_samples['confidence'].mean() if len(wrong_samples) > 0 else 0
                
                # 가장 자주 혼동되는 클래스
                most_confused_with = true_class_samples['predicted_target'].value_counts().drop(class_id, errors='ignore')
                most_confused_class = int(most_confused_with.index.values[0]) if len(most_confused_with) > 0 else None
                
                class_performance[class_id] = {
                    'class_name': class_name,
                    'total_samples': len(true_class_samples),
                    'correct_predictions': int(correct_predictions),
                    'accuracy': float(accuracy),
                    'precision': class_metrics.get('precision', 0.0),
                    'recall': class_metrics.get('recall', 0.0),
                    'f1_score': class_metrics.get('f1-score', 0.0),
                    'support': class_metrics.get('support', 0),
                    'avg_confidence_correct': float(avg_confidence_correct),
                    'avg_confidence_wrong': float(avg_confidence_wrong),
                    'confidence_gap': float(avg_confidence_correct - avg_confidence_wrong),
                    'most_confused_with': most_confused_class,
                    'most_confused_with_name': self.class_names.get(most_confused_class, 'None') if most_confused_class is not None else 'None'
                }
        
        analysis['class_performance'] = class_performance
        
        # 3. 혼동 매트릭스
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        analysis['confusion_matrix'] = cm.tolist()
        
        return analysis
    
    def correlate_with_corruption_data(self, class_analysis: Dict) -> Dict:
        """클래스 성능과 corruption 데이터 연관성 분석"""
        ic("Corruption 데이터와 성능 연관성 분석")
        
        # Corruption 분석 결과 로드
        corruption_files = {
            'train': self.corruption_dir / 'train_corruption_analysis.csv',
            'test': self.corruption_dir / 'test_corruption_analysis.csv',
            'comparison': self.corruption_dir / 'dataset_comparison.json'
        }
        
        correlations = {}
        
        # Corruption 데이터가 있는 경우에만 분석
        if all(f.exists() for f in corruption_files.values()):
            ic("Corruption 분석 결과 발견, 연관성 분석 진행")
            
            # 테스트 데이터 corruption 정보 로드
            test_corruption_df = pd.read_csv(corruption_files['test'])
            
            # 클래스별 corruption 특성 분석
            # (실제로는 파일명을 통해 클래스를 매핑해야 하지만, 여기서는 일반적인 corruption 패턴 분석)
            
            # 1. 회전에 취약한 클래스 식별
            if 'class_performance' in class_analysis:
                rotation_vulnerability = self._identify_rotation_vulnerable_classes(class_analysis['class_performance'])
                correlations['rotation_vulnerability'] = rotation_vulnerability
            
            # 2. 조명에 취약한 클래스 식별
            if 'class_performance' in class_analysis:
                lighting_vulnerability = self._identify_lighting_vulnerable_classes(class_analysis['class_performance'])
                correlations['lighting_vulnerability'] = lighting_vulnerability
            
            # 3. 블러에 취약한 클래스 식별
            if 'class_performance' in class_analysis:
                blur_vulnerability = self._identify_blur_vulnerable_classes(class_analysis['class_performance'])
                correlations['blur_vulnerability'] = blur_vulnerability
        else:
            ic("Corruption 분석 결과 없음, 성능 기반 추정만 수행")
            correlations = self._estimate_vulnerabilities_from_performance(class_analysis)
        
        return correlations
    
    def _identify_rotation_vulnerable_classes(self, class_performance: Dict) -> Dict:
        """회전에 취약한 클래스 식별"""
        vulnerable_classes = {}
        
        # 성능이 낮고 신뢰도 격차가 큰 클래스들을 회전 취약 클래스로 추정
        for class_id, perf in class_performance.items():
            vulnerability_score = 0
            
            # 낮은 정확도 (회전으로 인한 성능 저하 추정)
            if perf['accuracy'] < 0.7:
                vulnerability_score += (0.7 - perf['accuracy']) * 10
            
            # 신뢰도 격차 (올바른 예측과 틀린 예측의 신뢰도 차이)
            confidence_gap = perf.get('confidence_gap', 0)
            if confidence_gap < 0.2:  # 격차가 작으면 구분이 어려움을 의미
                vulnerability_score += (0.2 - confidence_gap) * 5
            
            vulnerable_classes[class_id] = {
                'class_name': perf['class_name'],
                'vulnerability_score': float(vulnerability_score),
                'evidence': {
                    'low_accuracy': perf['accuracy'] < 0.7,
                    'low_confidence_gap': confidence_gap < 0.2,
                    'accuracy': perf['accuracy'],
                    'confidence_gap': confidence_gap
                }
            }
        
        # 취약성 점수 기준 정렬
        sorted_vulnerable = sorted(vulnerable_classes.items(), 
                                 key=lambda x: x[1]['vulnerability_score'], reverse=True)
        
        return {
            'ranking': [(int(k), v) for k, v in sorted_vulnerable[:10]],  # 상위 10개
            'high_risk_classes': [int(k) for k, v in sorted_vulnerable[:5] if v['vulnerability_score'] > 2.0],
            'analysis_method': 'performance_based_estimation'
        }
    
    def _identify_lighting_vulnerable_classes(self, class_performance: Dict) -> Dict:
        """조명에 취약한 클래스 식별"""
        # 텍스트 기반 문서는 조명 변화에 더 취약할 것으로 추정
        # 여기서는 F1 점수가 낮고 정밀도/재현율 불균형이 있는 클래스를 조명 취약 클래스로 추정
        
        vulnerable_classes = {}
        
        for class_id, perf in class_performance.items():
            vulnerability_score = 0
            
            # F1 점수가 낮음
            if perf['f1_score'] < 0.8:
                vulnerability_score += (0.8 - perf['f1_score']) * 8
            
            # 정밀도와 재현율의 불균형 (조명 변화로 인한 feature 변화)
            precision = perf.get('precision', 0)
            recall = perf.get('recall', 0)
            if precision > 0 and recall > 0:
                imbalance = abs(precision - recall)
                if imbalance > 0.2:
                    vulnerability_score += imbalance * 3
            
            vulnerable_classes[class_id] = {
                'class_name': perf['class_name'],
                'vulnerability_score': float(vulnerability_score),
                'evidence': {
                    'low_f1': perf['f1_score'] < 0.8,
                    'precision_recall_imbalance': abs(precision - recall) > 0.2 if precision > 0 and recall > 0 else False,
                    'f1_score': perf['f1_score'],
                    'precision': precision,
                    'recall': recall
                }
            }
        
        sorted_vulnerable = sorted(vulnerable_classes.items(), 
                                 key=lambda x: x[1]['vulnerability_score'], reverse=True)
        
        return {
            'ranking': [(int(k), v) for k, v in sorted_vulnerable[:10]],
            'high_risk_classes': [int(k) for k, v in sorted_vulnerable[:5] if v['vulnerability_score'] > 1.5],
            'analysis_method': 'f1_precision_recall_based'
        }
    
    def _identify_blur_vulnerable_classes(self, class_performance: Dict) -> Dict:
        """블러에 취약한 클래스 식별"""
        # 세부적인 feature에 의존하는 클래스들은 블러에 더 취약
        # 여기서는 support가 낮고 성능이 불안정한 클래스를 블러 취약 클래스로 추정
        
        vulnerable_classes = {}
        
        for class_id, perf in class_performance.items():
            vulnerability_score = 0
            
            # 재현율이 특히 낮음 (블러로 인한 feature 손실)
            recall = perf.get('recall', 0)
            if recall < 0.7:
                vulnerability_score += (0.7 - recall) * 10
            
            # 잘못된 예측의 신뢰도가 높음 (블러로 인한 혼동)
            avg_confidence_wrong = perf.get('avg_confidence_wrong', 0)
            if avg_confidence_wrong > 0.6:  # 틀렸는데도 확신
                vulnerability_score += (avg_confidence_wrong - 0.6) * 5
            
            vulnerable_classes[class_id] = {
                'class_name': perf['class_name'],
                'vulnerability_score': float(vulnerability_score),
                'evidence': {
                    'low_recall': recall < 0.7,
                    'high_wrong_confidence': avg_confidence_wrong > 0.6,
                    'recall': recall,
                    'avg_confidence_wrong': avg_confidence_wrong
                }
            }
        
        sorted_vulnerable = sorted(vulnerable_classes.items(), 
                                 key=lambda x: x[1]['vulnerability_score'], reverse=True)
        
        return {
            'ranking': [(int(k), v) for k, v in sorted_vulnerable[:10]],
            'high_risk_classes': [int(k) for k, v in sorted_vulnerable[:5] if v['vulnerability_score'] > 2.0],
            'analysis_method': 'recall_confidence_based'
        }
    
    def _estimate_vulnerabilities_from_performance(self, class_analysis: Dict) -> Dict:
        """성능 데이터만으로 취약점 추정"""
        if 'class_performance' not in class_analysis:
            return {}
        
        return {
            'rotation_vulnerability': self._identify_rotation_vulnerable_classes(class_analysis['class_performance']),
            'lighting_vulnerability': self._identify_lighting_vulnerable_classes(class_analysis['class_performance']),
            'blur_vulnerability': self._identify_blur_vulnerable_classes(class_analysis['class_performance'])
        }
    
    def generate_augmentation_recommendations(self, correlations: Dict) -> Dict:
        """취약점 분석 기반 증강 권장사항 생성"""
        ic("클래스별 증강 권장사항 생성")
        
        recommendations = {
            'class_specific_strategies': {},
            'global_strategies': [],
            'progressive_training_plan': {}
        }
        
        # 1. 클래스별 맞춤 전략
        all_vulnerable_classes = set()
        
        # 회전 취약 클래스
        if 'rotation_vulnerability' in correlations:
            rotation_classes = correlations['rotation_vulnerability'].get('high_risk_classes', [])
            for class_id in rotation_classes:
                if class_id not in recommendations['class_specific_strategies']:
                    recommendations['class_specific_strategies'][class_id] = {
                        'class_name': self.class_names.get(class_id, f"class_{class_id}"),
                        'vulnerabilities': [],
                        'augmentation_focus': [],
                        'intensity_multiplier': 1.0
                    }
                
                recommendations['class_specific_strategies'][class_id]['vulnerabilities'].append('rotation')
                recommendations['class_specific_strategies'][class_id]['augmentation_focus'].append({
                    'type': 'rotation',
                    'priority': 'high',
                    'parameters': {
                        'range': '±45°',
                        'probability': 0.9,
                        'progressive_steps': ['±15°', '±30°', '±45°']
                    }
                })
                recommendations['class_specific_strategies'][class_id]['intensity_multiplier'] *= 1.5
            
            all_vulnerable_classes.update(rotation_classes)
        
        # 조명 취약 클래스
        if 'lighting_vulnerability' in correlations:
            lighting_classes = correlations['lighting_vulnerability'].get('high_risk_classes', [])
            for class_id in lighting_classes:
                if class_id not in recommendations['class_specific_strategies']:
                    recommendations['class_specific_strategies'][class_id] = {
                        'class_name': self.class_names.get(class_id, f"class_{class_id}"),
                        'vulnerabilities': [],
                        'augmentation_focus': [],
                        'intensity_multiplier': 1.0
                    }
                
                recommendations['class_specific_strategies'][class_id]['vulnerabilities'].append('lighting')
                recommendations['class_specific_strategies'][class_id]['augmentation_focus'].append({
                    'type': 'lighting',
                    'priority': 'high',
                    'parameters': {
                        'brightness_range': '[0.8, 1.8]',
                        'contrast_range': '[0.8, 1.2]',
                        'probability': 0.8
                    }
                })
                recommendations['class_specific_strategies'][class_id]['intensity_multiplier'] *= 1.3
            
            all_vulnerable_classes.update(lighting_classes)
        
        # 블러 취약 클래스
        if 'blur_vulnerability' in correlations:
            blur_classes = correlations['blur_vulnerability'].get('high_risk_classes', [])
            for class_id in blur_classes:
                if class_id not in recommendations['class_specific_strategies']:
                    recommendations['class_specific_strategies'][class_id] = {
                        'class_name': self.class_names.get(class_id, f"class_{class_id}"),
                        'vulnerabilities': [],
                        'augmentation_focus': [],
                        'intensity_multiplier': 1.0
                    }
                
                recommendations['class_specific_strategies'][class_id]['vulnerabilities'].append('blur')
                recommendations['class_specific_strategies'][class_id]['augmentation_focus'].append({
                    'type': 'blur',
                    'priority': 'medium',
                    'parameters': {
                        'gaussian_blur_sigma': '[0, 2.0]',
                        'motion_blur_limit': 7,
                        'probability': 0.5
                    }
                })
                recommendations['class_specific_strategies'][class_id]['intensity_multiplier'] *= 1.2
            
            all_vulnerable_classes.update(blur_classes)
        
        # 2. 전역 전략
        if len(all_vulnerable_classes) > self.num_classes * 0.6:  # 60% 이상의 클래스가 취약
            recommendations['global_strategies'].append("대부분 클래스가 취약하므로 전역적 강화 증강 필요")
        
        recommendations['global_strategies'].extend([
            "회전 증강을 최우선으로 적용 (Gemini 분석 기반)",
            "조명 증강을 두 번째 우선순위로 적용",
            "점진적 훈련으로 증강 강도 단계적 증가"
        ])
        
        # 3. 점진적 훈련 계획
        recommendations['progressive_training_plan'] = {
            'phase_1': {
                'duration': '30% of epochs',
                'focus': '기본 feature 학습',
                'augmentation_intensity': 'mild',
                'rotation_range': '±10°',
                'lighting_factor': '[0.9, 1.2]'
            },
            'phase_2': {
                'duration': '40% of epochs',
                'focus': '중간 강도 robustness',
                'augmentation_intensity': 'medium',
                'rotation_range': '±25°',
                'lighting_factor': '[0.8, 1.5]'
            },
            'phase_3': {
                'duration': '30% of epochs',
                'focus': '최대 robustness',
                'augmentation_intensity': 'full',
                'rotation_range': '±45°',
                'lighting_factor': '[0.8, 1.8]'
            }
        }
        
        return recommendations
    
    def visualize_class_analysis(self, class_analysis: Dict, correlations: Dict) -> Optional[Figure]:
        """클래스 분석 결과 시각화"""
        if 'class_performance' not in class_analysis:
            ic("성능 데이터 없음, 시각화 생략")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        class_performance = class_analysis['class_performance']
        
        # 1. 클래스별 정확도
        class_ids = list(class_performance.keys())
        accuracies = [class_performance[cid]['accuracy'] for cid in class_ids]
        class_names = [class_performance[cid]['class_name'] for cid in class_ids]
        
        axes[0].bar(range(len(class_ids)), accuracies)
        axes[0].set_title('클래스별 정확도')
        axes[0].set_xlabel('클래스')
        axes[0].set_ylabel('정확도')
        axes[0].set_xticks(range(len(class_ids)))
        axes[0].set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
        
        # 2. F1 점수 분포
        f1_scores = [class_performance[cid]['f1_score'] for cid in class_ids]
        axes[1].bar(range(len(class_ids)), f1_scores, color='orange')
        axes[1].set_title('클래스별 F1 점수')
        axes[1].set_xlabel('클래스')
        axes[1].set_ylabel('F1 점수')
        axes[1].set_xticks(range(len(class_ids)))
        axes[1].set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
        
        # 3. 신뢰도 격차
        confidence_gaps = [class_performance[cid]['confidence_gap'] for cid in class_ids]
        axes[2].bar(range(len(class_ids)), confidence_gaps, color='green')
        axes[2].set_title('클래스별 신뢰도 격차 (올바른 예측 - 틀린 예측)')
        axes[2].set_xlabel('클래스')
        axes[2].set_ylabel('신뢰도 격차')
        axes[2].set_xticks(range(len(class_ids)))
        axes[2].set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
        
        # 4. 회전 취약성 점수
        if 'rotation_vulnerability' in correlations:
            rotation_scores = {}
            for cid, vuln_data in correlations['rotation_vulnerability'].get('ranking', [])[:len(class_ids)]:
                rotation_scores[cid] = vuln_data['vulnerability_score']
            
            rot_scores = [rotation_scores.get(cid, 0) for cid in class_ids]
            axes[3].bar(range(len(class_ids)), rot_scores, color='red')
            axes[3].set_title('회전 취약성 점수')
            axes[3].set_xlabel('클래스')
            axes[3].set_ylabel('취약성 점수')
            axes[3].set_xticks(range(len(class_ids)))
            axes[3].set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
        
        # 5. 조명 취약성 점수
        if 'lighting_vulnerability' in correlations:
            lighting_scores = {}
            for cid, vuln_data in correlations['lighting_vulnerability'].get('ranking', [])[:len(class_ids)]:
                lighting_scores[cid] = vuln_data['vulnerability_score']
            
            light_scores = [lighting_scores.get(cid, 0) for cid in class_ids]
            axes[4].bar(range(len(class_ids)), light_scores, color='yellow')
            axes[4].set_title('조명 취약성 점수')
            axes[4].set_xlabel('클래스')
            axes[4].set_ylabel('취약성 점수')
            axes[4].set_xticks(range(len(class_ids)))
            axes[4].set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
        
        # 6. 샘플 수 분포
        sample_counts = [class_performance[cid]['total_samples'] for cid in class_ids]
        axes[5].bar(range(len(class_ids)), sample_counts, color='purple')
        axes[5].set_title('클래스별 샘플 수')
        axes[5].set_xlabel('클래스')
        axes[5].set_ylabel('샘플 수')
        axes[5].set_xticks(range(len(class_ids)))
        axes[5].set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, class_analysis: Dict, correlations: Dict, 
                              recommendations: Dict) -> str:
        """종합 요약 보고서 생성"""
        report_path = self.output_dir / 'class_analysis_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 클래스별 성능 분석 및 증강 전략 보고서\n")
            f.write("=" * 60 + "\n\n")
            
            # 1. 전체 요약
            f.write("## 1. 전체 요약\n")
            if 'class_performance' in class_analysis:
                class_perf = class_analysis['class_performance']
                avg_accuracy = np.mean([p['accuracy'] for p in class_perf.values()])
                avg_f1 = np.mean([p['f1_score'] for p in class_perf.values()])
                f.write(f"- 평균 클래스 정확도: {avg_accuracy:.3f}\n")
                f.write(f"- 평균 F1 점수: {avg_f1:.3f}\n")
                f.write(f"- 분석된 클래스 수: {len(class_perf)}\n\n")
            
            # 2. 취약 클래스 식별
            f.write("## 2. 취약 클래스 식별\n")
            
            # 회전 취약 클래스
            if 'rotation_vulnerability' in correlations:
                rot_classes = correlations['rotation_vulnerability'].get('high_risk_classes', [])
                f.write(f"### 회전 취약 클래스 ({len(rot_classes)}개):\n")
                for cid in rot_classes:
                    class_name = self.class_names.get(cid, f"class_{cid}")
                    f.write(f"- 클래스 {cid} ({class_name})\n")
                f.write("\n")
            
            # 조명 취약 클래스
            if 'lighting_vulnerability' in correlations:
                light_classes = correlations['lighting_vulnerability'].get('high_risk_classes', [])
                f.write(f"### 조명 취약 클래스 ({len(light_classes)}개):\n")
                for cid in light_classes:
                    class_name = self.class_names.get(cid, f"class_{cid}")
                    f.write(f"- 클래스 {cid} ({class_name})\n")
                f.write("\n")
            
            # 블러 취약 클래스
            if 'blur_vulnerability' in correlations:
                blur_classes = correlations['blur_vulnerability'].get('high_risk_classes', [])
                f.write(f"### 블러 취약 클래스 ({len(blur_classes)}개):\n")
                for cid in blur_classes:
                    class_name = self.class_names.get(cid, f"class_{cid}")
                    f.write(f"- 클래스 {cid} ({class_name})\n")
                f.write("\n")
            
            # 3. 권장 증강 전략
            f.write("## 3. 권장 증강 전략\n")
            
            if 'global_strategies' in recommendations:
                f.write("### 전역 전략:\n")
                for strategy in recommendations['global_strategies']:
                    f.write(f"- {strategy}\n")
                f.write("\n")
            
            if 'progressive_training_plan' in recommendations:
                plan = recommendations['progressive_training_plan']
                f.write("### 점진적 훈련 계획:\n")
                for phase, details in plan.items():
                    f.write(f"**{phase.upper()}** ({details['duration']}):\n")
                    f.write(f"  - 목표: {details['focus']}\n")
                    f.write(f"  - 회전 범위: {details['rotation_range']}\n")
                    f.write(f"  - 조명 변화: {details['lighting_factor']}\n\n")
            
            # 4. 클래스별 맞춤 전략
            f.write("## 4. 클래스별 맞춤 전략\n")
            if 'class_specific_strategies' in recommendations:
                class_strategies = recommendations['class_specific_strategies']
                for cid, strategy in class_strategies.items():
                    f.write(f"### 클래스 {cid} ({strategy['class_name']}):\n")
                    f.write(f"- 취약점: {', '.join(strategy['vulnerabilities'])}\n")
                    f.write(f"- 증강 강도 배수: {strategy['intensity_multiplier']:.1f}x\n")
                    for aug in strategy['augmentation_focus']:
                        f.write(f"- {aug['type']} 증강 ({aug['priority']} 우선순위)\n")
                    f.write("\n")
            
            # 5. 구현 우선순위
            f.write("## 5. 구현 우선순위\n")
            f.write("1. **회전 증강 구현** (Critical - 554% 성능 차이)\n")
            f.write("   - 즉시 ±15° 회전 적용\n")
            f.write("   - 점진적으로 ±45°까지 확장\n\n")
            f.write("2. **조명 증강 구현** (High - 과노출 문제)\n")
            f.write("   - 밝기 팩터 [0.8, 1.8] 적용\n")
            f.write("   - 대비 조정 포함\n\n")
            f.write("3. **노이즈 타입 변경** (Medium)\n")
            f.write("   - 가우시안 노이즈 비중 증가\n")
            f.write("   - 임펄스 노이즈 비중 감소\n\n")
            
            # 6. 다음 단계
            f.write("## 6. 다음 단계\n")
            f.write("1. Step C: Conservative Augmentation Test 진행\n")
            f.write("2. 기존 79% 성능 유지하며 점진적 개선\n")
            f.write("3. 클래스별 맞춤 증강 데이터 생성\n")
            f.write("4. Progressive training pipeline 구현\n")
        
        return str(report_path)
    
    def run_comprehensive_analysis(self, predictions_csv: str, 
                                 ground_truth_csv: Optional[str] = None):
        """종합 클래스 성능 분석 실행"""
        ic("종합 클래스 성능 분석 시작")
        
        # 1. 예측 결과 로드
        df = self.load_prediction_results(predictions_csv, ground_truth_csv)
        
        # 2. 클래스별 예측 분석
        class_analysis = self.analyze_class_predictions(df)
        
        # 3. Corruption 데이터와 연관성 분석
        correlations = self.correlate_with_corruption_data(class_analysis)
        
        # 4. 증강 권장사항 생성
        recommendations = self.generate_augmentation_recommendations(correlations)
        
        # 5. 시각화
        fig = self.visualize_class_analysis(class_analysis, correlations)
        if fig:
            viz_path = self.output_dir / 'class_performance_visualization.png'
            fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            viz_path = None
        
        # 6. 결과 저장
        # 클래스 분석 결과
        with open(self.output_dir / 'class_analysis.json', 'w', encoding='utf-8') as f:
            # JSON 직렬화를 위한 처리
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            import json
            json.dump(class_analysis, f, indent=2, ensure_ascii=False, default=convert_numpy)
        
        # 연관성 분석 결과
        with open(self.output_dir / 'vulnerability_correlations.json', 'w', encoding='utf-8') as f:
            json.dump(correlations, f, indent=2, ensure_ascii=False, default=convert_numpy)
        
        # 권장사항
        with open(self.output_dir / 'augmentation_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False, default=convert_numpy)
        
        # 7. 요약 보고서 생성
        summary_path = self.generate_summary_report(class_analysis, correlations, recommendations)
        
        ic("클래스 성능 분석 완료")
        ic(f"결과 저장 위치: {self.output_dir}")
        
        return {
            'class_analysis': str(self.output_dir / 'class_analysis.json'),
            'correlations': str(self.output_dir / 'vulnerability_correlations.json'),
            'recommendations': str(self.output_dir / 'augmentation_recommendations.json'),
            'visualization': str(viz_path) if viz_path else None,
            'summary': summary_path
        }


def main():
    """Fire CLI 인터페이스"""
    fire.Fire(ClassPerformanceAnalyzer)


if __name__ == "__main__":
    main()