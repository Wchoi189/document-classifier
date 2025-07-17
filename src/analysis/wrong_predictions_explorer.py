"""
src/analysis/wrong_predictions_explorer.py

잘못된 예측 탐색기 - 오분류된 샘플들의 시각적 분석 도구
Wrong predictions explorer - Visual analysis tool for misclassified samples
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict, Counter
import fire
from src.utils.config_utils import load_config, get_classification_metrics
from sklearn.metrics import classification_report, confusion_matrix


class WrongPredictionsExplorer:
    """오분류 샘플 분석을 위한 탐색 도구"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.setup_paths()
        self.load_class_info()
        
    def setup_paths(self):
        """경로 설정"""
        self.data_dir = Path(self.config['data']['root_dir'])
        self.output_dir = Path('outputs/wrong_predictions_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 데이터 디렉토리: {self.data_dir}")
        print(f"✅ 분석 결과 저장: {self.output_dir}")
    
    def load_class_info(self):
        """클래스 정보 로드"""
        meta_file = self.config['data']['meta_file']
        if os.path.exists(meta_file):
            self.meta_df = pd.read_csv(meta_file)
            self.class_names = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
            print(f"✅ 클래스 정보 로드: {len(self.class_names)}개 클래스")
        else:
            print(f"⚠️ 메타 파일 없음, 기본 클래스명 사용: {meta_file}")
            self.class_names = {i: f"class_{i}" for i in range(17)}
    
    def load_predictions(self, predictions_csv: str, ground_truth_csv: Optional[str] = None) -> pd.DataFrame:
        """예측 결과와 정답 로드"""
        print(f"📥 예측 결과 로드: {predictions_csv}")
        
        # 예측 결과 로드
        df_pred = pd.read_csv(predictions_csv)
        required_cols = ['filename', 'predicted_target', 'confidence']
        
        if not all(col in df_pred.columns for col in required_cols):
            raise ValueError(f"예측 CSV에 필수 컬럼이 없습니다: {required_cols}")
        
        # 정답 데이터가 있는 경우
        if ground_truth_csv and os.path.exists(ground_truth_csv):
            print(f"📥 정답 데이터 로드: {ground_truth_csv}")
            df_true = pd.read_csv(ground_truth_csv)
            
            # 파일명으로 조인 (확장자 제거)
            df_pred['join_key'] = df_pred['filename'].str.replace(r'\.(jpg|jpeg|png)$', '', regex=True)
            df_true['join_key'] = df_true['ID'].str.replace(r'\.(jpg|jpeg|png)$', '', regex=True)
            
            df_merged = pd.merge(df_pred, df_true, on='join_key', how='inner')
            
            if df_merged.empty:
                raise ValueError("예측 결과와 정답 데이터 간 매칭되는 파일이 없습니다.")
            
            print(f"✅ 매칭된 샘플: {len(df_merged)}개")
            return df_merged
        else:
            print("⚠️ 정답 데이터 없음 - 예측 결과만 분석")
            df_pred['target'] = -1  # 더미 정답
            return df_pred
    
    def identify_wrong_predictions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """잘못된 예측 식별 및 분석"""
        if 'target' not in df.columns:
            print("⚠️ 정답 라벨이 없어 오분류 분석을 건너뜁니다.")
            return df, {}
        
        # 오분류 식별
        df['is_correct'] = df['predicted_target'] == df['target']
        df['error_type'] = df.apply(
            lambda row: f"{self.class_names.get(row['target'], row['target'])} → {self.class_names.get(row['predicted_target'], row['predicted_target'])}" 
            if not row['is_correct'] else 'Correct', axis=1
        )
        
        wrong_preds = df[~df['is_correct']].copy()
        
        # 오분류 통계
        error_stats = {
            'total_samples': len(df),
            'correct_predictions': df['is_correct'].sum(),
            'wrong_predictions': len(wrong_preds),
            'accuracy': df['is_correct'].mean(),
            'error_rate': 1 - df['is_correct'].mean()
        }
        
        print(f"📊 오분류 분석 결과:")
        print(f"   전체 샘플: {error_stats['total_samples']}")
        print(f"   정확한 예측: {error_stats['correct_predictions']}")
        print(f"   잘못된 예측: {error_stats['wrong_predictions']}")
        print(f"   정확도: {error_stats['accuracy']:.3f}")
        
        return wrong_preds, error_stats
    
    def analyze_error_patterns(self, wrong_preds: pd.DataFrame) -> Dict:
        """오류 패턴 분석"""
        if wrong_preds.empty:
            return {}
        
        print("🔍 오류 패턴 분석 중...")
        
        patterns = {}
        
        # 1. 클래스별 오분류 빈도
        patterns['class_errors'] = wrong_preds.groupby('target').size().to_dict()
        
        # 2. 가장 자주 혼동되는 클래스 쌍
        error_pairs = wrong_preds.groupby(['target', 'predicted_target']).size().reset_index(name='count')
        patterns['confusion_pairs'] = error_pairs.nlargest(10, 'count').to_dict('records')
        
        # 3. 신뢰도별 오분류 분포
        confidence_bins = pd.cut(wrong_preds['confidence'], bins=[0, 0.5, 0.7, 0.9, 1.0], 
                               labels=['Very Low (0-0.5)', 'Low (0.5-0.7)', 'Medium (0.7-0.9)', 'High (0.9-1.0)'])
        patterns['confidence_distribution'] = confidence_bins.value_counts().to_dict()
        
        # 4. 낮은 신뢰도 예측 (임계값 이하)
        low_confidence_threshold = 0.7
        patterns['low_confidence_errors'] = len(wrong_preds[wrong_preds['confidence'] < low_confidence_threshold])
        patterns['high_confidence_errors'] = len(wrong_preds[wrong_preds['confidence'] >= low_confidence_threshold])
        
        return patterns
    
    def create_error_visualization(self, wrong_preds: pd.DataFrame, patterns: Dict) -> Figure:
        """오류 시각화 생성"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 클래스별 오분류 수 (서브플롯 1)
        ax1 = plt.subplot(2, 3, 1)
        if patterns.get('class_errors'):
            class_error_series = pd.Series(patterns['class_errors'])
            class_error_series.index = pd.Index([self.class_names.get(idx, f"Class_{idx}") for idx in class_error_series.index])
            class_error_series.plot(kind='bar', ax=ax1)
            ax1.set_title('클래스별 오분류 개수', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. 신뢰도 분포 (서브플롯 2)
        ax2 = plt.subplot(2, 3, 2)
        if not wrong_preds.empty:
            wrong_preds['confidence'].hist(bins=20, ax=ax2, alpha=0.7, color='red', label='Wrong')
            ax2.set_title('오분류 신뢰도 분포', fontsize=12)
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
        
        # 3. 혼동 매트릭스 (서브플롯 3-4, 큰 영역)
        ax3 = plt.subplot(2, 2, 2)
        if 'target' in wrong_preds.columns and not wrong_preds.empty:
            # 전체 데이터에 대한 혼동 매트릭스 필요 - 여기서는 오분류만 표시
            error_matrix = wrong_preds.groupby(['target', 'predicted_target']).size().unstack(fill_value=0)
            sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds', ax=ax3)
            ax3.set_title('오분류 혼동 매트릭스', fontsize=12)
        
        # 4. 가장 혼동되는 클래스 쌍 (서브플롯 5)
        ax4 = plt.subplot(2, 3, 5)
        if patterns.get('confusion_pairs'):
            pairs_df = pd.DataFrame(patterns['confusion_pairs'][:5])  # 상위 5개
            pairs_df['pair_label'] = pairs_df.apply(
                lambda row: f"{self.class_names.get(row['target'], row['target'])} → {self.class_names.get(row['predicted_target'], row['predicted_target'])}", 
                axis=1
            )
            pairs_df.plot(x='pair_label', y='count', kind='bar', ax=ax4)
            ax4.set_title('가장 자주 혼동되는 클래스 쌍', fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. 신뢰도 구간별 오분류 (서브플롯 6)
        ax5 = plt.subplot(2, 3, 6)
        if patterns.get('confidence_distribution'):
            conf_dist = pd.Series(patterns['confidence_distribution'])
            conf_dist.plot(kind='bar', ax=ax5, color='orange')
            ax5.set_title('신뢰도 구간별 오분류 분포', fontsize=12)
            ax5.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_sample_gallery(self, wrong_preds: pd.DataFrame, n_samples: int = 20) -> Figure:
        """오분류 샘플 갤러리 생성"""
        if wrong_preds.empty:
            print("⚠️ 표시할 오분류 샘플이 없습니다.")
            return None
        
        # 다양한 오류 유형에서 샘플 선택
        samples_to_show = []
        
        # 1. 높은 신뢰도 오분류 (모델이 확신했지만 틀린 경우)
        high_conf_wrong = wrong_preds[wrong_preds['confidence'] > 0.8].head(5)
        samples_to_show.extend(high_conf_wrong.to_dict('records'))
        
        # 2. 낮은 신뢰도 오분류 (모델이 애매해했던 경우)
        low_conf_wrong = wrong_preds[wrong_preds['confidence'] < 0.6].head(5)
        samples_to_show.extend(low_conf_wrong.to_dict('records'))
        
        # 3. 나머지는 랜덤 샘플링
        remaining_samples = wrong_preds[~wrong_preds.index.isin([s['filename'] for s in samples_to_show if 'filename' in s])]
        if len(remaining_samples) > 0:
            random_samples = remaining_samples.sample(min(n_samples - len(samples_to_show), len(remaining_samples)))
            samples_to_show.extend(random_samples.to_dict('records'))
        
        # 실제 이미지 로드 및 표시
        n_cols = 5
        n_rows = (len(samples_to_show) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(samples_to_show):
            if i >= n_rows * n_cols:
                break
                
            row, col = i // n_cols, i % n_cols
            
            # 이미지 로드
            img_path = self.data_dir / 'train' / sample['filename']
            if not img_path.exists():
                img_path = self.data_dir / 'test' / sample['filename']
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[row, col].imshow(img)
                    
                    # 제목 설정
                    true_class = self.class_names.get(sample.get('target', -1), 'Unknown')
                    pred_class = self.class_names.get(sample['predicted_target'], 'Unknown')
                    conf = sample['confidence']
                    
                    title = f"실제: {true_class}\n예측: {pred_class}\n신뢰도: {conf:.3f}"
                    axes[row, col].set_title(title, fontsize=10)
                else:
                    axes[row, col].text(0.5, 0.5, 'Image\nLoad\nError', 
                                      ha='center', va='center', transform=axes[row, col].transAxes)
            else:
                axes[row, col].text(0.5, 0.5, 'Image\nNot\nFound', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
            
            axes[row, col].axis('off')
        
        # 빈 subplot들 숨기기
        for i in range(len(samples_to_show), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('오분류 샘플 갤러리', fontsize=16)
        plt.tight_layout()
        return fig
    
    def create_confidence_analysis(self, df: pd.DataFrame) -> Dict:
        """신뢰도 기반 분석"""
        analysis = {}
        
        if df.empty:
            return analysis
        
        # 신뢰도 구간별 정확도
        confidence_ranges = [
            (0.0, 0.5, "매우 낮음"),
            (0.5, 0.7, "낮음"), 
            (0.7, 0.9, "보통"),
            (0.9, 1.0, "높음")
        ]
        
        range_analysis = {}
        for min_conf, max_conf, label in confidence_ranges:
            mask = (df['confidence'] >= min_conf) & (df['confidence'] < max_conf)
            subset = df[mask]
            
            if len(subset) > 0 and 'is_correct' in subset.columns:
                range_analysis[label] = {
                    'count': len(subset),
                    'accuracy': subset['is_correct'].mean(),
                    'avg_confidence': subset['confidence'].mean()
                }
            else:
                range_analysis[label] = {
                    'count': len(subset),
                    'accuracy': None,
                    'avg_confidence': subset['confidence'].mean() if len(subset) > 0 else 0
                }
        
        analysis['confidence_ranges'] = range_analysis
        
        # 신뢰도 임계값별 성능
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_analysis = {}
        
        for threshold in thresholds:
            high_conf = df[df['confidence'] >= threshold]
            if len(high_conf) > 0 and 'is_correct' in high_conf.columns:
                threshold_analysis[threshold] = {
                    'samples_above_threshold': len(high_conf),
                    'accuracy_above_threshold': high_conf['is_correct'].mean(),
                    'coverage': len(high_conf) / len(df)
                }
        
        analysis['threshold_analysis'] = threshold_analysis
        return analysis
    
    def generate_html_report(self, wrong_preds: pd.DataFrame, patterns: Dict, 
                           confidence_analysis: Dict, output_filename: Optional[str] = None):
        """HTML 보고서 생성"""
        if output_filename is None:
            output_filename = "wrong_predictions_report.html"
        
        html_path = self.output_dir / output_filename
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>오분류 분석 보고서</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          background-color: #e8f4f8; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .error-pair {{ margin: 5px 0; padding: 10px; background-color: #fff2f2; 
                             border-left: 4px solid #ff4444; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 오분류 분석 보고서</h1>
                <p>모델의 잘못된 예측에 대한 상세 분석</p>
            </div>
        """
        
        # 기본 통계
        if patterns:
            html_content += f"""
            <div class="section">
                <h2>📊 기본 통계</h2>
                <div class="metric">
                    <strong>전체 오분류:</strong> {patterns.get('wrong_predictions', 0)}개
                </div>
                <div class="metric">
                    <strong>오분류율:</strong> {patterns.get('error_rate', 0):.3f}
                </div>
                <div class="metric">
                    <strong>높은 신뢰도 오분류:</strong> {patterns.get('high_confidence_errors', 0)}개
                </div>
                <div class="metric">
                    <strong>낮은 신뢰도 오분류:</strong> {patterns.get('low_confidence_errors', 0)}개
                </div>
            </div>
            """
        
        # 가장 혼동되는 클래스 쌍
        if patterns.get('confusion_pairs'):
            html_content += """
            <div class="section">
                <h2>🔄 가장 혼동되는 클래스 쌍</h2>
            """
            for pair in patterns['confusion_pairs'][:10]:
                true_class = self.class_names.get(pair['target'], f"Class_{pair['target']}")
                pred_class = self.class_names.get(pair['predicted_target'], f"Class_{pair['predicted_target']}")
                html_content += f"""
                <div class="error-pair">
                    <strong>{true_class}</strong> → <strong>{pred_class}</strong>: {pair['count']}회
                </div>
                """
            html_content += "</div>"
        
        # 신뢰도 분석
        if confidence_analysis.get('confidence_ranges'):
            html_content += """
            <div class="section">
                <h2>📈 신뢰도 구간별 분석</h2>
                <table>
                    <tr><th>신뢰도 구간</th><th>샘플 수</th><th>정확도</th><th>평균 신뢰도</th></tr>
            """
            for range_name, data in confidence_analysis['confidence_ranges'].items():
                accuracy_str = f"{data['accuracy']:.3f}" if data['accuracy'] is not None else "N/A"
                html_content += f"""
                <tr>
                    <td>{range_name}</td>
                    <td>{data['count']}</td>
                    <td>{accuracy_str}</td>
                    <td>{data['avg_confidence']:.3f}</td>
                </tr>
                """
            html_content += "</table></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML 보고서 생성: {html_path}")
        return str(html_path)
    
    def generate_comprehensive_analysis(self, predictions_csv: str, ground_truth_csv: Optional[str] = None):
        """종합 오분류 분석 실행"""
        print("🚀 종합 오분류 분석 시작...")
        
        # 1. 데이터 로드
        df = self.load_predictions(predictions_csv, ground_truth_csv)
        
        # 2. 오분류 식별
        wrong_preds, error_stats = self.identify_wrong_predictions(df)
        
        if wrong_preds.empty:
            print("✅ 오분류가 없거나 정답 데이터가 없어 분석을 완료합니다.")
            return
        
        # 3. 오류 패턴 분석
        patterns = self.analyze_error_patterns(wrong_preds)
        patterns.update(error_stats)
        
        # 4. 신뢰도 분석
        confidence_analysis = self.create_confidence_analysis(df)
        
        # 5. 시각화 생성
        print("📊 시각화 생성 중...")
        error_viz = self.create_error_visualization(wrong_preds, patterns)
        viz_path = self.output_dir / 'error_analysis_visualization.png'
        error_viz.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(error_viz)
        
        gallery_path = None
        # 6. 샘플 갤러리 생성
        print("🖼️ 오분류 샘플 갤러리 생성 중...")
        gallery_fig = self.create_sample_gallery(wrong_preds, n_samples=20)
        if gallery_fig:
            gallery_path = self.output_dir / 'wrong_predictions_gallery.png'
            gallery_fig.savefig(gallery_path, dpi=300, bbox_inches='tight')
            plt.close(gallery_fig)
        
        # 7. HTML 보고서 생성
        print("📄 HTML 보고서 생성 중...")
        html_path = self.generate_html_report(wrong_preds, patterns, confidence_analysis)
        
        # 8. JSON 분석 결과 저장
        def convert_np(obj):
            if isinstance(obj, dict):
                return {convert_np(k): convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            else:
                return obj

        analysis_results = {
            'error_patterns': patterns,
            'confidence_analysis': confidence_analysis,
            'summary': {
                'total_wrong_predictions': int(len(wrong_preds)),
                'most_confused_classes': convert_np(patterns.get('confusion_pairs', [])[:5]),
                'recommendations': self._generate_recommendations(patterns, confidence_analysis)
            }
        }
        analysis_results = convert_np(analysis_results)
        
        json_path = self.output_dir / 'analysis_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print("\n✅ 종합 분석 완료!")
        print(f"📊 시각화: {viz_path}")
        print(f"🖼️ 갤러리: {gallery_path if 'gallery_path' is not None in locals() else 'N/A'}")
        print(f"📄 HTML 보고서: {html_path}")
        print(f"📋 JSON 결과: {json_path}")
        
        return {
            'visualization': str(viz_path),
            'gallery': str(gallery_path) if 'gallery_path' in locals() else None,
            'html_report': html_path,
            'json_results': str(json_path)
        }
    
    def _generate_recommendations(self, patterns: Dict, confidence_analysis: Dict) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        recommendations = []
        
        # 높은 신뢰도 오분류가 많은 경우
        if patterns.get('high_confidence_errors', 0) > patterns.get('low_confidence_errors', 0):
            recommendations.append("모델이 확신을 가지고 틀리는 경우가 많습니다. 데이터 품질이나 라벨링을 재검토해보세요.")
        
        # 특정 클래스에서 오분류가 집중된 경우
        if patterns.get('class_errors'):
            max_errors = max(patterns['class_errors'].values())
            total_errors = sum(patterns['class_errors'].values())
            if max_errors > total_errors * 0.3:  # 30% 이상이 한 클래스에 집중
                recommendations.append("특정 클래스에서 오분류가 집중되고 있습니다. 해당 클래스의 데이터 증강을 고려해보세요.")
        
        # 낮은 신뢰도 예측이 많은 경우
        if patterns.get('low_confidence_errors', 0) > patterns.get('high_confidence_errors', 0):
            recommendations.append("모델이 확신이 없는 예측이 많습니다. 모델 복잡도를 높이거나 더 많은 데이터가 필요할 수 있습니다.")
        
        # 혼동되는 클래스 쌍이 명확한 경우
        if patterns.get('confusion_pairs'):
            top_confusion = patterns['confusion_pairs'][0]
            if top_confusion['count'] > 5:  # 5회 이상 혼동
                recommendations.append(f"클래스 간 혼동이 자주 발생합니다. 유사한 클래스들의 구분 특징을 강화하는 증강 기법을 적용해보세요.")
        
        return recommendations


def main():
    """메인 함수 - Fire CLI 인터페이스"""
    fire.Fire(WrongPredictionsExplorer)


if __name__ == "__main__":
    main()