"""
src/analysis/wrong_predictions_explorer.py

잘못된 예측 탐색기 - 오분류된 샘플들의 시각적 분석 도구
Wrong predictions explorer - Visual analysis tool for misclassified samples
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
    
    def create_error_visualization(self, wrong_preds: pd.DataFrame, patterns: Dict) -> plt.Figure:
        """오류 시각화 생성"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 클래스별 오분류 수 (서브플롯 1)
        ax1 = plt.subplot(2, 3, 1)
        if patterns.get('class_errors'):
            class_error_series = pd.Series(patterns['class_errors'])
            class_error_series.index = [self.class_names.get(idx, f"Class_{idx}") for idx in class_error_series.index]
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
    
    def create_sample_gallery(self, wrong_preds: pd.DataFrame, n_samples: int = 20) -> plt.Figure:
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
        
        # 3. 나머지 랜덤 샘플
        remaining_samples = wrong_preds[~wrong_preds.index.isin(
            list(high_conf_wrong.index) + list(low_conf_wrong.index)
        )].sample(min(n_samples - len(samples_to_show), len(wrong_preds) - len(samples_to_show)), random_state=42)
        samples_to_show.extend(remaining_samples.to_dict('records'))
        
        # 실제 이미지 로드 및 갤러리 생성
        n_cols = 4
        n_rows = (len(samples_to_show) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample in enumerate(samples_to_show):
            row, col = idx // n_cols, idx % n_cols
            
            # 이미지 로드
            img_path = self.data_dir / 'train' / sample['filename']
            if not img_path.exists():
                img_path = self.data_dir / 'test' / sample['filename']
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[row, col].imshow(img)
                
                # 타이틀 정보
                true_class = self.class_names.get(sample.get('target', -1), 'Unknown')
                pred_class = self.class_names.get(sample['predicted_target'], 'Unknown')
                confidence = sample['confidence']
                
                title = f"실제: {true_class}\n예측: {pred_class}\n신뢰도: {confidence:.3f}"
                axes[row, col].set_title(title, fontsize=10)
            else:
                axes[row, col].text(0.5, 0.5, f"이미지 없음\n{sample['filename']}", 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
            
            axes[row, col].axis('off')
        
        # 빈 subplot 제거
        for idx in range(len(samples_to_show), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'오분류 샘플 갤러리 (총 {len(samples_to_show)}개)', fontsize=16)
        plt.tight_layout()
        return fig
    
    def create_confidence_analysis(self, df: pd.DataFrame) -> plt.Figure:
        """신뢰도 분석 차트 생성"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'is_correct' in df.columns:
            correct_preds = df[df['is_correct']]
            wrong_preds = df[~df['is_correct']]
            
            # 1. 신뢰도 분포 비교
            axes[0, 0].hist(correct_preds['confidence'], bins=30, alpha=0.7, label='정답', color='green')
            axes[0, 0].hist(wrong_preds['confidence'], bins=30, alpha=0.7, label='오답', color='red')
            axes[0, 0].set_title('정답 vs 오답 신뢰도 분포')
            axes[0, 0].set_xlabel('Confidence Score')
            axes[0, 0].legend()
            
            # 2. 신뢰도 구간별 정확도
            bins = np.arange(0, 1.1, 0.1)
            df['conf_bin'] = pd.cut(df['confidence'], bins=bins)
            accuracy_by_conf = df.groupby('conf_bin')['is_correct'].agg(['mean', 'count']).reset_index()
            
            axes[0, 1].bar(range(len(accuracy_by_conf)), accuracy_by_conf['mean'], 
                          alpha=0.7, color='blue')
            axes[0, 1].set_title('신뢰도 구간별 정확도')
            axes[0, 1].set_xlabel('Confidence Bins')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_xticks(range(len(accuracy_by_conf)))
            axes[0, 1].set_xticklabels([f"{bin.left:.1f}-{bin.right:.1f}" for bin in accuracy_by_conf['conf_bin']], 
                                     rotation=45)
        
        # 3. 클래스별 평균 신뢰도
        class_confidence = df.groupby('predicted_target')['confidence'].mean().sort_values(ascending=False)
        class_confidence.index = [self.class_names.get(idx, f"Class_{idx}") for idx in class_confidence.index]
        
        axes[1, 0].bar(range(len(class_confidence)), class_confidence.values, color='orange')
        axes[1, 0].set_title('클래스별 평균 예측 신뢰도')
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Average Confidence')
        axes[1, 0].set_xticks(range(len(class_confidence)))
        axes[1, 0].set_xticklabels(class_confidence.index, rotation=45, ha='right')
        
        # 4. 신뢰도 vs 정확도 산점도 (클래스별)
        if 'is_correct' in df.columns:
            class_stats = df.groupby('predicted_target').agg({
                'confidence': 'mean',
                'is_correct': 'mean'
            }).reset_index()
            
            axes[1, 1].scatter(class_stats['confidence'], class_stats['is_correct'], 
                             s=100, alpha=0.7, color='purple')
            axes[1, 1].set_title('클래스별 신뢰도 vs 정확도')
            axes[1, 1].set_xlabel('Average Confidence')
            axes[1, 1].set_ylabel('Accuracy')
            
            # 클래스 라벨 추가
            for idx, row in class_stats.iterrows():
                class_name = self.class_names.get(row['predicted_target'], f"C{row['predicted_target']}")
                axes[1, 1].annotate(class_name, (row['confidence'], row['is_correct']), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_html_report(self, wrong_preds: pd.DataFrame, patterns: Dict, error_stats: Dict) -> str:
        """HTML 형태의 상세 분석 보고서 생성"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>오분류 분석 보고서</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .stats {{ display: flex; justify-content: space-around; }}
                .stat-box {{ text-align: center; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .error-high {{ background-color: #ffcccc; }}
                .error-medium {{ background-color: #fff2cc; }}
                .error-low {{ background-color: #ccffcc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 오분류 분석 보고서</h1>
                <p>생성 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 전체 통계</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>{error_stats.get('total_samples', 0)}</h3>
                        <p>전체 샘플</p>
                    </div>
                    <div class="stat-box">
                        <h3>{error_stats.get('correct_predictions', 0)}</h3>
                        <p>정답 예측</p>
                    </div>
                    <div class="stat-box">
                        <h3>{error_stats.get('wrong_predictions', 0)}</h3>
                        <p>오답 예측</p>
                    </div>
                    <div class="stat-box">
                        <h3>{error_stats.get('accuracy', 0):.3f}</h3>
                        <p>정확도</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 주요 오류 패턴</h2>
        """
        
        # 혼동되는 클래스 쌍 표 추가
        if patterns.get('confusion_pairs'):
            html_content += """
                <h3>가장 자주 혼동되는 클래스 쌍</h3>
                <table>
                    <tr><th>실제 클래스</th><th>예측 클래스</th><th>오류 횟수</th></tr>
            """
            for pair in patterns['confusion_pairs'][:10]:
                true_class = self.class_names.get(pair['target'], f"Class_{pair['target']}")
                pred_class = self.class_names.get(pair['predicted_target'], f"Class_{pair['predicted_target']}")
                html_content += f"""
                    <tr>
                        <td>{true_class}</td>
                        <td>{pred_class}</td>
                        <td>{pair['count']}</td>
                    </tr>
                """
            html_content += "</table>"
        
        # 신뢰도 분석
        if patterns.get('confidence_distribution'):
            html_content += """
                <h3>신뢰도별 오분류 분포</h3>
                <table>
                    <tr><th>신뢰도 구간</th><th>오분류 개수</th></tr>
            """
            for conf_range, count in patterns['confidence_distribution'].items():
                html_content += f"<tr><td>{conf_range}</td><td>{count}</td></tr>"
            html_content += "</table>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>💡 개선 권장사항</h2>
                <ul>
                    <li>신뢰도가 높은데 틀린 예측들을 중점적으로 분석</li>
                    <li>자주 혼동되는 클래스 쌍에 대한 추가 특징 엔지니어링 고려</li>
                    <li>낮은 신뢰도 예측에 대한 임계값 조정 검토</li>
                    <li>오분류가 많은 클래스에 대한 추가 훈련 데이터 수집</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # HTML 파일 저장
        html_path = self.output_dir / 'detailed_analysis_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def run_comprehensive_analysis(self, 
                                  predictions_csv: str, 
                                  ground_truth_csv: Optional[str] = None,
                                  n_sample_images: int = 20):
        """종합 오분류 분석 실행"""
        print("🚀 종합 오분류 분석 시작...")
        
        # 1. 데이터 로드
        df = self.load_predictions(predictions_csv, ground_truth_csv)
        
        # 2. 오분류 식별
        wrong_preds, error_stats = self.identify_wrong_predictions(df)
        
        if wrong_preds.empty:
            print("✅ 모든 예측이 정확합니다!")
            return
        
        # 3. 오류 패턴 분석
        patterns = self.analyze_error_patterns(wrong_preds)
        
        # 4. 시각화 생성
        print("📊 시각화 생성 중...")
        
        # 오류 분석 차트
        error_viz = self.create_error_visualization(wrong_preds, patterns)
        error_viz.savefig(self.output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(error_viz)
        
        # 샘플 갤러리
        if len(wrong_preds) > 0:
            gallery_fig = self.create_sample_gallery(wrong_preds, n_sample_images)
            if gallery_fig:
                gallery_fig.savefig(self.output_dir / 'wrong_predictions_gallery.png', dpi=300, bbox_inches='tight')
                plt.close(gallery_fig)
        
        # 신뢰도 분석
        confidence_fig = self.create_confidence_analysis(df)
        confidence_fig.savefig(self.output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(confidence_fig)
        
        # 5. HTML 보고서 생성
        html_report = self.generate_html_report(wrong_preds, patterns, error_stats)
        
        # 6. JSON 결과 저장
        analysis_results = {
            'error_stats': error_stats,
            'patterns': patterns,
            'wrong_predictions_summary': {
                'total_wrong': len(wrong_preds),
                'high_confidence_wrong': len(wrong_preds[wrong_preds['confidence'] > 0.8]),
                'low_confidence_wrong': len(wrong_preds[wrong_preds['confidence'] < 0.5])
            }
        }
        
        with open(self.output_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        # 결과 요약 출력
        print("\n✅ 분석 완료!")
        print(f"📁 결과 저장 위치: {self.output_dir}")
        print(f"📄 HTML 보고서: {html_report}")
        print(f"📊 시각화 파일들:")
        print(f"   - error_analysis.png")
        print(f"   - wrong_predictions_gallery.png")
        print(f"   - confidence_analysis.png")
        
        return str(self.output_dir)


def main():
    """메인 함수 - Fire CLI 인터페이스"""
    fire.Fire(WrongPredictionsExplorer)


if __name__ == "__main__":
    main()