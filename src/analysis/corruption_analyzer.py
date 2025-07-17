"""
src/analysis/corruption_analyzer.py

테스트 데이터의 corruption 패턴을 분석하여 도메인 갭 원인을 파악하는 도구
Corruption pattern analyzer to identify domain gap causes in test data
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import fire
from icecream import ic
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set_theme(style="whitegrid", font="NanumGothic", font_scale=1.1, rc={'axes.unicode_minus': False}
)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.project_setup import setup_project_environment
setup_project_environment()
from src.utils.utils import convert_numpy_types
from src.utils.config_utils import load_config


class CorruptionAnalyzer:
    """테스트 데이터 corruption 패턴 분석기"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.setup_paths()
        ic("Corruption Analyzer 초기화 완료")
        
    def setup_paths(self):
        """경로 설정"""
        data_config = self.config.get('data', {})
        self.train_dir = Path(data_config.get('root_dir', 'data/dataset')) / 'train'
        self.test_dir = Path(data_config.get('root_dir', 'data/dataset')) / 'test'
        self.output_dir = Path('outputs/corruption_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        ic(f"훈련 데이터: {self.train_dir}")
        ic(f"테스트 데이터: {self.test_dir}")
        ic(f"결과 저장: {self.output_dir}")
    
    def analyze_blur_levels(self, image_path: str) -> Dict:
        """이미지의 블러 레벨 분석"""
        img = cv2.imread(image_path)
        if img is None:
            return {'blur_score': 0, 'blur_type': 'unknown'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 라플라시안 분산으로 블러 측정
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 그래디언트 크기로 블러 측정
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_mean = np.mean(grad_magnitude)
        
        # FFT 기반 고주파 성분 분석
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        high_freq_energy = np.mean(magnitude_spectrum[gray.shape[0]//4:3*gray.shape[0]//4, 
                                                     gray.shape[1]//4:3*gray.shape[1]//4])
        
        return {
            'blur_score': float(laplacian_var),
            'gradient_strength': float(grad_mean),
            'high_freq_energy': float(high_freq_energy),
            'blur_type': self._classify_blur_type(laplacian_var, grad_mean)
        }
    
    def _classify_blur_type(self, laplacian_var: float, grad_mean: float) -> str:
        """블러 타입 분류"""
        if laplacian_var < 50:
            return 'severe_blur'
        elif laplacian_var < 100:
            return 'moderate_blur'
        elif laplacian_var < 200:
            return 'mild_blur'
        else:
            return 'sharp'
    
    def analyze_noise_patterns(self, image_path: str) -> Dict:
        """노이즈 패턴 분석"""
        img = cv2.imread(image_path)
        if img is None:
            return {'noise_level': 0, 'noise_type': 'unknown'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 노이즈 추정
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        noise = gray.astype(np.float32) - denoised.astype(np.float32)
        noise_std = np.std(noise)
        
        # 임펄스 노이즈 감지 (salt and pepper)
        median_filtered = cv2.medianBlur(gray, 5)
        impulse_noise = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
        impulse_ratio = np.sum(impulse_noise > 20) / impulse_noise.size
        
        # 주파수 도메인에서 노이즈 특성 분석
        f_transform = np.fft.fft2(gray)
        power_spectrum = np.abs(f_transform)**2
        noise_floor = np.percentile(power_spectrum, 25)  # 하위 25% 주파수 성분
        
        return {
            'gaussian_noise_std': float(noise_std),
            'impulse_noise_ratio': float(impulse_ratio),
            'noise_floor': float(noise_floor),
            'noise_type': self._classify_noise_type(float(noise_std), float(impulse_ratio))
        }
    
    def _classify_noise_type(self, noise_std: float, impulse_ratio: float) -> str:
        """노이즈 타입 분류"""
        if impulse_ratio > 0.05:
            return 'impulse_noise'
        elif noise_std > 15:
            return 'high_gaussian_noise'
        elif noise_std > 8:
            return 'moderate_gaussian_noise'
        elif noise_std > 3:
            return 'low_gaussian_noise'
        else:
            return 'clean'
    
    def analyze_geometric_distortion(self, image_path: str) -> Dict:
        """기하학적 왜곡 분석"""
        img = cv2.imread(image_path)
        if img is None:
            return {'perspective_score': 0, 'rotation_angle': 0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 허프 변환으로 직선 감지
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            # for rho, theta in lines[:20]:  # 상위 20개 직선
            #     angle = theta * 180 / np.pi
            #     angles.append(angle)
            for line in lines[:20]:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                angles.append(angle)       
        # 원근 왜곡 점수 계산
        if angles:
            # 수직/수평선에서의 편차
            horizontal_angles = [a for a in angles if abs(a) < 45 or abs(a - 180) < 45]
            vertical_angles = [a for a in angles if abs(a - 90) < 45]
            
            h_deviation = np.std(horizontal_angles) if horizontal_angles else 0
            v_deviation = np.std(vertical_angles) if vertical_angles else 0
            perspective_score = (h_deviation + v_deviation) / 2
            
            # 전체적인 회전 각도 추정
            main_angle = np.median(angles) if angles else 0
            rotation_angle = min(abs(main_angle), abs(main_angle - 90), 
                               abs(main_angle - 180), abs(main_angle + 90))
        else:
            perspective_score = 0
            rotation_angle = 0
        
        return {
            'perspective_score': float(perspective_score),
            'rotation_angle': float(rotation_angle),
            'line_count': len(angles),
            'distortion_type': self._classify_distortion_type(float(perspective_score), float(rotation_angle))
        }
    
    def _classify_distortion_type(self, perspective_score: float, rotation_angle: float) -> str:
        """왜곡 타입 분류"""
        if perspective_score > 20:
            return 'severe_perspective'
        elif perspective_score > 10:
            return 'moderate_perspective'
        elif rotation_angle > 10:
            return 'rotation_dominant'
        elif rotation_angle > 5:
            return 'mild_rotation'
        else:
            return 'minimal_distortion'
    
    def analyze_lighting_conditions(self, image_path: str) -> Dict:
        """조명 조건 분석"""
        img = cv2.imread(image_path)
        if img is None:
            return {'brightness': 128, 'contrast': 50}
        
        # RGB와 LAB 색공간에서 분석
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # 밝기 분석
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # 대비 분석
        contrast = np.std(gray)
        
        # 다이나믹 레인지 분석
        dynamic_range = np.max(gray) - np.min(gray)
        
        # 히스토그램 분석
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_peak_count = len([i for i in range(1, 255) if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
        
        # 색온도 추정 (간단한 방법)
        b, g, r = cv2.split(img)
        color_temperature = np.mean(b) / (np.mean(r) + 1e-7)  # 블루/레드 비율
        
        return {
            'brightness': float(brightness),
            'brightness_std': float(brightness_std),
            'contrast': float(contrast),
            'dynamic_range': float(dynamic_range),
            'histogram_peaks': int(hist_peak_count),
            'color_temperature': float(color_temperature),
            'lighting_type': self._classify_lighting_type(float(brightness), float(contrast), float(dynamic_range))
        }
    
    def _classify_lighting_type(self, brightness: float, contrast: float, dynamic_range: float) -> str:
        """조명 타입 분류"""
        if brightness < 80:
            return 'underexposed'
        elif brightness > 180:
            return 'overexposed'
        elif contrast < 30:
            return 'low_contrast'
        elif dynamic_range < 100:
            return 'flat_lighting'
        else:
            return 'normal_lighting'
    
    def analyze_dataset(self, dataset_path: Path, dataset_name: str, 
                       max_samples: int = 200) -> pd.DataFrame:
        """데이터셋 전체 분석"""
        ic(f"{dataset_name} 데이터셋 분석 시작: {dataset_path}")
        
        # 이미지 파일 목록 수집
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(dataset_path.glob(ext)))
        
        if not image_files:
            ic(f"경고: {dataset_path}에서 이미지를 찾을 수 없음")
            return pd.DataFrame()
        
        # 샘플링 (너무 많으면 처리 시간이 오래 걸림)
        if len(image_files) > max_samples:
            image_files = np.random.choice(image_files, max_samples, replace=False)
            ic(f"샘플링: {len(image_files)}개 이미지 분석")
        
        results = []
        for i, img_path in enumerate(image_files):
            if i % 50 == 0:
                ic(f"진행률: {i}/{len(image_files)}")
            
            try:
                # 각종 corruption 분석
                blur_analysis = self.analyze_blur_levels(str(img_path))
                noise_analysis = self.analyze_noise_patterns(str(img_path))
                distortion_analysis = self.analyze_geometric_distortion(str(img_path))
                lighting_analysis = self.analyze_lighting_conditions(str(img_path))
                
                # 결과 통합
                result = {
                    'filename': img_path.name,
                    'dataset': dataset_name,
                    **blur_analysis,
                    **noise_analysis,
                    **distortion_analysis,
                    **lighting_analysis
                }
                
                results.append(result)
                
            except Exception as e:
                ic(f"오류 발생: {img_path.name} - {e}")
                continue
        
        df = pd.DataFrame(results)
        ic(f"{dataset_name} 분석 완료: {len(df)}개 샘플")
        return df
    
    def compare_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """훈련과 테스트 데이터셋 비교 분석"""
        ic("데이터셋 비교 분석 시작")
        
        comparison = {}
        
        # 수치형 컬럼들에 대한 통계 비교
        numeric_cols = ['blur_score', 'gradient_strength', 'gaussian_noise_std', 
                       'brightness', 'contrast', 'perspective_score', 'rotation_angle']
        
        for col in numeric_cols:
            if col in train_df.columns and col in test_df.columns:
                train_mean = train_df[col].mean()
                test_mean = test_df[col].mean()
                train_std = train_df[col].std()
                test_std = test_df[col].std()
                
                # t-test로 통계적 유의성 검정
                t_stat, p_value = stats.ttest_ind(train_df[col].dropna(), 
                                                test_df[col].dropna())
                
                comparison[col] = {
                    'train_mean': float(train_mean),
                    'test_mean': float(test_mean),
                    'train_std': float(train_std),
                    'test_std': float(test_std),
                    'difference': float(test_mean - train_mean),
                    'relative_difference': float((test_mean - train_mean) / train_mean * 100),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        # 카테고리형 변수 비교
        categorical_cols = ['blur_type', 'noise_type', 'distortion_type', 'lighting_type']
        
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                train_dist = train_df[col].value_counts(normalize=True).to_dict()
                test_dist = test_df[col].value_counts(normalize=True).to_dict()
                
                comparison[f'{col}_distribution'] = {
                    'train': train_dist,
                    'test': test_dist
                }
        
        return comparison
    
    def generate_corruption_profile(self, comparison: Dict) -> Dict:
        """corruption 프로필 생성 (증강 전략을 위한)"""
        ic("Corruption 프로필 생성")
        
        profile = {
            'critical_issues': [],
            'moderate_issues': [],
            'recommendations': []
        }
        
        # 임계값 설정
        for metric, data in comparison.items():
            if isinstance(data, dict) and 'relative_difference' in data:
                rel_diff = abs(data['relative_difference'])
                
                if rel_diff > 50 and data['significant']:  # 50% 이상 차이이고 통계적으로 유의
                    profile['critical_issues'].append({
                        'metric': metric,
                        'difference': data['relative_difference'],
                        'train_mean': data['train_mean'],
                        'test_mean': data['test_mean']
                    })
                elif rel_diff > 20 and data['significant']:  # 20% 이상 차이
                    profile['moderate_issues'].append({
                        'metric': metric,
                        'difference': data['relative_difference'],
                        'train_mean': data['train_mean'],
                        'test_mean': data['test_mean']
                    })
        
        # 권장사항 생성
        if any('blur' in issue['metric'] for issue in profile['critical_issues']):
            profile['recommendations'].append("블러 증강 강도를 크게 증가시켜야 함")
        
        if any('noise' in issue['metric'] for issue in profile['critical_issues']):
            profile['recommendations'].append("노이즈 증강을 추가하거나 강화해야 함")
        
        if any('perspective' in issue['metric'] for issue in profile['critical_issues']):
            profile['recommendations'].append("원근 왜곡 증강을 강화해야 함")
        
        if any('brightness' in issue['metric'] or 'contrast' in issue['metric'] 
               for issue in profile['critical_issues']):
            profile['recommendations'].append("조명 조건 증강을 다양화해야 함")
        
        return profile
    
    def visualize_comparison(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> plt.Figure:
        """비교 결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 주요 메트릭들 시각화
        metrics = ['blur_score', 'gaussian_noise_std', 'brightness', 
                  'contrast', 'perspective_score', 'rotation_angle']
        
        for i, metric in enumerate(metrics):
            if metric in train_df.columns and metric in test_df.columns:
                axes[i].hist(train_df[metric].dropna(), alpha=0.5, label='Train', bins=30)
                axes[i].hist(test_df[metric].dropna(), alpha=0.5, label='Test', bins=30)
                axes[i].set_title(f'{metric} 분포 비교')
                axes[i].set_xlabel(metric)
                axes[i].set_ylabel('빈도')
                axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    def run_comprehensive_analysis(self, max_samples: int = 200):
        """종합 corruption 분석 실행"""
        ic("종합 corruption 분석 시작")
        
        # 1. 각 데이터셋 분석
        train_df = self.analyze_dataset(self.train_dir, 'train', max_samples)
        test_df = self.analyze_dataset(self.test_dir, 'test', max_samples)
        
        if train_df.empty or test_df.empty:
            ic("오류: 데이터셋 분석 실패")
            return None
        
        # 2. 데이터셋 비교
        comparison = self.compare_datasets(train_df, test_df)
        
        # 3. Corruption 프로필 생성
        profile = self.generate_corruption_profile(comparison)
        
        # 4. 시각화
        fig = self.visualize_comparison(train_df, test_df)
        
        # 5. 결과 저장
        train_df.to_csv(self.output_dir / 'train_corruption_analysis.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_corruption_analysis.csv', index=False)
        
        with open(self.output_dir / 'dataset_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(comparison), f, indent=2, ensure_ascii=False)
        
        with open(self.output_dir / 'corruption_profile.json', 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(profile), f, indent=2, ensure_ascii=False)
        
        fig.savefig(self.output_dir / 'corruption_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 6. 요약 보고서 생성
        self.generate_summary_report(profile, comparison)
        
        ic("종합 분석 완료")
        ic(f"결과 저장 위치: {self.output_dir}")
        
        return {
            'train_analysis': str(self.output_dir / 'train_corruption_analysis.csv'),
            'test_analysis': str(self.output_dir / 'test_corruption_analysis.csv'),
            'comparison': str(self.output_dir / 'dataset_comparison.json'),
            'profile': str(self.output_dir / 'corruption_profile.json'),
            'visualization': str(self.output_dir / 'corruption_comparison.png'),
            'summary': str(self.output_dir / 'corruption_summary.txt')
        }
    
    def generate_summary_report(self, profile: Dict, comparison: Dict):
        """요약 보고서 생성"""
        report_path = self.output_dir / 'corruption_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Corruption Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("## Critical Issues (50%+ difference)\n")
            for issue in profile['critical_issues']:
                f.write(f"- {issue['metric']}: {issue['difference']:.1f}% 차이\n")
                f.write(f"  Train 평균: {issue['train_mean']:.3f}\n")
                f.write(f"  Test 평균: {issue['test_mean']:.3f}\n\n")
            
            f.write("## Moderate Issues (20%+ difference)\n")
            for issue in profile['moderate_issues']:
                f.write(f"- {issue['metric']}: {issue['difference']:.1f}% 차이\n")
            
            f.write("\n## Recommendations\n")
            for rec in profile['recommendations']:
                f.write(f"- {rec}\n")
            
            f.write(f"\n## Analysis Files Generated\n")
            f.write(f"- Train analysis: train_corruption_analysis.csv\n")
            f.write(f"- Test analysis: test_corruption_analysis.csv\n")
            f.write(f"- Comparison data: dataset_comparison.json\n")
            f.write(f"- Corruption profile: corruption_profile.json\n")
            f.write(f"- Visualization: corruption_comparison.png\n")


def main():
    """Fire CLI 인터페이스"""
    fire.Fire(CorruptionAnalyzer)


if __name__ == "__main__":
    main()