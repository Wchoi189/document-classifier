"""
src/utils/test_image_analyzer.py

테스트 이미지 분석기 - 가장 도전적이고 대표적인 테스트 샘플 자동 식별
Test image analyzer - Automatically identify most challenging and representative test samples
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
import fire
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from src.utils.config_utils import load_config
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure

# Set the font family to NanumGothic
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False


class TestImageAnalyzer:
    """테스트 이미지의 난이도와 특성을 분석하여 최적의 샘플을 선택하는 도구"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.setup_paths()
        
    def setup_paths(self):
        """경로 설정"""
        self.data_dir = Path(self.config['data']['root_dir'])
        self.test_dir = self.data_dir / 'test'
        self.train_dir = self.data_dir / 'train'
        self.output_dir = Path('outputs/test_image_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 테스트 디렉토리: {self.test_dir}")
        print(f"✅ 훈련 디렉토리: {self.train_dir}")
        print(f"✅ 분석 결과 저장: {self.output_dir}")
    
    def analyze_image_quality(self, image_path: str) -> Dict:
        """이미지 품질 분석 - 블러, 밝기, 대비 등"""
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        # RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 블러 정도 측정 (라플라시안 분산)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. 밝기 분석
        brightness = np.mean(img_rgb)
        brightness_std = np.std(img_rgb)
        
        # 3. 대비 분석 (RMS 대비)
        contrast = gray.std()
        
        # 4. 색상 분포 분석
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        
        # 히스토그램 엔트로피 (색상 다양성)
        hist_combined = np.concatenate([hist_b, hist_g, hist_r])
        hist_normalized = hist_combined / (hist_combined.sum() + 1e-7)
        color_entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-7))
        
        # 5. 가장자리 밀도
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        return {
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'brightness_std': float(brightness_std),
            'contrast': float(contrast),
            'color_entropy': float(color_entropy),
            'edge_density': float(edge_density),
            'width': img.shape[1],
            'height': img.shape[0],
            'aspect_ratio': img.shape[1] / img.shape[0]
        }
    
    def detect_perspective_distortion(self, image_path: str) -> Dict:
        """원근 왜곡 정도 감지"""
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 허프 변환으로 직선 감지
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        line_angles = []
        if lines is not None:
            for rho, theta in lines[:20]:  # 상위 20개 직선만 분석
                angle = theta * 180 / np.pi
                line_angles.append(angle)
        
        # 2. 각도 분산 (높을수록 더 왜곡됨)
        angle_variance = np.var(line_angles) if line_angles else 0
        
        # 3. 주요 방향성 분석 (0도, 90도에서 얼마나 벗어났는지)
        if line_angles:
            # 수직/수평에서의 편차
            horizontal_deviation = min([abs(angle) for angle in line_angles if abs(angle) < 45], default=45)
            vertical_deviation = min([abs(angle - 90) for angle in line_angles if abs(angle - 90) < 45], default=45)
            perspective_score = min(horizontal_deviation, vertical_deviation)
        else:
            perspective_score = 0
        
        return {
            'line_count': len(line_angles),
            'angle_variance': float(angle_variance),
            'perspective_score': float(perspective_score),  # 낮을수록 더 정직한 문서
            'has_strong_lines': len(line_angles) > 10
        }
    
    def analyze_document_structure(self, image_path: str) -> Dict:
        """문서 구조 분석 - 텍스트 영역, 여백 등"""
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. 텍스트 영역 추정 (어두운 영역 = 텍스트)
        # 적응형 임계값으로 텍스트 영역 분리
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        text_pixels = np.sum(adaptive_thresh == 0)  # 검은색 픽셀 = 텍스트
        text_ratio = text_pixels / (h * w)
        
        # 2. 여백 분석 (가장자리 영역의 밝기)
        border_size = min(w, h) // 20  # 이미지 크기의 5%
        top_border = np.mean(gray[:border_size, :])
        bottom_border = np.mean(gray[-border_size:, :])
        left_border = np.mean(gray[:, :border_size])
        right_border = np.mean(gray[:, -border_size:])
        avg_border_brightness = np.mean([top_border, bottom_border, left_border, right_border])
        
        # 3. 중앙 vs 가장자리 밝기 차이
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)
        brightness_contrast = abs(center_brightness - avg_border_brightness)
        
        # 4. 수평/수직 투영으로 텍스트 라인 감지
        horizontal_projection = np.sum(adaptive_thresh == 0, axis=1)
        vertical_projection = np.sum(adaptive_thresh == 0, axis=0)
        
        # 텍스트 라인 수 추정 (피크 감지)
        from scipy.signal import find_peaks
        peaks_h, _ = find_peaks(horizontal_projection, height=w*0.05)  # 최소 높이 조건
        estimated_text_lines = len(peaks_h)
        
        return {
            'text_ratio': float(text_ratio),
            'avg_border_brightness': float(avg_border_brightness),
            'center_brightness': float(center_brightness),
            'brightness_contrast': float(brightness_contrast),
            'estimated_text_lines': int(estimated_text_lines),
            'document_completeness': float(min(1.0, text_ratio * 2))  # 문서 완성도 추정
        }
    
    def calculate_challenge_score(self, image_analysis: Dict) -> float:
        """이미지의 도전적 정도를 0-1 스케일로 계산"""
        score_components = []
        
        # 1. 블러 점수 (낮을수록 더 도전적)
        blur_score = image_analysis.get('blur_score', 100)
        blur_challenge = max(0, (100 - blur_score) / 100)  # 100 이하일 때 도전적
        score_components.append(blur_challenge * 0.2)
        
        # 2. 밝기 극값 (너무 어둡거나 밝으면 도전적)
        brightness = image_analysis.get('brightness', 128)
        brightness_challenge = abs(brightness - 128) / 128  # 128에서 멀수록 도전적
        score_components.append(brightness_challenge * 0.15)
        
        # 3. 대비 (너무 낮으면 도전적)
        contrast = image_analysis.get('contrast', 50)
        contrast_challenge = max(0, (50 - contrast) / 50)
        score_components.append(contrast_challenge * 0.15)
        
        # 4. 원근 왜곡 (높을수록 도전적)
        perspective_score = image_analysis.get('perspective_score', 0)
        perspective_challenge = min(1.0, perspective_score / 30)  # 30도 이상이면 최대 도전적
        score_components.append(perspective_challenge * 0.25)
        
        # 5. 문서 완성도 (낮을수록 도전적)
        completeness = image_analysis.get('document_completeness', 1.0)
        completeness_challenge = 1.0 - completeness
        score_components.append(completeness_challenge * 0.15)
        
        # 6. 텍스트 비율 (너무 낮거나 높으면 도전적)
        text_ratio = image_analysis.get('text_ratio', 0.5)
        optimal_text_ratio = 0.3  # 최적 텍스트 비율
        text_challenge = abs(text_ratio - optimal_text_ratio) / optimal_text_ratio
        score_components.append(min(1.0, text_challenge) * 0.1)
        
        total_challenge_score = sum(score_components)
        return min(1.0, total_challenge_score)
    
    def analyze_all_test_images(self) -> pd.DataFrame:
        """모든 테스트 이미지 분석"""
        print("🔍 모든 테스트 이미지 분석 중...")
        
        test_files = list(self.test_dir.glob('*.jpg')) + list(self.test_dir.glob('*.png'))
        
        if not test_files:
            raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {self.test_dir}")
        
        results = []
        
        for i, img_path in enumerate(test_files):
            if i % 50 == 0:  # 진행 상황 출력
                print(f"   진행: {i}/{len(test_files)}")
            
            try:
                # 기본 이미지 분석
                quality_analysis = self.analyze_image_quality(str(img_path))
                perspective_analysis = self.detect_perspective_distortion(str(img_path))
                structure_analysis = self.analyze_document_structure(str(img_path))
                
                # 모든 분석 결과 결합
                combined_analysis = {
                    'filename': img_path.name,
                    'file_path': str(img_path),
                    **quality_analysis,
                    **perspective_analysis,
                    **structure_analysis
                }
                
                # 도전 점수 계산
                combined_analysis['challenge_score'] = self.calculate_challenge_score(combined_analysis)
                
                results.append(combined_analysis)
                
            except Exception as e:
                print(f"⚠️ 분석 실패: {img_path.name} - {e}")
                continue
        
        df_results = pd.DataFrame(results)
        print(f"✅ {len(df_results)}개 이미지 분석 완료")
        
        return df_results
    
    def select_representative_samples(self, df_analysis: pd.DataFrame, 
                                    n_samples: int = 20) -> pd.DataFrame:
        """대표적인 샘플 선택 (다양한 난이도와 특성을 고려)"""
        print(f"🎯 대표 샘플 {n_samples}개 선택 중...")
        
        selected_samples = []
        
        # 1. 도전적 난이도별 샘플 (40%)
        challenge_samples = int(n_samples * 0.4)
        
        # 매우 도전적 (상위 20%)
        very_challenging = df_analysis.nlargest(len(df_analysis)//5, 'challenge_score')
        selected_samples.extend(very_challenging.sample(min(challenge_samples//2, len(very_challenging))).to_dict('records'))
        
        # 중간 도전적 (중간 30%)
        medium_challenging = df_analysis[(df_analysis['challenge_score'] >= df_analysis['challenge_score'].quantile(0.35)) & 
                                       (df_analysis['challenge_score'] <= df_analysis['challenge_score'].quantile(0.65))]
        selected_samples.extend(medium_challenging.sample(min(challenge_samples//2, len(medium_challenging))).to_dict('records'))
        
        # 2. 특정 문제 유형별 샘플 (60%)
        remaining_samples = n_samples - len(selected_samples)
        
        # 블러 문제
        blurry_samples = df_analysis.nsmallest(len(df_analysis)//4, 'blur_score')
        if len(blurry_samples) > 0:
            selected_samples.extend(blurry_samples.sample(min(2, len(blurry_samples))).to_dict('records'))
        
        # 밝기 문제 (너무 어둡거나 밝음)
        brightness_issues = df_analysis[(df_analysis['brightness'] < 80) | (df_analysis['brightness'] > 180)]
        if len(brightness_issues) > 0:
            selected_samples.extend(brightness_issues.sample(min(2, len(brightness_issues))).to_dict('records'))
        
        # 원근 왜곡 문제
        perspective_issues = df_analysis.nlargest(len(df_analysis)//4, 'perspective_score')
        if len(perspective_issues) > 0:
            selected_samples.extend(perspective_issues.sample(min(3, len(perspective_issues))).to_dict('records'))
        
        # 문서 불완전성 문제
        incomplete_docs = df_analysis.nsmallest(len(df_analysis)//4, 'document_completeness')
        if len(incomplete_docs) > 0:
            selected_samples.extend(incomplete_docs.sample(min(2, len(incomplete_docs))).to_dict('records'))
        
        # 나머지는 랜덤 다양성 확보
        already_selected = {s['filename'] for s in selected_samples}
        remaining_df = df_analysis[~df_analysis['filename'].isin(already_selected)]
        if len(remaining_df) > 0 and len(selected_samples) < n_samples:
            additional_needed = n_samples - len(selected_samples)
            selected_samples.extend(remaining_df.sample(min(additional_needed, len(remaining_df))).to_dict('records'))
        
        # 중복 제거 및 최종 선택
        unique_samples = []
        seen_filenames = set()
        
        for sample in selected_samples:
            if sample['filename'] not in seen_filenames:
                unique_samples.append(sample)
                seen_filenames.add(sample['filename'])
            
            if len(unique_samples) >= n_samples:
                break
        
        result_df = pd.DataFrame(unique_samples)
        print(f"✅ {len(result_df)}개 대표 샘플 선택 완료")
        
        return result_df
    def create_sample_gallery(self, selected_samples: pd.DataFrame) -> Figure:
        """선택된 샘플들의 갤러리 생성"""
        """선택된 샘플들의 갤러리 생성"""
        n_samples = len(selected_samples)
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx, sample) in enumerate(selected_samples.iterrows()):
            if i >= n_rows * n_cols:
                break
                
            row, col = i // n_cols, i % n_cols
            
            # 이미지 로드
            img = cv2.imread(sample['file_path'])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img)
                
                # 제목 설정 (주요 특성 표시)
                title = f"{sample['filename']}\n"
                title += f"도전점수: {sample['challenge_score']:.2f}\n"
                title += f"블러: {sample['blur_score']:.0f}, "
                title += f"원근: {sample['perspective_score']:.0f}"
                
                axes[row, col].set_title(title, fontsize=9)
            else:
                axes[row, col].text(0.5, 0.5, 'Load\nError', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
            
            axes[row, col].axis('off')
        
        # 빈 subplot들 숨기기
        for i in range(n_samples, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('선택된 대표 테스트 샘플들', fontsize=16)
        plt.tight_layout()
        return fig
    
    def generate_selection_report(self, analysis_df: pd.DataFrame, 
                                selected_df: pd.DataFrame) -> str:
        """선택 보고서 생성"""
        report_path = self.output_dir / 'test_image_selection_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 테스트 이미지 선택 보고서\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"## 전체 분석 결과\n")
            f.write(f"- 전체 테스트 이미지: {len(analysis_df)}개\n")
            f.write(f"- 선택된 대표 샘플: {len(selected_df)}개\n\n")
            
            f.write(f"## 도전 점수 분포\n")
            f.write(f"- 평균 도전 점수: {analysis_df['challenge_score'].mean():.3f}\n")
            f.write(f"- 최고 도전 점수: {analysis_df['challenge_score'].max():.3f}\n")
            f.write(f"- 최저 도전 점수: {analysis_df['challenge_score'].min():.3f}\n\n")
            
            f.write(f"## 선택된 샘플의 특성\n")
            f.write(f"- 평균 도전 점수: {selected_df['challenge_score'].mean():.3f}\n")
            f.write(f"- 블러 문제 샘플: {len(selected_df[selected_df['blur_score'] < 50])}개\n")
            f.write(f"- 원근 왜곡 샘플: {len(selected_df[selected_df['perspective_score'] > 10])}개\n")
            f.write(f"- 밝기 문제 샘플: {len(selected_df[(selected_df['brightness'] < 80) | (selected_df['brightness'] > 180)])}개\n\n")
            
            f.write(f"## 선택된 파일 목록\n")
            for idx, sample in selected_df.iterrows():
                f.write(f"- {sample['filename']}: 도전점수 {sample['challenge_score']:.3f}\n")
            
            f.write(f"\n## 사용 방법\n")
            f.write(f"1. 이 파일들을 시각적 검증에 활용\n")
            f.write(f"2. 증강 기법의 효과를 이 샘플들로 테스트\n")
            f.write(f"3. 모델 성능 개선 시 이 샘플들로 검증\n")
        
        return str(report_path)
    
    def run_comprehensive_analysis(self, n_samples: int = 20):
        """종합 테스트 이미지 분석 실행"""
        print("🚀 종합 테스트 이미지 분석 시작...")
        
        # 1. 모든 테스트 이미지 분석
        analysis_df = self.analyze_all_test_images()
        
        # 2. 대표 샘플 선택
        selected_df = self.select_representative_samples(analysis_df, n_samples)
        
        # 3. 결과 저장
        analysis_path = self.output_dir / 'full_test_analysis.csv'
        analysis_df.to_csv(analysis_path, index=False)
        
        selected_path = self.output_dir / 'selected_representative_samples.csv'
        selected_df.to_csv(selected_path, index=False)
        
        # 4. 갤러리 생성
        print("🖼️ 선택된 샘플 갤러리 생성 중...")
        gallery_fig = self.create_sample_gallery(selected_df)
        gallery_path = self.output_dir / 'representative_samples_gallery.png'
        gallery_fig.savefig(gallery_path, dpi=300, bbox_inches='tight')
        plt.close(gallery_fig)
        
        # 5. 보고서 생성
        report_path = self.generate_selection_report(analysis_df, selected_df)
        
        print("\n✅ 종합 분석 완료!")
        print(f"📊 전체 분석: {analysis_path}")
        print(f"🎯 선택된 샘플: {selected_path}")
        print(f"🖼️ 갤러리: {gallery_path}")
        print(f"📄 보고서: {report_path}")
        
        return {
            'full_analysis': str(analysis_path),
            'selected_samples': str(selected_path),
            'gallery': str(gallery_path),
            'report': report_path
        }


def main():
    """메인 함수 - Fire CLI 인터페이스"""
    fire.Fire(TestImageAnalyzer)


if __name__ == "__main__":
    main()