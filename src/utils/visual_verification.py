"""
src/utils/visual_verification.py

시각적 검증 스크립트 - 증강된 훈련 이미지와 실제 테스트 조건 비교
Visual verification script - Compare augmented training images with real test conditions
"""

import os
import cv2
import numpy as np

from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random
from PIL import Image
import fire

from src.data.augmentation import get_configurable_transforms, get_train_transforms, get_valid_transforms
from src.utils.config_utils import load_config
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the font family to NanumGothic
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False



class VisualVerificationTool:
    """훈련 데이터 증강과 테스트 조건 비교를 위한 시각적 검증 도구"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.setup_paths()
        self.setup_transforms()
        
    def setup_paths(self):
        """경로 설정"""
        self.train_dir = Path(self.config['data']['root_dir']) / 'train'
        self.test_dir = Path(self.config['data']['root_dir']) / 'test'
        self.output_dir = Path('outputs/visual_verification')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 훈련 이미지 디렉토리: {self.train_dir}")
        print(f"✅ 테스트 이미지 디렉토리: {self.test_dir}")
        print(f"✅ 출력 디렉토리: {self.output_dir}")
    
    def setup_transforms(self):
        """다양한 증강 전략 설정"""
        img_size = self.config['data']['image_size']
        mean = self.config['data']['mean']
        std = self.config['data']['std']
        
        # 기본 증강 (기존)
        self.transform_basic = get_train_transforms(img_size, img_size, mean, std)
        
        # 문서 특화 증강 (약함)
        self.transform_document = get_configurable_transforms(
            img_size, img_size, mean, std, 
            {'strategy': 'document', 'intensity': 0.5}
        )
        
        # 강력한 증강 (테스트 조건 시뮬레이션)
        self.transform_robust = get_configurable_transforms(
            img_size, img_size, mean, std,
            {'strategy': 'robust', 'intensity': 0.8}
        )
        
        # 검증용 (증강 없음)
        self.transform_valid = get_valid_transforms(img_size, img_size, mean, std)
        
        print("✅ 모든 증강 전략 설정 완료")
    
    def load_sample_images(self, n_samples: int = 5) -> Tuple[List[np.ndarray], List[str]]:
        """훈련 이미지 샘플 로드"""
        train_files = list(self.train_dir.glob('*.jpg'))
        if len(train_files) == 0:
            train_files = list(self.train_dir.glob('*.png'))
        
        if len(train_files) == 0:
            raise FileNotFoundError(f"훈련 이미지를 찾을 수 없습니다: {self.train_dir}")
        
        # 랜덤 샘플링
        selected_files = random.sample(train_files, min(n_samples, len(train_files)))
        
        images = []
        filenames = []
        
        for file_path in selected_files:
            # OpenCV로 이미지 로드 (RGB 변환)
            img = cv2.imread(str(file_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                filenames.append(file_path.name)
                print(f"✅ 로드됨: {file_path.name} - 크기: {img.shape}")
            else:
                print(f"⚠️ 로드 실패: {file_path.name}")
        
        return images, filenames
    
    def load_test_samples(self, n_samples: int = 5) -> Tuple[List[np.ndarray], List[str]]:
        """테스트 이미지 샘플 로드"""
        test_files = list(self.test_dir.glob('*.jpg'))
        if len(test_files) == 0:
            test_files = list(self.test_dir.glob('*.png'))
        
        if len(test_files) == 0:
            print(f"⚠️ 테스트 이미지를 찾을 수 없습니다: {self.test_dir}")
            return [], []
        
        # 랜덤 샘플링
        selected_files = random.sample(test_files, min(n_samples, len(test_files)))
        
        images = []
        filenames = []
        
        for file_path in selected_files:
            img = cv2.imread(str(file_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                filenames.append(file_path.name)
                print(f"✅ 테스트 이미지 로드됨: {file_path.name}")
        
        return images, filenames
    
    def apply_augmentation(self, image: np.ndarray, transform) -> np.ndarray:
        """증강 적용 및 시각화를 위한 역정규화"""
        try:
            # Albumentations 적용
            augmented = transform(image=image)
            tensor_img = augmented['image']
            
            # 텐서를 numpy로 변환 (C, H, W) -> (H, W, C)
            if hasattr(tensor_img, 'numpy'):
                img_np = tensor_img.numpy().transpose(1, 2, 0)
            else:
                img_np = tensor_img.permute(1, 2, 0).numpy()
            
            # 역정규화 (정규화된 이미지를 원래 범위로)
            mean = np.array(self.config['data']['mean'])
            std = np.array(self.config['data']['std'])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            
            return img_np
            
        except Exception as e:
            print(f"❌ 증강 적용 실패: {e}")
            # 실패시 원본 이미지 반환 (0-1 범위로 정규화)
            return image.astype(np.float32) / 255.0
    
    def create_comparison_grid(self, 
                             original_images: List[np.ndarray], 
                             filenames: List[str],
                             test_images: Optional[List[np.ndarray]] = None,
                             test_filenames: Optional[List[str]] = None) -> plt.Figure:
        """비교 그리드 생성"""
        
        n_train = len(original_images)
        n_test = len(test_images) if test_images else 0
        
        # 그리드 크기 계산: 훈련 이미지당 4개 증강 + 테스트 이미지들
        cols = 5  # 원본 + 3개 증강 + 1개 여백
        rows = n_train + (1 if n_test > 0 else 0)  # 훈련 이미지 행들 + 테스트 이미지 행
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 훈련 이미지들과 증강 비교
        for i, (img, filename) in enumerate(zip(original_images, filenames)):
            # 원본 이미지
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'원본\n{filename}', fontsize=10)
            axes[i, 0].axis('off')
            
            # 기본 증강
            aug_basic = self.apply_augmentation(img, self.transform_basic)
            axes[i, 1].imshow(aug_basic)
            axes[i, 1].set_title('기본 증강', fontsize=10)
            axes[i, 1].axis('off')
            
            # 문서 증강
            aug_document = self.apply_augmentation(img, self.transform_document)
            axes[i, 2].imshow(aug_document)
            axes[i, 2].set_title('문서 증강', fontsize=10)
            axes[i, 2].axis('off')
            
            # 강력한 증강
            aug_robust = self.apply_augmentation(img, self.transform_robust)
            axes[i, 3].imshow(aug_robust)
            axes[i, 3].set_title('강력한 증강', fontsize=10)
            axes[i, 3].axis('off')
            
            # 빈 공간
            axes[i, 4].axis('off')
        
        # 테스트 이미지들 표시 (마지막 행)
        if n_test > 0:
            test_row = n_train
            for j in range(cols):
                if j < len(test_images):
                    axes[test_row, j].imshow(test_images[j])
                    axes[test_row, j].set_title(f'테스트 샘플\n{test_filenames[j]}', fontsize=10)
                    axes[test_row, j].axis('off')
                else:
                    axes[test_row, j].axis('off')
        
        plt.suptitle('훈련 데이터 증강 vs 실제 테스트 조건 비교', fontsize=16, y=0.98)
        plt.tight_layout()
        return fig
    
    def analyze_augmentation_coverage(self, 
                                    original_images: List[np.ndarray], 
                                    n_variations: int = 10) -> Dict:
        """증강 커버리지 분석 - 다양한 증강 결과 생성"""
        print(f"🔍 증강 커버리지 분석 중... (변형 {n_variations}개)")
        
        analysis_results = {
            'brightness_range': [],
            'blur_levels': [],
            'rotation_angles': [],
            'perspective_scores': []
        }
        
        for img in original_images:
            brightness_values = []
            blur_scores = []
            
            # 여러 번 증강 적용하여 다양성 측정
            for _ in range(n_variations):
                aug_img = self.apply_augmentation(img, self.transform_robust)
                
                # 밝기 측정
                brightness = np.mean(aug_img)
                brightness_values.append(brightness)
                
                # 블러 정도 측정 (라플라시안 분산)
                gray = cv2.cvtColor((aug_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(blur_score)
            
            analysis_results['brightness_range'].append((min(brightness_values), max(brightness_values)))
            analysis_results['blur_levels'].append((min(blur_scores), max(blur_scores)))
        
        return analysis_results
    
    def generate_comprehensive_report(self, n_train_samples: int = 5, n_test_samples: int = 5):
        """종합 시각적 검증 보고서 생성"""
        print("🚀 종합 시각적 검증 보고서 생성 중...")
        
        # 1. 훈련 이미지 로드
        print("\n📥 훈련 이미지 로드 중...")
        train_images, train_filenames = self.load_sample_images(n_train_samples)
        
        # 2. 테스트 이미지 로드
        print("\n📥 테스트 이미지 로드 중...")
        test_images, test_filenames = self.load_test_samples(n_test_samples)
        
        # 3. 비교 그리드 생성
        print("\n📊 비교 그리드 생성 중...")
        comparison_fig = self.create_comparison_grid(
            train_images, train_filenames, 
            test_images, test_filenames
        )
        
        # 4. 결과 저장
        output_path = self.output_dir / 'augmentation_vs_test_comparison.png'
        comparison_fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(comparison_fig)
        
        # 5. 증강 커버리지 분석
        print("\n🔍 증강 커버리지 분석 중...")
        coverage_analysis = self.analyze_augmentation_coverage(train_images)
        
        # 6. 분석 결과 요약 생성
        self.create_analysis_summary(coverage_analysis, len(train_images), len(test_images))
        
        print(f"\n✅ 보고서 생성 완료!")
        print(f"📁 출력 파일: {output_path}")
        print(f"📁 요약 파일: {self.output_dir / 'analysis_summary.txt'}")
        
        return str(output_path)
    
    def create_analysis_summary(self, coverage_analysis: Dict, n_train: int, n_test: int):
        """분석 요약 텍스트 파일 생성"""
        summary_path = self.output_dir / 'analysis_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 시각적 검증 분석 요약\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"## 기본 정보\n")
            f.write(f"- 분석된 훈련 이미지: {n_train}개\n")
            f.write(f"- 분석된 테스트 이미지: {n_test}개\n")
            f.write(f"- 증강 전략: basic, document, robust\n\n")
            
            f.write(f"## 증강 커버리지 분석\n")
            if coverage_analysis['brightness_range']:
                brightness_ranges = coverage_analysis['brightness_range']
                min_brightness = min([r[0] for r in brightness_ranges])
                max_brightness = max([r[1] for r in brightness_ranges])
                f.write(f"- 밝기 변화 범위: {min_brightness:.3f} ~ {max_brightness:.3f}\n")
            
            if coverage_analysis['blur_levels']:
                blur_ranges = coverage_analysis['blur_levels']
                min_blur = min([r[0] for r in blur_ranges])
                max_blur = max([r[1] for r in blur_ranges])
                f.write(f"- 블러 레벨 범위: {min_blur:.1f} ~ {max_blur:.1f}\n")
            
            f.write(f"\n## 권장사항\n")
            f.write(f"1. 생성된 비교 이미지를 확인하여 테스트 조건과 일치하는지 검토\n")
            f.write(f"2. 증강 강도가 부족하면 config.yaml에서 intensity 값 증가\n")
            f.write(f"3. 증강이 과도하면 intensity 값 감소\n")
            f.write(f"4. 특정 증강 기법이 효과적이면 해당 전략 사용\n")


def main():
    """메인 함수 - Fire CLI 인터페이스"""
    fire.Fire(VisualVerificationTool)


if __name__ == "__main__":
    main()