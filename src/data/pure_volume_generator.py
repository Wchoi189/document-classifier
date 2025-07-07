# src/data/pure_volume_generator.py
"""
Pure Volume Dataset Generator - No Rotation Hypothesis Test
순수 볼륨 증강 (회전 없음) - 가설 검증용 데이터셋 생성기
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from tqdm import tqdm
from icecream import ic
import fire
from collections import defaultdict
import shutil
from datetime import datetime


class PureVolumeGenerator:
    """회전 없는 순수 볼륨 증강 데이터셋 생성기"""
    
    def __init__(self, 
                 source_dir: str = "data/raw",
                 output_base_dir: str = "data/augmented_datasets",
                 csv_file: str = "data/raw/metadata/train.csv",
                 meta_file: str = "data/raw/metadata/meta.csv"):
        
        self.source_dir = Path(source_dir)
        self.output_base_dir = Path(output_base_dir)
        self.csv_file = csv_file
        self.meta_file = meta_file
        
        # 데이터 로드
        self.df = pd.read_csv(csv_file)
        self.meta_df = pd.read_csv(meta_file)
        
        # 클래스 정보
        self.class_info = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
        self.class_distribution = self.df['target'].value_counts().sort_index()
        
        ic("🚀 Pure Volume Generator 초기화")
        ic(f"원본 샘플 수: {len(self.df)}")
        ic(f"클래스 수: {len(self.class_info)}")
        # ic(f"클래스 분포:\n{self.class_distribution.to_dict()}")
        ic(self.df.columns)

    def get_pure_volume_augmentation(self) -> A.Compose:
        """
        순수 볼륨 증강 파이프라인 (회전 없음)
        
        Focus Areas (from corruption analysis):
        - Brightness/Contrast: Test data 11.2% brighter
        - Noise: Switch from impulse to gaussian (test data dominant)
        - Quality: Blur, compression variations
        - NO ROTATION: Avoid artificial rotation artifacts
        """
        
        return A.Compose([
            # 🔆 조명 개선 (Test data 11.2% brighter than train)
            A.OneOf([
                # A.RandomBrightnessContrast(
                #     brightness_limit=0.15,  # 밝기 조정 (테스트 데이터 매칭)
                #     contrast_limit=0.15,
                #     p=0.8
                # ),
                A.RandomGamma(
                    gamma_limit=(85, 115),  # 감마 보정
                    p=0.6
                ),
                A.CLAHE(
                    clip_limit=2.0,  # 대비 제한 적응형 히스토그램 균등화
                    tile_grid_size=(8, 8),
                    p=0.4
                )
            ], p=0.7),
            
            # 🔊 노이즈 타입 전환 (Impulse → Gaussian)
            # A.OneOf([
            #     A.GaussNoise(
            #         mean_range=(10.0, 50.0),  # 테스트 데이터의 가우시안 노이즈 매칭
            #         mean=0,
            #         p=0.8
            #     ),
            #     A.ISONoise(
            #         color_shift=(0.01, 0.05),  # ISO 노이즈 시뮬레이션
            #         intensity=(0.1, 0.5),
            #         p=0.6
            #     ),
            #     A.MultiplicativeNoise(
            #         multiplier=[0.9, 1.1],
            #         per_channel=True,
            #         p=0.4
            #     )
            # ], p=0.6),
            
            # 📷 품질 변화 (자연스러운 스캔/촬영 조건)
            # A.OneOf([
            #     A.MotionBlur(
            #         blur_limit=3,  # 경미한 모션 블러
            #         p=0.5
            #     ),
            #     A.GaussianBlur(
            #         blur_limit=3,  # 경미한 가우시안 블러
            #         p=0.5
            #     ),
            #     A.MedianBlur(
            #         blur_limit=3,  # 노이즈 제거 효과
            #         p=0.3
            #     )
            # ], p=0.5),
            
            # 📄 압축/품질 아티팩트
            # A.OneOf([
            #     A.ImageCompression(
            #         quality_range=(85,100),  # 높은 품질 유지
            #         p=0.7
            #     ),
            #     A.Downscale(
            #        scale_range=(0.9,0.99),  # 경미한 해상도 감소
            #         interpolation_pair=cv2.INTER_LINEAR,
            #         p=0.5
            #     )
            # ], p=0.4),
            
            # 🎨 색상 미세 조정 (문서 스캔 조건 시뮬레이션)
            # A.OneOf([
            #     A.HueSaturationValue(
            #         hue_shift_limit=5,      # 미세한 색조 변화
            #         sat_shift_limit=10,     # 채도 조정
            #         val_shift_limit=5,      # 명도 조정
            #         p=0.6
            #     ),
                # A.RGBShift(
                #     r_shift_limit=10,
                #     g_shift_limit=10,
                #     b_shift_limit=10,
                #     p=0.5
                # ),
            #     A.ChannelShuffle(p=0.1)  # 극히 드물게 채널 섞기
            # ], p=0.3),
            
            # 🔄 최종 전처리
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            )
        ])

    def generate_pure_volume_dataset(self, 
                                   multiplier: int = 3,
                                   output_name: str = "pure_volume_3X_no_rotation") -> Path:
        """
        순수 볼륨 데이터셋 생성 (회전 없음)
        
        Args:
            multiplier: 데이터 증강 배수 (기본 20배)
            output_name: 출력 디렉토리 이름
        
        Returns:
            생성된 데이터셋 경로
        """
        
        ic(f"🎯 순수 볼륨 데이터셋 생성 시작 (배수: {multiplier})")
        
        # 출력 디렉토리 설정
        output_dir = self.output_base_dir / output_name
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        metadata_dir = output_dir / "metadata"
        
        # 디렉토리 생성
        for dir_path in [train_dir, val_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            ic(f"📁 디렉토리 생성: {dir_path}")
        
        # 클래스별 서브디렉토리 생성
        # for class_id in self.class_info.keys():
        #     (train_dir / str(class_id)).mkdir(exist_ok=True)
        #     (val_dir / str(class_id)).mkdir(exist_ok=True)
        
        # 증강 파이프라인 설정
        augmentation = self.get_pure_volume_augmentation()
        
        # 진행 상황 추적
        train_records = []
        val_records = []
        generation_stats = {
            'total_generated': 0,
            'train_generated': 0,
            'val_generated': 0,
            'class_distribution': defaultdict(int),
            'generation_time': datetime.now().isoformat(),
            'strategy': 'pure_volume_no_rotation',
            'multiplier': multiplier,
            'augmentation_details': {
                'rotation': False,
                'brightness_contrast': True,
                'noise_gaussian': True,
                'quality_variations': True,
                'color_adjustments': True
            }
        }
        
        # 클래스별 데이터 생성
        for class_id in tqdm(self.class_info.keys(), desc="클래스별 처리"):
            class_samples = self.df[self.df['target'] == class_id]
            original_count = len(class_samples)
            target_count = original_count * multiplier
            
            ic(f"🎲 클래스 {class_id} ({self.class_info[class_id]}): {original_count} → {target_count}")
            
            generated_count = 0
            
            # 각 원본 이미지에 대해 증강 생성
            for _, row in tqdm(class_samples.iterrows(), 
                             desc=f"클래스 {class_id} 증강", 
                             total=len(class_samples),
                             leave=False):
                
                # 원본 이미지 로드
                # img_path = self.source_dir / "train" / str(class_id) / row['ID']
                img_path = self.source_dir / "train" / row['ID']
                
                if not img_path.exists():
                    ic(f"⚠️ 이미지 없음: {img_path}")
                    continue
                
                image = cv2.imread(str(img_path))
                if image is None:
                    ic(f"⚠️ 이미지 로드 실패: {img_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 각 원본에 대해 multiplier만큼 증강 생성
                for aug_idx in range(multiplier):
                    try:
                        # 증강 적용
                        augmented = augmentation(image=image)
                        aug_image = augmented['image']
                        
                        # 텐서를 다시 이미지로 변환
                        if hasattr(aug_image, 'numpy'):
                            aug_image = aug_image.numpy()
                        
                        # 정규화 해제 및 이미지 형식 복원
                        if aug_image.dtype != np.uint8:
                            # 정규화 해제
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            
                            if len(aug_image.shape) == 3 and aug_image.shape[0] == 3:
                                # CHW → HWC 변환
                                aug_image = np.transpose(aug_image, (1, 2, 0))
                            
                            # 정규화 해제
                            aug_image = aug_image * std + mean
                            aug_image = np.clip(aug_image * 255, 0, 255).astype(np.uint8)
                        
                        # 저장 경로 결정 (80/20 분할)
                        is_validation = (generated_count % 5 == 4)  # 20% 검증용
                        base_dir = val_dir if is_validation else train_dir
                        
                        # 파일명 생성
                        base_name = Path(row['ID']).stem
                        new_filename = f"{base_name}_aug_{aug_idx:03d}.jpg"
                        save_path = base_dir / new_filename
                        
                        # 이미지 저장
                        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(save_path), aug_image_bgr, 
                                  [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        # 메타데이터 기록
                        record = {
                        'ID': new_filename,      # 'image_name' 키를 'ID'로 변경
                        'target': class_id  
                        }
                        if is_validation:
                            val_records.append(record)
                            generation_stats['val_generated'] += 1
                        else:
                            train_records.append(record)
                            generation_stats['train_generated'] += 1
                        
                        generated_count += 1
                        generation_stats['total_generated'] += 1
                        generation_stats['class_distribution'][class_id] += 1
                        
                    except Exception as e:
                        ic(f"⚠️ 증강 실패: {row['ID']}, 오류: {e}")
                        continue
        
        # 메타데이터 CSV 파일 저장
        train_df = pd.DataFrame(train_records)
        val_df = pd.DataFrame(val_records)
        
        train_csv_path = metadata_dir / "train.csv"
        val_csv_path = metadata_dir / "val.csv"
        
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)
        
        # 메타데이터 복사
        meta_csv_path = metadata_dir / "meta.csv"
        shutil.copy2(self.meta_file, meta_csv_path)
        
        # 생성 통계 저장
        stats_path = output_dir / "generation_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(generation_stats, f, indent=2, ensure_ascii=False)
        
        # 결과 요약
        ic("✅ 순수 볼륨 데이터셋 생성 완료!")
        ic(f"📊 총 생성: {generation_stats['total_generated']}")
        ic(f"🎯 훈련용: {generation_stats['train_generated']}")
        ic(f"🔍 검증용: {generation_stats['val_generated']}")
        ic(f"💾 저장 위치: {output_dir}")
        
        return output_dir

    def create_training_config(self, dataset_path: Path) -> Path:
        """순수 볼륨 데이터셋용 훈련 설정 파일 생성"""
        
        config_content = f"""# @package _global_
defaults:
  - _self_

name: "pure-volume-no-rotation-test"
description: "순수 볼륨 데이터셋 테스트 (회전 없음) - 3x 배수"
tags: ["pure-volume", "no-rotation", "hypothesis-test", "3X"]

seed: 42
device: 'cuda'

# Pure Volume Dataset
data:
  root_dir: "{dataset_path}"
  csv_file: "{dataset_path}/metadata/train.csv"
  meta_file: "{dataset_path}/metadata/meta.csv"
  val_csv_file: "{dataset_path}/metadata/val.csv"
  val_root_dir: "{dataset_path}/val"
  
  image_size: 224
  val_size: 0.0  # 별도 validation 파일 사용
  num_workers: 4
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# 런타임 증강 비활성화 (데이터에 이미 적용됨)
augmentation:
  enabled: false
  strategy: "none"
  intensity: 0.0

model:
  name: "resnet50"
  pretrained: true

train:
  epochs: 25  # 더 많은 데이터로 안정적 훈련
  batch_size: 32  # 더 큰 배치 크기
  mixed_precision: true
  
  early_stopping:
    patience: 8
    metric: 'val_f1'
    mode: 'max'

optimizer:
  name: 'AdamW'
  learning_rate: 0.0001
  weight_decay: 0.0001

scheduler:
  name: 'CosineAnnealingLR'
  T_max: 25

wandb:
  username: wchoi189
  enabled: true
  project: "document-classifier-pure-volume"
  name: "pure-volume-3X-no-rotation-test"
  tags: ["pure-volume", "no-rotation", "3X", "hypothesis-test"]
  notes: "순수 볼륨 증강 테스트 - 회전 없음으로 가설 검증"

paths:
  output_dir: "outputs/pure_volume_test"
  model_dir: "outputs/pure_volume_test/models"

logging:
  checkpoint_dir: "outputs/pure_volume_test/checkpoints"
"""
        
        # 설정 파일 저장
        config_path = Path("configs/experiment/pure_volume_no_rotation.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        ic(f"📄 훈련 설정 파일 생성: {config_path}")
        return config_path


def main():
    """CLI 진입점"""
    fire.Fire(PureVolumeGenerator)


if __name__ == "__main__":
    main()