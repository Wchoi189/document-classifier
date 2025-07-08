# src/data/dataset_multiplier.py
"""
Dataset Multiplication Engine
대용량 증강 데이터셋 생성기 - 체계적이고 효율적인 데이터 확장
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


class DatasetMultiplier:
    """체계적인 데이터셋 증강 및 저장 시스템"""
    
    def __init__(self, 
                 source_dir: str = "data/raw",
                 output_base_dir: str = "data/augmented_datasets",
                 csv_file: str = "data/raw/metadata/train.csv",
                 meta_file: str = "data/raw/metadata/meta.csv"):
        """
        초기화
        Args:
            source_dir: 원본 이미지 디렉토리
            output_base_dir: 증강 데이터셋 저장 기본 디렉토리
            csv_file: 원본 CSV 파일
            meta_file: 메타데이터 파일
        """
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
        
        ic("📊 Dataset Multiplier 초기화 완료")
        ic(f"원본 샘플 수: {len(self.df)}")
        ic(f"클래스 수: {len(self.class_info)}")
        ic(f"클래스별 분포: {dict(self.class_distribution.head())}")

    def generate_progressive_datasets(self, base_multiplier: int = 10):
        """Progressive training용 3단계 데이터셋 생성"""
        
        progressive_configs = [
            ("phase1_mild_20deg", "phase1_mild", base_multiplier),
            ("phase2_variety_60deg", "phase2_variety", base_multiplier),
            ("phase3_full_90deg", "phase3_full", base_multiplier)
        ]
        
        results = {}
        
        for dataset_name, strategy, multiplier in progressive_configs:
            ic(f"\n🎯 {dataset_name} 생성 시작")
            try:
                output_path = self.save_augmented_dataset(dataset_name, strategy, multiplier)
                results[dataset_name] = {
                    'status': 'success',
                    'path': output_path,
                    'strategy': strategy,
                    'multiplier': multiplier
                }
                ic(f"✅ {dataset_name} 완료")
            except Exception as e:
                ic(f"❌ {dataset_name} 실패: {e}")
                results[dataset_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results   
        
    def create_augmentation_strategy(self, strategy_name: str, intensity: float = 0.7) -> A.Compose:
        """증강 전략 생성"""
        if strategy_name == "phase1_mild":
            # Phase 1: Mild Acclimation (±20°)
            return A.Compose([
                A.Rotate(
                    limit=20, 
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(255, 255, 255), 
                    p=0.8
                ),
                A.GaussNoise(
                    std_range=(0.01, 0.05), 
                    p=0.3
                ),
            ])
            
        elif strategy_name == "phase2_variety":
            # Phase 2: Introduce Variety (±60° + discrete 90°)
            return A.Compose([
                A.OneOf([
                    # Continuous rotations
                    A.Rotate(
                        limit=60, 
                        border_mode=cv2.BORDER_CONSTANT, 
                        fill=(255, 255, 255), 
                        p=1.0
                    ),
                    # Discrete 90° rotations
                    A.RandomRotate90(p=1.0),
                ], p=0.9),               
                A.Perspective(
                    scale=(0.01, 0.04), 
                    keep_size=True, 
                    p=0.3
                ),
            ])
            
        elif strategy_name == "phase3_full":
            # Phase 3: Full Robustness (±90°)
            return A.Compose([
                A.OneOf([
                    # Full range continuous rotations
                    A.Rotate(
                        limit=90, 
                        border_mode=cv2.BORDER_CONSTANT, 
                        fill=(255, 255, 255), 
                        p=1.0
                    ),
                    # Discrete 90° rotations
                    A.RandomRotate90(p=1.0),
                    # Horizontal/Vertical flips
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ], p=0.95),                
                A.Perspective(
                    scale=(0.01, 0.04), 
                    keep_size=True, 
                    p=0.3
                ),
            ])
               
        elif strategy_name == "volume_focused":
            # V1: 대용량 생성 - 다양한 변형
            return A.Compose([
                A.OneOf([
                    A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1.0), # Added parameter
                    A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1.0), # Added parameter
                    A.Rotate(limit=60, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1.0), # Added parameter
                    A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1.0), # Added parameter
                ], p=0.8),
                A.Perspective(
                    scale=(0.01, 0.04),
                    keep_size=True,
                    fill=(255, 255, 255),  # Added parameter
                    p=0.4
                ),
            ])
            
        elif strategy_name == "test_focused":
            # V2: 테스트 조건 시뮬레이션
            return A.Compose([
            # 회전 중심 (554% 차이 해결)
            A.Rotate(
            limit=25, 
            border_mode=cv2.BORDER_CONSTANT, 
            p=0.9
            ),
            
            
            # 원근 왜곡
            A.Perspective(
            scale=(0.05, 0.15), 
            keep_size=True, 
            p=0.5
            ),
            ])
        elif strategy_name == "balanced":
            # V3: 클래스 균형 - 보수적 증강
            return A.Compose([
            A.Rotate(
                limit=20, 
                border_mode=cv2.BORDER_CONSTANT, 
                p=0.7
            ),
            A.Perspective(
                scale=(0.01, 0.04), 
                keep_size=True, 
                p=0.3
            ),
            ])
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def calculate_multiplication_targets(self, strategy: str, target_multiplier: int) -> Dict[int, int]:
        """클래스별 증강 목표 계산"""
        targets = {}
        
        if strategy == "balanced":
            # 모든 클래스를 최대 클래스 크기로 맞춤
            max_class_size = self.class_distribution.max()
            target_size = max_class_size * target_multiplier
            
            for class_id, current_size in self.class_distribution.items():
                targets[class_id] = target_size
                
        else:
            # 균등 증강
            for class_id, current_size in self.class_distribution.items():
                targets[class_id] = current_size * target_multiplier
        
        return targets
    
    def generate_augmented_samples(self, 
                                 image_path: str, 
                                 transform: A.Compose, 
                                 num_augmentations: int,
                                 base_filename: str) -> List[Tuple[np.ndarray, str]]:
        """단일 이미지에서 여러 증강 샘플 생성"""
        
        # 원본 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            ic(f"⚠️ 이미지 로드 실패: {image_path}")
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_samples = []
        
        # 원본 포함
        augmented_samples.append((image, f"{base_filename}_original.jpg"))
        
        # 증강 샘플 생성
        for i in range(num_augmentations - 1):  # -1 because we include original
            try:
                augmented = transform(image=image)
                aug_image = augmented['image']
                aug_filename = f"{base_filename}_aug_{i:03d}.jpg"
                augmented_samples.append((aug_image, aug_filename))
            except Exception as e:
                ic(f"⚠️ 증강 실패: {base_filename}, aug {i}: {e}")
                continue
        
        return augmented_samples
    
    def save_augmented_dataset(self, 
                             dataset_name: str,
                             strategy: str,
                             target_multiplier: int,
                             batch_size: int = 100) -> str:
        """증강 데이터셋 생성 및 저장"""
        
        ic(f"🚀 {dataset_name} 데이터셋 생성 시작")
        ic(f"전략: {strategy}, 배수: {target_multiplier}x")
        
        # 출력 디렉토리 설정
        output_dir = self.output_base_dir / dataset_name
        images_dir = output_dir / "train"
        metadata_dir = output_dir / "metadata"
        
        # 디렉토리 생성
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 증강 전략 및 목표 설정
        transform = self.create_augmentation_strategy(strategy)
        multiplication_targets = self.calculate_multiplication_targets(strategy, target_multiplier)
        
        # 새 CSV 데이터 준비
        new_csv_data = []
        generation_stats = defaultdict(int)
        
        # 클래스별 처리
        total_classes = len(self.class_distribution)
        
        for class_idx, (class_id, current_count) in enumerate(self.class_distribution.items()):
            target_count = multiplication_targets[class_id]
            augmentations_per_image = target_count // current_count
            
            ic(f"클래스 {class_id} ({self.class_info[class_id]}): "
               f"{current_count} → {target_count} ({augmentations_per_image}x)")
            
            # 해당 클래스 이미지들 가져오기
            class_images = self.df[self.df['target'] == class_id]
            
            # 배치 처리
            batch_count = 0
            batch_images = []
            
            with tqdm(total=len(class_images), 
                     desc=f"클래스 {class_id} 처리중",
                     leave=False) as pbar:
                
                for _, row in class_images.iterrows():
                    source_image_path = self.source_dir / "train" / row['ID']
                    base_filename = Path(row['ID']).stem
                    
                    # 증강 샘플 생성
                    augmented_samples = self.generate_augmented_samples(
                        str(source_image_path),
                        transform,
                        augmentations_per_image,
                        f"class_{class_id}_{base_filename}"
                    )
                    
                    # 배치에 추가
                    for aug_image, aug_filename in augmented_samples:
                        batch_images.append((aug_image, aug_filename, class_id))
                        
                        # CSV 데이터 추가
                        new_csv_data.append({
                            'ID': aug_filename,
                            'target': class_id
                        })
                    
                    # 배치 저장
                    if len(batch_images) >= batch_size:
                        self._save_batch(batch_images, images_dir)
                        generation_stats[class_id] += len(batch_images)
                        batch_images = []
                        batch_count += 1
                    
                    pbar.update(1)
                
                # 남은 배치 저장
                if batch_images:
                    self._save_batch(batch_images, images_dir)
                    generation_stats[class_id] += len(batch_images)
        
        # 메타데이터 저장
        self._save_metadata(new_csv_data, metadata_dir, dataset_name, generation_stats)
        
        # 생성 요약
        total_generated = sum(generation_stats.values())
        ic(f"✅ {dataset_name} 데이터셋 생성 완료")
        ic(f"총 생성 샘플: {total_generated:,}")
        ic(f"저장 위치: {output_dir}")
        
        return str(output_dir)
    
    def _save_batch(self, batch_images: List[Tuple[np.ndarray, str, int]], output_dir: Path):
        """이미지 배치 저장"""
        for image, filename, class_id in batch_images:
            output_path = output_dir / filename
            
            # RGB to BGR for cv2
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)
    
    def _save_metadata(self, csv_data: List[Dict], metadata_dir: Path, 
                      dataset_name: str, stats: Dict[int, int]):
        """메타데이터 파일들 저장"""
        
        # 새 train.csv 저장
        new_df = pd.DataFrame(csv_data)
        train_csv_path = metadata_dir / "train.csv"
        new_df.to_csv(train_csv_path, index=False)
        
        # 원본 meta.csv 복사
        meta_csv_path = metadata_dir / "meta.csv"
        shutil.copy2(self.meta_file, meta_csv_path)
        
        # 생성 통계 저장
        stats_path = metadata_dir / "generation_stats.json"
        generation_info = {
            'dataset_name': dataset_name,
            'generation_timestamp': datetime.now().isoformat(),
            'total_samples': sum(stats.values()),
            'original_samples': len(self.df),
            'multiplication_factor': sum(stats.values()) / len(self.df),
            'class_statistics': {
                str(class_id): {
                    'generated_count': count,
                    'class_name': self.class_info[class_id]
                } for class_id, count in stats.items()
            }
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, indent=2, ensure_ascii=False)
        
        ic(f"📋 메타데이터 저장: {metadata_dir}")
    
    def generate_all_variants(self):
        """모든 데이터셋 변형 생성"""
        
        variants = [
            ("v1_volume_20x", "volume_focused", 20),
            ("v2_test_focused_10x", "test_focused", 10),
            ("v3_balanced_15x", "balanced", 15),
            ("v3_double_2x", "double", 2)
        ]
        
        results = {}
        
        for dataset_name, strategy, multiplier in variants:
            ic(f"\n🎯 {dataset_name} 생성 시작")
            try:
                output_path = self.save_augmented_dataset(dataset_name, strategy, multiplier)
                results[dataset_name] = {
                    'status': 'success',
                    'path': output_path,
                    'strategy': strategy,
                    'multiplier': multiplier
                }
                ic(f"✅ {dataset_name} 완료")
            except Exception as e:
                ic(f"❌ {dataset_name} 실패: {e}")
                results[dataset_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 전체 요약 저장
        summary_path = self.output_base_dir / "generation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        ic(f"🎉 모든 변형 생성 완료. 요약: {summary_path}")
        return results
    def generate_stratified_kfold_datasets(self, k: int = 5, multiplier: int = 10, 
                                        strategy: str = "phase1_mild"):
        """
        Stratified K-fold 데이터셋 생성 - 소스 레벨에서 분할 후 증강
        
        Args:
            k: 폴드 수
            multiplier: 증강 배수
            strategy: 증강 전략
        """
        ic(f"🎯 Stratified {k}-fold 데이터셋 생성 시작")
        ic(f"전략: {strategy}, 배수: {multiplier}x")
        
        from sklearn.model_selection import StratifiedKFold
        
        # 클래스별로 소스 이미지 그룹화
        source_groups = defaultdict(list)
        for _, row in self.df.iterrows():
            source_groups[row['target']].append(row['ID'])
        
        # 각 클래스에서 소스 이미지 기준으로 K-fold 분할
        kfold_splits = []
        
        for fold_idx in range(k):
            train_sources = []
            val_sources = []
            
            for class_id, sources in source_groups.items():
                # 클래스 내에서 K-fold 분할
                fold_size = len(sources) // k
                start_idx = fold_idx * fold_size
                end_idx = start_idx + fold_size if fold_idx < k-1 else len(sources)
                
                # 현재 폴드를 validation으로
                val_sources.extend(sources[start_idx:end_idx])
                # 나머지를 training으로
                train_sources.extend(sources[:start_idx] + sources[end_idx:])
            
            kfold_splits.append({
                'fold': fold_idx,
                'train_sources': train_sources,
                'val_sources': val_sources
            })
        
        ic(f"✅ {k}개 폴드 분할 완료")
        
        # 각 폴드별 데이터셋 생성
        results = {}
        
        for split_info in kfold_splits:
            fold_idx = split_info['fold']
            dataset_name = f"{strategy}_fold_{fold_idx}"
            
            ic(f"\n📁 Fold {fold_idx} 데이터셋 생성 중...")
            
            try:
                fold_path = self._generate_fold_dataset(
                    split_info, 
                    dataset_name, 
                    strategy, 
                    multiplier
                )
                
                results[dataset_name] = {
                    'status': 'success',
                    'path': fold_path,
                    'fold': fold_idx,
                    'train_sources': len(split_info['train_sources']),
                    'val_sources': len(split_info['val_sources'])
                }
                ic(f"✅ Fold {fold_idx} 완료: {fold_path}")
                
            except Exception as e:
                ic(f"❌ Fold {fold_idx} 실패: {e}")
                results[dataset_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'fold': fold_idx
                }
        
        return results

    def _generate_fold_dataset(self, split_info: Dict, dataset_name: str, 
                            strategy: str, multiplier: int) -> str:
        """단일 폴드 데이터셋 생성"""
        
        # 출력 디렉토리 설정
        output_dir = self.output_base_dir / dataset_name
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        metadata_dir = output_dir / "metadata"
        
        for dir_path in [train_dir, val_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 증강 전략 생성
        transform = self.create_augmentation_strategy(strategy)
        
        # Train/Val CSV 데이터 준비
        train_csv_data = []
        val_csv_data = []
        
        # Training set 생성 (증강 적용)
        ic(f"📊 Training set 생성 중 ({len(split_info['train_sources'])} 소스)")
        for source_filename in tqdm(split_info['train_sources'], desc="Training 증강"):
            # 원본 데이터에서 해당 소스 찾기
            source_row = self.df[self.df['ID'] == source_filename].iloc[0]
            source_path = self.source_dir / "train" / source_filename
            
            if not source_path.exists():
                ic(f"⚠️ 소스 파일 없음: {source_path}")
                continue
            
            # 증강 샘플 생성
            base_filename = Path(source_filename).stem
            class_id = source_row['target']
            
            augmented_samples = self.generate_augmented_samples(
                str(source_path),
                transform,
                multiplier,
                f"fold_train_class_{class_id}_{base_filename}"
            )
            
            # 증강된 샘플들 저장
            for aug_image, aug_filename in augmented_samples:
                # 이미지 저장
                output_path = train_dir / aug_filename
                image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), image_bgr)
                
                # CSV 데이터 추가
                train_csv_data.append({
                    'ID': aug_filename,
                    'target': class_id
                })
        
        # Validation set 생성 (원본만, 증강 없음)
        ic(f"📊 Validation set 생성 중 ({len(split_info['val_sources'])} 소스)")
        for source_filename in tqdm(split_info['val_sources'], desc="Validation 복사"):
            source_row = self.df[self.df['ID'] == source_filename].iloc[0]
            source_path = self.source_dir / "train" / source_filename
            
            if not source_path.exists():
                continue
            
            # 원본 이미지 복사 (증강 없음)
            val_filename = f"val_original_{source_filename}"
            val_path = val_dir / val_filename
            shutil.copy2(source_path, val_path)
            
            # CSV 데이터 추가
            val_csv_data.append({
                'ID': val_filename,
                'target': source_row['target']
            })
        
        # 메타데이터 저장
        self._save_fold_metadata(train_csv_data, val_csv_data, metadata_dir, dataset_name)
        
        ic(f"✅ Fold 데이터셋 생성 완료: {output_dir}")
        ic(f"   Training samples: {len(train_csv_data)}")
        ic(f"   Validation samples: {len(val_csv_data)}")
        
        return str(output_dir)

    def _save_fold_metadata(self, train_data: List[Dict], val_data: List[Dict], 
                        metadata_dir: Path, dataset_name: str):
        """K-fold 메타데이터 저장"""
        
        # Train/Val CSV 분리 저장
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        train_df.to_csv(metadata_dir / "train.csv", index=False)
        val_df.to_csv(metadata_dir / "val.csv", index=False)
        
        # 원본 meta.csv 복사
        shutil.copy2(self.meta_file, metadata_dir / "meta.csv")
        
        # 폴드 정보 저장
        fold_info = {
            'dataset_name': dataset_name,
            'generation_timestamp': datetime.now().isoformat(),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'data_leakage': 'prevented',
            'split_method': 'source_level_stratified'
        }
        
        with open(metadata_dir / "fold_info.json", 'w', encoding='utf-8') as f:
            json.dump(fold_info, f, indent=2, ensure_ascii=False)

def main():
    """Fire CLI 인터페이스"""
    fire.Fire(DatasetMultiplier)


if __name__ == "__main__":
    main()