# src/data/csv_dataset.py - ENHANCED VERSION
"""
Enhanced CSV Document Dataset with Cross-Phase Validation Support
교차 단계 검증을 지원하는 향상된 CSV 문서 데이터셋
"""

import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from icecream import ic


class CSVDocumentDataset(Dataset):
    """
    Enhanced Document dataset with cross-phase validation support
    교차 단계 검증을 지원하는 향상된 문서 데이터셋
    """
    
    def __init__(self, 
                 root_dir, 
                 csv_file, 
                 meta_file, 
                 transform=None, 
                 split='train', 
                 val_size=0.2, 
                 seed=42,
                 val_root_dir=None,
                 val_csv_file=None,
                 val_meta_file=None):
        """
        Args:
            root_dir: 훈련 이미지 데이터셋의 루트 디렉토리
            csv_file: 훈련 CSV 파일
            meta_file: 메타데이터 파일
            transform: 이미지 변환
            split: 'train', 'val', 또는 'all'
            val_size: 검증 세트 비율 (교차 단계 검증 사용 시 무시됨)
            seed: 랜덤 시드
            val_root_dir: 별도 검증 데이터셋의 루트 디렉토리 (교차 단계용)
            val_csv_file: 별도 검증 CSV 파일 (교차 단계용)
            val_meta_file: 별도 검증 메타 파일 (교차 단계용, 선택사항)
        """
        self.transform = transform
        self.split = split
        
        # 🔧 **핵심 수정**: 이미지 경로를 일관되게 관리하기 위한 속성
        self.image_dir = None
        
        self.is_cross_phase = val_csv_file is not None
        
        # 메타데이터 로드 (클래스 정보는 훈련용 메타 파일 기준으로 통일)
        self.meta_df = pd.read_csv(meta_file)
        self.target_to_class = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
        self.classes = [self.target_to_class[i] for i in sorted(self.target_to_class.keys())]
        
        if self.is_cross_phase:
            ic("🔄 Cross-phase validation mode 감지됨")
            self._setup_cross_phase_validation(root_dir, csv_file, val_root_dir, val_csv_file)
        else:
            ic("📊 Standard single-dataset validation mode")
            self._setup_standard_validation(root_dir, csv_file, val_size, seed)

        # 최종 설정된 데이터프레임과 경로로 검증 수행
        self._verify_dataset()
    
    def _setup_cross_phase_validation(self, train_root, train_csv, val_root, val_csv):
        """교차 단계 검증 설정"""
        if self.split == 'train':
            self.df = pd.read_csv(train_csv)
            # 🔧 **수정**: 'train' 폴더까지 포함한 전체 경로를 image_dir로 설정
            self.image_dir = Path(train_root) / 'train'
            ic(f"✅ 훈련 데이터 로드: {len(self.df)}개 샘플, 경로: {self.image_dir}")
        elif self.split == 'val':
            self.df = pd.read_csv(val_csv)
            # 🔧 **수정**: 'val' 폴더까지 포함한 전체 경로를 image_dir로 설정
            self.image_dir = Path(val_root) / 'val'
            ic(f"🔄 교차 단계 검증 데이터 로드: {len(self.df)}개 샘플, 경로: {self.image_dir}")
        else: # 'all'
            self.df = pd.read_csv(train_csv)
            self.image_dir = Path(train_root) / 'train'

    def _setup_standard_validation(self, root_dir, csv_file, val_size, seed):
        """표준 단일 데이터셋 검증 설정"""
        full_df = pd.read_csv(csv_file)
        ic(f"표준 데이터셋 로드: {len(full_df)} 샘플")

        # 🔧 **수정**: 표준 모드에서도 이미지 디렉토리 경로는 동일
        self.image_dir = Path(root_dir) / 'train'

        if self.split in ['train', 'val'] and val_size > 0:
            train_df, val_df = train_test_split(
                full_df, 
                test_size=val_size, 
                random_state=seed, 
                stratify=full_df['target']
            )
            self.df = train_df if self.split == 'train' else val_df
        else:
            self.df = full_df

        ic(f"최종 데이터셋 크기 ('{self.split}'): {len(self.df)}개 샘플, 경로: {self.image_dir}")

    def _verify_dataset(self):
        """데이터셋 설정 검증 (경로 및 샘플 파일)"""
        if self.df.empty:
            ic("⚠️ 데이터프레임이 비어있어 검증을 건너뜁니다.")
            return

        ic("🔍 데이터셋 설정 확인 중...")
        sample_files = self.df['ID'].head(3).tolist()
        missing_count = 0
        
        for filename in sample_files:
            # 🔧 **수정**: 일관된 self.image_dir 사용
            img_path = self.image_dir / filename
            if not img_path.exists():
                missing_count += 1
                ic(f"⚠️ 파일 없음: {img_path}")
        
        if missing_count == 0:
            ic(f"✅ 샘플 파일 검증 완료 ({len(sample_files)}개 확인)")
        else:
            ic(f"⚠️ {missing_count}/{len(sample_files)} 샘플 파일 누락")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 🔧 **수정**: 경로 구성 로직을 self.image_dir로 단순화
        img_path = self.image_dir / row['ID']
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, row['target']

    def get_info(self):
        """데이터셋 정보 반환"""
        return {
            'dataset_size': len(self.df),
            'num_classes': len(self.classes),
            'split': self.split,
            'is_cross_phase': self.is_cross_phase,
        }


# 🔧 유틸리티 함수: 교차 단계 검증 감지
def is_cross_phase_config(config):
    """
    설정에서 교차 단계 검증 사용 여부 감지
    
    Args:
        config: 데이터 설정 딕셔너리
        
    Returns:
        bool: 교차 단계 검증 사용 여부
    """
    data_config = config.get('data', {})
    return data_config.get('val_csv_file') is not None


def create_cross_phase_datasets(config, train_transforms, valid_transforms):
    """
    교차 단계 검증 데이터셋 생성 유틸리티
    
    Args:
        config: 전체 설정 딕셔너리
        train_transforms: 훈련용 변환
        valid_transforms: 검증용 변환
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    data_config = config['data']
    
    ic("🔄 교차 단계 검증 데이터셋 생성 중...")
    
    # 훈련 데이터셋
    train_dataset = CSVDocumentDataset(
        root_dir=data_config['root_dir'],
        csv_file=data_config['csv_file'],
        meta_file=data_config['meta_file'],
        transform=train_transforms,
        split='train',
        val_size=0.0,  # 교차 단계에서는 분할하지 않음
        seed=config['seed'],
        val_root_dir=data_config.get('val_root_dir'),
        val_csv_file=data_config.get('val_csv_file'),
        val_meta_file=data_config.get('val_meta_file')
    )
    
    # 검증 데이터셋 (별도 데이터)
    val_dataset = CSVDocumentDataset(
        root_dir=data_config['root_dir'],  # 기본값
        csv_file=data_config['csv_file'],   # 기본값
        meta_file=data_config['meta_file'],
        transform=valid_transforms,
        split='val',
        val_size=0.0,
        seed=config['seed'],
        val_root_dir=data_config.get('val_root_dir'),
        val_csv_file=data_config.get('val_csv_file'),
        val_meta_file=data_config.get('val_meta_file')
    )
    
    # 데이터셋 정보 출력
    train_info = train_dataset.get_info()
    val_info = val_dataset.get_info()
    
    ic(f"✅ 훈련 데이터셋: {train_info['dataset_size']}개 샘플")
    ic(f"🔄 검증 데이터셋: {val_info['dataset_size']}개 샘플 (교차 단계)")
    
    return train_dataset, val_dataset