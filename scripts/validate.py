# scripts/debug_dataset_paths.py
"""
Debug script to check K-fold dataset structure and paths
K-fold 데이터셋 구조와 경로 확인용 디버그 스크립트
"""

import pandas as pd
from pathlib import Path
from icecream import ic
import fire


class DatasetPathDebugger:
    """데이터셋 경로 구조 디버깅"""
    
    def check_phase1_structure(self):
        """Phase 1 데이터셋 구조 확인"""
        ic("🔍 Phase 1 데이터셋 구조 확인")
        
        base_path = Path("data/augmented_datasets/phase1_mild_fold_0")
        
        # 기본 구조 확인
        ic(f"기본 경로 존재: {base_path.exists()}")
        
        if base_path.exists():
            ic("📁 디렉토리 구조:")
            for item in base_path.iterdir():
                ic(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Train 디렉토리 확인
        train_dir = base_path / "train"
        ic(f"Train 디렉토리 존재: {train_dir.exists()}")
        
        if train_dir.exists():
            train_files = list(train_dir.glob("*.jpg"))
            ic(f"Train 이미지 수: {len(train_files)}")
            if train_files:
                ic(f"첫 번째 파일: {train_files[0].name}")
                ic(f"마지막 파일: {train_files[-1].name}")
        
        # Val 디렉토리 확인
        val_dir = base_path / "val"
        ic(f"Val 디렉토리 존재: {val_dir.exists()}")
        
        if val_dir.exists():
            val_files = list(val_dir.glob("*.jpg"))
            ic(f"Val 이미지 수: {len(val_files)}")
            if val_files:
                ic(f"첫 번째 파일: {val_files[0].name}")
        
        # CSV 파일 확인
        metadata_dir = base_path / "metadata"
        if metadata_dir.exists():
            train_csv = metadata_dir / "train.csv"
            val_csv = metadata_dir / "val.csv"
            
            if train_csv.exists():
                train_df = pd.read_csv(train_csv)
                ic(f"Train CSV 행 수: {len(train_df)}")
                ic(f"Train CSV 샘플: {train_df['ID'].head(3).tolist()}")
            
            if val_csv.exists():
                val_df = pd.read_csv(val_csv)
                ic(f"Val CSV 행 수: {len(val_df)}")
                ic(f"Val CSV 샘플: {val_df['ID'].head(3).tolist()}")
    
    def check_phase2_structure(self):
        """Phase 2 데이터셋 구조 확인"""
        ic("🔍 Phase 2 데이터셋 구조 확인")
        
        base_path = Path("data/augmented_datasets/phase2_variety_fold_0")
        
        # 기본 구조 확인
        ic(f"기본 경로 존재: {base_path.exists()}")
        
        if base_path.exists():
            ic("📁 디렉토리 구조:")
            for item in base_path.iterdir():
                ic(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Val 디렉토리 확인 (cross-phase validation용)
        val_dir = base_path / "val"
        ic(f"Val 디렉토리 존재: {val_dir.exists()}")
        
        if val_dir.exists():
            val_files = list(val_dir.glob("*.jpg"))
            ic(f"Val 이미지 수: {len(val_files)}")
            if val_files:
                ic(f"첫 번째 파일: {val_files[0].name}")
                ic(f"마지막 파일: {val_files[-1].name}")
        
        # CSV 파일 확인
        metadata_dir = base_path / "metadata"
        if metadata_dir.exists():
            val_csv = metadata_dir / "val.csv"
            
            if val_csv.exists():
                val_df = pd.read_csv(val_csv)
                ic(f"Phase 2 Val CSV 행 수: {len(val_df)}")
                ic(f"Phase 2 Val CSV 샘플: {val_df['ID'].head(3).tolist()}")
    
    def check_file_mapping(self):
        """CSV와 실제 파일 매핑 확인"""
        ic("🔍 CSV와 실제 파일 매핑 확인")
        
        # Phase 1 Train 확인
        train_csv_path = Path("data/augmented_datasets/phase1_mild_fold_0/metadata/train.csv")
        train_dir = Path("data/augmented_datasets/phase1_mild_fold_0/train")
        
        if train_csv_path.exists() and train_dir.exists():
            train_df = pd.read_csv(train_csv_path)
            sample_files = train_df['ID'].head(5).tolist()
            
            ic("Phase 1 Train 파일 존재 확인:")
            for filename in sample_files:
                file_path = train_dir / filename
                ic(f"  {filename}: {'✅' if file_path.exists() else '❌'} ({file_path})")
        
        # Phase 2 Val 확인
        val_csv_path = Path("data/augmented_datasets/phase2_variety_fold_0/metadata/val.csv")
        val_dir = Path("data/augmented_datasets/phase2_variety_fold_0/val")
        
        if val_csv_path.exists() and val_dir.exists():
            val_df = pd.read_csv(val_csv_path)
            sample_files = val_df['ID'].head(5).tolist()
            
            ic("Phase 2 Val 파일 존재 확인:")
            for filename in sample_files:
                file_path = val_dir / filename
                ic(f"  {filename}: {'✅' if file_path.exists() else '❌'} ({file_path})")
    
    def suggest_fix(self):
        """경로 문제 해결 방안 제시"""
        ic("💡 경로 문제 해결 방안")
        
        # 실제 구조 파악
        self.check_phase1_structure()
        self.check_phase2_structure()
        self.check_file_mapping()
        
        ic("🔧 가능한 해결책:")
        ic("1. 이미지 파일이 올바른 위치에 있는지 확인")
        ic("2. CSV 파일의 ID 컬럼과 실제 파일명 일치 확인")
        ic("3. Dataset loader의 경로 구성 로직 수정")


def main():
    fire.Fire(DatasetPathDebugger)


if __name__ == "__main__":
    main()