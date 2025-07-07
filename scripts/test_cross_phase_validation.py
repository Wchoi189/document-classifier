# scripts/test_cross_phase_validation.py
"""
Cross-Phase Validation Test Script
교차 단계 검증 기능 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
os.chdir(project_root)

import fire
from icecream import ic
from src.utils.config_utils import load_config
from src.data.csv_dataset import CSVDocumentDataset, is_cross_phase_config, create_cross_phase_datasets
from src.data.augmentation import get_valid_transforms


class CrossPhaseValidationTester:
    """교차 단계 검증 기능 테스트"""
    
    def test_enhanced_dataset_loader(self, 
                                   config_path: str = "configs/experiment/progressive_cross_phase_validation/phase1_rotation_mild.yaml"):
        """
        향상된 데이터셋 로더 테스트
        
        Args:
            config_path: 교차 단계 검증 설정 파일
        """
        ic("🧪 Enhanced Dataset Loader 테스트 시작")
        
        try:
            # 설정 로드
            config = load_config(config_path)
            ic(f"설정 파일 로드: {config_path}")
            
            # 교차 단계 검증 감지
            is_cross_phase = is_cross_phase_config(config)
            ic(f"교차 단계 검증 감지: {is_cross_phase}")
            
            if not is_cross_phase:
                ic("❌ 교차 단계 검증 설정이 감지되지 않음")
                return False
            
            # 변환 생성
            valid_transforms = get_valid_transforms(
                height=config['data']['image_size'],
                width=config['data']['image_size'],
                mean=config['data']['mean'],
                std=config['data']['std']
            )
            
            # 교차 단계 데이터셋 생성
            ic("🔄 교차 단계 데이터셋 생성 중...")
            train_dataset, val_dataset = create_cross_phase_datasets(
                config, valid_transforms, valid_transforms
            )
            
            # 데이터셋 정보 검증
            train_info = train_dataset.get_info()
            val_info = val_dataset.get_info()
            
            ic("📊 훈련 데이터셋 정보:")
            ic(f"   크기: {train_info['dataset_size']}")
            ic(f"   클래스 수: {train_info['num_classes']}")
            ic(f"   교차 단계: {train_info['is_cross_phase']}")
            
            ic("🎯 검증 데이터셋 정보:")
            ic(f"   크기: {val_info['dataset_size']}")
            ic(f"   클래스 수: {val_info['num_classes']}")
            ic(f"   교차 단계: {val_info['is_cross_phase']}")
            
            # 샘플 데이터 로드 테스트
            ic("🔍 샘플 데이터 로드 테스트...")
            
            train_sample = train_dataset[0]
            val_sample = val_dataset[0]
            
            ic(f"훈련 샘플 형태: {train_sample[0].shape}")
            ic(f"검증 샘플 형태: {val_sample[0].shape}")
            
            ic("✅ Enhanced Dataset Loader 테스트 성공!")
            return True
            
        except Exception as e:
            ic(f"❌ 테스트 실패: {e}")
            import traceback
            ic("상세 오류:", traceback.format_exc())
            return False

    def test_standard_dataset_compatibility(self, 
                                          config_path: str = "configs/experiment/document_classifier_0701.yaml"):
        """
        표준 데이터셋 호환성 테스트 (기존 기능 보존 확인)
        
        Args:
            config_path: 표준 설정 파일
        """
        ic("🧪 Standard Dataset 호환성 테스트 시작")
        
        try:
            # 설정 로드
            config = load_config(config_path)
            
            # 표준 모드 감지
            is_cross_phase = is_cross_phase_config(config)
            ic(f"교차 단계 검증 감지: {is_cross_phase}")
            
            if is_cross_phase:
                ic("⚠️ 이 설정은 교차 단계 검증용입니다")
                return False
            
            # 표준 데이터셋 생성
            dataset = CSVDocumentDataset(
                root_dir=config['data']['root_dir'],
                csv_file=config['data']['csv_file'],
                meta_file=config['data']['meta_file'],
                split='train',
                val_size=0.2,
                seed=42
            )
            
            ic(f"✅ 표준 데이터셋 생성 성공: {len(dataset)}개 샘플")
            
            # 샘플 로드 테스트
            sample = dataset[0]
            ic(f"샘플 형태: {sample[0].shape}")
            
            ic("✅ Standard Dataset 호환성 테스트 성공!")
            return True
            
        except Exception as e:
            ic(f"❌ 호환성 테스트 실패: {e}")
            return False
    
    def compare_validation_approaches(self):
        """
        표준 vs 교차 단계 검증 비교
        """
        ic("🔬 검증 방식 비교 테스트")
        
        # 표준 검증 테스트
        standard_success = self.test_standard_dataset_compatibility()
        
        # 교차 단계 검증 테스트  
        cross_phase_success = self.test_enhanced_dataset_loader()
        
        ic(f"표준 검증: {'✅ 성공' if standard_success else '❌ 실패'}")
        ic(f"교차 단계 검증: {'✅ 성공' if cross_phase_success else '❌ 실패'}")
        
        if standard_success and cross_phase_success:
            ic("🎉 모든 검증 방식이 정상 작동!")
            return True
        else:
            ic("⚠️ 일부 검증 방식에 문제가 있습니다")
            return False
    
    def create_test_config(self):
        """
        테스트용 교차 단계 검증 설정 생성
        """
        ic("📝 테스트 설정 파일 생성 중...")
        
        config_content = """# Test Cross-Phase Validation Config
defaults:
  - _self_

name: "test-cross-phase-validation"
description: "Test cross-phase validation functionality"
tags: ["test", "cross-phase", "validation"]

seed: 42
device: 'cuda'

# Test cross-phase validation
data:
  # Train on Phase 1 data
  root_dir: "data/augmented_datasets/phase1_mild_fold_0"/train
  csv_file: "data/augmented_datasets/phase1_mild_fold_0/metadata/train.csv"
  meta_file: "data/augmented_datasets/phase1_mild_fold_0/metadata/meta.csv"
  
  # Validate on Phase 2 data (harder conditions)
  val_root_dir: "data/augmented_datasets/phase2_variety_fold_0/val"
  val_csv_file: "data/augmented_datasets/phase2_variety_fold_0/metadata/val.csv"
  
  image_size: 224
  val_size: 0.0  # Not used in cross-phase mode
  num_workers: 0
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# Test settings
train:
  epochs: 3  # Quick test
  batch_size: 8
  
wandb:
  enabled: false  # Disable for testing
"""
        
        # 테스트 설정 파일 저장
        test_config_dir = Path("configs/test")
        test_config_dir.mkdir(exist_ok=True)
        
        test_config_path = test_config_dir / "cross_phase_test.yaml"
        with open(test_config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        ic(f"✅ 테스트 설정 파일 생성: {test_config_path}")
        return str(test_config_path)


def main():
    """Fire CLI 인터페이스"""
    fire.Fire(CrossPhaseValidationTester)


if __name__ == "__main__":
    main()


# 사용 예시:
# python scripts/test_cross_phase_validation.py test_enhanced_dataset_loader
# python scripts/test_cross_phase_validation.py compare_validation_approaches
# python scripts/test_cross_phase_validation.py create_test_config