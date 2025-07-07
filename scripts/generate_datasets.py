# scripts/generate_datasets.py
"""
Dataset Generation CLI Script
대용량 증강 데이터셋 생성을 위한 명령행 인터페이스
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
from src.data.dataset_multiplier import DatasetMultiplier


class DatasetGenerationCLI:
    """데이터셋 생성을 위한 CLI 인터페이스"""
    
    def __init__(self):
        self.multiplier = DatasetMultiplier()
        ic("🚀 Dataset Generation CLI 초기화 완료")
    
    def generate_single(self, 
                       dataset_name: str,
                       strategy: str = "volume_focused",
                       multiplier: int = 10,
                       batch_size: int = 100):
        """
        단일 데이터셋 변형 생성
        
        Args:
            dataset_name: 데이터셋 이름 (예: v1_volume_20x)
            strategy: 증강 전략 (volume_focused, test_focused, balanced)
            multiplier: 증강 배수
            batch_size: 배치 크기 (메모리 사용량 조절)
        
        Example:
            python scripts/generate_datasets.py generate_single \
                --dataset_name=test_experiment \
                --strategy=volume_focused \
                --multiplier=5
        """
        ic(f"📊 단일 데이터셋 생성: {dataset_name}")
        ic(f"전략: {strategy}, 배수: {multiplier}x")
        
        try:
            output_path = self.multiplier.save_augmented_dataset(
                dataset_name=dataset_name,
                strategy=strategy,
                target_multiplier=multiplier,
                batch_size=batch_size
            )
            
            ic(f"✅ 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            ic(f"❌ 생성 실패: {e}")
            raise
    
    def generate_all(self):
        """
        모든 데이터셋 변형을 순차적으로 생성
        
        - v1_volume_20x: 대용량 다양성 중심 (20배 증강)
        - v2_test_focused_10x: 테스트 조건 시뮬레이션 (10배 증강)  
        - v3_balanced_15x: 클래스 균형 맞춤 (15배 증강)
        
        Example:
            python scripts/generate_datasets.py generate_all
        """
        ic("🎯 모든 데이터셋 변형 생성 시작")
        
        results = self.multiplier.generate_all_variants()
        
        # 결과 요약 출력
        ic("📋 생성 결과 요약:")
        for dataset_name, result in results.items():
            status = result['status']
            if status == 'success':
                ic(f"  ✅ {dataset_name}: {result['path']}")
            else:
                ic(f"  ❌ {dataset_name}: {result['error']}")
        
        return results
    
    def generate_quick_test(self, multiplier: int = 2):
        """
        빠른 테스트용 소규모 데이터셋 생성
        
        Args:
            multiplier: 증강 배수 (기본 2배)
        
        Example:
            python scripts/generate_datasets.py generate_quick_test --multiplier=3
        """
        ic(f"⚡ 빠른 테스트 데이터셋 생성 ({multiplier}x)")
        
        return self.generate_single(
            dataset_name=f"quick_test_{multiplier}x",
            strategy="volume_focused",
            multiplier=multiplier,
            batch_size=50
        )
    
    def generate_volume_focused(self, multiplier: int = 20):
        """
        대용량 다양성 중심 데이터셋 생성
        
        Args:
            multiplier: 증강 배수 (기본 20배)
        """
        return self.generate_single(
            dataset_name=f"v1_volume_{multiplier}x",
            strategy="volume_focused",
            multiplier=multiplier
        )
    
    def generate_test_focused(self, multiplier: int = 10):
        """
        테스트 조건 시뮬레이션 데이터셋 생성
        
        Args:
            multiplier: 증강 배수 (기본 10배)
        """
        return self.generate_single(
            dataset_name=f"v2_test_focused_{multiplier}x",
            strategy="test_focused",
            multiplier=multiplier
        )
    
    def generate_balanced(self, multiplier: int = 15):
        """
        클래스 균형 맞춤 데이터셋 생성
        
        Args:
            multiplier: 증강 배수 (기본 15배)
        """
        return self.generate_single(
            dataset_name=f"v3_balanced_{multiplier}x",
            strategy="balanced",
            multiplier=multiplier
        )
    
    def check_storage_space(self):
        """
        필요한 저장 공간 추정
        """
        ic("💾 저장 공간 요구사항 분석")
        
        # 대략적인 계산
        original_samples = len(self.multiplier.df)
        avg_image_size_mb = 0.5  # JPG 파일 평균 크기 추정
        
        datasets = [
            ("v1_volume_20x", 20),
            ("v2_test_focused_10x", 10),
            ("v3_balanced_15x", 15)
        ]
        
        total_space_gb = 0
        
        ic("📊 추정 저장 공간:")
        for name, multiplier in datasets:
            samples = original_samples * multiplier
            space_gb = (samples * avg_image_size_mb) / 1024
            total_space_gb += space_gb
            ic(f"  {name}: {samples:,} 샘플, ~{space_gb:.1f}GB")
        
        ic(f"📦 총 예상 필요 공간: ~{total_space_gb:.1f}GB")
        ic(f"✅ 사용자 가용 공간: 100GB+ (충분함)")
        
        return total_space_gb
    
    def show_dataset_info(self):
        """
        현재 데이터셋 정보 표시
        """
        ic("📊 원본 데이터셋 정보:")
        ic(f"총 샘플: {len(self.multiplier.df)}")
        ic(f"클래스 수: {len(self.multiplier.class_info)}")
        
        ic("클래스별 분포:")
        for class_id, count in self.multiplier.class_distribution.items():
            class_name = self.multiplier.class_info[class_id]
            ic(f"  클래스 {class_id} ({class_name}): {count}개")

    def generate_progressive(self, multiplier: int = 10):
        """
        Progressive rotation training용 3단계 데이터셋 생성
        
        Args:
            multiplier: 각 단계별 증강 배수
        
        Example:
            python scripts/generate_datasets.py generate_progressive --multiplier=10
        """
        ic(f"🎯 Progressive rotation datasets 생성 ({multiplier}x)")
        
        return self.multiplier.generate_progressive_datasets(multiplier)    
    
    def generate_kfold(self, k: int = 5, multiplier: int = 10, strategy: str = "phase1_mild"):
        """
        Stratified K-fold 데이터셋 생성
        
        Args:
            k: 폴드 수
            multiplier: 증강 배수  
            strategy: 증강 전략
        
        Example:
            python scripts/generate_datasets.py generate_kfold --k=5 --multiplier=5 --strategy=phase1_mild
        """
        ic(f"🎯 Stratified {k}-fold 데이터셋 생성")
        
        return self.multiplier.generate_stratified_kfold_datasets(k, multiplier, strategy)

def main():
    """메인 실행 함수"""
    fire.Fire(DatasetGenerationCLI)


if __name__ == "__main__":
    main()