
import sys
import os
from pathlib import Path

# 🔧 프로젝트 루트 경로 설정 (항상 첫 번째로)
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)
os.chdir(project_root)
# test_conservative.py 파일 생성
from src.training.conservative_augmentation_tester import ConservativeAugmentationTester

tester = ConservativeAugmentationTester()
result = tester.run_conservative_augmentation_test(
    baseline_checkpoint="outputs/models/model_epoch_20.pth",
    quick_epochs=3
)
print(result)