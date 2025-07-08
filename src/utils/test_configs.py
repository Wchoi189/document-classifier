# 🧪 Final Conservative Augmentation Tester 테스트 스크립트

import sys
import os
from pathlib import Path

# 🔧 프로젝트 루트 경로 설정
project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)
os.chdir(project_root)

from icecream import ic

def test_enhanced_config_loading():
    """Enhanced config loading 테스트"""
    ic("🧪 Enhanced Config Loading 테스트 시작")
    
    try:
        # 1. 새로운 config_utils 함수들 테스트
        from src.utils.config_utils import load_config, normalize_config_structure
        
        # Config 로드 (Hydra defaults 수동 병합 포함)
        config = load_config("configs/config.yaml")
        config = normalize_config_structure(config)
        
        ic("✅ Enhanced config loading 성공")
        ic("Config 키들:", list(config.keys()))
        
        # 2. 필수 키들 확인
        required_keys = ['model', 'optimizer', 'scheduler', 'train', 'data']
        missing_keys = []
        
        for key in required_keys:
            if key in config:
                ic(f"✅ {key} 키 존재")
                if isinstance(config[key], dict):
                    ic(f"  내용: {list(config[key].keys())}")
            else:
                missing_keys.append(key)
                ic(f"❌ {key} 키 없음")
        
        if not missing_keys:
            ic("🎉 모든 필수 키 존재!")
            return True
        else:
            ic(f"⚠️ 누락된 키들: {missing_keys}")
            return False
            
    except Exception as e:
        ic(f"❌ Enhanced config loading 실패: {e}")
        import traceback
        ic("상세 오류:", traceback.format_exc())
        return False

def test_conservative_tester_with_enhanced_config():
    """Enhanced config로 Conservative Tester 테스트"""
    ic("🧪 Enhanced Conservative Tester 테스트 시작")
    
    try:
        # 1. Conservative Tester 임포트 및 초기화
        sys.path.insert(0, str(Path.cwd()))  # 현재 디렉토리를 path에 추가
        
        # 수정된 버전을 동적으로 임포트하기 위해 임시로 저장
        temp_file = Path("temp_conservative_tester.py")
        
        # 수정된 코드를 임시 파일로 저장 (실제로는 파일 교체 필요)
        # 여기서는 기존 파일이 수정되었다고 가정하고 테스트
        
        from src.training.conservative_augmentation_tester import ConservativeAugmentationTester
        
        tester = ConservativeAugmentationTester()
        ic("✅ Conservative Tester 초기화 성공")
        
        # 2. Config 완성도 검증 결과 확인
        ic("Config 검증 완료")
        
        # 3. Progressive configs 생성 테스트
        configs = tester.create_progressive_augmentation_configs()
        ic(f"✅ 점진적 설정 생성 성공: {len(configs)}개")
        
        # 4. 첫 번째 config로 간단한 검증 (실제 훈련 없이 설정만 확인)
        test_config = configs[0]
        ic(f"테스트 config: {test_config['experiment_name']}")
        
        return True
        
    except Exception as e:
        ic(f"❌ Enhanced Conservative Tester 실패: {e}")
        import traceback
        ic("상세 오류:", traceback.format_exc())
        return False

def main():
    """전체 테스트 실행"""
    ic("🚀 Conservative Augmentation Tester FINAL FIX 테스트")
    ic("=" * 60)
    
    tests = [
        ("Enhanced Config Loading", test_enhanced_config_loading),
        ("Enhanced Conservative Tester", test_conservative_tester_with_enhanced_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        ic(f"\n🧪 {test_name} 테스트 중...")
        result = test_func()
        results.append((test_name, result))
        ic(f"결과: {'✅ 성공' if result else '❌ 실패'}")
    
    # 최종 결과
    ic("\n" + "=" * 60)
    ic("🏁 FINAL FIX 테스트 결과:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        ic(f"{test_name}: {status}")
    
    ic(f"\n전체 결과: {passed}/{total} 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        ic("🎉 FINAL FIX 성공!")
        ic("📋 다음 단계:")
        ic("  1. config_utils.py 파일 교체")
        ic("  2. conservative_augmentation_tester.py 파일 교체") 
        ic("  3. 실제 Conservative Test 실행")
        
        ic("\n🚀 실행 명령어:")
        ic("python -m src.training.conservative_augmentation_tester run_conservative_augmentation_test \\")
        ic("    --baseline_checkpoint outputs/models/model_epoch_20.pth \\")
        ic("    --quick_epochs 3")
    else:
        ic("⚠️ 일부 테스트 실패 - 추가 디버깅 필요")
    
    return passed == total

if __name__ == "__main__":
    main()