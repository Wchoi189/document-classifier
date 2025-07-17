#!/bin/bash
# scripts/test_pure_volume_hypothesis.sh
# 순수 볼륨 가설 검증 스크립트

echo "🚀 순수 볼륨 가설 검증 시작 (회전 없음)"
echo "=================================="

# 1. 순수 볼륨 데이터셋 생성
echo "📊 1단계: 순수 볼륨 데이터셋 생성 (20x, 회전 없음)"
python -c "
from src.data.pure_volume_generator import PureVolumeGenerator
generator = PureVolumeGenerator()
dataset_path = generator.generate_pure_volume_dataset(multiplier=20, output_name='pure_volume_20x_no_rotation')
config_path = generator.create_training_config(dataset_path)
print(f'✅ 데이터셋 생성 완료: {dataset_path}')
print(f'✅ 설정 파일 생성: {config_path}')
"

# 2. 훈련 실행
echo ""
echo "🎯 2단계: 순수 볼륨 데이터셋으로 훈련"
python scripts/train.py experiment=pure_volume_no_rotation

# 3. 예측 실행  
echo ""
echo "🔮 3단계: 테스트 데이터에 대한 예측"
python scripts/predict.py run --input_path data/raw/test --use-last --output_suffix "_pure_volume_test"

echo ""
echo "🏁 순수 볼륨 가설 검증 완료!"
echo "=================================="
echo "📋 다음 단계:"
echo "   1. WandB에서 훈련 메트릭 확인"
echo "   2. 생성된 예측 파일로 대회 제출"
echo "   3. 대회 점수와 79% 기준점 비교"
echo ""
echo "🎯 가설 검증 포인트:"
echo "   - Validation F1: 80-90% 범위 (90%+ 아님)"  
echo "   - Competition score: >79% (기준점 개선)"
echo "   - 회전 없는 순수 볼륨 증강의 효과 확인"