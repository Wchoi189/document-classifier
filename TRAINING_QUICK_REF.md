
🔥 학습 팀을 위한 빠른 참조 가이드
==================================================

📁 데이터셋 파일:
  • train.csv: 1,570개 샘플, 17개 클래스
  • meta.csv: 클래스 ID와 이름 매핑
  • 이미지: data/dataset/train/*.jpg

⚙️  권장 설정 업데이트:
  • image_size: 224
  • batch_size: 32
  • model: resnet34
  • use_weighted_sampling: False

🎯 주요 과제:
  • 클래스 불균형 (비율: 2.2)
  • 다양한 이미지 크기
  • 샘플 중 0.0% 파일 누락

💡 학습 팁:
  • 정확도(accuracy)와 F1-점수(F1-score) 모두 모니터링
  • 계층적 교차 검증 (stratified validation) 사용
  • 조기 종료 (early stopping) 구현
  • 심각한 불균형 시 포컬 손실 (focal loss) 고려

🚀 학습 시작 준비:
  python train.py --config config/config.yaml --debug
