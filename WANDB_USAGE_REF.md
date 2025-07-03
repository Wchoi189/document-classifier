
## 🚀 Complete Setup Instructions

### 1단계: WandB 설치
```bash
# environment.yml에 추가
pip install wandb

# 또는 직접 설치
pip install wandb

# WandB 로그인 (최초 한 번만)
wandb login

```

### 2단계: 설정 파일 업데이트
**configs/config.yaml에 다음 내용을 추가하세요:**

```yaml
# WandB 설정 추가
wandb:
  enabled: true
  project: "document-classifier"
  entity: null  # WandB 사용자 이름 (선택 사항)
  name: null    # 실행 이름 자동 생성
  tags: ["resnet50", "document-classification", "imbalanced"]
  notes: "17개 클래스 문서 분류 - 클래스 불균형 2.2:1"

  # 로깅 설정
  log_frequency: 10        # N 배치마다 로그 기록
  log_images: true         # 예측 샘플 이미지 로그 기록
  log_model: true          # 모델 아티팩트 저장
  log_gradients: false     # 그래디언트 norm 모니터링
  log_confusion_matrix: true

  # 고급 기능
  watch_model: true        # 모델 구조 모니터링
  log_code: true           # 코드 스냅샷 저장

# 기존 설정은 그대로 유지됩니다...
seed: 42
device: 'cuda'
# ... 나머지 부분은 변경 없음
```

### 3단계: 학습 스크립트 업데이트
**`train.py`를 수정하세요:**

```python
# 이 라인을:
from src.trainer.trainer import Trainer

# 이렇게 변경:
from src.trainer.wandb_trainer import WandBTrainer

# 그리고 이 라인을:
trainer = Trainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config)

# 이렇게 변경:
trainer = WandBTrainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config)

```

## 🎯 사용 예시

### WandB로 학습하기
```bash
# config.yaml에서 WandB를 활성화한 후:
python -m scripts.train --config configs/config0701.yaml

# WandB 임시 비활성화:
python -m scripts.train --config configs/config.yaml --wandb-disabled

# 오프라인으로 실행 (나중에 동기화):
python -m scripts.train --config configs/config.yaml --wandb-offline
```

### WandB로 예측하기
```bash
# WandB 로깅과 함께 예측:
python -m predict predict_images_wandb checkpoints/best_model.pth data/dataset/test/ --wandb-project document-classifier

# WandB 없이 예측:
python -m predict predict_images_wandb checkpoints/best_model.pth data/dataset/test/

# 일반 예측 (WandB 사용 안 함):
python -m predict predict_images checkpoints/best_model.pth data/dataset/test/

# last_model.pth로 예측 실행:
python -m predict predict_images checkpoints/last_model.pth data/dataset/test/ --output my_last_predictions.csv

```

### 하이퍼파라미터 스윕 (Sweep)
```bash
# 스윕 생성:
wandb sweep sweeps/sweep_config.yaml

# 스윕 에이전트 실행:
wandb agent <sweep_id>

```

## 📊 What You'll Get in WandB Dashboard

### 1. **Real-time Training Monitoring**
- ✅ Loss curves (train/validation)
- ✅ Accuracy and F1-score trends  
- ✅ Learning rate scheduling
- ✅ GPU memory usage
- ✅ Training time per epoch

### 2. **Model Performance Analysis**
- ✅ Interactive confusion matrices
- ✅ Per-class accuracy breakdown
- ✅ Sample predictions with images
- ✅ Confidence score distributions

### 3. **Experiment Comparison**
- ✅ Side-by-side metric comparison
- ✅ Hyperparameter correlation analysis
- ✅ Model architecture comparison
- ✅ Training efficiency metrics

### 4. **Advanced Features**
- ✅ Model artifacts (automatic saving)
- ✅ Code versioning
- ✅ Hyperparameter sweeps
- ✅ Team collaboration

## 🔧 Quick Start Checklist

1. **✅ Install WandB**: `pip install wandb`
2. **✅ Login**: `wandb login`
3. **✅ Add config**: Update `configs/config.yaml` with wandb section
4. **✅ Create trainer**: Add `trainer/wandb_trainer.py`
5. **✅ Update train.py**: Import `WandBTrainer` instead of `Trainer`
6. **✅ Update predict.py**: Add `predict_images_wandb` function
7. **✅ Run training**: `python train.

