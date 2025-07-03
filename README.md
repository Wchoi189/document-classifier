문서 이미지를 분석하고 분류하기 위한 딥러닝 프로젝트입니다.

## 🚀 프로젝트 구조

```
document-classifier/
├── src/                      # 📦 핵심 소스 코드 (Core source code)
│   ├── data/                 # 💾 데이터 로딩, 전처리, 증강 (Data loading, preprocessing, augmentation)
│   ├── models/               # 🧠 모델 아키텍처 정의 (Model architectures)
│   ├── training/             # 🏋️ 훈련 루프, 트레이너, 콜백 (Training loops, trainers, callbacks)
│   ├── utils/                # 🛠️ 보조 유틸리티 함수 (Utilities)
│   └── inference/            # 🔮 추론 및 예측 (Inference & prediction)
│
├── configs/                  # ⚙️ 실험 설정 YAML (Experiment configs)
│   ├── base.yaml             # └─ 기본 설정 (Base config)
│   └── resnet50.yaml         # └─ ResNet50 설정 (ResNet50 config)
│
├── data/                     # 📊 데이터 (Data)
│   ├── raw/                  # └─ 원본 데이터 (Raw data)
│   └── processed/            # └─ 전처리 데이터 (Processed data)
│
├── notebooks/                # 📝 EDA/분석 노트북 (Jupyter notebooks)
├── scripts/                  # ▶️ 실행 스크립트 (Run scripts)
│   ├── train.py              # └─ 모델 훈련 (Train)
│   ├── predict.py            # └─ 예측 (Predict)
│   └── evaluate.py           # └─ 평가 (Evaluate)
│
├── tests/                    # ✅ 테스트 (Tests)
├── outputs/                  # 📤 결과물 (Outputs)
│   ├── models/               # └─ 모델 가중치 (Model weights)
│   ├── predictions/          # └─ 예측 결과 (Predictions)
│   └── figures/              # └─ 시각화 (Figures)
│
├── environment.yml           # 🌐 Conda 환경 설정 (Conda env)
├── requirements.txt          # 📄 Pip 요구사항 (Pip requirements)
├── project_setup.py          # 🚀 환경 초기화 (Setup script)
├── setup.py                  # 🏗️ 설치 스크립트 (Install script)
├── README.md                 # 📖 프로젝트 문서 (Docs)
└── .gitignore                # 🙈 Git 추적 제외 (Git ignore)
```

---

## ⚙️ 설치 및 환경 설정

### Conda 환경 (권장)

1. **Anaconda/Miniconda 설치**  
    [Anaconda](https://www.anaconda.com/products/distribution) 또는 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 설치

2. **Conda 환경 생성 및 활성화**
    ```bash
    conda env create -f environment.yml
    conda activate doc-classifier-env
    ```

### Pip 환경 (대안)

1. **가상 환경 생성 및 활성화**
    ```bash
    python -m venv .venv
    # Linux/MacOS
    source .venv/bin/activate
    # Windows
    # .\.venv\Scripts\activate
    ```

2. **의존성 설치**
    ```bash
    pip install -r requirements.txt
    ```

---
## 3. 추론 및 분석 (Inference & Analysis)
훈련된 모델을 사용하여 예측을 실행하고 결과를 분석합니다.

### 기본 예측 (Standard Prediction)
다음은 가장 기본적인 예측 명령어입니다.

```bash
# 특정 모델 체크포인트를 사용하여 예측
python -m scripts.predict run --input_path data/raw/test --checkpoint_path outputs/models/best_model.pth

# 가장 마지막에 훈련된 모델을 사용하여 예측
python -m scripts.predict run --input_path data/raw/test --use-last
```

### 예측 옵션 (Prediction Options)
기본 run 명령어에 플래그를 추가하여 동작을 변경할 수 있습니다.

```bash
# WandB 로깅 활성화
python -m scripts.predict run --input_path data/raw/test --use-last --wandb

# 다른 설정 파일 사용
python -m scripts.predict run --input_path data/raw/test --use-last --config_path configs/experiment_2.yaml
```

### 특수 목적 명령어 (Specialized Commands)
```bash
# 배치 예측 (파일 목록으로 예측)
python -m scripts.predict batch --checkpoint_path outputs/models/best_model.pth --input_list_file my_inputs.txt

# 결과 분석 (생성된 CSV 파일 분석)
python -m scripts.predict analyze --predictions_csv outputs/predictions/predictions_HHMM.csv
```

### 4. 데이터 분석 (Exploratory Data Analysis)
Jupyter Lab을 실행하여 데이터 분석 노트북을 확인할 수 있습니다.

```bash
# Jupyter Lab 실행
jupyter lab
```
브라우저에서 `notebooks/` 디렉토리로 이동하여 노트북 파일을 여세요.
> **모든 명령어는 프로젝트 루트 디렉토리에서 실행하세요.**

---
