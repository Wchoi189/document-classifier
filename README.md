# 문서 이미지 분류기 (Document Image Classifier)
문서 이미지를 분석하고 분류하기 위한 딥러닝 프로젝트입니다.

## 🚀 프로젝트 구조

```
doc-classifier/
├── checkpoints/         # 훈련된 모델 가중치
├── config/              # 설정 파일 (config.yaml)
├── data/                # 데이터셋 및 증강 코드
├── inference/           # 추론 로직
├── logs/                # 훈련 로그
├── models/              # 모델 아키텍처
├── notebooks/           # EDA 및 분석 노트북
├── outputs/             # 예측 결과
├── trainer/             # 훈련 로직
├── utils/               # 유틸리티 함수 (메트릭, 시각화)
├── predict.py           # 추론 스크립트
├── README.md            # 프로젝트 문서
├── environment.yml      # Conda 환경 설정 파일 (권장)
├── requirements.txt     # Pip 요구사항 파일
└── train.py             # 훈련 스크립트
```

## ⚙️ 설치 및 환경 설정

이 프로젝트는 **Conda**를 사용하여 환경을 관리하는 것을 권장합니다. Conda는 Python 버전뿐만 아니라 CUDA와 같은 복잡한 라이브러리 의존성까지 안정적으로 관리해줍니다.

### 권장 방법: Conda와 `environment.yml` 사용하기

이 방법은 프로젝트에 필요한 모든 패키지(PyTorch, CUDA 포함)를 한 번에 설치하여 가장 안정적이고 재현성 높은 환경을 보장합니다.

1.  **Anaconda 또는 Miniconda 설치**
    아직 설치하지 않았다면, [Anaconda](https://www.anaconda.com/products/distribution) 또는 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)를 먼저 설치해주세요.

2.  **Conda 환경 생성**
    프로젝트의 루트 디렉토리에서 다음 명령어를 실행하여 `environment.yml` 파일로부터 `doc-classifier-env`라는 이름의 Conda 환경을 생성합니다.

    ```bash
    conda env create -f environment.yml
    ```

3.  **생성된 환경 활성화**
    ```bash
    conda activate doc-classifier-env
    ```

### 보조 방법: `pip`과 `requirements.txt` 사용하기

Conda를 사용하지 않는 환경에서는 `pip`을 사용하여 Python 패키지만 설치할 수 있습니다. 이 방법은 Python과 CUDA가 이미 시스템에 올바르게 설치되어 있다고 가정합니다.

1.  **Python 가상 환경 생성 및 활성화**
    ```bash
    # 가상 환경 생성
    python -m venv .venv

    # Linux/MacOS
    source .venv/bin/activate

    # Windows
    # .\.venv\Scripts\activate
    ```

2.  **의존성 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

## 💡 사용법

### 1. 설정 구성
`config/config.yaml` 파일을 열어 데이터셋 경로(`root_dir`)와 훈련에 필요한 하이퍼파라미터를 수정하세요.

### 2. 탐색적 데이터 분석 (EDA)
데이터의 특징을 파악하고 전처리 방향을 결정하기 위해 Jupyter Notebook을 실행합니다. 먼저, 프로젝트의 Conda 환경이 활성화되었는지 확인하세요.

```bash
# (doc-classifier-env)
jupyter lab notebooks/01_EDA.ipynb

3. 모델 훈련
다음 명령어로 모델 훈련을 시작합니다. 훈련 로그는 logs/ 디렉토리에, 모델 가중치는 checkpoints/ 디렉토리에 저장됩니다.

# (doc-classifier-env)
python train.py --config config/config.yaml

4. 추론 실행
훈련된 모델을 사용하여 새로운 이미지에 대한 예측을 수행합니다.

# (doc-classifier-env)
python predict.py --checkpoint checkpoints/best_model.pth --input /path/to/your/image_or_dir --output outputs/predictions.csv
