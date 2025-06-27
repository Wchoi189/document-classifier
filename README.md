# Doc Classifer
문서 이미지 분류를 위한 딥러닝 프로젝트입니다.

## 프로젝트 구조

```
doc-classifer/
├── checkpoints/          # 훈련된 모델 가중치
├── config/               # 설정 파일 (config.yaml)
├── data/                 # 데이터셋 및 증강 코드
├── inference/            # 추론 로직
├── logs/                 # 훈련 로그
├── models/               # 모델 아키텍처
├── notebooks/            # EDA 및 분석 노트북
├── outputs/              # 예측 결과
├── trainer/              # 훈련 로직
├── utils/                # 유틸리티 함수 (메트릭, 시각화)
├── predict.py            # 추론 스크립트
├── README.md             # 프로젝트 문서
├── requirements.txt      # 필요한 패키지
└── train.py              # 훈련 스크립트
```

## 설치

### 옵션 1: venv 사용

```bash
# 저장소 클론 (필요한 경우)
# git clone [repository_url]
# cd doc-classifer

# 가상환경 생성 및 활성화
python -m venv doc-venv

# Linux/MacOS
source doc-venv/bin/activate

# Windows
# .\doc-venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 옵션 2: pyenv 사용

```bash
# Python 3.10.11 설치
pyenv install 3.10.11

# 가상환경 생성
pyenv virtualenv 3.10.11 doc-pyenv

# 프로젝트에 로컬 환경 설정
pyenv local doc-pyenv

# 환경 확인
pyenv version

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 1. 설정 구성
데이터셋 경로(`root_dir`)와 하이퍼파라미터를 설정하기 위해 `config/config.yaml`을 수정하세요.

### 2. 탐색적 데이터 분석
데이터 탐색을 위해 Jupyter Notebook을 실행하세요:

```bash
jupyter lab notebooks/01_EDA.ipynb
```

### 3. 모델 훈련
다음 명령어로 훈련을 시작하세요. 결과는 `logs/`와 `checkpoints/`에 저장됩니다:

```bash
python train.py --config config/config.yaml
```

### 4. 추론 실행
훈련된 모델을 사용하여 예측을 수행하세요:

```bash
python predict.py --checkpoint checkpoints/best_model.pth --input /path/to/image_or_dir --output outputs/predictions.csv
```