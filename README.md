# 📄 Document Image Classification with Deep Learning

## Team

| ![이경도](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이승민](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최웅비](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이상원](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김재덕](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이경도](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_2/tree/james)             |            [이승민](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_2/tree/lsw)             |            [최웅비](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_2/tree/wb2x)             |            [이상원](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_2/)             |            [김재덕](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_2/)             |
|                            팀장, 모델 아키텍처 설계                            |                            데이터 전처리 및 증강                            |                            분석 도구 개발                            |                            실험 관리 및 최적화                            |                            평가 및 시각화                            |

## 0. Overview

### Environment
- **OS**: Ubuntu 20.04 LTS
- **Python**: 3.8+
- **GPU**: CUDA 11.8+ compatible
- **Framework**: PyTorch 2.0+, Hydra 1.2+, WandB

### Requirements
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Hydra-core >= 1.2.0
- wandb >= 0.15.0
- opencv-python >= 4.7.0
- albumentations >= 1.3.0
- pandas >= 1.5.0
- scikit-learn >= 1.2.0
- timm >= 0.9.0
- fire >= 0.5.0
- icecream >= 2.1.3

## 1. Competition Info

### Overview
17개 문서 유형을 분류하는 이미지 기반 딥러닝 경진대회입니다. 다양한 문서 스타일, 회전, 조명 조건에서의 강건한 분류 성능이 요구됩니다. 현재 79% 테스트 정확도를 달성했으며, 도메인 적응을 통한 성능 개선을 목표로 합니다.

### Timeline
- **시작일**: 2024년 12월 1일
- **최종 제출 마감**: 2025년 2월 28일
- **중간 평가**: 2025년 1월 15일
- **최종 발표**: 2025년 3월 5일

## 2. Components

### Directory

```
document-classifier/
├── src/                          # 📦 핵심 소스 코드
│   ├── data/                     # 💾 데이터 로딩, 전처리, 증강
│   │   ├── csv_dataset.py        # CSV 기반 데이터셋 로더
│   │   ├── augmentation.py       # 문서별 맞춤 증강
│   │   └── dataset_multiplier.py # K-fold 데이터셋 생성
│   ├── models/                   # 🧠 모델 아키텍처
│   │   ├── model.py              # ResNet50/ConvNeXt 모델
│   │   └── arcface.py            # ArcFace 손실 함수
│   ├── training/                 # 🏋️ 훈련 시스템
│   │   ├── trainer.py            # 기본 트레이너
│   │   └── conservative_augmentation_tester.py # 점진적 증강 테스터
│   ├── analysis/                 # 📊 분석 도구
│   │   ├── corruption_analyzer.py      # 도메인 갭 분석
│   │   ├── class_performance_analyzer.py # 클래스별 성능 분석
│   │   └── wrong_predictions_explorer.py # 오분류 탐색
│   ├── utils/                    # 🛠️ 유틸리티
│   │   ├── config_utils.py       # Hydra 설정 관리
│   │   ├── visual_verification.py # 시각적 검증
│   │   └── test_image_analyzer.py # 테스트 이미지 분석
│   └── inference/                # 🔮 추론 및 예측
│       ├── predictor.py          # 배치 예측 시스템
│       └── batch.py              # 대량 추론 처리
├── configs/                      # ⚙️ 실험 설정
│   ├── config.yaml               # 기본 설정
│   ├── experiment/               # 실험별 설정
│   │   ├── quick_debug.yaml      # 빠른 디버깅 (3 에포크)
│   │   ├── production_robust.yaml # 프로덕션 훈련 (30+ 에포크)
│   │   └── progressive_cross_phase_validation/ # 점진적 교차 검증
│   └── model/                    # 모델별 설정
│       ├── resnet50.yaml         # ResNet50 설정
│       ├── convnextv2.yaml       # ConvNeXt 설정
│       └── efficientnet.yaml     # EfficientNet 설정
├── data/                         # 📊 데이터
│   ├── raw/                      # 원본 데이터
│   │   ├── train/                # 훈련 이미지 (1,570장)
│   │   ├── test/                 # 테스트 이미지 (3,140장)
│   │   └── metadata/             # CSV 메타데이터
│   ├── augmented_datasets/       # 증강된 데이터셋
│   │   ├── v1_volume_10x/        # 10배 증강 데이터셋
│   │   ├── phase1_mild_fold_0/   # 1단계 경미한 증강
│   │   └── phase2_variety_fold_0/ # 2단계 다양한 증강
│   └── processed/                # 전처리된 데이터
├── scripts/                      # ▶️ 실행 스크립트
│   ├── train.py                  # Hydra 기반 모델 훈련
│   ├── predict.py                # Fire 기반 예측 실행
│   ├── generate_datasets.py      # 데이터셋 생성 CLI
│   └── setup-dev-user.sh         # 개발 환경 설정
├── notebooks/                    # 📝 EDA 및 분석 노트북
├── outputs/                      # 📤 결과물
│   ├── models/                   # 모델 가중치
│   │   ├── best_model.pth        # 최고 성능 모델
│   │   └── last_model.pth        # 최신 모델
│   ├── predictions/              # 예측 결과 (타임스탬프별)
│   ├── corruption_analysis/      # 손상 분석 결과
│   ├── class_performance_analysis/ # 클래스 성능 분석
│   └── visual_verification/      # 시각적 검증 결과
├── docs/                         # 📚 문서
│   ├── PROJECT_DIGEST.md         # 프로젝트 요약 (35K 토큰 압축)
│   ├── OVERVIEW.md               # 개요 및 아키텍처
│   └── USAGE_GUIDE.md            # 사용법 가이드
├── environment.yml               # Conda 환경 설정
├── requirements.txt              # Pip 요구사항
└── README.md                     # 프로젝트 문서
```

## 3. Data Description

### Dataset Overview
- **총 샘플 수**: 4,710장 (훈련: 1,570장, 테스트: 3,140장)
- **클래스 수**: 17개 문서 유형
- **이미지 규격**: 다양한 해상도 (중앙값: 443×591)
- **클래스 불균형**: 2.2:1 비율 (최대 100장, 최소 46장)
- **도메인 갭**: 훈련-테스트 간 회전각도 554% 차이

### EDA

**핵심 발견사항**:
- **회전 불일치**: 훈련 평균 1.92° vs 테스트 평균 12.57° (554% 차이)
- **조명 차이**: 테스트 데이터의 46% 과노출 vs 훈련 데이터의 20%
- **노이즈 유형 변화**: 훈련(임펄스 노이즈 59.5%) → 테스트(가우시안 노이즈 75.5%)
- **종횡비**: 0.75 중앙값, 대부분 세로형 문서

**클래스 분포**:
- **대형 클래스** (100장): 클래스 0,2,3,4,5,6,7,8,9,10,11,12,15,16
- **소형 클래스**: 클래스 1 (46장), 클래스 14 (50장), 클래스 13 (74장)

### Data Processing

**전처리 파이프라인**:
1. **이미지 정규화**: ImageNet 통계 사용 (mean=[0.485, 0.456, 0.406])
2. **크기 조정**: 224×224 (ResNet50) / 384×384 (ConvNeXt)
3. **메타데이터 검증**: 누락 파일 0% 확인

**증강 전략** (점진적 적용):
- **1단계**: ±15° 회전, 경미한 조명 변화 (강도 0.5)
- **2단계**: ±25° 회전, 중간 조명 변화 (강도 0.6)
- **3단계**: ±45° 회전, 전체 조명 변화 (강도 0.8)

**K-fold 교차검증**:
- 계층적 5-fold 분할
- 소스 레벨 분할로 데이터 누수 방지
- 클래스 균형 유지

## 4. Modeling

### Model Description

**주요 모델**: ResNet50 (2,350만 파라미터)
- ImageNet 사전 훈련된 가중치 사용
- ArcFace 손실 함수로 특징 학습 강화
- 17개 클래스를 위한 커스텀 분류 헤드

**대안 모델**: ConvNeXtV2-Base
- 384×384 해상도 지원
- 향상된 정확도와 효율성
- Drop Path 정규화 적용

**선택 이유**:
- 문서 이미지의 세밀한 특징 추출 능력
- 다양한 회전과 조명 조건에서의 강건성
- 전이 학습을 통한 빠른 수렴

### Modeling Process

**훈련 설정**:
- **옵티마이저**: AdamW (lr=0.0001, weight_decay=0.0001)
- **배치 크기**: 32 (ResNet50) / 16 (ConvNeXt)
- **에포크**: 25-30 (조기 종료 사용)
- **혼합 정밀도**: 활성화 (메모리 효율성)

**검증 전략**:
- 교차 단계 검증 (Cross-phase validation)
- 다른 증강 단계 데이터로 검증
- WandB를 통한 실시간 모니터링

**성능 추적**:
- 훈련/검증 손실 및 정확도
- F1-score, 정밀도, 재현율
- 클래스별 성능 매트릭스
- 혼동 매트릭스 시각화

## 5. Result

### Leader Board

**현재 성능**:
- **테스트 정확도**: 79%
- **훈련 정확도**: 89%
- **도메인 갭**: 10%
- **F1-score**: 0.78 (macro-average)

**순위**: 상위 25% (구체적 순위는 대회 종료 후 공개)

**성능 개선 목표**:
- 목표 테스트 정확도: 85-90%
- 도메인 갭 축소: 10% → 5% 이하
- 클래스별 균형 개선

### Presentation

- [최종 발표 자료](docs/presentation/final_presentation.pdf)
- [중간 발표 자료](docs/presentation/midterm_presentation.pdf)
- [기술 보고서](docs/technical_report.pdf)

## 6. Advanced Features & Analysis Tools

### 🔍 Corruption Analysis
```bash
# 포괄적인 손상 분석 실행
python -m src.analysis.corruption_analyzer run_comprehensive_analysis
```
- 훈련-테스트 간 도메인 갭 정량화
- 회전, 밝기, 블러, 노이즈 분석
- 시각적 비교 리포트 생성

### 📊 Class Performance Analysis
```bash
# 클래스별 성능 분석
python -m src.analysis.class_performance_analyzer analyze_class_performance
```
- 취약 클래스 식별
- 성능-손상 상관관계 분석
- 개선 우선순위 제안

### 🔍 Wrong Predictions Explorer
```bash
# 오분류 탐색 (예측 파일 필요)
python -m src.analysis.wrong_predictions_explorer explore_wrong_predictions \
    outputs/predictions/predictions_1234.csv
```
- HTML 갤러리 형태의 오분류 분석
- 패턴 식별 및 시각화
- 개선점 도출

### 🎯 Visual Verification
```bash
# 시각적 검증 도구
python -m src.utils.visual_verification run_visual_verification \
    --config_path configs/experiment/production_robust.yaml
```
- 증강된 데이터와 실제 테스트 조건 비교
- 증강 강도 검증
- 도메인 적응 효과 확인

## 7. Quick Start Guide

### 환경 설정
```bash
# Conda 환경 생성
conda env create -f environment.yml
conda activate doc-classifier-env

# 또는 Pip 설치
pip install -r requirements.txt
```

### 빠른 훈련 (디버깅)
```bash
# 3 에포크 빠른 디버깅
python scripts/train.py experiment=quick_debug

# 프로덕션 훈련 (30 에포크)
python scripts/train.py experiment=production_robust
```

### 예측 실행
```bash
# 최신 모델로 예측
python scripts/predict.py run --input_path data/raw/test --use-last

# 특정 체크포인트로 예측
python scripts/predict.py run --input_path data/raw/test \
    --checkpoint_path outputs/models/best_model.pth
```

### 분석 도구 실행
```bash
# 종합 데이터 분석
python -m src.analysis.corruption_analyzer run_comprehensive_analysis

# 클래스 성능 분석
python -m src.analysis.class_performance_analyzer analyze_class_performance
```

## 8. Configuration Management

### Hydra 기반 설정
- **기본 설정**: `configs/config.yaml`
- **실험별 설정**: `configs/experiment/`
- **모델별 설정**: `configs/model/`

### 실험 예시
```bash
# 다양한 실험 설정 사용
python scripts/train.py experiment=convnext_baseline
python scripts/train.py experiment=phase1_kfold_training
python scripts/train.py model=efficientnet train.epochs=50
```

## 9. Performance Optimization

### 현재 최적화 기법
- **혼합 정밀도 훈련**: 메모리 사용량 50% 감소
- **점진적 증강**: 안전한 성능 개선
- **교차 단계 검증**: 과적합 방지
- **ArcFace 손실**: 특징 학습 강화

### 계획된 개선사항
- **GradCAM 시각화**: 모델 해석성 향상
- **테스트 시간 증강**: 추론 정확도 향상
- **앙상블 모델**: 다중 모델 결합
- **하이퍼파라미터 자동 튜닝**: Optuna 연동

## etc

### Reference
- [ResNet 논문](https://arxiv.org/abs/1512.03385) - Deep Residual Learning
- [ConvNeXt 논문](https://arxiv.org/abs/2201.03545) - A ConvNet for the 2020s
- [ArcFace 논문](https://arxiv.org/abs/1801.07698) - Additive Angular Margin Loss
- [Albumentations](https://albumentations.ai/) - 데이터 증강 라이브러리
- [Hydra](https://hydra.cc/) - 설정 관리 프레임워크
- [WandB](https://wandb.ai/) - 실험 추적 플랫폼

### Acknowledgments
- **Upstage AI Lab** - 인프라 및 기술 지원
- **패스트캠퍼스** - 교육 프로그램 제공
- **팀원들** - 협업과 지식 공유