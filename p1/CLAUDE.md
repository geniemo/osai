# Project 1: Semantic Segmentation — Claude Guidelines

이 파일은 `p1/` 내부 작업 시 루트 `osai/CLAUDE.md`에 **추가로** 로드됩니다. 루트 공통 지침은 그대로 유지하면서 프로젝트 특화 규칙을 적용합니다.

## 1. Task

- **Semantic Segmentation**: 모든 픽셀을 semantic class로 분류하는 per-pixel classification
- **기본 metric**: mIoU (IoU per class including background → 평균)
  - `IoU = TP / (TP + FP + FN)`
- **ignore label** (VOC masks에서 255) → **loss와 mIoU 계산 모두에서 제외**
- **Loss**: CE loss 기본, 추가 loss(dice 등) 사용 가능
- **마감**: 5/5 23:59
- **배점**: 수업 전체의 35%

## 2. Library Policy

### 허용

- AI: **PyTorch, TorchVision**
- Vision: **OpenCV, Pillow, Scikit-Image, Matplotlib**, etc.
- Tracking: **WandB**

### 금지

- HuggingFace libraries (Datasets, TIMM, Hub, Evaluate, etc.)
- modeling, data processing, augmentation, training, evaluation을 지원하는 3rd party (PyTorch Lightning, Accelerate, **Albumentations**, etc.)
- 불확실한 경우 교수님에게 질문

### AI 도구

ChatGPT, Claude Code, Gemini 사용 장려. **리포트에 사용 내역 기록 필수**.

## 3. Data & Split Policy

사용 가능: **ImageNet-1K, MS-COCO, Pascal-VOC** (TorchVision 지원).

> **DO NOT USE THE VALIDATION SET FOR TRAINING** (원문 대문자 강조)

### Pascal-VOC: 20 classes + 1 background
- 여러 해의 VOC train split 병합 가능
- **val/test annotation을 학습에 사용 금지**

### MS-COCO: 80 classes + 1 background
- VOC 20개 클래스 포함 (class name mapping 필요, `pycocotools` 설치)
- 추가 학습용으로 사용 가능

### 평가
- 교수님이 별도 special dataset 준비
- **4/22 수업에서 test images 공지 예정**

## 4. Pretrained & Modeling Constraints

- **CNN만 사용** (no RNN, no Transformer)
- **Pretrained 정책**:
  - TorchVision이 지원하는 **image classification** pretrained 모델만 사용 가능
  - TorchVision의 semantic segmentation pretrained weight **사용 금지**
  - ImageNet-1K image classification으로 학습된 pretrained weight 사용 가능
  - **Quantized 모델 사용 금지**

## 5. Evaluation & Submission

### 채점 공식

```
S = S_mIoU × S_FLOPs + S_Code + S_Report
```

**중요**: S_mIoU × S_FLOPs는 **곱**이므로 둘 중 하나가 0이면 성능 점수 전체가 0이 됩니다.

- **S_mIoU**: closed test set mIoU 기준, 0-3-4-5 scale (absolute grading)
- **S_FLOPs**: 입력 `(1, 3, 480, 640)` 기준 측정, 0-3-4-5 scale
- **S_Code**: 5점 — clear module structure, reproducible scripts, proper checkpoint loading/saving, readable implementation, no hard-coded test-set assumptions, 수업 내용 활용
- **S_Report**: 5점 — model architecture, training recipe, validation results, ablation/trial history, failure case analysis, WandB evidence

### 미공지 사항

- **Threshold (0/3/4/5 기준값)**: 4/29 수업에서 공지 예정
- **Test images, submission site**: 4/22 수업에서 공지 예정

### 리포트

- PDF, **6페이지 이하**, **11pt** 본문
- WandB log + Overview page 캡처 필수
- **마감 후 수업 시간에 WandB 페이지 직접 확인** (공정성 검증)

### Zip 구조

```
2025xxxxxx_project01.zip
├── src/
├── checkpoints/model.pth
├── submit/
│   ├── img/   (빈 폴더로 제출)
│   └── pred/
├── 2025xxxxx_project01_report.pdf
├── pyproject.toml
└── README.md
```

### README.md 필수 항목

- 학습, 의존성 설치, 추론 실행 방법
- FLOPs 측정 방법
- 재현 방법: `submit/img/*.jpg` → `submit/pred/*.png` (동일 파일명, 예: 0001.jpg → 0001.png)

## 6. Reproducibility

### 워크플로우

- **데스크탑 (5070 Ti)**: 다양한 실험을 통해 **최적 솔루션** (아키텍처 + 레시피 + augmentation + loss 조합) 탐색
- **Colab L4 또는 T4**: 확정된 솔루션을 **전체 파이프라인(데이터→학습→평가) 처음부터 재실행** → 이 결과가 제출물

### 재현의 의미

"같은 설계를 재실행"하는 것이지 **"같은 mIoU 숫자가 나와야"** 하는 것이 아닙니다. GPU 비결정성 등으로 ±1-2% 편차는 자연스럽습니다.

### 필수 사항

- WandB overview에 **T4/L4 GPU 증거 필수**
- Colab Pro+ 사용
- **Resumable training**: 체크포인트 중간 저장 + resume 지원 (model/optimizer/scheduler/scaler/epoch/RNG state 포함, WandB run resume)

### dtype 주의

| GPU | bfloat16 | 권장 |
|---|---|---|
| T4 (Turing sm_75) | ❌ 미지원 | fp16 |
| L4 (Ada sm_89) | ✅ | fp16 또는 bf16 |
| 5070 Ti (Blackwell sm_120, CUDA 12.8+) | ✅ | fp16 또는 bf16 |

**안전한 선택**: fp16으로 통일 (T4에서도 동작).
