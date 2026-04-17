# Agent Team Design — OSAI Workspace

> 성균관대 Open-Source AI Practice 수업의 과제/프로젝트에서 활용할 Claude Code 에이전트 팀 설계 문서.
> 브레인스토밍 일자: 2026-04-15 ~ 2026-04-18

---

## 1. File Structure

```
/home/park/workspace/osai/
├── CLAUDE.md                          # 루트 — 수업 전체 공통 지침
├── .gitignore
├── .claude/
│   ├── settings.json                  # 팀 기능 활성화
│   └── agents/                        # 12개 에이전트 정의 (project scope)
│       ├── model-architect.md
│       ├── loss-designer.md
│       ├── data-augmentation-engineer.md
│       ├── training-strategist.md
│       ├── efficiency-optimizer.md
│       ├── miou-specialist.md
│       ├── submission-checker.md
│       ├── colab-reproducer.md
│       ├── skeleton-guardian.md
│       ├── wandb-inspector.md
│       ├── math-tutor.md
│       └── debugger-competitor.md
├── p1/                                # Project 1
│   ├── CLAUDE.md                      # Project 1 특화 지침
│   └── ...
├── w2/ ~ w5/                          # 주간 과제
│   └── CLAUDE.md                      # 각 과제 특화 지침 (필요 시)
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-18-agent-team-design.md
└── README.md
```

### Scope 결정

- **Project scope** (`.claude/agents/`): 이 워크스페이스가 OSAI 수업 전용이므로 프로젝트 scope로 충분.
- **User scope** (`~/.claude/agents/`): 다른 과목이나 개인 프로젝트에서도 쓰려면 user scope로 이동.

### Git 정책

- `.claude/agents/*.md`, `.claude/settings.json`, `CLAUDE.md` → **git 포함** (데스크탑에서 clone 시 자동 적용)
- `.claude/teams/`, `.claude/tasks/` → **gitignore** (런타임 상태, 머신별 독립)

---

## 2. CLAUDE.md Architecture

Claude Code는 CLAUDE.md를 **계층적으로 로드**:
- `osai/CLAUDE.md` → 항상 로드
- `osai/p1/CLAUDE.md` → `p1/` 내부 작업 시 추가 로드
- `osai/w5/CLAUDE.md` → `w5/` 내부 작업 시 추가 로드

팀원도 동일하게 작업 디렉토리의 CLAUDE.md를 로드함. 단, 팀원은 리더의 대화 기록은 상속받지 않으므로 **CLAUDE.md가 모든 팀원에게 공유되는 유일한 지침서**.

### 2.1 Root `osai/CLAUDE.md` — 수업 전체 공통 (4개 섹션)

#### § 1. Environments

3머신 워크플로우:
- **노트북** (WSL2): 설계, 계획, 팀 구성
- **데스크탑** (RTX 5070 Ti): 실험 탐색, 다양한 ablation → 최적 솔루션 발견
- **Colab Pro+** (T4 또는 L4): 공식 제출용 학습 1회 실행

동기화: Git + GitHub. 코드와 Claude 설정 파일은 git으로 관리.
구체 재현 정책(Colab 필수 여부)은 각 과제/프로젝트의 CLAUDE.md에서 결정.

#### § 2. Agent Team Usage

판단 기준:
- **팀**: 2개 이상 에이전트가 서로의 결과를 보고 토론해야 할 때, 또는 같은 문제를 다른 관점에서 동시 탐색할 때
- **단발 subagent**: 하나의 에이전트가 독립적으로 답을 내면 충분할 때
- **단발 subagent 연속**: 여러 체크를 순차 실행하면 되는 경우

토큰 비용 경고: 팀은 팀원 수 × 컨텍스트만큼 토큰 소모. Max5 요금제에서 opus 팀원 다수는 usage 급격히 소진.
Hooks: 운영 중 점진적 추가 방침. 반복되는 실수/누락 패턴 발견 시 자동화.

#### § 3. Communication Style

- **깊고 정확한 응답 선호**. 성급히 단순화하지 말 것.
- 트레이드오프, 가정, 빠진 고려사항을 명시. 확신 낮은 부분은 그렇게 표기.
- 수학/선형대수 설명: **구체적 숫자 예시와 직관적 비유** 활용. 추상 수식보다 "이 행렬이 실제로 뭘 하는지" 보여주기.

#### § 4. Common Code Rules

- 교수님 스켈레톤 코드의 **주석, docstring, 시그니처를 절대 삭제하지 않는다**. `pass`나 `return None` 등 placeholder만 교체.
- 스켈레톤에 없는 **print/debug 출력을 추가하지 않는다**. 출력은 스켈레톤에서 명시적으로 요구한 경우에만.

### 2.2 Project/Assignment CLAUDE.md Template — 6개 섹션

| # | 섹션 | 설명 |
|---|---|---|
| 1 | **Task** | 과제/프로젝트 설명, metric, 마감, 배점 |
| 2 | **Library Policy** | 이 과제에 허용/금지 라이브러리 |
| 3 | **Data & Split Policy** | 데이터셋, 학습/검증 split 규칙 |
| 4 | **Pretrained & Modeling Constraints** | 모델링 제약 (CNN only 등), pretrained 정책 |
| 5 | **Evaluation & Submission** | 채점 공식, 제출 규정, zip 구조 |
| 6 | **Reproducibility** | Colab 재현 필요 여부, resume 정책 |

해당 과제에 없는 섹션은 생략.

### 2.3 `p1/CLAUDE.md` — Project 1: Semantic Segmentation

#### § 1. Task

- Semantic Segmentation: 모든 픽셀을 semantic class로 분류하는 per-pixel classification
- 기본 metric: **mIoU** (IoU per class including background → 평균)
  - `IoU = TP / (TP + FP + FN)`
- **ignore label** (VOC masks에서 255) → **loss와 mIoU 계산 모두에서 제외**
- CE loss 기본, 추가 loss (dice 등) 사용 가능
- 마감: 5/5 23:59, 배점: 35%

#### § 2. Library Policy

허용:
- AI: PyTorch, TorchVision
- Vision: OpenCV, Pillow, Scikit-Image, Matplotlib, etc.
- Tracking: WandB

금지:
- HuggingFace libraries (Datasets, TIMM, Hub, Evaluate, etc.)
- modeling, data processing, augmentation, training, evaluation을 지원하는 3rd party (PyTorch Lightning, Accelerate, Albumentations, etc.)
- 불확실하면 교수님에게 질문

AI 도구 (ChatGPT, Claude Code, Gemini) 사용 장려 → 리포트에 사용 내역 기록.

#### § 3. Data & Split Policy

- 사용 가능: **ImageNet-1K, MS-COCO, Pascal-VOC** (TorchVision 지원)
- **DO NOT USE THE VALIDATION SET FOR TRAINING**
- Pascal-VOC: 20 classes + 1 background
  - 여러 해 VOC train split 병합 가능
  - val/test annotation 학습 사용 금지
- MS-COCO: 80 classes + 1 background
  - VOC 20개 클래스 포함 (class name mapping 필요, pycocotools)
  - 추가 학습용으로 사용 가능
- 평가: 교수님이 별도 special dataset 준비 (4/22 수업에서 test images 공지 예정)

#### § 4. Pretrained & Modeling Constraints

- **CNN만 사용 (no RNN, no Transformer)**
- Pretrained 정책:
  - TorchVision이 지원하는 **image classification** pretrained 모델만 사용 가능
  - TorchVision의 semantic segmentation pretrained weight **사용 금지**
  - ImageNet-1K image classification으로 학습된 pretrained weight 사용 가능
  - **Quantized 모델 사용 금지**

#### § 5. Evaluation & Submission

채점 공식: **`S = S_mIoU × S_FLOPs + S_Code + S_Report`**
- **S_mIoU × S_FLOPs는 곱** → 둘 중 하나가 0이면 성능 점수 전체 0
- S_mIoU: closed test set mIoU, 0-3-4-5 scale (absolute grading)
- S_FLOPs: 입력 `(1, 3, 480, 640)` 기준 측정, 0-3-4-5 scale
- S_Code: 5점 — clear module structure, reproducible scripts, proper checkpoint, readable, no hard-coded test assumptions, 수업 내용 활용
- S_Report: 5점 — architecture, training recipe, validation results, ablation history, failure analysis, WandB evidence

미공지 사항:
- Threshold (0/3/4/5 기준값): **4/29 수업에서 공지 예정**
- Test images, submission site: **4/22 수업에서 공지 예정**

리포트: PDF, 6페이지 이하, 11pt 본문
WandB: log + Overview page 캡처 필수, **마감 후 수업 시간에 WandB 페이지 직접 확인**

Zip 구조:
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

README.md 필수 항목:
- 학습, 의존성 설치, 추론 실행 방법
- FLOPs 측정 방법
- 재현 방법: `submit/img/*.jpg` → `submit/pred/*.png` (동일 파일명, 예: 0001.jpg → 0001.png)

#### § 6. Reproducibility

- 데스크탑(5070 Ti)에서 **다양한 실험을 통해 최적 솔루션을 탐색**
- 확정된 솔루션을 **Colab L4 또는 T4에서 전체 파이프라인(데이터→학습→평가) 처음부터 재실행** → 이 결과가 제출물
- WandB overview에 T4/L4 GPU 증거 필수
- Colab Pro+ 사용
- **Resumable training**: 체크포인트 중간 저장 + resume 지원 필수 (model/optimizer/scheduler/scaler/epoch/RNG state 포함, WandB run resume)

---

## 3. Agent Definitions (12)

모든 정의는 `.claude/agents/*.md`에 저장. **팀 팀원으로도, 단발 subagent로도** 동일 파일 참조.

### 모델링 전담 (5개)

#### 3.1 `model-architect` — opus

```
description: CNN segmentation 모델 아키텍처 설계 및 성능-효율 트레이드오프 분석
```

- 역할: 전체 모델 아키텍처 설계. TorchVision pretrained backbone 선택 가이드, neck/head 구조 제안, 성능-효율 균형 분석
- 지식: TorchVision classification models 특성 (params, FLOPs, feature map 크기), segmentation neck (FPN, ASPP, PPM 등), upsampling 전략 (bilinear, deconv, PixelShuffle 등)
- 제약 인지: CNN만, TorchVision classification pretrained만, segmentation pretrained 금지, quantized 금지

#### 3.2 `loss-designer` — opus

```
description: Segmentation loss 함수 설계 — CE, Dice, Focal, auxiliary loss 조합 및 ignore label 처리
```

- 역할: loss 조합 설계, ignore label(255) 처리 검증, class imbalance 대응, auxiliary loss 배치 위치 제안
- 지식: CE/Dice/Focal/Boundary loss 특성, class frequency 기반 weighting, deep supervision

#### 3.3 `data-augmentation-engineer` — sonnet

```
description: 허용 라이브러리(OpenCV/PIL/torchvision) 기반 데이터 파이프라인 및 augmentation 설계
```

- 역할: DataLoader 구현, VOC/COCO 전처리, class name mapping, augmentation 파이프라인 구성, segmentation mask 동기화 (geometric transform 시 mask도 동일 변환)
- 지식: torchvision.transforms v2, OpenCV geometric/color transforms, PIL operations, joint image-mask transform 패턴
- 제약 인지: Albumentations 등 3rd party augmentation 라이브러리 금지

#### 3.4 `training-strategist` — opus

```
description: Optimizer, LR scheduler, mixed precision, EMA 등 학습 전략 설계 및 resumable training 구현
```

- 역할: optimizer 선택 (SGD vs AdamW), LR schedule (Cosine/Poly/Step + warmup), mixed precision (fp16 — T4 호환), grad clipping, EMA, batch size 결정, **resumable training 구현** (checkpoint에 model/optimizer/scheduler/scaler/epoch/RNG state 포함, WandB run resume)
- 제약 인지: 최종 학습은 Colab T4/L4에서 실행, resume 필수

#### 3.5 `efficiency-optimizer` — opus

```
description: FLOPs 예산 내 성능 극대화 — 모델 경량화 설계 및 FLOPs 측정
```

- 역할: FLOPs 측정 (입력 `(1,3,480,640)` 기준), 경량화 기법 (depthwise separable conv, channel scaling, depth scaling, dilated rate 조정), S_mIoU × S_FLOPs 곱 관계를 고려한 최적점 탐색
- 지식: thop/fvcore 등 FLOPs 측정 도구, 경량 CNN 설계 패턴

### 검증·제출 (4개)

#### 3.6 `miou-specialist` — sonnet

```
description: mIoU 계산 구현 정확성 검증 — ignore label, per-class IoU, background 포함 여부
```

- 역할: mIoU 구현 코드 검증, ignore label(255) 제외 확인, background class 포함 확인, class별 IoU 분석 (약한 class 식별)
- 지식: confusion matrix 기반 IoU 계산, TP/FP/FN 정의, edge case (해당 class가 GT에 없는 경우)

#### 3.7 `submission-checker` — sonnet

```
description: 제출 규정 준수 검증 — zip 구조, README 필수 항목, 파일명 매핑, 리포트 규격
```

- 역할: zip 구조 검증, README.md 필수 항목 확인, 파일명 매핑 (0001.jpg → 0001.png), 리포트 규격 (PDF/6p/11pt), pyproject.toml 존재, 스켈레톤 코드 주석 보존 여부 제출 시점 최종 체크

#### 3.8 `colab-reproducer` — sonnet

```
description: 로컬에서 확정된 솔루션의 Colab T4/L4 전체 파이프라인 재현 검증
```

- 역할: 데스크탑에서 탐색한 최적 솔루션이 Colab에서 데이터→학습→평가 전체 동작하는지 검증. dtype 호환 (T4는 bfloat16 미지원 → fp16), CUDA/PyTorch 버전 호환, 의존성 (uv.lock) Colab 동작 확인, resume 동작 검증

#### 3.9 `skeleton-guardian` — haiku

```
description: 교수님 스켈레톤 코드의 주석, docstring, 시그니처 보존 검증
```

- 역할: diff 기반으로 스켈레톤 원본 대비 주석/docstring/시그니처 변경 여부 체크, placeholder (`pass`, `return None`) 교체 외의 변경 경고
- 주 사용처: 주간 과제 (Project 1에서는 스켈레톤이 없으면 미사용)

### 지원 (2개)

#### 3.10 `wandb-inspector` — sonnet

```
description: WandB 로그 점검 — Overview 캡처 필수 항목, GPU 증거, config 기록, 리포트 증거 누락 확인
```

- 역할: WandB run에 필요한 정보 (loss curve, mIoU curve, learning rate, GPU info, config) 기록 검증, Overview page에서 리포트 증거 식별, **GPU 증거(T4/L4) 확인**, S_Report 요구사항과 매칭

#### 3.11 `math-tutor` — opus

```
description: 선형대수, 확률, 최적화 등 DL 수학 개념을 구체적 예시와 직관적 비유로 설명
```

- 역할: 수업 내용 중 수학적 개념 설명, 구체 숫자 예시, 직관적 비유 활용
- 사용자 특성 인지: 선형대수에 약함, 추상적 수식보다 "이 행렬이 실제로 뭘 하는지" 예시가 효과적

### 팀 전용 (1개)

#### 3.12 `debugger-competitor` — opus

```
description: 경쟁 가설 기반 병렬 디버깅 — 팀 기능으로 3-5명 소환해 원인 대립 조사
```

- 역할: 하나의 버그/문제에 대해 서로 다른 가설을 세우고 독립 조사. 다른 팀원의 가설에 적극적으로 반박 시도. 증거 기반으로 수렴.
- 팀 전용: 단발 subagent로는 효과 없음 (팀원 간 직접 통신·반박이 핵심)

### 모델 배정 요약

| 모델 | 에이전트 (개수) | 이유 |
|---|---|---|
| **opus** (6) | model-architect, loss-designer, training-strategist, efficiency-optimizer, math-tutor, debugger-competitor | 설계 판단력, 트레이드오프 분석, 깊은 추론 필요 |
| **sonnet** (5) | data-augmentation-engineer, miou-specialist, submission-checker, colab-reproducer, wandb-inspector | 구현 검증, 측정, 규칙 기반 체크 |
| **haiku** (1) | skeleton-guardian | 단순 diff 기반 패턴 매칭 |

### 에이전트 경계 검증

겹침 분석:
- model-architect ↔ efficiency-optimizer: **명확**. architect = "무엇을 만들지", optimizer = "어떻게 줄일지"
- loss-designer ↔ training-strategist: **명확**. loss = 모델 출력단 설계, training = 학습 루프
- loss-designer ↔ miou-specialist: **명확**. loss = 학습 목적 함수, miou = 평가 metric 구현
- submission-checker ↔ colab-reproducer: **명확**. submission = 형식 검증, colab = 실행 검증
- wandb-inspector ↔ colab-reproducer: **경계 조정 완료**. wandb-inspector가 GPU 증거 확인 일원화, colab-reproducer는 환경 호환성만

빠진 영역 해결:
- 데이터 파이프라인 (DataLoader/전처리/class mapping) → `data-augmentation-engineer`에 범위 확장으로 해결
- 리포트 드래프팅 → 메인 세션(리더)이 사용자와 대화하며 작성. 별도 에이전트 불필요

---

## 4. Settings & Hooks

### `.claude/settings.json` (프로젝트 scope — git 포함)

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### `~/.claude.json` (전역 — 각 머신, git 미포함)

`teammateMode: "auto"` (기본값, 변경 불필요):
- tmux 세션 안 → 분할 창
- 그 외 → in-process (Shift+Down으로 팀원 순환)

### Hooks 방침

지금은 미설정. 실제 팀 운영 후 반복되는 문제 패턴 발견 시 점진 추가:
- `TeammateIdle`: 팀원 유휴 시 결과 요약 강제
- `TaskCompleted`: 작업 완료 시 자동 검증 (lint, type check)
- `TaskCreated`: 작업 생성 시 형식/범위 검증

---

## 5. Usage Recipes

### 5.1 팀 vs 단발 판단 기준

| 조건 | 메커니즘 | 이유 |
|---|---|---|
| 2개 이상 에이전트가 서로 토론 필요 | **팀** | 팀원 간 직접 통신이 핵심 가치 |
| 같은 문제를 다른 관점에서 동시 탐색 | **팀** | 병렬 + 상호 반박 |
| 하나의 에이전트가 독립 답변 충분 | **단발 subagent** | 토큰 절약 |
| 여러 체크를 순차 실행 | **단발 subagent 연속** | 팀 병렬성 불필요 |

### 5.2 Project 1 Phase-by-Phase

| Phase | 시점 | 구성 | 메커니즘 |
|---|---|---|---|
| 1. 아키텍처 설계 | 프로젝트 초반 | architect + efficiency + loss | **팀** |
| 2. 데이터 + 평가 | 설계 확정 후 | data-aug + miou | 단발 병렬 |
| 3. 학습 루프 | 파이프라인 완성 후 | training-strategist | 단발 |
| 4. 실험 탐색 | 반복 | architect + training + efficiency | **팀** |
| 5. 디버깅 | 필요 시 | debugger × 3-5 | **팀** |
| 6. Colab 재현 | 솔루션 확정 후 | colab-reproducer | 단발 |
| 7. 제출 점검 | 마감 전 | submission + wandb + colab | **팀** |

### 5.3 프롬프트 예시

> 모델을 지정하지 않으면 에이전트 정의의 `model` 필드가 적용됨. 비용 관리가 필요하면 `"Use Sonnet for each teammate"` 추가 가능.

**아키텍처 설계 팀:**
```text
아키텍처 설계 팀을 만들어줘. Use Sonnet for each teammate.
- model-architect: TorchVision classification pretrained 중 backbone 후보 3개,
  각각에 어울리는 neck/head 구조 제안
- efficiency-optimizer: 각 후보의 FLOPs를 (1,3,480,640) 기준으로 추정,
  S_mIoU × S_FLOPs 곱 관계에서 최적점 분석
- loss-designer: 각 아키텍처에 맞는 loss 조합 제안 (CE 기본 + 보조 loss)
서로 결과 공유하고 토론해서 최종 추천안 1-2개로 수렴해줘.
```

**디버깅 팀:**
```text
디버깅 팀을 만들어줘. debugger-competitor 타입으로 4명 소환.
문제: validation mIoU가 20 epoch 이후로 45%에서 안 올라감.
train loss는 계속 내려가는데 val mIoU가 정체.
각자 다른 가설을 세우고 코드/로그를 조사해서 서로 반박하면서 원인 찾아줘.
```

**실험 방향 토론:**
```text
실험 방향 토론 팀을 만들어줘.
- model-architect: 현재 mIoU 65%, backbone을 바꿔야 할지 neck을 개선할지
- training-strategist: LR schedule이나 augmentation 변경으로 올릴 수 있는지
- efficiency-optimizer: 현재 FLOPs 대비 성능이 적절한지, 경량화 여지
현재 WandB 로그와 validation 결과를 분석하고 다음 실험 우선순위를 정해줘.
```

**제출 전 점검:**
```text
제출 전 점검 팀을 만들어줘.
- submission-checker: zip 구조, README 필수 항목, 파일명 매핑, 리포트 규격
- wandb-inspector: WandB Overview에 GPU 증거(T4/L4), loss/mIoU curve,
  config 기록 완성도, 리포트에 넣을 증거 누락
- colab-reproducer: Colab에서 실제 재현했을 때 환경 문제 없었는지 최종 확인
서로 결과 공유하고, 빠진 것 있으면 지적해줘.
```

### 5.4 주간 과제 활용

| 상황 | 에이전트 | 메커니즘 |
|---|---|---|
| 스켈레톤 코드 구현 후 주석 보존 확인 | `skeleton-guardian` | 단발 |
| 수업 개념 이해 (convolution, backprop 등) | `math-tutor` | 단발 |
| 구현이 안 될 때 원인 탐색 | `debugger-competitor` × 3명 | 팀 |

---

## 6. Environment & Reproducibility

### 3머신 동기화

| 데이터 종류 | 노트북 | 데스크탑 | Colab | 전송 방법 |
|---|---|---|---|---|
| 코드 + Claude 설정 | ✅ 작성 | ✅ 실행 | ✅ 재현 | **Git + GitHub** |
| 데이터셋 (VOC/COCO) | ✗ | ✅ 다운로드 | TorchVision auto 또는 Drive 마운트 | 현지 다운로드 |
| 체크포인트 (.pth) | ✗ | ✅ 탐색용 생성 | ✅ 공식 학습으로 생성 | 각자 생성 (전송 불필요) |
| WandB 로그 | - | ✅ 탐색 기록 | ✅ 공식 기록 | 계정 기반 자동 |
| `.claude/teams/*` 런타임 | 각자 | 각자 | 각자 | **gitignore** |

### Colab 재현 정책 (Project 1)

- 데스크탑에서 다양한 실험을 통해 **최적 솔루션** (아키텍처 + 레시피 + augmentation + loss 조합) 탐색
- 확정된 솔루션의 **코드와 config**를 Colab으로 가져가서 **처음부터 다시 학습**
- Colab에서 생성된 체크포인트와 WandB run이 **제출물**
- 데스크탑 체크포인트는 제출과 무관
- "재현"은 **"같은 설계를 재실행"**하는 것이지 **"같은 mIoU 숫자가 나와야"** 하는 것이 아님 (GPU 비결정성 등으로 ±1-2% 편차 자연스러움)

### CUDA/dtype 호환성

| GPU | 아키텍처 | bfloat16 | 비고 |
|---|---|---|---|
| T4 | Turing sm_75 | ❌ | fp16만 |
| L4 | Ada sm_89 | ✅ | |
| 5070 Ti | Blackwell sm_120 | ✅ | CUDA 12.8+ |

안전한 선택: **fp16으로 통일** (T4에서도 동작).

---

## 7. Key Decisions Log

| 결정 | 선택 | 이유 |
|---|---|---|
| CLAUDE.md 구조 | 계층형 (루트 + 과제별) | 과제마다 Library/Data/Rules가 다름 |
| 에이전트 scope | Project (`.claude/agents/`) | 워크스페이스가 OSAI 전용 |
| 역할 구체성 | OSAI 수업 특화 (B) | 수업 제약이 에이전트에 내장 → 매번 주입 불필요 |
| 에이전트 수 | 12개 | 모델링 5 + 검증 4 + 지원 2 + 팀전용 1 |
| model 배정 | opus 6 / sonnet 5 / haiku 1 | 원안 유지, Max5 요금제에서 호출 시점에 override 가능 |
| 팀 vs subagent | 양쪽 호환 정의, 호출 시 결정 | 정의 파일은 동일, CLAUDE.md에 판단 기준 가이드 |
| Hooks | 운영 후 점진 추가 | 아직 팀 운영 경험 없음 |
| 체크포인트 전송 | 불필요 | Colab에서 처음부터 재학습 |
| 학습 시간 제약 | 없음 | Colab Pro+ + resumable training |
| Colab 재현 의미 | 전체 파이프라인 재실행 | WandB overview에 T4/L4 GPU 증거 필요 |
