# P2: FFHQ-1024 GAN — Design

**Date**: 2026-06-01
**Deadline**: 2026-06-09 23:59 (이메일 제출, khshim@skku.edu)
**Strategy**: **D-256** — StyleGAN2-style 256 native scratch + bilinear 1024 wrapper
**Hardware**: Colab Pro+ **A100** (952 units → ~79h 가용)
**Fallback policy**: D-256 학습 실패 시 사용자와 별도 결정 (옵션 C — baseline 확장 — 을 그 시점에 구체화)

---

## 1. 목표와 제약

### 점수 목표
- **현실 목표**: FID 5~15 (256 native, bilinear 1024 후) → FID 점수 **10점** (FID < 40 구간)
- 총점 목표 **28~33점 / 35**, 기댓값 27~28점
- 단순화/단일-config 전략으로 분산 줄이고 학습 시간 79h 풀로 활용

### Hard 제약 (project02.pdf)
- Generator ≤ **40M params** (초과 시 5점 0). StyleGAN2 256 G는 ~24M으로 안전.
- z: 512-dim, output: 3×1024×1024 (bilinear wrapper로 256→1024)
- valid/test 학습 금지, 외부 데이터 금지, **pretrained 금지**
- 허용 lib: PyTorch, TorchVision, OpenCV/PIL/skimage/matplotlib, WandB, pytorch-fid
- 금지: HuggingFace, Lightning, Accelerate, Albumentations
- 교수님 제공 baseline은 활용 가능하나 **D-256 path에서는 사용 안 함** (StyleGAN과 ResNet baseline은 weight 호환 X)

### 제출
- `2020314315_project02.zip` = `src/` + `checkpoints/model.pth` + `2020314315_project02_report.pdf` (6p, 11pt) + `pyproject.toml` + `README.md`
- README에 학습/생성 방법 + 사용한 GPU(A100) 명시
- Generator만 제출 (Discriminator 제외)
- 리더보드 별도 ONNX 제출: `(B,512) → (B,3,1024,1024)` (256 native G + bilinear wrapper)

---

## 2. 아키텍처: StyleGAN2 (skip-G, modulated conv)

256 native scratch. baseline ResNet GAN과는 완전히 다른 아키텍처.

### Generator (StyleGAN2 skip generator)

```
z (512) → MappingNet → w (512)

const 4×4 (학습 가능 input) → Synthesis stages:
  4 → 8 → 16 → 32 → 64 → 128 → 256
각 stage: ModulatedConv (style=w_i) → AddNoise → BiasAct(LeakyReLU 0.2 + √2)
각 stage end: ToRGB(modulated 1×1, style=w_i) → bilinear upsample → 이전 RGB 누적 (skip)

최종 RGB at 256×256, range [-1, 1] (tanh 안 씀 — StyleGAN 관례)
```

### MappingNet
- 8 layers, FC 512→512
- LeakyReLU(0.2) + √2 gain
- **LR multiplier 0.01** (필수, 빠뜨리면 발산)
- Equalized LR 적용

### Channels (resolution → channels)
StyleGAN2 official FFHQ-256 패턴 적용:
```
4: 512, 8: 512, 16: 512, 32: 512,
64: 512, 128: 256, 256: 128
```

### Param budget (G)
- MappingNet (8 × 512×512): ~2.1M
- Const input: 4×4×512 = 8K
- Synthesis blocks (modulated 3×3 conv × 2 per stage + noise scale + bias):
  - 4: only style + bias (no upsample), 512×512×9 ≈ 2.4M
  - 8: 1 up + 1 same = 2× 512×512×9 = 4.7M
  - 16: 4.7M
  - 32: 4.7M
  - 64: 4.7M
  - 128: 512×256×9 + 256×256×9 = 1.2M + 0.6M = 1.8M
  - 256: 256×128×9 + 128×128×9 = 0.3M + 0.15M = 0.45M
- ToRGB (modulated 1×1 per stage): 약 0.5M total
- AddNoise scale, bias: ~10K
- **합계 ≈ 26M params** (40M 한참 미만)

### Discriminator (StyleGAN2 residual)

```
RGB 256×256 → FromRGB → Conv stages:
  256 → 128 → 64 → 32 → 16 → 8 → 4
각 stage: ResBlockDown (conv 3×3 → conv 3×3 → avgpool 2×) with √2 residual scale
MinibatchStd (group 4) at 4×4
FinalConv 3×3 + FinalLinear → scalar
```

- Spectral Norm 미적용 (StyleGAN2 관례: R1 regularization으로 충분)
- Equalized LR 적용
- 채널: G와 대칭
- **D ≈ 24M params** (D는 hard threshold 없음)

### 핵심 구현 요소 체크리스트

- [x] **Equalized learning rate** (모든 Conv2d, Linear에 runtime scale = He init)
- [x] **Modulated conv** (per-sample weight modulation + demodulation, group conv 트릭)
- [x] **Mapping LR multiplier 0.01**
- [x] **Skip-G ToRGB sum** (각 stage의 RGB를 bilinear upsample 후 누적)
- [x] **AddNoise per layer** (learnable scale)
- [x] **BiasAct** (bias + leakyReLU + √2 gain)
- [x] **MinibatchStd** at D's 4×4 (group 4)
- [x] **Path Length Regularization** for G (lazy every 8 G steps, target ~ EMA)
- [x] **R1 Regularization** for D (lazy every 16 D steps, γ=10)
- [x] **EMA G** for sampling (half-life by image count)
- [x] **Style mixing** at training (prob 0.9, random crossover layer)
- [x] **Truncation trick** at inference (ψ=0.7) — optional, FID 측정 시 ψ=1.0

---

## 3. 학습 일정 (D-256 단일 phase)

phase 없음. 단일 256 native config로 단일 학습.

### 핵심 hyperparameters (StyleGAN2-ADA 공식 + baseline 환경 조합)

```yaml
generator:
  z_dim: 512
  w_dim: 512
  resolution: 256
  channels: {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128}
  mapping_layers: 8
  mapping_lr_mul: 0.01
  style_mixing_prob: 0.9

discriminator:
  resolution: 256
  channels: {256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
  minibatch_std_group: 4

training:
  train_zip: data/train_50k_256.zip
  resolution: 256
  batch_size: 32                      # A100 40GB + mixed precision
  total_images: 14_000_000            # 79h × ~180 img/s 추정 상한
  num_workers: 8
  flip: true
  precision: bf16                     # A100에서 필수

  lr_g: 0.002                         # StyleGAN2 official
  lr_d: 0.002
  beta1: 0.0
  beta2: 0.99
  weight_decay: 0.0

  r1_gamma: 10.0
  r1_lazy_every: 16
  pl_weight: 2.0
  pl_lazy_every: 8
  pl_decay: 0.01                      # PL target EMA decay

  grad_clip_g: 0.0                    # StyleGAN은 보통 clip 안 함
  grad_clip_d: 0.0

  ema_kimg: 20                        # half-life ≈ 20k images (StyleGAN2 default for 256)

  augment: diffaug                    # 'color,translation' (cutout 제외)
  augment_prob: 1.0                   # DiffAug는 항상 적용

  ckpt_every: 200_000                 # 매 200k images
  fid_every: 500_000                  # 매 500k images
  fid_n_samples: 8000
  sample_every: 50_000
  sample_n: 64
  sample_seed: 12345
  seed: 42

wandb:
  project: ffhqgen-skku-p2
  name: d256-stylegan2-scratch
  mode: online

out:
  run_dir: runs/d256_main
```

### 학습량 예산
- 14M images / 32 batch = ~437k steps
- A100 mixed precision throughput 추정: **150~200 img/s @ 256 batch 32**
- 14M / 175 = ~22h GPU time (이론). Colab overhead 포함해서 **~25h**
- 79h 풀로 쓰면 25M images까지 가능. 14M로 잡은 건 보수적 추정 (수렴이 빠르면 일찍 중단).
- **stopping criteria**: FID가 5회 연속 측정에서 0.5 미만 변화 (수렴 신호)

### 학습 안정성 측정 (WandB)
- `loss/D_total`, `loss/G`, `loss/R1`, `loss/PL`
- `D_out/real_mean`, `D_out/fake_mean`
- `grad_norm/G`, `grad_norm/D`
- `pl_mean_path_length` (PL 정규화 target)
- `throughput/imgs_per_sec`
- `fid` (커스텀 측정 후 log)
- samples grid 50k images마다 wandb image

---

## 4. ABORT GATE (D-256 → C 별도 구체화 전환 기준)

다음 중 하나라도 발생 시 즉시 학습 중단:

| 시점 | 조건 | 의미 |
|---|---|---|
| 1h dry-run | throughput < 100 img/s @ bs32 mixed | 구현 비효율, 시간 부족 누적 |
| 6h (~3M images) | FID > 60 | 학습 망가짐 |
| 12h (~7M images) | FID > 30 | 수렴 속도 너무 느림 |
| 언제든 | G loss sustained > 5 (5 logging window 연속) | 학습 발산 |
| 언제든 | grad_norm/G or grad_norm/D > 100 sustained | gradient explosion |
| 언제든 | NaN/Inf in loss/grad | 즉시 abort |
| 언제든 | reproducible OOM (batch 16/8까지 줄였는데도) | 메모리 모델 결함 |

abort 시:
1. WandB run finalize
2. 마지막 정상 ckpt 보존
3. 사용자에게 abort 사유 + WandB 링크 보고
4. 사용자 결정 → 옵션 C(baseline 확장) 별도 구체화 시작

---

## 5. 코드베이스 구조

```
p2/
├── CLAUDE.md                       # p2 특화 지침 (root CLAUDE.md에 add-on)
├── README.md                        # 제출용 (학습/생성 방법, 사용 GPU)
├── pyproject.toml                   # 제출 의존성
├── data/                           # gitignore. train_50k_256.zip 사용
├── checkpoints/                    # gitignore (.gitkeep만 유지)
│   ├── ffhq256_baseline.pt         # 교수님 제공 (D path에서 사용 안 함, fallback 대비 보관)
│   └── model.pth                   # 제출용 (최종 EMA G state)
├── runs/                           # gitignore. WandB local, samples
├── configs/
│   └── d256.yaml                   # 단일 config
├── src/
│   ├── __init__.py
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── ops.py                  # EqualLinear, EqualConv2d, ModulatedConv2d, AddNoise, BiasAct
│   │   ├── mapping.py              # MappingNet
│   │   ├── synthesis.py            # SynthesisNet (skip-G structure)
│   │   ├── generator.py            # Generator = Mapping + Synthesis
│   │   ├── discriminator.py        # ResBlockDown D
│   │   └── ema.py                  # EMA G (image-count half-life)
│   ├── losses.py                   # ns_logistic_g/d, r1_penalty, path_length_penalty
│   ├── augment.py                  # DiffAug color+translation (baseline 그대로)
│   ├── dataset.py                  # ZipImageDataset (baseline 그대로)
│   ├── fid.py                      # pytorch-fid wrapper, real-stats cache
│   ├── transfer.py                 # 빈 placeholder (D path에서는 불필요, C fallback 대비)
│   └── utils.py                    # seed/io/timing helpers
├── train.py                        # 단일 진입점
├── generate.py                     # sample grid (지정 ckpt → png)
├── eval_fid.py                     # ckpt → 8k samples → FID
├── export_onnx.py                  # G(256 native) + bilinear 1024 wrapper → submission.onnx
├── package_submission.py           # zip 패키징
└── colab/
    ├── COLAB.md                    # Colab 운영 가이드 (Drive mount, A100 reconnect/resume)
    └── colab_p2_d256.ipynb         # 단일 노트북
```

p1 폴더와 일관된 구조. baseline의 `src/{model,losses,augment,dataset}.py`는 별도 보관(`p2/baseline/`)하여 fallback 시 참조용으로 유지.

---

## 6. Path Length Regularization 상세 (StyleGAN2 핵심)

```python
def path_length_penalty(G, w, pl_mean: torch.Tensor, decay: float = 0.01):
    """Lazy PL — 8 G steps마다.
    
    w: (B, num_layers, w_dim) mapping 출력
    pl_mean: running EMA of path lengths (scalar, persistent buffer)
    """
    noise = torch.randn_like(G_out) / sqrt(H * W)
    pl_grads = autograd.grad(
        (G(w) * noise).sum(), w, create_graph=True
    )[0]
    pl_lengths = pl_grads.square().sum(dim=2).mean(dim=1).sqrt()
    pl_mean_new = pl_mean.lerp(pl_lengths.mean(), decay)
    pl_mean.copy_(pl_mean_new.detach())
    pl_penalty = (pl_lengths - pl_mean_new).square().mean()
    return pl_penalty  # 곱하기 pl_weight (2.0) × pl_lazy_every (8)
```

`pl_mean`을 학습 시작 시 0으로 초기화, 점차 lengths의 EMA로 수렴.

---

## 7. Modulated Conv 상세 (StyleGAN2 핵심)

```python
class ModulatedConv2d(nn.Module):
    """Per-sample weight modulation + demodulation.
    
    Implementation trick: reshape weight as group convolution.
    """
    def __init__(self, in_ch, out_ch, kernel, style_dim, demodulate=True, up=False):
        # weight (out_ch, in_ch, k, k), equalized LR scale
        # affine: w → style (in_ch) via Linear

    def forward(self, x, w):
        B, C, H, W = x.shape
        style = self.affine(w) + 1  # (B, in_ch), bias init 1
        weight = self.weight * style.view(B, 1, C, 1, 1)  # (B, out_ch, in_ch, k, k)
        if self.demodulate:
            d = (weight.square().sum(dim=[2,3,4]) + 1e-8).rsqrt()  # (B, out_ch)
            weight = weight * d.view(B, -1, 1, 1, 1)
        # Group conv trick:
        weight = weight.view(B * self.out_ch, C, k, k)
        x = x.view(1, B * C, H, W)
        x = F.conv2d(x, weight, groups=B, padding=k//2)
        x = x.view(B, self.out_ch, H_out, W_out)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
```

**check**: ONNX export에 group conv가 dynamic batch에서 잘 trace되는지 확인 필요 (opset 17). 안되면 batch dim 따로 unroll 또는 일반 conv with loop.

---

## 8. 데이터 파이프라인

- 학습: `ZipImageDataset(train_50k_256.zip, flip=True)` — baseline 그대로
- valid: `valid_10k_256.zip` — FID 측정 시 real-side 통계용
- valid는 학습에 절대 사용 안 함

### FID real-stats 1회 캐시

```bash
# valid_10k_256.zip → 디렉토리로 풀고 (또는 zip-stream으로 직접)
python -m pytorch_fid /path/to/valid_dir --save-stats p2/checkpoints/fid_stats_256.npz
```

이후 self-FID 측정은:
```bash
# G로 8k 샘플 dump → 디렉토리
python eval_fid.py --ckpt runs/d256_main/ckpt_xxx.pt --n 8000 --out /tmp/samples
python -m pytorch_fid /tmp/samples p2/checkpoints/fid_stats_256.npz
```

리더보드 FID와의 차이는 ±2~3 예상 (test set vs valid set).

---

## 9. ONNX 제출 export

baseline `export_onnx.py`의 `SubmissionWrapper`를 약간 수정:
- 우리 Generator는 256 native, w 사용. `forward(z)` 직접 받음
- StyleGAN G의 forward는 내부에서 mapping → synthesis → RGB
- Wrapper는 G output을 bilinear 1024로 resize

```python
class SubmissionWrapper(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G
    def forward(self, z):
        x = self.G(z)  # (B, 3, 256, 256), range [-1, 1]
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        return x
```

ONNX export 위험: ModulatedConv의 group conv가 dynamic batch에서 opset 17에 trace 가능한지 D1 dry-run에서 확인. 안 되면 export-only 모드에서 batch=1 unroll.

---

## 10. Resumable training (Colab 핵심)

baseline `train.py`의 resume 패턴 거의 그대로 재활용:
- ckpt에 `G_state, D_state, G_ema_state, optG_state, optD_state, rng_state, wandb_run_id, pl_mean (PL target buffer), images_seen, step`
- `--resume <ckpt>`로 bit-for-bit 복원
- 24h Colab session 만료 시 자동 disconnect → 재접속 후 `--resume` 한 줄로 재개

**추가 안전장치**:
- `ckpt_every: 200_000` → 매 200k images (~20분에 1회)
- 학습 시작 시 마지막 ckpt 자동 감지 + resume prompt
- Drive backup: 매 1M images마다 last 3 ckpts를 Drive에 rsync

---

## 11. WandB

- project: `ffhqgen-skku-p2`
- run: `d256-stylegan2-scratch` (단일 run)
- config: cfg 전체 dump
- main metrics:
  - `loss/D_total`, `loss/D_real`, `loss/D_fake`, `loss/G`
  - `loss/R1`, `loss/PL`
  - `pl_mean_path_length`
  - `D_out/real_mean`, `D_out/fake_mean`
  - `grad_norm/G`, `grad_norm/D`
  - `throughput/imgs_per_sec`
  - `fid` (커스텀, 500k마다)
- samples grid 50k images마다 wandb image
- 리포트용 overview 캡처 1장 (학습 종료 시)

---

## 12. Schedule (9일)

| Day | Date | 작업 |
|---|---|---|
| **D1** | 06-01 (오늘) | 코드 작성 (StyleGAN2 G/D/Mapping/ModConv/EMA/PL/R1) + 데스크탑 5분 dry-run + Colab launch |
| D2 | 06-02 | 학습 모니터링 (1h throughput / 6h FID / 12h FID gate). 정상이면 계속. WandB 매일 확인. 리포트 골격 작성 |
| D3 | 06-03 | 학습 ~36h 누적. 8M images 도달 예상. FID 측정 |
| D4 | 06-04 | 학습 ~60h 누적. 11M images 도달 예상. FID 측정 |
| D5 | 06-05 | 학습 ~75h 누적. 13~14M images 도달. 수렴 확인 |
| D6 | 06-06 | 학습 종료 (또는 수렴 조기 종료). 최종 FID 측정. ONNX export 검증 |
| D7 | 06-07 | 중간 리더보드 제출. 리포트 본문 작성 (architecture, recipe, FID curve, failure cases) |
| D8 | 06-08 | 리포트 마무리, WandB 캡처, 제출 zip dry-run |
| D9 | 06-09 | 최종 검증 + 이메일 제출 (cushion ~6h) |

사람 시간의 대부분은 D6~D9의 리포트 + 패키징. 학습은 자동.

---

## 13. 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| StyleGAN2 구현 정확도 미달 (가장 큰 위험) | 학습 발산/수렴 실패 | 1h dry-run에서 loss/grad/throughput 다 정상 범위 확인. ABORT gate로 빠른 cutting |
| ModulatedConv ONNX export 실패 | 제출 불가 | D1 dry-run에 ONNX export sanity 포함. 실패 시 batch=1 unroll export |
| Path Length reg가 NaN 발생 | 학습 정지 | pl_grads에 finite check, NaN 발견 시 해당 step 건너뛰기 (lazy reg이므로 영향 적음) |
| Colab A100 가용성 (할당 실패) | 학습 시작 못함 | L4로 임시 시작 → A100 가용 시 resume. 단 L4는 throughput 절반 → 학습 시간 부족 가능 |
| Colab 24h session 만료 | 학습 중단 | resume 즉시 가능. ckpt_every=200k로 손실 최소화 |
| 컴퓨트 유닛 952 초과 사용 | 학습 강제 종료 | 학습 진행도와 unit 잔량 매일 점검. 70h 시점에 강제 종료 + 그 ckpt로 제출 |
| Param > 40M (실수) | 5점 0 | train.py 시작 시 assert, package_submission.py에서도 재확인 |
| bilinear 1024로 FID는 OK인데 학생 quality 점수 낮음 | 학생 quality 1~2점 | 받아들임. StyleGAN의 얼굴 자연스러움이 일부 상쇄 기대 |
| ADA 미구현으로 FFHQ-50k에서 overfit | FID 정체 | DiffAug로 시작 → 만약 D loss가 너무 빨리 0에 가까워지면 ADA 추가 구현 (시간 여유 있을 때) |

---

## 14. 코드 품질 / 리포트 5+5점 체크리스트

### 코드 5점
- [x] Clear module structure → `src/networks/{ops,mapping,synthesis,generator,discriminator,ema}.py` 모듈 분리
- [x] Reproducible training and evaluation scripts → `train.py`, `eval_fid.py` 단일 진입점
- [x] Proper checkpoint loading and saving → baseline의 검증된 resume 패턴 차용
- [x] Readable implementation → docstring + 타입 hint
- [x] No hard-coded test-set assumptions → 데이터셋 경로는 yaml config
- [x] Try to use many things learned in class → optimizer, EMA, augmentation, regularization (R1, PL), modulation 등

### 리포트 5점 (6 pages, 11pt PDF)
- [ ] Model architecture → StyleGAN2 diagram + channel table + param count
- [ ] Training recipe → hyperparam 표 + 79h on A100 + DiffAug + R1 + PL
- [ ] Validation results → FID 시계열 + 최종 best (자체 측정 + 리더보드)
- [ ] Ablation or trial history → 이번엔 ablation 없음. 시간순 FID 곡선 1장으로 갈음
- [ ] Failure case analysis → 학습 후 EMA G 64 샘플 grid에서 실패 케이스 골라 1페이지
- [ ] WandB evidence → overview 캡처 + 시계열 그래프

p1 리포트 패턴 재활용.

---

## 15. 의존성 (`pyproject.toml`)

```toml
[project]
name = "p2-ffhq-gan"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "torchvision>=0.16",
    "numpy>=1.24",
    "pillow>=10.0",
    "pyyaml>=6.0",
    "wandb>=0.16",
    "onnx>=1.15",
    "onnxruntime>=1.16",
    "pytorch-fid>=0.3",
    "scipy>=1.11",
]
```

---

## 16. 의사결정 기록 (왜 이렇게 했는지)

- **D-256 단독 (D-1024 안 함)**: 79h를 256에 풀로 투입 → StyleGAN2 FFHQ-256 수렴값(FID 5~10) 도달. bilinear 1024 wrapper는 pytorch-fid의 Inception 299×299 resize 때문에 FID에 미치는 영향 미미. 1024 native는 메모리/속도 압박이 커서 같은 시간에 FID가 더 안 좋게 끝날 위험.
- **C 옵션을 동시 작성 안 함**: 단일 코드 베이스로 단순화 + 학습 시간 집중. D 실패 시 별도 의사결정 후 C를 그 시점에 구체화.
- **StyleGAN2 (1) 채택, StyleGAN3 안 함**: StyleGAN3는 alias-free filters로 더 무겁고 학습 더 오래 걸림. 50k 이미지 + 79h A100에서는 StyleGAN2가 sweet spot.
- **DiffAug로 시작, ADA는 옵션**: DiffAug는 baseline에서 검증됨, 구현 단순. ADA는 적응 augmentation prob 관리 복잡. 학습 안정성 우선.
- **bf16 mixed precision**: A100에서 fp32 대비 3배 throughput. StyleGAN2-ADA 공식 코드도 mixed precision 표준.
- **lr 0.002, β2=0.99**: StyleGAN2-ADA 공식 hyperparam. baseline의 lr=1e-3, β2=0.9는 ResNet GAN 기준이라 사용 안 함.
- **EMA half-life 20k images**: StyleGAN2 default for 256. baseline의 10k는 ResNet GAN의 더 작은 capacity 모델 기준.
- **Style mixing 0.9**: StyleGAN2 default. disentanglement + 일반화.
- **Tanh 없음**: StyleGAN2 G는 마지막에 tanh를 안 씀. 출력 범위는 학습으로 [-1, 1] 근방에 수렴.
- **Truncation trick은 FID 측정 시 ψ=1**: FID 본래 정의에 맞춤. ψ=0.7은 quality assessment 시 사용 가능.
