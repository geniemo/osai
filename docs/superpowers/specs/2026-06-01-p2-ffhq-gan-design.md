# P2: FFHQ-1024 GAN — Design

**Date**: 2026-06-01
**Deadline**: 2026-06-09 23:59 (이메일 제출, khshim@skku.edu)
**Strategy**: C — 하이브리드 (256 fine-tune → 512 native → 1024 native short phase)
**Workflow**: 데스크탑 dry-run만, Colab Pro+ single resumable long run

---

## 1. 목표와 제약

### 점수 목표
- 현실적 목표: **FID 45~50 → 8점**, 최대 욕심 FID <45 → 9점
- baseline 무손실(FID 66.1, 4점) + 추가 4~5점 = 안전선
- 학생 quality 점수는 1024 native 디테일로 평균 이상 노림

### Hard 제약 (project02.pdf)
- Generator ≤ 40M params (초과 시 5점 0)
- z: 512-dim, output: 3×1024×1024
- valid/test 학습 금지, 외부 데이터 금지, pretrained 금지
- 허용 lib: PyTorch, TorchVision, OpenCV/PIL/skimage/matplotlib, WandB, pytorch-fid
- 금지 lib: HuggingFace, Lightning, Accelerate, Albumentations 등 third-party
- 기존 repo 시작 금지 (단, 교수님 제공 baseline은 활용 가능)

### 제출
- `2020314315_project02.zip` = `src/` + `checkpoints/model.pth` + `2020314315_project02_report.pdf` (6p, 11pt) + `pyproject.toml` + `README.md`
- README에 학습/생성 방법 명시
- Generator만 제출 (Discriminator 제외)
- 리더보드는 별도 ONNX 제출: `(B,512) → (B,3,1024,1024)`

---

## 2. 아키텍처

### Generator (총 21.6M 예상, 40M 한참 미만)

baseline G256(21.2M)을 그대로 받고 끝단에 두 블록 추가. 채널은 baseline의 halving 패턴을 깨고 **고해상도에 더 많이 할당** (저해상도는 baseline 그대로 두므로 transfer-init이 자연스러움).

```
resolutions: [4, 8, 16, 32, 64, 128, 256, 512, 1024]
channels:
  4: 512, 8: 512, 16: 512, 32: 512,    # baseline (변경 X)
  64: 256, 128: 128, 256: 64,           # baseline (변경 X)
  512: 64, 1024: 32                    # NEW
attention_resolutions: [32]            # baseline 유지
norm_type: gn, gn_groups: 32
```

추가되는 모듈:
- `ResBlockUp(64→64)` 256→512 (new): conv1 3×3 64×64 + conv2 3×3 64×64 + skip Identity = ~73K
- `ResBlockUp(64→32)` 512→1024 (new): conv1 64×32 + conv2 32×32 + skip 1×1 64→32 = ~30K
- 새 `to_rgb`: Conv2d(32, 3, k=3) = ~0.9K
- 새 `out_norm`: GroupNorm(32, 32) = 64

총 추가 ≈ 105K. 기존 to_rgb(64→3) + out_norm(GN64)은 256 phase에서만 사용되고 512+ phase에서는 비활성화 (혹은 제거).

**최종 G ≈ 21.3M params**, 40M 대비 18.7M 여유 ⇒ 필요시 ch[512]/ch[1024]를 더 늘릴 수 있음 (예: 96/48까지 안전).

### Discriminator (대칭 확장)

```
resolutions: [1024, 512, 256, 128, 64, 32, 16, 8, 4]
channels:
  1024: 32, 512: 64,                   # NEW (G 대칭)
  256: 64, 128: 128, 64: 256,           # baseline
  32: 512, 16: 512, 8: 512, 4: 512     # baseline
use_spectral_norm: true
minibatch_std_group: 4
attention_resolutions: [32]
```

- `ResBlockDown(32→64)` 1024→512 (new): SN 적용, ~30K
- `ResBlockDown(64→64)` 512→256 (new): ~73K
- 새 `from_rgb`: Conv2d(3, 32, k=3) = ~0.9K
- baseline의 from_rgb(3→64)는 256 phase에서만 사용

총 D ≈ 20.3M. D는 hard threshold 없으므로 자유롭게 확장 가능. 필요시 ch[1024]/ch[512]를 64/128로 키워 capacity 보강.

### Phase별 활성 sub-network

Phase는 **resolution을 켰다 끄는** 방식으로 동작:

| Phase | G output | D input | active G modules | active D modules |
|---|---|---|---|---|
| 1 (256 ft) | 256 | 256 | baseline 그대로 | baseline 그대로 |
| 2 (512 main) | 512 | 512 | baseline + 256→512 block + 새 to_rgb_512 | from_rgb_512 + 512→256 block + baseline 이후 stage |
| 3 (1024 sharp) | 1024 | 1024 | baseline + 256→512 + 512→1024 + to_rgb_1024 | from_rgb_1024 + 1024→512 + 이후 |

phase 전환 시:
- G: 기존 to_rgb를 떼고 새 to_rgb를 붙임 (random init), 새 ResBlockUp도 random init
- D: 대칭으로 from_rgb 교체, 새 ResBlockDown random init
- baseline 부분 weights는 그대로 유지

---

## 3. 학습 일정 (Phase-driven training)

### Phase 1: 256 fine-tune (warm-up)
- **목적**: optimizer/EMA 상태 재구축, DiffAug + R1 학습 흐름 확인. baseline은 이미 5M images 학습됨 → 짧게.
- **데이터**: `train_50k_256.zip`
- **batch_size**: 64
- **이미지 수**: **150k** (~2k step)
- **init**: baseline `ffhq256_baseline.pt` → strict load (G/D/G_ema)
- **종료 조건**: 150k images 도달 → phase 전환

### Phase 2: 512 native main (핵심 품질 단계)
- **목적**: 1024 디테일의 70%가 결정되는 메인 단계
- **데이터**: `train_50k_512.zip`
- **batch_size**: 32 (L4 24GB 기준 안전)
- **이미지 수**: **1.2M**
- **init**: Phase 1 final ckpt에서 256까지 그대로, 256→512 ResBlockUp + 새 to_rgb는 random init. D 측 대칭.
- **종료 조건**: 1.2M images 도달

### Phase 3: 1024 sharpening (short)
- **목적**: 고해상도 디테일 마무리. 1024 학습은 메모리/속도 부담 크므로 단계 길이 짧게.
- **데이터**: `train_50k_1024-003.zip`
- **batch_size**: 8 (L4 24GB 기준)
- **이미지 수**: **300k**
- **init**: Phase 2 final ckpt에서 512까지 그대로, 512→1024 ResBlockUp + 새 to_rgb random init. D 대칭.
- **종료 조건**: 300k images 도달

### 총 이미지 예산
- ~1.65M images 학습
- L4 throughput 추정 (fp32, single GPU):
  - 256/batch64: ~150 img/s → 150k images ≈ 17분
  - 512/batch32: ~40 img/s → 1.2M images ≈ 8.3시간
  - 1024/batch8: ~7 img/s → 300k images ≈ 12시간
- 합계 약 **21시간** + Colab 세션 reconnect overhead. 9일 일정에 충분.

### Hyperparameters (baseline에서 그대로, 변경 금지)
```yaml
lr_g: 1e-3
lr_d: 1e-3
beta1: 0.0
beta2: 0.9          # 0.99 X
weight_decay: 0.0
r1_gamma: 10.0
r1_lazy_every: 16
grad_clip_g: 10.0
grad_clip_d: 100.0  # SN으로 사실상 무한
ema_half_life: 10_000
precision: fp32     # bf16은 phase 2 안정성 검증 후 옵션
augment: color,translation
flip: true
sample_seed: 12345
seed: 42
```

phase 전환 시 optimizer는 새로 만든다 (새 블록의 1st moment 초기화). 이전 phase의 baseline 부분 optimizer state는 의도적으로 버린다 (해상도 변경 = grad 통계 변화).

---

## 4. 코드베이스 구조

`p2/` 디렉토리 내부:

```
p2/
├── CLAUDE.md                # p2 특화 지침 (root CLAUDE.md에 add-on)
├── README.md                # 제출용 README (학습/생성 방법)
├── pyproject.toml           # 제출 의존성 명세
├── data/                    # gitignore. train_*.zip, valid_*.zip 압축 그대로
├── checkpoints/             # gitignore (.gitkeep만 유지)
│   ├── ffhq256_baseline.pt  # 교수님 제공
│   └── model.pth            # 제출용 final ckpt (Phase 3 EMA)
├── runs/                    # gitignore. WandB local, samples grid
├── configs/
│   ├── phase1_256.yaml
│   ├── phase2_512.yaml
│   └── phase3_1024.yaml
├── src/
│   ├── __init__.py
│   ├── model.py             # baseline model.py 확장 (Generator/Discriminator/EMA)
│   ├── losses.py            # baseline 그대로 (ns_logistic + r1)
│   ├── augment.py           # baseline DiffAug 그대로
│   ├── dataset.py           # baseline ZipImageDataset 그대로 + valid_zip 옵션
│   ├── transfer.py          # NEW: phase 전환 시 가중치 부분 로딩
│   ├── fid.py               # NEW: pytorch-fid wrapper, real-stats cache
│   └── utils.py             # seed/io/timing helpers
├── train.py                 # baseline train.py 확장 (phase 인식, transfer init)
├── generate.py              # baseline 그대로 (sample grid)
├── export_onnx.py           # baseline 그대로
├── eval_fid.py              # NEW: ckpt → samples 디렉토리 → FID 계산
├── package_submission.py    # NEW: p1 패턴, zip 패키징
└── colab/
    ├── COLAB.md             # Colab 운영 가이드 (resume, drive mount)
    └── colab_p2_main.ipynb  # 단일 노트북, phase 순차 실행
```

p1 폴더와 동등한 구조를 유지하여 일관성 확보. 기존 baseline zip의 `src/`, `train.py`, `export_onnx.py`는 **그대로 시작 코드**로 사용하고 변경분만 추가/수정.

### 변경 vs 유지

| 파일 | 정책 |
|---|---|
| `src/model.py` | baseline 그대로 + 새 GeneratorConfig 한 줄 추가. 빌딩 블록 (ResBlockUp/Down, SelfAttn, MinibatchStd, EMA) 수정 X |
| `src/losses.py` | 그대로 |
| `src/augment.py` | 그대로 |
| `src/dataset.py` | 그대로 (valid_zip은 별도 import에서 처리) |
| `train.py` | phase 인식 + transfer-init 호출만 추가. CLI에 `--phase {1,2,3}` 옵션 |
| `export_onnx.py` | 그대로. config-driven Generator 인스턴스화에 대응되도록 main 함수만 작은 수정 |
| `generate.py` | 그대로 |
| 신규 `src/transfer.py` | 핵심 — phase N+1로 갈 때 phase N의 G/D state를 부분 로드 |
| 신규 `src/fid.py` | pytorch-fid CLI 호출 래퍼, 실측 stats 캐시 |
| 신규 `eval_fid.py` | ckpt → 8k samples → FID 측정 (자기 점검용) |
| 신규 `package_submission.py` | p1 패턴 따라 zip 패키징 |

---

## 5. Phase 전환 (transfer-init)

`src/transfer.py`의 두 핵심 함수:

### `load_g_partial(G_new, prev_ckpt)`
- `prev_ckpt["G_state"]` (or `G_ema_state`) 의 키 중 새 G와 모양이 일치하는 키만 load
- shape mismatch나 새 키 (e.g. `stages.N`, `to_rgb`, `out_norm`)는 random init 유지
- 명시적 로깅: "Loaded X keys, skipped Y keys (shape mismatch / missing in target)"

### `load_d_partial(D_new, prev_ckpt)`
- 동일. D는 resolutions 순서가 역순임에 주의 (1024→...→4)
- 새 from_rgb와 새 ResBlockDown은 random init

phase 전환은 노트북 셀에서 명시적 sequence로:

```python
# phase 1 → phase 2
ckpt_p1 = torch.load("runs/p1/final.pt", weights_only=True)
G2 = Generator(GeneratorConfig.from_dict(yaml.load(phase2_cfg)["generator"]))
D2 = Discriminator(...)
G2_ema = EMA(G2, half_life=10_000)
load_g_partial(G2, ckpt_p1)
load_g_partial(G2_ema.shadow, ckpt_p1)  # EMA도 같은 baseline에서 출발
load_d_partial(D2, ckpt_p1)
# 새 optimizer
# train.py 호출 (resume 아님 — 새 phase의 첫 launch)
```

---

## 6. 데이터 파이프라인

- 학습: `ZipImageDataset` (baseline 그대로)
- valid: `valid_10k_{256,512,1024}.zip` — FID 측정 시 real-side 통계용
- **valid는 학습에 절대 사용 안 함** (project02.pdf 제약)
- 이미지 전처리: `[-1, 1]` 범위, horizontal flip 50%, DiffAug(color+translation)는 D 입력 직전에만
- num_workers: 8 (Colab은 4로 조정 가능)

---

## 7. FID 자기 측정 (`src/fid.py`, `eval_fid.py`)

리더보드 FID와 일치시키기 위해 `pytorch-fid`의 Inception V3을 그대로 사용:

```bash
# 실 통계 1회 캐시
python -m pytorch_fid p2/data/valid_10k_256_dir --save-stats p2/checkpoints/fid_stats_256.npz
python -m pytorch_fid p2/data/valid_10k_512_dir --save-stats p2/checkpoints/fid_stats_512.npz
python -m pytorch_fid p2/data/valid_10k_1024_dir --save-stats p2/checkpoints/fid_stats_1024.npz
```

각 phase 후반에서 EMA G로 8k samples 생성 → 디렉토리 저장 → `pytorch_fid` 호출. valid는 학습에 안 쓰므로 FID 측정에는 valid set이 적합 (분포가 train과 동일).

**중요**: 리더보드 FID는 valid set 기준일 가능성이 높음 (test는 grader 전용). 자체 측정은 valid 사용으로 통일.

각 phase에서 측정 주기:
- Phase 1: 종료시 1회 (sanity)
- Phase 2: 매 200k images마다 (총 6회)
- Phase 3: 매 100k images마다 (총 3회)

WandB에 FID를 시계열로 log.

---

## 8. ONNX 제출 export

baseline `export_onnx.py`의 `SubmissionWrapper`는 bilinear 1024 resize를 항상 적용 → 우리 G가 1024를 native로 내보내도 안전 (resize는 no-op).

```python
# 최종 단계
G = Generator(GeneratorConfig.from_dict(phase3_cfg["generator"]))
state = torch.load("p2/checkpoints/model.pth", weights_only=True)["G_ema_state"]
G.load_state_dict(state)
export_to_onnx(G, "p2/checkpoints/model.onnx")
```

검증:
```python
import onnxruntime as ort
sess = ort.InferenceSession("p2/checkpoints/model.onnx")
out = sess.run(None, {"z": np.random.randn(4, 512).astype(np.float32)})[0]
assert out.shape == (4, 3, 1024, 1024)
assert -1.05 <= out.min() and out.max() <= 1.05  # tanh + 약간의 fp 오차
```

리더보드 제출 시점 = phase 3 종료 직후. 5/20~6/9 사이 교수님이 occasional quality score를 매길 수 있다고 했으므로, **phase 2 종료 시점에 1회 중간 ONNX 제출**도 고려.

---

## 9. Resumable training (Colab 핵심)

baseline `train.py`는 이미 `--resume`을 정확히 구현:
- G/D/G_ema/optG/optD/RNG/wandb_run_id 모두 복원
- async_save_checkpoint로 비차단 저장

추가 신경 쓸 점:
- Colab disconnect 시 마지막 ckpt만 살아남음 → `ckpt_every: 50_000` 으로 빈도 ↑ (baseline 100k → 50k)
- Drive 마운트 시 동기화 지연 → run_dir는 Colab 로컬에 두고 주기적으로 `cp` 또는 `rsync`로 Drive에 백업
- 학습 launch 셀과 resume 셀을 노트북에서 분리 (헷갈림 방지)

phase 전환은 resume이 아니라 **새 phase의 first launch** — `--init-from` 대신 `transfer.load_*_partial` 사용. CLI 인터페이스에 `--phase-init <prev_ckpt>` 옵션 추가.

---

## 10. WandB

- 단일 project: `ffhqgen-skku-p2`
- run 3개:
  - `p2-phase1-256-ft`
  - `p2-phase2-512-native`
  - `p2-phase3-1024-sharp`
- 각 run에 phase 별 cfg 전체 dump (`wandb.config = cfg`)
- 주요 metric: `loss/D_total`, `loss/G`, `loss/R1`, `D_out/real_mean`, `D_out/fake_mean`, `grad_norm/G`, `grad_norm/D`, `throughput/imgs_per_sec`, `fid` (커스텀 측정 후 log)
- samples grid를 phase별 50k images마다 wandb image로 업로드

p1 패턴 따라 리포트에 WandB overview 캡처 첨부 (학생 quality 점수 + 리포트 5점 직결).

---

## 11. Schedule (9일)

| Day | Date | 작업 |
|---|---|---|
| D1 | 06-01 (오늘) | 설계 문서 confirm → 코드 골격 → 5분 데스크탑 dry-run (256 ft 단계 한 step) → Colab Phase 1 launch |
| D2 | 06-02 | Phase 1 완료 (자동) → Phase 2 launch. 데스크탑에서 리포트 골격 작성 시작 |
| D3~D6 | 06-03 ~ 06-06 | Phase 2 학습 (반나절~1일). Colab disconnect 시 24h마다 resume. 매일 FID 측정/WandB 확인. 리포트 본문 작성 |
| D7 | 06-07 | Phase 2 완료 → Phase 3 launch. 중간 ONNX 1회 leaderboard 제출 (quality score 노출용) |
| D8 | 06-08 | Phase 3 완료 → 최종 FID 측정 → ONNX 재export → 리더보드 최종 갱신. 리포트 마무리. 제출 zip 빌드 dry-run |
| D9 | 06-09 | 최종 검증 + 이메일 제출 (cushion 4시간 이상 확보) |

각 phase 학습은 자동 → 사람 시간은 거의 들지 않음. 사람 시간의 80%는 리포트 + 코드 정리 + WandB capture.

---

## 12. 위험과 대응

| 위험 | 대응 |
|---|---|
| Colab disconnect로 ckpt 손실 | ckpt_every=50k + Drive backup 주기 + 마지막 ckpt name 노트북 셀에 기록 |
| Phase 2 학습 divergence | baseline hyperparam 유지가 1차 방어. 발생 시 즉시 stop, 이전 ckpt에서 lr 50% 줄여 재개 (single retry). 그래도 실패 시 phase 2 길이를 줄이고 phase 3 skip → bilinear 1024로 제출 (= 옵션 A로 fallback) |
| 1024 학습 OOM | batch 4로 감축, 그래도 안 되면 1024 phase skip → ONNX wrapper bilinear (옵션 A 동작) |
| Param > 40M (실수) | train.py 시작 시 assert, package_submission.py에서도 재확인 |
| ONNX export 실패 | 데스크탑에서 phase 1 ckpt로 사전 검증 (D1 dry-run에 포함) |
| 학생 quality 점수 외곡 | 5/20~6/9 occasional scoring 단계에 중간 ONNX 1회 제출 (D7) → 거기서 받는 피드백으로 phase 3 마무리 조정 |
| Colab Pro+ 크레딧 소진 | 데스크탑 RTX 5070 Ti를 백업으로 보관 (사용자가 켜두기만 하면 됨). 단 정책상 데스크탑 사용은 사용자 승인 후 |

---

## 13. 코드 품질 / 리포트 5+5점 체크리스트

### 코드 5점 (project02.pdf 기준)
- [x] Clear module structure → p1과 동일한 구조
- [x] Reproducible training and evaluation scripts → train.py + eval_fid.py 단일 진입점
- [x] Proper checkpoint loading and saving → baseline의 검증된 resume 패턴
- [x] Readable implementation → docstring/타입 보존, baseline 스타일 유지
- [x] No hard-coded test-set assumptions → 데이터셋 경로는 모두 yaml config
- [x] Try to use many things learned in class → BN/IN/GN, optimizer, EMA, augmentation, regularization, SN, R1 모두 활용

### 리포트 5점 (6 pages, 11pt)
- [ ] Model architecture → diagram + channel table
- [ ] Training recipe → hyperparam 표 + phase별 시간/이미지 수
- [ ] Validation results → FID 시계열 + best score
- [ ] Ablation or trial history → 이번엔 ablation 없으나 phase별 FID 곡선이 trial history 역할
- [ ] Failure case analysis → 학습 후 EMA G로 64 샘플 grid, 실패 케이스 골라 분석 1페이지
- [ ] WandB evidence → 3 phase overview 캡처 + 시계열 그래프

p1 리포트 패턴(`p1/docs/report.md`)을 base로 재활용.

---

## 14. 의존성 (`pyproject.toml`)

baseline 그대로 + 자체 측정용:
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
    "scipy>=1.11",  # pytorch-fid 의존
]
```

---

## 15. 의사결정 기록 (왜 이렇게 했는지)

- **C를 선택한 이유**: A는 1024 native가 없어 학생 quality 점수가 baseline 외관에 발이 묶임. B는 progressive 3단계 native 전체 학습이 9일+단일 GPU에 무리. C는 baseline의 head-start를 살리면서 1024 native 디테일을 phase 3 짧은 학습으로 추가 확보.
- **D(StyleGAN scratch)를 안 한 이유**: baseline 5M-images 학습 가중치를 버려야 함. 9일 + 단일 GPU + 디버깅 사이클 0 환경에서 StyleGAN 재구현은 baseline FID 66.1조차 못 넘길 위험 실재.
- **Channel halving 규칙(...256:64, 512:32, 1024:16)을 안 따른 이유**: G의 40M 한도까지 18.7M 여유가 있음. 고해상도에 더 많은 채널을 할당하는 게 FID에 유리.
- **bf16 안 쓰는 이유**: baseline README에 "bf16 trained fine for ~3M images then a late spike was easier to diagnose in fp32"라는 명시. 9일에 디버깅 사이클이 없으므로 안정성 우선.
- **EMA G로 FID/제출하는 이유**: baseline README와 일치. EMA는 GAN 학습의 지터를 평탄화.
- **Phase 1 (256 ft)를 굳이 하는 이유**: baseline은 자체 학습이 끝난 상태지만 우리 환경(Colab L4)에서 optimizer 초기화 + DiffAug + R1 흐름이 정상 동작하는지 짧게 확인. 또한 baseline은 EMA half-life 10k 단위로 학습된 상태 — 같은 코드로 짧게 재학습하면 다음 phase 진입 시 weight 분포가 정렬된 상태로 시작.
