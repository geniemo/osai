# Project 1 — Semantic Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pascal-VOC 21-class semantic segmentation pipeline (train → infer → ONNX export → 3-channel submission) achieving high mIoU under FLOPs budget at input (1, 3, 480, 640).

**Architecture:** Modular B(ResNet-50 + DeepLabV3+, OS=16) default + A(MobileNetV3-Large + LR-ASPP) backup, swappable via YAML. 2-stage training (COCO+VOC mixed pretrain → VOC trainaug finetune) with EMA, AMP fp16, resumable. Custom ONNX FLOPs counter. Three submission channels: codebase zip + 1000-PNG zip + structure-only ONNX.

**Tech Stack:** Python 3.12, uv, PyTorch >=2.7 (CUDA 12.8 binary for Blackwell sm_120), TorchVision (classification pretrained only), pycocotools, OpenCV, Pillow, WandB, onnx, pytest.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-04-24-p1-segmentation-design.md`
- Constraints: `osai/CLAUDE.md`, `p1/CLAUDE.md`
- Agent team mapping: `docs/superpowers/specs/2026-04-18-agent-team-design.md`
- Reference impls: `w4/models/deeplab_v3.py`, `w4/utils/compute_utils.py`, `w5/train.py`

---

## File Structure

### Created files (45)

```
p1/
├── pyproject.toml                                  # uv project, no HF
├── README.md                                       # 학습/설치/추론/FLOPs/재현/제출
├── tests/
│   ├── __init__.py
│   ├── test_seed.py
│   ├── test_metrics.py
│   ├── test_flops_pytorch.py
│   ├── test_flops_onnx.py
│   ├── test_checkpoint.py
│   ├── test_transforms.py
│   ├── test_voc.py
│   ├── test_coco.py
│   ├── test_data_builder.py
│   ├── test_models.py
│   ├── test_loss.py
│   ├── test_train_smoke.py
│   ├── test_infer_smoke.py
│   └── test_package_submission.py
└── src/
    ├── __init__.py
    ├── train.py                                    # 학습 진입점
    ├── eval.py                                     # val mIoU
    ├── infer.py                                    # input → output (TTA)
    ├── export_onnx.py                              # model → ONNX
    ├── measure_flops.py                            # PyTorch + ONNX FLOPs
    ├── package_submission.py                       # output → 채점용 zip 검증/패키징
    ├── config/
    │   ├── default.yaml                            # B default
    │   ├── light.yaml                              # A backup
    │   └── colab.yaml                              # Colab override
    ├── data/
    │   ├── __init__.py
    │   ├── builder.py                              # build_dataloaders
    │   ├── voc.py                                  # Pascal-VOC trainaug + val
    │   ├── coco.py                                 # COCO + VOC mapping + cache
    │   ├── transforms.py                           # joint image-mask
    │   └── download.py                             # 자동 다운로드
    ├── models/
    │   ├── __init__.py
    │   ├── builder.py                              # build_model swap
    │   ├── seg_model.py                            # SegmentationModel + export_mode
    │   ├── backbones/
    │   │   ├── __init__.py
    │   │   ├── resnet.py                           # ResNet50 OS=16
    │   │   └── mobilenet.py                        # MobileNetV3-Large dilated
    │   ├── necks/
    │   │   ├── __init__.py
    │   │   ├── aspp.py                             # ASPP DLv3+
    │   │   └── lr_aspp.py                          # LR-ASPP
    │   ├── heads/
    │   │   ├── __init__.py
    │   │   └── deeplabv3plus.py                    # decoder + classifier
    │   └── aux/
    │       ├── __init__.py
    │       └── fcn_head.py                         # 보조 CE head
    ├── losses/
    │   ├── __init__.py
    │   └── seg_loss.py                             # CE + Dice + Aux
    └── utils/
        ├── __init__.py
        ├── metrics.py                              # ConfusionMatrix mIoU
        ├── flops.py                                # PyTorch + ONNX counters
        ├── checkpoint.py                           # full state save/load
        ├── seed.py                                 # set_seed + RNG state
        └── viz.py                                  # mask 시각화
```

### Empty placeholder dirs (committed via `.gitkeep`)

```
p1/checkpoints/
p1/submit/img/                                      # PDF 요구: 빈 폴더
p1/submit/pred/
```

### Gitignored runtime dirs (이미 .gitignore에 있음)

```
p1/input/                                           # 1000 test images
p1/output/                                          # 추론 결과 누적
p1/data/                                            # VOC, COCO 다운로드 위치
p1/img.zip                                          # 원본 test image zip
```

---

## Phase 0 — Setup (Tasks 1-2)

### Task 1: pyproject.toml + uv sync

**Files:**
- Create: `p1/pyproject.toml`
- Create: `p1/.python-version`

- [ ] **Step 1: Write `p1/pyproject.toml`**

```toml
[project]
name = "osai-p1"
version = "0.1.0"
description = "OSAI Project 1 — Pascal-VOC semantic segmentation"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.7.0",
    "torchvision>=0.22.0",
    "numpy>=2.0",
    "pillow>=10.0",
    "opencv-python-headless>=4.10",
    "wandb>=0.18",
    "pyyaml>=6.0",
    "pycocotools>=2.0",
    "onnx>=1.17",
    "tqdm>=4.66",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `p1/.python-version`**

```
3.12
```

- [ ] **Step 3: Run `uv sync` and verify GPU**

Run: `cd p1 && uv sync`
Expected: 모든 패키지 설치 성공, `.venv/` 생성

Run: `cd p1 && uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"`
Expected: `2.x.x True NVIDIA GeForce RTX 5070 Ti (12, 0)` (sm_120 = capability (12, 0))

만약 capability 출력이 다르거나 cuda available이 False면 STOP — Colab Plan B로 전환 (spec §10 risk).

- [ ] **Step 4: Commit**

```bash
git add p1/pyproject.toml p1/.python-version p1/uv.lock
git commit -m "p1: init uv project with PyTorch CUDA 12.8 binary"
```

---

### Task 2: Project skeleton (디렉토리 + __init__.py + .gitkeep)

**Files:** 45 placeholder files. Use script.

- [ ] **Step 1: Create directory structure script**

```bash
cd p1

# src tree
mkdir -p src/config src/data src/models/backbones src/models/necks src/models/heads src/models/aux src/losses src/utils
mkdir -p tests
mkdir -p checkpoints submit/img submit/pred

# __init__.py 파일들
touch src/__init__.py src/data/__init__.py src/models/__init__.py
touch src/models/backbones/__init__.py src/models/necks/__init__.py
touch src/models/heads/__init__.py src/models/aux/__init__.py
touch src/losses/__init__.py src/utils/__init__.py
touch tests/__init__.py

# placeholder .gitkeep
touch checkpoints/.gitkeep submit/img/.gitkeep submit/pred/.gitkeep
```

- [ ] **Step 2: Verify structure**

Run: `cd p1 && find . -type d -not -path './.venv*' -not -path './.git*' | sort`
Expected: 위 구조 그대로 출력

- [ ] **Step 3: Commit**

```bash
git add p1/src p1/tests p1/checkpoints p1/submit
git commit -m "p1: add project skeleton (dirs + __init__.py + .gitkeep)"
```

---

## Phase 1 — Foundation Utilities (Tasks 3-7)

### Task 3: `src/utils/seed.py`

**Files:**
- Create: `p1/src/utils/seed.py`
- Test: `p1/tests/test_seed.py`

- [ ] **Step 1: Write failing test**

`p1/tests/test_seed.py`:
```python
import random
import numpy as np
import torch
from src.utils import seed as seed_mod


def test_set_seed_makes_random_deterministic():
    seed_mod.set_seed(42)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_mod.set_seed(42)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_get_set_rng_state_roundtrip():
    seed_mod.set_seed(0)
    state = seed_mod.get_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_mod.set_seed(123)               # 다른 시드로 교란
    seed_mod.set_rng_state(state)        # 복원
    actual = (random.random(), np.random.rand(), torch.rand(1).item())
    assert actual == expected


def test_worker_init_seeds_workers_distinctly():
    seed_mod.set_seed(0)
    seed_mod.worker_init_fn(0)
    s0 = (random.random(), np.random.rand())
    seed_mod.set_seed(0)
    seed_mod.worker_init_fn(1)
    s1 = (random.random(), np.random.rand())
    assert s0 != s1
```

- [ ] **Step 2: Run test (should fail)**

Run: `cd p1 && uv run pytest tests/test_seed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.seed'`

- [ ] **Step 3: Implement `src/utils/seed.py`**

```python
"""Reproducibility helpers: seed control + RNG state get/set."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
```

- [ ] **Step 4: Run test (should pass)**

Run: `cd p1 && uv run pytest tests/test_seed.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/utils/seed.py p1/tests/test_seed.py
git commit -m "p1: add seed utilities with RNG state save/load"
```

---

### Task 4: `src/utils/metrics.py` — ConfusionMatrix mIoU

**Files:**
- Create: `p1/src/utils/metrics.py`
- Test: `p1/tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

`p1/tests/test_metrics.py`:
```python
import torch
from src.utils.metrics import SegMetric


def test_perfect_prediction_yields_iou_one():
    m = SegMetric(num_classes=3, ignore_index=255)
    pred = torch.tensor([[0, 1, 2], [0, 1, 2]])
    target = torch.tensor([[0, 1, 2], [0, 1, 2]])
    m.update(pred, target)
    miou, per_class = m.compute()
    assert miou == 1.0
    assert per_class == [1.0, 1.0, 1.0]


def test_ignore_label_excluded_from_iou():
    m = SegMetric(num_classes=2, ignore_index=255)
    # target에 255 픽셀 → 그 위치는 통계에서 빠져야 함
    pred = torch.tensor([[0, 1, 0], [1, 0, 1]])
    target = torch.tensor([[0, 1, 255], [1, 0, 255]])
    m.update(pred, target)
    miou, _ = m.compute()
    # 255를 빼면 4 픽셀 모두 정확 → mIoU = 1.0
    assert miou == 1.0


def test_class_absent_from_gt_excluded_from_mean():
    # 클래스 2가 GT에도 pred에도 없으면 nanmean으로 평균에서 제외
    m = SegMetric(num_classes=3)
    pred = torch.tensor([[0, 1], [0, 1]])
    target = torch.tensor([[0, 1], [0, 1]])
    m.update(pred, target)
    miou, per_class = m.compute()
    # class 0, 1만 IoU=1, class 2는 NaN → mean = 1.0
    assert miou == 1.0
    assert per_class[0] == 1.0
    assert per_class[1] == 1.0
    assert per_class[2] != per_class[2]  # NaN check


def test_partial_overlap_iou():
    m = SegMetric(num_classes=2)
    # class 0: TP=1, FP=1, FN=1 → IoU = 1/3
    # class 1: TP=1, FP=1, FN=1 → IoU = 1/3
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 0, 1])
    m.update(pred, target)
    miou, per_class = m.compute()
    assert abs(per_class[0] - 1/3) < 1e-6
    assert abs(per_class[1] - 1/3) < 1e-6
    assert abs(miou - 1/3) < 1e-6
```

- [ ] **Step 2: Run test (should fail)**

Run: `cd p1 && uv run pytest tests/test_metrics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/utils/metrics.py`**

```python
"""Segmentation mIoU via confusion matrix accumulation.

Standard formula: IoU_c = TP_c / (TP_c + FP_c + FN_c).
- ignore_index 픽셀은 누적에서 제외 (loss와 일치).
- denom=0 클래스는 NaN → nanmean으로 평균에서 제외.
"""
from __future__ import annotations

from typing import Tuple, List

import torch


class SegMetric:
    def __init__(self, num_classes: int = 21, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # pred, target: any shape, dtype long (or int)
        pred = pred.flatten()
        target = target.flatten()
        valid = target != self.ignore_index
        pred = pred[valid].long()
        target = target[valid].long()
        # bincount-based confusion matrix (vectorized)
        idx = target * self.num_classes + pred
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += binc.view(self.num_classes, self.num_classes).to(self.cm.device)

    def compute(self) -> Tuple[float, List[float]]:
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.tensor(float("nan")))
        miou = torch.nanmean(iou).item()
        return miou, iou.tolist()
```

- [ ] **Step 4: Run test (should pass)**

Run: `cd p1 && uv run pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/utils/metrics.py p1/tests/test_metrics.py
git commit -m "p1: add SegMetric (mIoU) with ignore_index and nanmean handling"
```

---

### Task 5: `src/utils/flops.py` — PyTorch FLOPs counter

**Files:**
- Create: `p1/src/utils/flops.py` (PyTorch part only; ONNX part in Task 6)
- Test: `p1/tests/test_flops_pytorch.py`

기반: `w4/utils/compute_utils.py`. 그대로 가져오되 import 경로 + ONNX FLOPs 함수와 합칠 단일 모듈로.

- [ ] **Step 1: Write failing test**

`p1/tests/test_flops_pytorch.py`:
```python
import torch
import torch.nn as nn

from src.utils.flops import count_pytorch_flops


def test_single_conv_flops():
    # 1x1 conv: in_ch=3 → out_ch=8, input (1, 3, 4, 4) → output (1, 8, 4, 4)
    # MAC = 1*8*4*4*3*1*1 = 384
    m = nn.Conv2d(3, 8, kernel_size=1, bias=False)
    flops = count_pytorch_flops(m, input_size=(1, 3, 4, 4))
    assert flops == 384


def test_3x3_conv_with_groups():
    # depthwise 3x3 conv: in=8, out=8, groups=8, input (1, 8, 4, 4)
    # MAC = 1*8*4*4*(8/8)*3*3 = 1152
    m = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)
    flops = count_pytorch_flops(m, input_size=(1, 8, 4, 4))
    assert flops == 1152


def test_linear_flops():
    # linear: 16 → 4, input (1, 16) → MAC = 1*16*4 = 64
    m = nn.Linear(16, 4, bias=False)
    flops = count_pytorch_flops(m, input_size=(1, 16))
    assert flops == 64
```

- [ ] **Step 2: Run test (fail)**

Run: `cd p1 && uv run pytest tests/test_flops_pytorch.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement PyTorch FLOPs portion of `src/utils/flops.py`**

```python
"""FLOPs counters: PyTorch (forward hooks) + ONNX (graph traversal).

PyTorch counter는 Conv2d/Linear의 MAC만 카운트 (×2 안 함).
w4/utils/compute_utils.py 기반, w4 컨벤션 보존.

ONNX counter (count_onnx_flops)는 다음 task에서 추가.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _conv2d_flops(layer: nn.Conv2d, output: torch.Tensor) -> int:
    n, cout, hout, wout = output.shape
    kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * (layer.in_channels // layer.groups)
    return int(n * cout * hout * wout * kernel_ops)


def _linear_flops(layer: nn.Linear, output: torch.Tensor) -> int:
    n = output.shape[0]
    return int(n * layer.in_features * layer.out_features)


def count_pytorch_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 480, 640),
    device: str = "cpu",
) -> int:
    """Conv2d + Linear FLOPs (MAC). bias 무시. w4 컨벤션."""
    flops: Dict[int, int] = {}
    hooks = []

    def conv_hook(layer, _inp, out):
        flops[id(layer)] = _conv2d_flops(layer, out)

    def linear_hook(layer, _inp, out):
        flops[id(layer)] = _linear_flops(layer, out)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(input_size, device=device)
        model.to(device)(dummy)
    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    return int(sum(flops.values()))
```

- [ ] **Step 4: Run test (pass)**

Run: `cd p1 && uv run pytest tests/test_flops_pytorch.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/utils/flops.py p1/tests/test_flops_pytorch.py
git commit -m "p1: add PyTorch FLOPs counter (port of w4 compute_utils)"
```

---

### Task 6: `src/utils/flops.py` — ONNX FLOPs counter (extend)

**Files:**
- Modify: `p1/src/utils/flops.py` (add ONNX functions)
- Test: `p1/tests/test_flops_onnx.py`

- [ ] **Step 1: Write failing test**

`p1/tests/test_flops_onnx.py`:
```python
import io
import torch
import torch.nn as nn

from src.utils.flops import count_onnx_flops, count_pytorch_flops


def _export_to_temp(model: nn.Module, input_size, path):
    model.eval()
    dummy = torch.zeros(input_size)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["output"],
        opset_version=17, dynamic_axes=None,
    )


def test_single_conv_onnx_matches_pytorch(tmp_path):
    m = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    onnx_path = tmp_path / "single_conv.onnx"
    _export_to_temp(m, (1, 3, 16, 16), onnx_path)

    py_flops = count_pytorch_flops(m, (1, 3, 16, 16))
    onnx_flops, breakdown = count_onnx_flops(str(onnx_path), input_shape=(1, 3, 16, 16))
    assert py_flops == onnx_flops
    assert breakdown["Conv"] == py_flops


def test_resnet_block_onnx_within_5pct_of_pytorch(tmp_path):
    # 작은 cnn: conv → bn → relu → conv
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(16)
            self.c2 = nn.Conv2d(16, 8, 1, bias=False)
        def forward(self, x):
            return self.c2(torch.relu(self.b1(self.c1(x))))

    m = Tiny()
    onnx_path = tmp_path / "tiny.onnx"
    _export_to_temp(m, (1, 3, 32, 32), onnx_path)

    py = count_pytorch_flops(m, (1, 3, 32, 32))
    onx, _ = count_onnx_flops(str(onnx_path), input_shape=(1, 3, 32, 32))
    rel_diff = abs(py - onx) / py
    assert rel_diff < 0.05, f"PyTorch={py} ONNX={onx} diff={rel_diff:.3f}"
```

- [ ] **Step 2: Run test (fail)**

Run: `cd p1 && uv run pytest tests/test_flops_onnx.py -v`
Expected: FAIL — `count_onnx_flops` not defined

- [ ] **Step 3: Append ONNX counter to `src/utils/flops.py`**

```python
# === ONNX FLOPs counter (custom, no 3rd-party FLOPs lib) ===

from collections import defaultdict
from typing import Optional, Tuple as _T

import onnx
from onnx import shape_inference, numpy_helper


def _get_attr(node, name: str, default):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode()
    return default


def _build_shape_map(model: onnx.ModelProto) -> dict:
    shapes = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else None)
        shapes[vi.name] = dims
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def _conv_flops(node, shape_map) -> int:
    out = shape_map.get(node.output[0])
    w = shape_map.get(node.input[1])
    if out is None or w is None or any(x is None for x in out + w):
        return 0
    n, c_out, h_out, w_out = out
    _, c_in_per_g, k_h, k_w = w
    return n * c_out * h_out * w_out * c_in_per_g * k_h * k_w


def _gemm_flops(node, shape_map) -> int:
    a = shape_map.get(node.input[0])
    out = shape_map.get(node.output[0])
    if a is None or out is None or any(x is None for x in a + out):
        return 0
    m, k = a[-2], a[-1]
    n = out[-1]
    return m * k * n


def _matmul_flops(node, shape_map) -> int:
    return _gemm_flops(node, shape_map)


def count_onnx_flops(
    onnx_path: str,
    input_shape: _T[int, int, int, int] = (1, 3, 480, 640),
) -> _T[int, dict]:
    """ONNX 그래프 FLOPs (MAC). Conv/Gemm/MatMul만. BN/ReLU/Add/Resize는 0.

    가중치 제거된 model_structure.onnx도 OK — initializer.dims는 유지됨.
    """
    model = onnx.load(onnx_path)

    # 입력 shape 명시 (가변일 경우)
    inp = model.graph.input[0]
    for i, val in enumerate(input_shape):
        inp.type.tensor_type.shape.dim[i].dim_value = val

    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shape_map = _build_shape_map(inferred)

    total = 0
    breakdown: dict = defaultdict(int)
    for node in inferred.graph.node:
        op = node.op_type
        if op == "Conv":
            f = _conv_flops(node, shape_map)
        elif op == "Gemm":
            f = _gemm_flops(node, shape_map)
        elif op == "MatMul":
            f = _matmul_flops(node, shape_map)
        else:
            f = 0
        total += f
        breakdown[op] += f

    return total, dict(breakdown)
```

- [ ] **Step 4: Run test (pass)**

Run: `cd p1 && uv run pytest tests/test_flops_onnx.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/utils/flops.py p1/tests/test_flops_onnx.py
git commit -m "p1: add ONNX FLOPs counter via graph traversal"
```

---

### Task 7: `src/utils/checkpoint.py` — full state save/load

**Files:**
- Create: `p1/src/utils/checkpoint.py`
- Test: `p1/tests/test_checkpoint.py`

- [ ] **Step 1: Write failing test**

`p1/tests/test_checkpoint.py`:
```python
import torch
import torch.nn as nn

from src.utils.checkpoint import save_full, load_full, save_model_only


def _build_state():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 0.5 ** s)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    # 한 step 실행해서 state 채움
    for _ in range(2):
        opt.zero_grad()
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        sched.step()
    return model, opt, sched, scaler


def test_save_load_roundtrip(tmp_path):
    model, opt, sched, scaler = _build_state()
    ema = nn.Linear(4, 2); ema.load_state_dict(model.state_dict())
    path = tmp_path / "ckpt.pth"
    save_full(
        path=str(path), iter_count=100, stage=1,
        model=model, ema_model=ema, optimizer=opt, scheduler=sched, scaler=scaler,
        best_miou=0.5, wandb_run_id="abc", config={"foo": 1},
    )

    new_model = nn.Linear(4, 2)
    new_ema = nn.Linear(4, 2)
    new_opt = torch.optim.SGD(new_model.parameters(), lr=999.0)
    new_sched = torch.optim.lr_scheduler.LambdaLR(new_opt, lr_lambda=lambda s: 1.0)
    new_scaler = torch.amp.GradScaler('cuda', enabled=False)
    meta = load_full(
        path=str(path), model=new_model, ema_model=new_ema,
        optimizer=new_opt, scheduler=new_sched, scaler=new_scaler,
    )
    # state restored
    assert meta["iter"] == 100
    assert meta["stage"] == 1
    assert meta["best_miou"] == 0.5
    assert meta["wandb_run_id"] == "abc"
    assert meta["config"] == {"foo": 1}
    # optimizer momentum buffer transferred
    assert "momentum_buffer" in new_opt.state[list(new_opt.state.keys())[0]]


def test_save_model_only_strips_extras(tmp_path):
    model = nn.Linear(4, 2)
    path = tmp_path / "model.pth"
    save_model_only(str(path), model)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    assert "weight" in state and "bias" in state
```

- [ ] **Step 2: Run test (fail)**

Run: `cd p1 && uv run pytest tests/test_checkpoint.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/utils/checkpoint.py`**

```python
"""Checkpoint helpers: full training state vs model-only."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utils.seed import get_rng_state, set_rng_state


def save_full(
    *,
    path: str,
    iter_count: int,
    stage: int,
    model: nn.Module,
    ema_model: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    best_miou: float,
    wandb_run_id: Optional[str],
    config: Dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "iter": iter_count,
        "stage": stage,
        "model_state": model.state_dict(),
        "ema_state": ema_model.state_dict() if ema_model is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_miou": best_miou,
        "rng_state": get_rng_state(),
        "wandb_run_id": wandb_run_id,
        "config": config,
    }
    torch.save(state, path)


def load_full(
    *,
    path: str,
    model: nn.Module,
    ema_model: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state"])
    if ema_model is not None and state.get("ema_state") is not None:
        ema_model.load_state_dict(state["ema_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    set_rng_state(state["rng_state"])
    return {
        "iter": state["iter"],
        "stage": state["stage"],
        "best_miou": state["best_miou"],
        "wandb_run_id": state["wandb_run_id"],
        "config": state["config"],
    }


def save_model_only(path: str, model: nn.Module) -> None:
    """제출용: model_state만 저장 (가벼움, EMA model로 호출)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
```

- [ ] **Step 4: Run test (pass)**

Run: `cd p1 && uv run pytest tests/test_checkpoint.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/utils/checkpoint.py p1/tests/test_checkpoint.py
git commit -m "p1: add full-state checkpoint save/load with RNG"
```

---

## Phase 2 — Data Pipeline (Tasks 8-12)

### Task 8: `src/data/transforms.py` — joint image-mask transforms

**Files:**
- Create: `p1/src/data/transforms.py`
- Test: `p1/tests/test_transforms.py`

- [ ] **Step 1: Write failing test**

`p1/tests/test_transforms.py`:
```python
import torch
from PIL import Image
import numpy as np

from src.data.transforms import build_train_transform, build_val_transform


def _fake_pair(h=300, w=400):
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    mask = Image.fromarray(np.random.randint(0, 21, (h, w), dtype=np.uint8))
    return img, mask


def test_train_transform_returns_tensors_with_correct_dtypes():
    t = build_train_transform(crop_size=128, scale_range=(0.5, 2.0))
    img, mask = _fake_pair()
    out_img, out_mask = t(img, mask)
    assert isinstance(out_img, torch.Tensor)
    assert out_img.dtype == torch.float32
    assert out_img.shape == (3, 128, 128)
    assert out_mask.dtype == torch.long
    assert out_mask.shape == (128, 128)


def test_mask_values_in_valid_range_after_transform():
    t = build_train_transform(crop_size=128, scale_range=(0.5, 2.0))
    img, mask = _fake_pair()
    _, out_mask = t(img, mask)
    # 0~20 (class) 또는 255 (ignore, 패딩)
    unique = set(out_mask.unique().tolist())
    valid = set(range(21)) | {255}
    assert unique <= valid


def test_image_normalized_to_imagenet_stats():
    t = build_train_transform(crop_size=128, scale_range=(1.0, 1.0))  # no scale
    # 단색 이미지 (mean=0.485 → 정규화 후 0)
    arr = np.full((300, 400, 3), int(0.485 * 255), dtype=np.uint8)
    arr[..., 1] = int(0.456 * 255)
    arr[..., 2] = int(0.406 * 255)
    img = Image.fromarray(arr)
    mask = Image.fromarray(np.zeros((300, 400), dtype=np.uint8))
    out_img, _ = t(img, mask)
    # 평균이 0 근처 (정규화 후)
    assert abs(out_img.mean().item()) < 0.05


def test_val_transform_preserves_size_and_no_aug():
    t = build_val_transform()
    img, mask = _fake_pair(h=200, w=300)
    out_img, out_mask = t(img, mask)
    # val은 원본 크기 유지
    assert out_img.shape == (3, 200, 300)
    assert out_mask.shape == (200, 300)
```

- [ ] **Step 2: Run test (fail)**

Run: `cd p1 && uv run pytest tests/test_transforms.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/data/transforms.py`**

```python
"""Joint image-mask transforms using torchvision.transforms.v2.

Mask는 tv_tensors.Mask로 wrap → NEAREST + ignore=255 fill 자동 처리.
Color/blur/erasing 등 image-only 변환은 자동으로 mask 영향 없음.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _PairWrapper:
    """Compose v2 transform이 (image, mask) tuple을 받도록 wrap."""

    def __init__(self, transform: v2.Transform) -> None:
        self.transform = transform

    def __call__(self, image, mask) -> Tuple[Tensor, Tensor]:
        # PIL → tv_tensors
        img_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)
        out_img, out_mask = self.transform(img_tv, mask_tv)
        return out_img.as_subclass(torch.Tensor), out_mask.as_subclass(torch.Tensor).long()


def build_train_transform(
    crop_size: int = 480,
    scale_range: Tuple[float, float] = (0.5, 2.0),
) -> _PairWrapper:
    pipeline = v2.Compose([
        v2.ScaleJitter(target_size=(crop_size, crop_size), scale_range=scale_range, antialias=True),
        v2.RandomCrop(size=crop_size, pad_if_needed=True, fill={tv_tensors.Image: 0, tv_tensors.Mask: 255}),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.RandomGrayscale(p=0.1),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        v2.RandomErasing(p=0.25),
    ])
    return _PairWrapper(pipeline)


def build_val_transform() -> _PairWrapper:
    pipeline = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])
    return _PairWrapper(pipeline)
```

- [ ] **Step 4: Run test (pass)**

Run: `cd p1 && uv run pytest tests/test_transforms.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/data/transforms.py p1/tests/test_transforms.py
git commit -m "p1: add joint image-mask transforms (torchvision v2)"
```

---

### Task 9: `src/data/voc.py` — Pascal-VOC dataset

**Files:**
- Create: `p1/src/data/voc.py`
- Test: `p1/tests/test_voc.py` (smoke test, marked slow)

VOC 2012 trainaug ID list 생성: VOC 2012 train (1464) + SBD (9118) – val overlap.

- [ ] **Step 1: Write smoke test (skip if data missing)**

`p1/tests/test_voc.py`:
```python
import os
from pathlib import Path

import pytest
import torch

from src.data.voc import VOCSegDataset
from src.data.transforms import build_val_transform

VOC_ROOT = Path(os.environ.get("VOC_ROOT", "data/voc"))
HAS_VOC = (VOC_ROOT / "VOCdevkit/VOC2012/JPEGImages").exists()


@pytest.mark.skipif(not HAS_VOC, reason="VOC data not present")
def test_voc_val_loads_one_sample():
    ds = VOCSegDataset(root=str(VOC_ROOT), split="val", transform=build_val_transform())
    assert len(ds) == 1449
    img, mask = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape[0] == 3
    assert mask.dtype == torch.long
    # mask values in [0, 20] or 255
    unique = set(mask.unique().tolist())
    assert unique <= set(range(21)) | {255}
```

- [ ] **Step 2: Run test**

Run: `cd p1 && uv run pytest tests/test_voc.py -v`
Expected: SKIPPED (data 없음 — Phase 5에서 다운 후 다시)

- [ ] **Step 3: Implement `src/data/voc.py`**

```python
"""Pascal-VOC 2012 segmentation dataset (trainaug + val).

trainaug = VOC 2012 train (1464) + SBD extra (9118) – val overlap (= 10582 total).
SBD ID list는 download.py가 다운로드 후 'train_aug.txt'로 저장.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VOCSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,                       # "trainaug" or "val"
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.devkit = self.root / "VOCdevkit" / "VOC2012"

        # ID list 파일 경로
        if split == "trainaug":
            id_file = self.devkit / "ImageSets" / "Segmentation" / "train_aug.txt"
            mask_dir = self.root / "SegmentationClassAug"   # SBD merged masks
        elif split == "val":
            id_file = self.devkit / "ImageSets" / "Segmentation" / "val.txt"
            mask_dir = self.devkit / "SegmentationClass"
        else:
            raise ValueError(f"Unknown split: {split}")

        with open(id_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.img_dir = self.devkit / "JPEGImages"
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img = Image.open(self.img_dir / f"{img_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{img_id}.png")  # palette PNG, [0..20] + 255
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask
```

- [ ] **Step 4: Commit**

```bash
git add p1/src/data/voc.py p1/tests/test_voc.py
git commit -m "p1: add VOC trainaug/val dataset"
```

---

### Task 10: `src/data/coco.py` — COCO with VOC class mapping + mask cache

**Files:**
- Create: `p1/src/data/coco.py`
- Test: `p1/tests/test_coco.py` (smoke + unit)

- [ ] **Step 1: Write tests**

`p1/tests/test_coco.py`:
```python
import os
from pathlib import Path
import numpy as np
import pytest

from src.data.coco import COCO_TO_VOC, build_voc_mask_from_anns


def test_coco_to_voc_mapping_complete():
    # 정확히 20개 클래스 매핑
    assert len(COCO_TO_VOC) == 20
    # VOC class id는 1..20 모두 등장
    voc_ids = set(COCO_TO_VOC.values())
    assert voc_ids == set(range(1, 21))


def test_build_voc_mask_voc_class_painted():
    # 모의 annotation: cat (COCO 17 → VOC 8) 픽셀 마스크
    h, w = 4, 4
    cat_mask = np.zeros((h, w), dtype=np.uint8)
    cat_mask[:2, :2] = 1
    anns = [{"category_id": 17, "binary_mask": cat_mask}]  # 모의 annotation
    out = build_voc_mask_from_anns(anns, h, w)
    assert out[0, 0] == 8         # cat → VOC 8
    assert out[3, 3] == 0         # background


def test_build_voc_mask_non_voc_class_becomes_ignore():
    # COCO 51 (bowl) — VOC에 없음 → 255
    h, w = 4, 4
    bowl_mask = np.zeros((h, w), dtype=np.uint8)
    bowl_mask[:2, :2] = 1
    anns = [{"category_id": 51, "binary_mask": bowl_mask}]
    out = build_voc_mask_from_anns(anns, h, w)
    assert out[0, 0] == 255       # ignore
    assert out[3, 3] == 0         # background
```

- [ ] **Step 2: Run test (fail)**

Run: `cd p1 && uv run pytest tests/test_coco.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/data/coco.py`**

```python
"""COCO 2017 train (VOC subset) dataset.

핵심:
- 20-class hard-coded mapping (COCO id → VOC id)
- non-VOC class object → 255 (ignore) — false negative 방지
- mask 사전 캐싱: data/coco/coco_voc_masks/{image_id}.png (1회만 ~30-60분)
- 캐시된 mask가 없으면 지연 생성
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# torchvision/DeepLab 표준 매핑
COCO_TO_VOC = {
    1:15, 2:2, 3:7, 4:14, 5:1, 6:6, 7:19, 9:4, 16:3, 17:8,
    18:12, 19:13, 20:17, 21:10, 44:5, 62:9, 63:18, 64:16, 67:11, 72:20,
}


def build_voc_mask_from_anns(anns: List[dict], h: int, w: int) -> np.ndarray:
    """anns: list of {"category_id": int, "binary_mask": (h,w) uint8 0/1}.
    실제 사용 시 pycocotools annToMask 결과를 binary_mask로 전달.
    """
    out = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        m = ann["binary_mask"]
        cat = ann["category_id"]
        if cat in COCO_TO_VOC:
            out[m == 1] = COCO_TO_VOC[cat]
        else:
            out[m == 1] = 255
    return out


class COCOSegDataset(Dataset):
    def __init__(
        self,
        coco_root: str,
        split: str = "train2017",
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        from pycocotools.coco import COCO  # lazy import

        self.coco_root = Path(coco_root)
        self.split = split
        self.transform = transform
        self.img_dir = self.coco_root / split
        ann_file = self.coco_root / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(ann_file))
        # VOC 클래스 1개라도 포함된 이미지만 keep
        voc_cat_ids = list(COCO_TO_VOC.keys())
        keep_ids = set()
        for cid in voc_cat_ids:
            keep_ids.update(self.coco.getImgIds(catIds=[cid]))
        self.image_ids = sorted(keep_ids)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.coco_root / "coco_voc_masks")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _build_mask(self, image_id: int, h: int, w: int) -> np.ndarray:
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns_raw = self.coco.loadAnns(ann_ids)
        anns = [{"category_id": a["category_id"], "binary_mask": self.coco.annToMask(a)} for a in anns_raw]
        return build_voc_mask_from_anns(anns, h, w)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        cache_path = self.cache_dir / f"{image_id}.png"
        if cache_path.exists():
            mask_arr = np.array(Image.open(cache_path))
        else:
            mask_arr = self._build_mask(image_id, info["height"], info["width"])
            Image.fromarray(mask_arr, mode="L").save(cache_path)
        mask = Image.fromarray(mask_arr, mode="L")
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask
```

- [ ] **Step 4: Run test (pass)**

Run: `cd p1 && uv run pytest tests/test_coco.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/data/coco.py p1/tests/test_coco.py
git commit -m "p1: add COCO dataset with VOC class mapping + mask cache"
```

---

### Task 11: `src/data/download.py` — auto download VOC + SBD + COCO

**Files:**
- Create: `p1/src/data/download.py`

자동 다운로드 (재현성 핵심). 큰 파일이라 직접 호출.

- [ ] **Step 1: Implement `src/data/download.py`**

```python
"""자동 다운로드 + SBD merge + COCO mask cache 생성.

VOC 2012: TorchVision auto download.
SBD: berkeley mirror에서 수동 다운 + train_aug.txt 생성.
COCO: cocodataset.org에서 train2017 + annotations 다운 (~25GB).

호출:
    python -m src.data.download --voc-root data/voc --coco-root data/coco
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


SBD_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# 표준 trainaug ID list URL (AdvSemiSeg / SBD merged 결과)
TRAINAUG_URL = "https://www.dropbox.com/scl/fi/8gqxa8wxwa0jpe5xkcjzz/train_aug.txt?rlkey=ssvd0w6jrhqg5sl4hh7m6n9bp&dl=1"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} exists")
        return
    print(f"[get] {url} → {dest}")
    urlretrieve(url, dest)


def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} → {target}")
    if archive.suffix == ".tar" or str(archive).endswith(".tgz") or str(archive).endswith(".tar.gz"):
        with tarfile.open(archive) as tf:
            tf.extractall(target)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)


def download_voc(voc_root: Path) -> None:
    archive = voc_root / "VOCtrainval_11-May-2012.tar"
    _download(VOC_URL, archive)
    _extract(archive, voc_root)


def download_sbd_and_merge(voc_root: Path) -> None:
    """SBD를 다운로드 → benchmark/dataset/cls/*.mat 마스크를 PNG로 변환 →
    VOCdevkit/VOC2012/SegmentationClassAug에 통합. train_aug.txt도 다운."""
    sbd_archive = voc_root / "benchmark.tgz"
    _download(SBD_URL, sbd_archive)
    sbd_extract = voc_root / "sbd"
    if not (sbd_extract / "benchmark_RELEASE").exists():
        _extract(sbd_archive, sbd_extract)

    # MAT → PNG 변환
    out_mask_dir = voc_root / "VOCdevkit" / "VOC2012" / "SegmentationClassAug"
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    src_voc_mask = voc_root / "VOCdevkit" / "VOC2012" / "SegmentationClass"
    if src_voc_mask.exists():
        for p in src_voc_mask.iterdir():
            shutil.copy2(p, out_mask_dir / p.name)

    sbd_cls = sbd_extract / "benchmark_RELEASE" / "dataset" / "cls"
    if sbd_cls.exists():
        from scipy.io import loadmat
        from PIL import Image
        import numpy as np
        for mat_file in sbd_cls.glob("*.mat"):
            png_path = out_mask_dir / f"{mat_file.stem}.png"
            if png_path.exists():
                continue
            data = loadmat(mat_file)
            mask = data["GTcls"][0][0][1].astype("uint8")
            Image.fromarray(mask, mode="L").save(png_path)

    # train_aug.txt
    seg_dir = voc_root / "VOCdevkit" / "VOC2012" / "ImageSets" / "Segmentation"
    train_aug_path = seg_dir / "train_aug.txt"
    if not train_aug_path.exists():
        try:
            _download(TRAINAUG_URL, train_aug_path)
        except Exception as e:
            print(f"[warn] train_aug.txt download failed: {e}", file=sys.stderr)
            print("[warn] Build manually from SBD train + VOC train, excluding VOC val")


def download_coco(coco_root: Path) -> None:
    train_zip = coco_root / "train2017.zip"
    ann_zip = coco_root / "annotations_trainval2017.zip"
    _download(COCO_TRAIN_URL, train_zip)
    _download(COCO_ANN_URL, ann_zip)
    if not (coco_root / "train2017").exists():
        _extract(train_zip, coco_root)
    if not (coco_root / "annotations").exists():
        _extract(ann_zip, coco_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", default="data/voc")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--skip-voc", action="store_true")
    parser.add_argument("--skip-coco", action="store_true")
    args = parser.parse_args()
    voc_root = Path(args.voc_root)
    coco_root = Path(args.coco_root)
    if not args.skip_voc:
        download_voc(voc_root)
        download_sbd_and_merge(voc_root)
    if not args.skip_coco:
        download_coco(coco_root)
    print("[done]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add scipy to deps (SBD .mat 읽기용)**

`p1/pyproject.toml` dependencies에 추가:
```toml
"scipy>=1.13",
```

Run: `cd p1 && uv sync`
Expected: scipy 설치 성공

- [ ] **Step 3: Verify download script syntax**

Run: `cd p1 && uv run python -c "from src.data.download import main; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add p1/src/data/download.py p1/pyproject.toml p1/uv.lock
git commit -m "p1: add VOC/SBD/COCO download script"
```

---

### Task 12: `src/data/builder.py` — DataLoader builder + isolation guard

**Files:**
- Create: `p1/src/data/builder.py`
- Test: `p1/tests/test_data_builder.py`

- [ ] **Step 1: Write failing test**

`p1/tests/test_data_builder.py`:
```python
import pytest
from src.data.builder import build_dataloaders, _assert_not_test_path


def test_isolation_guard_rejects_input_path():
    # voc_root에 'input/' 포함 시 assertion 실패
    with pytest.raises(AssertionError, match="must not include 'input/'"):
        _assert_not_test_path("p1/input/test_public", role="voc_root")


def test_isolation_guard_accepts_data_path():
    _assert_not_test_path("data/voc", role="voc_root")           # OK
    _assert_not_test_path("./data/coco", role="coco_root")       # OK


def test_isolation_guard_blocks_relative_paths_too():
    with pytest.raises(AssertionError):
        _assert_not_test_path("./input/anywhere", role="coco_root")
```

- [ ] **Step 2: Run test (fail)**

Run: `cd p1 && uv run pytest tests/test_data_builder.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/data/builder.py`**

```python
"""DataLoader builder + test set 격리 가드 (개발 안전망)."""
from __future__ import annotations

from pathlib import PurePath
from typing import Tuple

from torch.utils.data import ConcatDataset, DataLoader

from src.data.coco import COCOSegDataset
from src.data.transforms import build_train_transform, build_val_transform
from src.data.voc import VOCSegDataset
from src.utils.seed import worker_init_fn


def _assert_not_test_path(path: str, *, role: str) -> None:
    """train data root는 'input/' 디렉토리를 포함해선 안 됨 (test set 격리)."""
    parts = PurePath(path).parts
    assert "input" not in parts, (
        f"{role}={path!r} must not include 'input/' — test set 격리 위반. "
        f"data 경로는 'data/voc' 등 input/ 외부에."
    )


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """cfg["data"]에서 path/배치/워커 읽어 train+val DataLoader 생성.

    cfg["data"]["stage"] = 1: COCO+VOC mixed
    cfg["data"]["stage"] = 2: VOC trainaug only
    """
    data_cfg = cfg["data"]
    voc_root = data_cfg["voc_root"]
    coco_root = data_cfg["coco_root"]
    _assert_not_test_path(voc_root, role="voc_root")
    _assert_not_test_path(coco_root, role="coco_root")

    train_t = build_train_transform(
        crop_size=data_cfg["crop_size"],
        scale_range=tuple(data_cfg["scale_range"]),
    )
    val_t = build_val_transform()

    voc_train = VOCSegDataset(root=voc_root, split="trainaug", transform=train_t)
    voc_val = VOCSegDataset(root=voc_root, split="val", transform=val_t)

    stage = data_cfg.get("stage", 1)
    if stage == 1:
        coco_train = COCOSegDataset(coco_root=coco_root, split="train2017", transform=train_t)
        train_ds = ConcatDataset([coco_train, voc_train])
    else:
        train_ds = voc_train

    common = dict(
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=data_cfg.get("persistent_workers", True),
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(voc_val, batch_size=1, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
```

- [ ] **Step 4: Run test (pass)**

Run: `cd p1 && uv run pytest tests/test_data_builder.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/data/builder.py p1/tests/test_data_builder.py
git commit -m "p1: add data builder with test_public/ isolation guard"
```

---

## Phase 3 — Model B (Tasks 13-19)

### Task 13: `src/models/backbones/resnet.py` — ResNet-50 OS=16

**Files:**
- Create: `p1/src/models/backbones/resnet.py`
- Test: `p1/tests/test_models.py` (extend later)

- [ ] **Step 1: Implement**

```python
"""ResNet-50 backbone for segmentation.

TorchVision IMAGENET1K_V2 pretrained, layer4를 dilated conv로 OS=32→16.
forward는 (c2, c5) 반환 — c2는 DLv3+ decoder의 low-level skip용.
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    LOW_CHANNELS = 256
    HIGH_CHANNELS = 2048

    def __init__(self, output_stride: int = 16, pretrained: bool = True) -> None:
        super().__init__()
        if output_stride == 16:
            replace = [False, False, True]
        elif output_stride == 8:
            replace = [False, True, True]
        else:
            raise ValueError(f"output_stride must be 8 or 16, got {output_stride}")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights, replace_stride_with_dilation=replace)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # 256 ch, H/4 (low-level)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4   # 2048 ch, H/16 (OS=16) — dilated

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)        # low-level
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)       # high-level
        return c2, c5
```

- [ ] **Step 2: Smoke test**

Run: `cd p1 && uv run python -c "
import torch
from src.models.backbones.resnet import ResNet50Backbone
m = ResNet50Backbone(output_stride=16, pretrained=False)
m.eval()
x = torch.zeros(1, 3, 480, 640)
with torch.no_grad():
    c2, c5 = m(x)
print('c2:', tuple(c2.shape), 'c5:', tuple(c5.shape))
assert c2.shape == (1, 256, 120, 160), c2.shape
assert c5.shape == (1, 2048, 30, 40), c5.shape
print('ok')
"`
Expected: `c2: (1, 256, 120, 160) c5: (1, 2048, 30, 40)\nok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/models/backbones/resnet.py
git commit -m "p1: add ResNet-50 backbone (OS=16, dilated layer4)"
```

---

### Task 14: `src/models/necks/aspp.py` — ASPP

**Files:** Create `p1/src/models/necks/aspp.py`

- [ ] **Step 1: Implement** (w4/models/deeplab_v3.py ASPP class 기반, channel arg 추가)

```python
"""Atrous Spatial Pyramid Pooling (DeepLabV3+ style).

5 branches (1×1, 3×3 d=6, d=12, d=18, GAP) → concat → 1×1 → BN → ReLU → Dropout.
Reference: w4/models/deeplab_v3.py.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


def _conv1x1(c_in: int, c_out: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)


def _conv3x3(c_in: int, c_out: int, dilation: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=3, padding=dilation, dilation=dilation, bias=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates: List[int] = (6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        # 1×1
        self.branches.append(nn.Sequential(_conv1x1(in_channels, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        # 3×3 dilated
        for r in rates:
            self.branches.append(nn.Sequential(_conv3x3(in_channels, out_channels, r), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        # GAP branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        n_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            _conv1x1(out_channels * n_branches, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        outs = [b(x) for b in self.branches]
        outs.append(interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, dim=1))
```

- [ ] **Step 2: Smoke**

Run: `cd p1 && uv run python -c "
import torch
from src.models.necks.aspp import ASPP
m = ASPP(2048, 256, rates=[6,12,18]).eval()
x = torch.zeros(1, 2048, 30, 40)
with torch.no_grad(): y = m(x)
print(tuple(y.shape))
assert y.shape == (1, 256, 30, 40)
print('ok')
"`
Expected: `(1, 256, 30, 40)\nok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/models/necks/aspp.py
git commit -m "p1: add ASPP neck (5 branches, rates [6,12,18])"
```

---

### Task 15: `src/models/heads/deeplabv3plus.py` — Decoder

**Files:** Create `p1/src/models/heads/deeplabv3plus.py`

- [ ] **Step 1: Implement**

```python
"""DeepLabV3+ decoder: low-level skip + ASPP feature fusion.

ASPP_out (H/16) ─upsample×4─→ + low_level (1×1 → 48ch) ─→ concat (304) ─→
3×3 conv → 256 → 3×3 conv → 256 → 1×1 conv → num_classes ─upsample×4─→ logits at input H,W.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, low_in_channels: int, aspp_out_channels: int, num_classes: int, low_proj_channels: int = 48) -> None:
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_in_channels, low_proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_proj_channels),
            nn.ReLU(inplace=True),
        )
        merged = aspp_out_channels + low_proj_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(merged, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, aspp_out: Tensor, low_level: Tensor, output_size: tuple) -> Tensor:
        low = self.low_proj(low_level)
        h_low, w_low = low.shape[2:]
        aspp_up = interpolate(aspp_out, size=(h_low, w_low), mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x
```

- [ ] **Step 2: Smoke**

Run: `cd p1 && uv run python -c "
import torch
from src.models.heads.deeplabv3plus import DeepLabV3PlusHead
h = DeepLabV3PlusHead(low_in_channels=256, aspp_out_channels=256, num_classes=21).eval()
aspp = torch.zeros(1, 256, 30, 40)
low = torch.zeros(1, 256, 120, 160)
with torch.no_grad(): y = h(aspp, low, output_size=(480, 640))
print(tuple(y.shape))
assert y.shape == (1, 21, 480, 640)
print('ok')
"`
Expected: `(1, 21, 480, 640)\nok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/models/heads/deeplabv3plus.py
git commit -m "p1: add DeepLabV3+ decoder head with low-level skip"
```

---

### Task 16: `src/models/aux/fcn_head.py` — auxiliary CE head

**Files:** Create `p1/src/models/aux/fcn_head.py`

- [ ] **Step 1: Implement**

```python
"""FCN-style auxiliary head for deep supervision (학습 전용).

ONNX export 전 SegmentationModel.export_mode()가 이 head를 비활성화.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


class FCNHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, mid_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor, output_size: tuple) -> Tensor:
        y = self.conv(x)
        return interpolate(y, size=output_size, mode="bilinear", align_corners=False)
```

- [ ] **Step 2: Commit**

```bash
git add p1/src/models/aux/fcn_head.py
git commit -m "p1: add FCN aux head for deep supervision"
```

---

### Task 17: `src/models/seg_model.py` — combiner with export_mode()

**Files:** Create `p1/src/models/seg_model.py`

- [ ] **Step 1: Implement**

```python
"""SegmentationModel: backbone + neck + head + (optional) aux head.

forward 반환:
- training: (main_logits, aux_logits) tuple
- inference (after export_mode()): main_logits only
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


class SegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        aux_head: Optional[nn.Module] = None,
        aux_in_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.aux_head = aux_head
        self._export = False     # ONNX export 모드 플래그

    def export_mode(self) -> "SegmentationModel":
        """ONNX export 직전 호출. aux head 비활성화 + main logits만 반환."""
        self._export = True
        # aux_head는 메모리 절약을 위해 제거 가능 (state_dict는 유지하지만 forward 미사용)
        return self

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        h, w = x.shape[2:]
        low, high = self.backbone(x)
        feat = self.neck(high)
        main_logits = self.head(feat, low, output_size=(h, w))
        if self._export or self.aux_head is None or not self.training:
            return main_logits
        # 학습 시: c4 (layer3 출력) 필요 → backbone에서 별도 노출 필요한데
        # 우리 구조는 (c2, c5)만 반환. aux는 c5에서 분기 (DLv3 표준은 c4지만 단순화).
        aux_logits = self.aux_head(high, output_size=(h, w))
        return main_logits, aux_logits
```

- [ ] **Step 2: Commit**

```bash
git add p1/src/models/seg_model.py
git commit -m "p1: add SegmentationModel with export_mode for ONNX"
```

---

### Task 18: `src/models/builder.py` — model swap dispatcher

**Files:**
- Create: `p1/src/models/builder.py`
- Test: `p1/tests/test_models.py`

- [ ] **Step 1: Write smoke test**

`p1/tests/test_models.py`:
```python
import torch
import pytest

from src.models.builder import build_model


def test_build_resnet_deeplabv3plus_b_forward_shape():
    cfg = {
        "model": {
            "backbone": "resnet50",
            "head": "deeplabv3plus",
            "num_classes": 21,
            "output_stride": 16,
            "pretrained": False,         # 테스트는 weights 없이
            "use_aux": True,
        }
    }
    m = build_model(cfg).eval()
    x = torch.zeros(1, 3, 480, 640)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 21, 480, 640)


def test_build_with_aux_returns_tuple_in_training():
    cfg = {
        "model": {
            "backbone": "resnet50",
            "head": "deeplabv3plus",
            "num_classes": 21,
            "output_stride": 16,
            "pretrained": False,
            "use_aux": True,
        }
    }
    m = build_model(cfg).train()
    x = torch.zeros(1, 3, 480, 640)
    out = m(x)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (1, 21, 480, 640)
    assert out[1].shape == (1, 21, 480, 640)


def test_export_mode_removes_aux_output():
    cfg = {
        "model": {
            "backbone": "resnet50", "head": "deeplabv3plus",
            "num_classes": 21, "output_stride": 16, "pretrained": False, "use_aux": True,
        }
    }
    m = build_model(cfg).eval().export_mode()
    x = torch.zeros(1, 3, 480, 640)
    with torch.no_grad():
        y = m(x)
    assert isinstance(y, torch.Tensor)
```

- [ ] **Step 2: Implement `src/models/builder.py`**

```python
"""build_model(cfg) — backbone/neck/head swap by yaml key."""
from __future__ import annotations

from src.models.backbones.resnet import ResNet50Backbone
from src.models.heads.deeplabv3plus import DeepLabV3PlusHead
from src.models.necks.aspp import ASPP
from src.models.aux.fcn_head import FCNHead
from src.models.seg_model import SegmentationModel


def build_model(cfg: dict) -> SegmentationModel:
    mc = cfg["model"]
    bb_name = mc["backbone"]
    head_name = mc["head"]
    num_classes = mc["num_classes"]
    output_stride = mc.get("output_stride", 16)
    pretrained = mc.get("pretrained", True)
    use_aux = mc.get("use_aux", True)

    if bb_name == "resnet50":
        backbone = ResNet50Backbone(output_stride=output_stride, pretrained=pretrained)
        low_ch, high_ch = ResNet50Backbone.LOW_CHANNELS, ResNet50Backbone.HIGH_CHANNELS
    elif bb_name == "mobilenet_v3_large":
        from src.models.backbones.mobilenet import MobileNetV3LargeBackbone
        backbone = MobileNetV3LargeBackbone(pretrained=pretrained)
        low_ch, high_ch = MobileNetV3LargeBackbone.LOW_CHANNELS, MobileNetV3LargeBackbone.HIGH_CHANNELS
    else:
        raise ValueError(f"Unknown backbone: {bb_name}")

    if head_name == "deeplabv3plus":
        rates = mc.get("aspp_rates", [6, 12, 18])
        neck = ASPP(in_channels=high_ch, out_channels=256, rates=tuple(rates))
        head = DeepLabV3PlusHead(low_in_channels=low_ch, aspp_out_channels=256, num_classes=num_classes)
    elif head_name == "lraspp":
        from src.models.necks.lr_aspp import LRASPPHead
        neck = nn.Identity()
        head = LRASPPHead(low_in=low_ch, high_in=high_ch, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown head: {head_name}")

    aux_head = FCNHead(in_channels=high_ch, num_classes=num_classes) if use_aux else None
    return SegmentationModel(backbone, neck, head, aux_head)
```

- [ ] **Step 3: Run tests**

Run: `cd p1 && uv run pytest tests/test_models.py -v`
Expected: 3 passed (network 다운로드 없이 — pretrained=False)

- [ ] **Step 4: Commit**

```bash
git add p1/src/models/builder.py p1/tests/test_models.py
git commit -m "p1: add model builder with backbone/head swap"
```

---

### Task 19: ONNX export smoke test

**Files:** Test only (no new src file).

- [ ] **Step 1: Smoke test**

Run: `cd p1 && uv run python -c "
import torch, onnx
from src.models.builder import build_model
cfg = {'model': {'backbone':'resnet50','head':'deeplabv3plus','num_classes':21,'output_stride':16,'pretrained':False,'use_aux':True}}
m = build_model(cfg).eval().export_mode()
dummy = torch.zeros(1,3,480,640)
torch.onnx.export(m, dummy, '/tmp/p1_smoke.onnx',
    input_names=['input'], output_names=['logits'],
    dynamic_axes=None, opset_version=17)
mp = onnx.load('/tmp/p1_smoke.onnx')
onnx.checker.check_model(mp)
print('graph nodes:', len(mp.graph.node))
print('inputs:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in mp.graph.input])
print('outputs:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in mp.graph.output])
print('ok')
"`
Expected: 노드 수 출력, input shape (1,3,480,640), output (1,21,480,640), `ok`. (체크 실패 시 STOP — opset/op 호환성 문제)

- [ ] **Step 2: FLOPs measure smoke**

Run: `cd p1 && uv run python -c "
from src.utils.flops import count_onnx_flops
total, breakdown = count_onnx_flops('/tmp/p1_smoke.onnx', (1,3,480,640))
print(f'Total: {total/1e9:.2f} GFLOPs')
for op, f in sorted(breakdown.items(), key=lambda kv: -kv[1]):
    if f > 0:
        print(f'  {op}: {f/1e9:.2f} G')
"`
Expected: 약 40-50 GFLOPs (Conv 우세). spec §6.6 추정과 일치 확인.

- [ ] **Step 3: Commit (no code change, just verification — skip)**

---

## Phase 4 — Loss + Config (Tasks 20-21)

### Task 20: `src/losses/seg_loss.py`

**Files:**
- Create: `p1/src/losses/seg_loss.py`
- Test: `p1/tests/test_loss.py`

- [ ] **Step 1: Write failing test**

`p1/tests/test_loss.py`:
```python
import torch
from src.losses.seg_loss import SegLoss


def test_perfect_prediction_yields_low_loss():
    crit = SegLoss(num_classes=21, ignore_index=255, dice_weight=0.5, aux_weight=0.4)
    target = torch.randint(0, 21, (2, 32, 32))
    # one-hot → high logit on correct class
    logits = torch.zeros(2, 21, 32, 32)
    logits.scatter_(1, target.unsqueeze(1), 10.0)
    loss = crit(logits, None, target)
    assert loss.item() < 0.5


def test_ignore_pixels_excluded_from_loss():
    crit = SegLoss(num_classes=2, ignore_index=255, dice_weight=0.0, aux_weight=0.0)
    # 모든 픽셀이 ignore → loss 0 (또는 NaN 안 나야 함)
    logits = torch.randn(1, 2, 4, 4)
    target = torch.full((1, 4, 4), 255)
    loss = crit(logits, None, target)
    assert torch.isfinite(loss)


def test_aux_loss_added_when_provided():
    crit = SegLoss(num_classes=2, ignore_index=255, dice_weight=0.0, aux_weight=1.0)
    target = torch.randint(0, 2, (1, 4, 4))
    main = torch.randn(1, 2, 4, 4)
    aux = torch.randn(1, 2, 4, 4)
    l_with = crit(main, aux, target)
    l_without = crit(main, None, target)
    assert l_with > l_without
```

- [ ] **Step 2: Run (fail)**

Run: `cd p1 && uv run pytest tests/test_loss.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
"""Segmentation loss = CE(main) + dice_weight × Dice(main) + aux_weight × CE(aux).

ignore_index=255는 CE는 자동, Dice는 수동 mask로 제외.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Per-class dice (background 포함), ignore mask 적용, 평균."""

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W), target: (N, H, W)
        valid = (target != self.ignore_index).unsqueeze(1).float()  # (N,1,H,W)
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        target_oh = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_oh = target_oh * valid                                # (N,C,H,W)
        probs = F.softmax(logits, dim=1) * valid

        dims = (0, 2, 3)
        intersect = (probs * target_oh).sum(dim=dims)
        denom = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2 * intersect + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 21,
        ignore_index: int = 255,
        dice_weight: float = 0.5,
        aux_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight

    def forward(self, main_logits: torch.Tensor, aux_logits: Optional[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(main_logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(main_logits, target)
        if aux_logits is not None and self.aux_weight > 0:
            loss = loss + self.aux_weight * self.ce(aux_logits, target)
        return loss
```

- [ ] **Step 4: Run (pass)**

Run: `cd p1 && uv run pytest tests/test_loss.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/losses/seg_loss.py p1/tests/test_loss.py
git commit -m "p1: add CE+Dice+Aux segmentation loss with ignore=255"
```

---

### Task 21: `src/config/default.yaml`

**Files:** Create `p1/src/config/default.yaml`

- [ ] **Step 1: Write config**

```yaml
# Default config: B (ResNet-50 + DeepLabV3+, OS=16)

seed: 42

device: cuda

data:
  voc_root: ./data/voc
  coco_root: ./data/coco
  batch_size: 16
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  crop_size: 480
  scale_range: [0.5, 2.0]
  stage: 1                   # 1=COCO+VOC, 2=VOC trainaug only (자동 전이)

model:
  backbone: resnet50
  head: deeplabv3plus
  num_classes: 21
  output_stride: 16
  pretrained: true
  use_aux: true
  aspp_rates: [6, 12, 18]

loss:
  ignore_index: 255
  dice_weight: 0.5
  aux_weight: 0.4

optimizer:
  type: sgd
  base_lr: 0.01              # stage 1 default; stage 2는 자동 0.001
  momentum: 0.9
  weight_decay: 0.0005
  backbone_lr_mult: 0.1

scheduler:
  type: poly
  power: 0.9
  warmup_iters: 1000         # stage 1; stage 2는 자동 500

training:
  stage1_iters: 80000
  stage2_iters: 30000
  stage2_base_lr: 0.001
  stage2_warmup: 500
  log_interval: 50
  val_interval_stage1: 5000
  val_interval_stage2: 2000
  ckpt_interval: 5000
  amp_dtype: fp16
  ema_decay: 0.9999
  grad_clip: 1.0

wandb:
  entity: g1nie-sungkyunkwan-university
  project: osai-p1-local
  run_name_prefix: dev
  tags: [desktop, voc+coco]

paths:
  ckpt_dir: ./checkpoints
  training_state: ./checkpoints/training_state.pth
  best_ckpt: ./checkpoints/best.pth
  final_ckpt: ./checkpoints/model.pth
```

- [ ] **Step 2: Sanity load**

Run: `cd p1 && uv run python -c "
import yaml
cfg = yaml.safe_load(open('src/config/default.yaml'))
print(cfg['model']['backbone'], cfg['training']['stage1_iters'])
"`
Expected: `resnet50 80000`

- [ ] **Step 3: Commit**

```bash
git add p1/src/config/default.yaml
git commit -m "p1: add default config (B: resnet50 + deeplabv3+)"
```

---

## Phase 5 — Training Loop (Tasks 22-28)

### Task 22: `src/train.py` — basic single-stage loop (no AMP/EMA/WandB yet)

**Files:** Create `p1/src/train.py` (initial, will be extended in next tasks)

- [ ] **Step 1: Implement minimal train.py**

```python
"""Training entrypoint. 처음 commit은 단일 stage 기본 loop;
후속 task에서 AMP/EMA/resume/WandB/2-stage 추가."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import LambdaLR

from src.data.builder import build_dataloaders
from src.losses.seg_loss import SegLoss
from src.models.builder import build_model
from src.utils.metrics import SegMetric
from src.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    base_lr = cfg["optimizer"]["base_lr"]
    bb_mult = cfg["optimizer"]["backbone_lr_mult"]
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return torch.optim.SGD(
        [
            {"params": backbone_params, "lr": base_lr * bb_mult},
            {"params": other_params, "lr": base_lr},
        ],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )


def make_poly_scheduler(opt: torch.optim.Optimizer, total_iters: int, warmup_iters: int, power: float = 0.9) -> LambdaLR:
    def lam(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return (1.0 - progress) ** power
    return LambdaLR(opt, lr_lambda=lam)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, num_classes=21) -> tuple[float, list]:
    metric = SegMetric(num_classes=num_classes, ignore_index=255)
    model.eval()
    for img, mask in loader:
        img = img.to(device); mask = mask.to(device)
        out = model(img)
        if isinstance(out, tuple): out = out[0]
        pred = out.argmax(dim=1)
        metric.update(pred, mask)
    model.train()
    return metric.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    total_iters = cfg["training"]["stage1_iters"]
    scheduler = make_poly_scheduler(optimizer, total_iters, cfg["scheduler"]["warmup_iters"], cfg["scheduler"]["power"])
    criterion = SegLoss(
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["loss"]["ignore_index"],
        dice_weight=cfg["loss"]["dice_weight"],
        aux_weight=cfg["loss"]["aux_weight"],
    )

    iter_count = 0
    data_iter = iter(train_loader)
    model.train()
    log_interval = cfg["training"]["log_interval"]
    while iter_count < total_iters:
        try:
            img, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img, mask = next(data_iter)
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(img)
        main_logits, aux_logits = (out if isinstance(out, tuple) else (out, None))
        loss = criterion(main_logits, aux_logits, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["training"]["grad_clip"])
        optimizer.step()
        scheduler.step()
        iter_count += 1
        if iter_count % log_interval == 0:
            print(f"iter {iter_count}/{total_iters} loss={loss.item():.4f} lr={optimizer.param_groups[1]['lr']:.5f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `cd p1 && uv run python -c "import src.train; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/train.py
git commit -m "p1: add basic train.py loop (no AMP/EMA/resume yet)"
```

---

### Task 23: Add AMP fp16 to train.py

**Files:** Modify `p1/src/train.py`

- [ ] **Step 1: Replace train loop with AMP-enabled version**

Edit `main()` body — replace the inner training while-loop region:

```python
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["training"]["amp_dtype"] == "fp16"))
    amp_dtype = torch.float16 if cfg["training"]["amp_dtype"] == "fp16" else torch.bfloat16
    
    iter_count = 0
    data_iter = iter(train_loader)
    model.train()
    log_interval = cfg["training"]["log_interval"]
    while iter_count < total_iters:
        try:
            img, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img, mask = next(data_iter)
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out = model(img)
            main_logits, aux_logits = (out if isinstance(out, tuple) else (out, None))
            loss = criterion(main_logits, aux_logits, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        iter_count += 1
        if iter_count % log_interval == 0:
            print(f"iter {iter_count}/{total_iters} loss={loss.item():.4f} lr={optimizer.param_groups[1]['lr']:.5f}")
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.train; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/train.py
git commit -m "p1: add AMP fp16 + GradScaler to training loop"
```

---

### Task 24: Add EMA to train.py

**Files:** Modify `p1/src/train.py`

- [ ] **Step 1: Add EMA helper**

Add to top-level of `src/train.py` (above `main`):

```python
import copy

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)
```

In `main()`, after model construction:

```python
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()
    ema_decay = cfg["training"]["ema_decay"]
```

In the training loop, after `scheduler.step()`:

```python
        update_ema(ema_model, model, ema_decay)
```

- [ ] **Step 2: Verify import**

Run: `cd p1 && uv run python -c "import src.train; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/train.py
git commit -m "p1: add EMA model with decay 0.9999"
```

---

### Task 25: Add resume + checkpoint save to train.py

**Files:** Modify `p1/src/train.py`

- [ ] **Step 1: Add resume logic + ckpt save**

In `main()`, after building model/optimizer/scheduler/scaler/ema:

```python
    from src.utils.checkpoint import save_full, load_full, save_model_only

    ckpt_dir = Path(cfg["paths"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    training_state_path = Path(cfg["paths"]["training_state"])
    best_ckpt_path = Path(cfg["paths"]["best_ckpt"])
    final_ckpt_path = Path(cfg["paths"]["final_ckpt"])

    iter_count = 0
    best_miou = 0.0
    wandb_run_id = None  # Task 26에서 set
    if training_state_path.exists():
        meta = load_full(
            path=str(training_state_path),
            model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )
        iter_count = meta["iter"]
        best_miou = meta["best_miou"]
        wandb_run_id = meta["wandb_run_id"]
        print(f"[resume] from iter {iter_count}, best_miou={best_miou:.4f}")
```

In the training loop, after `update_ema(...)`:

```python
        if iter_count % cfg["training"]["ckpt_interval"] == 0:
            save_full(
                path=str(training_state_path), iter_count=iter_count, stage=cfg["data"]["stage"],
                model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_miou=best_miou, wandb_run_id=wandb_run_id, config=cfg,
            )
        val_interval = cfg["training"][f"val_interval_stage{cfg['data']['stage']}"]
        if iter_count % val_interval == 0:
            miou_ema, _ = evaluate(ema_model, val_loader, device)
            print(f"  [val] mIoU_ema={miou_ema:.4f}")
            if miou_ema > best_miou:
                best_miou = miou_ema
                torch.save({"ema_state": ema_model.state_dict(), "iter": iter_count, "miou": miou_ema}, best_ckpt_path)

    save_model_only(str(final_ckpt_path), ema_model)
    print(f"[done] final ckpt → {final_ckpt_path}, best_miou={best_miou:.4f}")
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.train; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/train.py
git commit -m "p1: add resumable checkpoint save/load + best ckpt tracking"
```

---

### Task 26: Add WandB integration to train.py

**Files:** Modify `p1/src/train.py`

- [ ] **Step 1: Add wandb.init + logging**

Add at top of `main()` after `set_seed`:

```python
    import wandb
    from datetime import datetime
    run_name = f"{cfg['wandb']['run_name_prefix']}_{datetime.now().strftime('%m%d')}_{cfg['model']['backbone']}_stage{cfg['data']['stage']}"
    run = wandb.init(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        name=run_name,
        config=cfg,
        id=wandb_run_id,                # set after resume below if exists
        resume="allow",
        tags=cfg["wandb"]["tags"] + [f"stage{cfg['data']['stage']}"],
    )
    wandb_run_id = run.id
```

Note: this needs `wandb_run_id` defined first. Reorder so resume happens before `wandb.init`, or initialize `wandb_run_id = None` before resume block.

In the training loop, replace the print statement:

```python
        if iter_count % log_interval == 0:
            wandb.log({
                "step": iter_count,
                "train/loss": loss.item(),
                "lr/backbone": optimizer.param_groups[0]["lr"],
                "lr/head": optimizer.param_groups[1]["lr"],
            }, step=iter_count)
```

After validation:

```python
            wandb.log({"val/mIoU_ema": miou_ema, "val/best_mIoU": best_miou}, step=iter_count)
```

After model build, log GPU + params:

```python
    run.summary["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    n_params = sum(p.numel() for p in model.parameters())
    run.summary["params/total"] = n_params
```

At end:

```python
    run.finish()
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.train; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/train.py
git commit -m "p1: add WandB integration with resume + GPU evidence"
```

---

### Task 27: Add 2-stage dispatch to train.py

**Files:** Modify `p1/src/train.py`

- [ ] **Step 1: Refactor `main()` to call `run_stage(cfg, stage)` twice**

Replace `main()` with:

```python
def run_stage(cfg: dict, stage: int) -> None:
    """단일 stage 학습. stage 2는 base_lr/iters/warmup이 다르고, stage 1 best ckpt를 자동 로드."""
    # cfg를 stage 별로 변형 (얕은 복사)
    cfg = {**cfg, "data": {**cfg["data"], "stage": stage}}
    if stage == 2:
        cfg["optimizer"] = {**cfg["optimizer"], "base_lr": cfg["training"]["stage2_base_lr"]}
        cfg["scheduler"] = {**cfg["scheduler"], "warmup_iters": cfg["training"]["stage2_warmup"]}
        total_iters = cfg["training"]["stage2_iters"]
    else:
        total_iters = cfg["training"]["stage1_iters"]

    # ...(이전 main() 본문, total_iters 사용)...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, default=None, help="1 or 2; 생략 시 둘 다 순차 실행")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage is None:
        # Stage 1 → Stage 2 자동 전이
        run_stage(cfg, 1)
        # Stage 2 시작 시 stage 1 best ckpt를 model로 로드
        # (training_state는 stage 1용이라 stage 2 시작에선 새로 시작)
        run_stage(cfg, 2)
    else:
        run_stage(cfg, args.stage)
```

Stage 2 진입 시 best ckpt를 model에 로드하는 로직 추가 (run_stage 안):
```python
    if stage == 2 and Path(cfg["paths"]["best_ckpt"]).exists() and not Path(cfg["paths"]["training_state"]).exists():
        bc = torch.load(cfg["paths"]["best_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(bc["ema_state"])
        ema_model.load_state_dict(bc["ema_state"])
        print(f"[stage2] loaded best ckpt from stage1 (mIoU={bc['miou']:.4f})")
```

또한 stage 전환 시 `training_state.pth`를 삭제하거나 stage 별로 다른 파일명으로 분리 — 단순함을 위해 stage 별 path:
```python
    cfg["paths"] = {**cfg["paths"], "training_state": cfg["paths"]["training_state"].replace(".pth", f"_stage{stage}.pth")}
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.train; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/train.py
git commit -m "p1: add 2-stage dispatch with best ckpt transfer"
```

---

### Task 28: Smoke test — 100 iter run on real data

**Files:** No new code; verify only.

전제: VOC 데이터가 다운되어 있어야 함 (`download.py` 실행 후) — 없으면 SKIP.

- [ ] **Step 1: Make smoke config**

Create `p1/src/config/smoke.yaml`:
```yaml
seed: 42
device: cuda
data:
  voc_root: ./data/voc
  coco_root: ./data/coco
  batch_size: 4
  num_workers: 2
  pin_memory: true
  persistent_workers: false
  crop_size: 256
  scale_range: [0.75, 1.5]
  stage: 2                   # VOC only (COCO 다운 안 됐어도 동작)
model:
  backbone: resnet50
  head: deeplabv3plus
  num_classes: 21
  output_stride: 16
  pretrained: true
  use_aux: true
  aspp_rates: [6, 12, 18]
loss:
  ignore_index: 255
  dice_weight: 0.5
  aux_weight: 0.4
optimizer:
  type: sgd
  base_lr: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  backbone_lr_mult: 0.1
scheduler:
  type: poly
  power: 0.9
  warmup_iters: 10
training:
  stage1_iters: 100
  stage2_iters: 100
  stage2_base_lr: 0.001
  stage2_warmup: 10
  log_interval: 10
  val_interval_stage1: 100
  val_interval_stage2: 100
  ckpt_interval: 50
  amp_dtype: fp16
  ema_decay: 0.99
  grad_clip: 1.0
wandb:
  entity: g1nie-sungkyunkwan-university
  project: osai-p1-local
  run_name_prefix: smoke
  tags: [smoke]
paths:
  ckpt_dir: ./checkpoints/smoke
  training_state: ./checkpoints/smoke/state.pth
  best_ckpt: ./checkpoints/smoke/best.pth
  final_ckpt: ./checkpoints/smoke/model.pth
```

- [ ] **Step 2: Run smoke**

Run: `cd p1 && WANDB_MODE=offline uv run python -m src.train --config src/config/smoke.yaml --stage 2`
Expected: 100 iter 완료, ckpt 파일 생성, val mIoU 출력 (작은 값이라도 finite)

- [ ] **Step 3: Resume smoke (interrupt-style)**

Manually delete only `state.pth`, re-run → `[resume] from iter 100` 또는 새로 시작 후 동일 수행.
Or: run again same command, since iter == total → 즉시 `[done]` 출력.

- [ ] **Step 4: Commit smoke config (config 자체는 dev 도구)**

```bash
git add p1/src/config/smoke.yaml
git commit -m "p1: add smoke config (100 iter, small batch)"
```

---

## Phase 6 — Eval + Inference (Tasks 29-30)

### Task 29: `src/eval.py`

**Files:** Create `p1/src/eval.py`

- [ ] **Step 1: Implement**

```python
"""Eval entrypoint: model.pth (or ckpt) → val mIoU + per-class IoU."""
from __future__ import annotations

import argparse

import torch

from src.data.builder import build_dataloaders
from src.models.builder import build_model
from src.train import evaluate, load_config
from src.utils.metrics import SegMetric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    cfg["data"]["stage"] = 2  # val만 필요
    _, val_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = state.get("ema_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    miou, per_class = evaluate(model, val_loader, device, num_classes=cfg["model"]["num_classes"])
    print(f"mIoU: {miou:.4f}")
    for i, v in enumerate(per_class):
        print(f"  class {i:2d}: {v:.4f}" if v == v else f"  class {i:2d}: NaN")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.eval; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/eval.py
git commit -m "p1: add eval.py for val mIoU + per-class IoU"
```

---

### Task 30: `src/infer.py` — TTA inference (input → output PNGs)

**Files:** Create `p1/src/infer.py`

- [ ] **Step 1: Implement**

```python
"""Inference with multi-scale TTA + hflip. input_dir/*.jpg → output_dir/*.png."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.builder import build_model
from src.train import load_config


def _round_up(x: float, multiple: int = 32) -> int:
    return int(math.ceil(x / multiple) * multiple)


def _preprocess(img: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


@torch.no_grad()
def tta_predict(model, img_pil: Image.Image, device, scales=(0.5, 0.75, 1.0, 1.25, 1.5)) -> np.ndarray:
    W, H = img_pil.size
    accum = torch.zeros(21, H, W, device=device)
    for scale in scales:
        Hs = _round_up(H * scale, 32); Ws = _round_up(W * scale, 32)
        img_t = _preprocess(img_pil, (Ws, Hs)).to(device)
        for hflip in (False, True):
            x = torch.flip(img_t, dims=[-1]) if hflip else img_t
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            logits = logits.float().softmax(dim=1)
            if hflip: logits = torch.flip(logits, dims=[-1])
            accum += logits.squeeze(0)
    pred = accum.argmax(dim=0).to(torch.uint8).cpu().numpy()
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = state.get("ema_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval(); model.export_mode()

    inp = Path(args.input); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    files = sorted(inp.glob("*.jpg"))
    scales = (1.0,) if args.no_tta else (0.5, 0.75, 1.0, 1.25, 1.5)
    for f in tqdm(files):
        img = Image.open(f).convert("RGB")
        pred = tta_predict(model, img, device, scales=scales)
        Image.fromarray(pred, mode="L").save(out / f"{f.stem}.png", format="PNG")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.infer; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/infer.py
git commit -m "p1: add infer.py with multi-scale TTA + hflip"
```

---

## Phase 7 — ONNX + Submission (Tasks 31-34)

### Task 31: `src/export_onnx.py`

**Files:** Create `p1/src/export_onnx.py`

- [ ] **Step 1: Implement**

```python
"""ONNX export entrypoint. EMA model → 가중치 제거된 ONNX (10MB 이하)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnx
import torch

from src.models.builder import build_model
from src.train import load_config


def export(cfg: dict, ckpt_path: str, out_path: str, opset: int = 17) -> None:
    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("ema_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval(); model.export_mode()
    dummy = torch.zeros(1, 3, 480, 640)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes=None, opset_version=opset, do_constant_folding=True,
    )

    # 가중치 제거 (교수님 제공 코드)
    m = onnx.load(out_path)
    for init in m.graph.initializer:
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.ClearField("int32_data")
        init.ClearField("int64_data")
    onnx.save(m, out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    assert size_mb <= 10.0, f"ONNX too large: {size_mb:.2f}MB"
    print(f"[ok] {out_path} ({size_mb:.2f}MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    export(cfg, args.ckpt, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.export_onnx; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/export_onnx.py
git commit -m "p1: add export_onnx.py (EMA + strip weights, 10MB check)"
```

---

### Task 32: `src/measure_flops.py`

**Files:** Create `p1/src/measure_flops.py`

- [ ] **Step 1: Implement**

```python
"""FLOPs 측정. PyTorch counter (학습 중 sanity) + ONNX counter (채점 ground truth)."""
from __future__ import annotations

import argparse

from src.train import load_config
from src.utils.flops import count_pytorch_flops, count_onnx_flops


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--onnx", default=None)
    args = p.parse_args()

    if args.onnx:
        total, breakdown = count_onnx_flops(args.onnx, (1, 3, 480, 640))
        print(f"[ONNX] {args.onnx}: {total/1e9:.2f} GFLOPs")
        for op, f in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            if f > 0:
                print(f"    {op}: {f/1e9:.3f} G")

    if args.config and args.ckpt:
        import torch
        from src.models.builder import build_model
        cfg = load_config(args.config)
        m = build_model(cfg)
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = state.get("ema_state", state) if isinstance(state, dict) else state
        m.load_state_dict(sd)
        m.export_mode().eval()
        f_py = count_pytorch_flops(m, (1, 3, 480, 640))
        print(f"[PyTorch] {f_py/1e9:.2f} GFLOPs (sanity check)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity**

Run: `cd p1 && uv run python -c "import src.measure_flops; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add p1/src/measure_flops.py
git commit -m "p1: add measure_flops.py (PyTorch + ONNX cross-check)"
```

---

### Task 33: `src/package_submission.py`

**Files:**
- Create: `p1/src/package_submission.py`
- Test: `p1/tests/test_package_submission.py`

- [ ] **Step 1: Write tests**

`p1/tests/test_package_submission.py`:
```python
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.package_submission import validate_pred_dir, package_zip


def _make_pred_dir(d: Path, n: int, valid: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.zeros((100, 100), dtype=np.uint8)
        if not valid:
            arr[0, 0] = 99    # invalid (outside [0,20])
        Image.fromarray(arr, mode="L").save(d / f"{i:03d}.png")


def test_validate_accepts_correct_dir(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 1000)
    validate_pred_dir(d)  # should not raise


def test_validate_rejects_wrong_count(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 999)
    with pytest.raises(AssertionError, match="1000"):
        validate_pred_dir(d)


def test_validate_rejects_invalid_pixel(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 1000, valid=False)
    with pytest.raises(AssertionError, match=r"\[0, 20\]"):
        validate_pred_dir(d, sample_check=10)


def test_package_zip_flat_and_correct_files(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 1000)
    out_zip = tmp_path / "submission.zip"
    package_zip(d, out_zip)
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert len(names) == 1000
    assert all("/" not in n for n in names)
    assert names[0] == "000.png"
    assert names[-1] == "999.png"
```

- [ ] **Step 2: Run (fail)**

Run: `cd p1 && uv run pytest tests/test_package_submission.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
"""채점용 PNG zip 검증/패키징.

요구:
- 정확히 1000 파일
- 이름 ^\d{3}\.png$
- flat root (하위 폴더 없음)
- 픽셀 값 [0, 20] 정수
- 압축해제 ≤500MB
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


_PNG_RE = re.compile(r"^\d{3}\.png$")


def validate_pred_dir(pred_dir: Union[str, Path], sample_check: int = 50) -> None:
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.iterdir())
    files = [f for f in files if f.is_file() and f.suffix == ".png"]
    assert len(files) == 1000, f"expected 1000 files, got {len(files)}"
    for f in files:
        assert _PNG_RE.match(f.name), f"bad name: {f.name}"
    # 픽셀 값 검사 (전체 또는 샘플)
    targets = files if sample_check >= len(files) else files[:: max(1, len(files) // sample_check)]
    for f in targets:
        arr = np.array(Image.open(f))
        assert arr.dtype == np.uint8, f"{f.name}: dtype {arr.dtype}"
        assert arr.min() >= 0 and arr.max() <= 20, f"{f.name}: pixel out of [0, 20]"
    # 압축 해제 크기 추정
    total = sum(f.stat().st_size for f in files)
    assert total <= 500_000_000, f"total size {total/1e6:.1f}MB > 500MB"


def package_zip(pred_dir: Union[str, Path], out_zip: Union[str, Path]) -> None:
    pred_dir = Path(pred_dir); out_zip = Path(out_zip)
    files = sorted([f for f in pred_dir.iterdir() if f.is_file() and f.suffix == ".png"])
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    validate_pred_dir(args.pred)
    package_zip(args.pred, args.out)
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run (pass)**

Run: `cd p1 && uv run pytest tests/test_package_submission.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add p1/src/package_submission.py p1/tests/test_package_submission.py
git commit -m "p1: add submission packaging with validation"
```

---

### Task 34: End-to-end smoke (10K + 5K iter on real data + full pipeline)

**Files:** No new code. Manual run.

전제: VOC 데이터 다운로드 완료. COCO는 옵션 (없으면 stage 1 skip).

- [ ] **Step 1: Quick stage 2 only run (10K iter)**

`p1/src/config/quick.yaml` 생성 (smoke.yaml 변형):
- stage1_iters: 0
- stage2_iters: 10000
- batch_size: 16
- crop_size: 480
- log_interval: 100

Run: `cd p1 && uv run python -m src.train --config src/config/quick.yaml --stage 2`
Expected: ~30분 (5070 Ti) 후 완료, val mIoU 출력 (>20% expected after 10K iter on VOC)

- [ ] **Step 2: Eval**

Run: `cd p1 && uv run python -m src.eval --config src/config/quick.yaml --ckpt checkpoints/quick/model.pth`
Expected: mIoU 출력

- [ ] **Step 3: Inference on test_public**

Run: `cd p1 && uv run python -m src.infer --config src/config/quick.yaml --ckpt checkpoints/quick/model.pth --input input/test_public --output output/pred_quick`
Expected: 1000 PNG 파일 생성 (~30분 with TTA)

- [ ] **Step 4: ONNX export + FLOPs**

Run: `cd p1 && uv run python -m src.export_onnx --config src/config/quick.yaml --ckpt checkpoints/quick/model.pth --out /tmp/model_struct.onnx`
Expected: `[ok] /tmp/model_struct.onnx (~1-2MB)`

Run: `cd p1 && uv run python -m src.measure_flops --onnx /tmp/model_struct.onnx`
Expected: ~40-50 GFLOPs 출력

- [ ] **Step 5: Package PNG zip**

Run: `cd p1 && uv run python -m src.package_submission --pred output/pred_quick --out /tmp/submission_pred.zip`
Expected: `[ok] /tmp/submission_pred.zip` + 검증 통과

- [ ] **Step 6: Commit quick config**

```bash
git add p1/src/config/quick.yaml
git commit -m "p1: add quick config (10K iter sanity)"
```

---

## Phase 8 — Backup A: MobileNetV3-Large + LR-ASPP (Tasks 35-37)

### Task 35: `src/models/backbones/mobilenet.py`

**Files:** Create `p1/src/models/backbones/mobilenet.py`

- [ ] **Step 1: Implement**

```python
"""MobileNetV3-Large backbone for segmentation (LR-ASPP 기반).

dilated 마지막 inverted residual block들 → effective OS=16.
forward (low_level, high_level) 반환:
- low_level: features[~4] 출력 (40ch, H/8)
- high_level: features 끝 (960ch, H/16, dilated)
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torch import Tensor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3LargeBackbone(nn.Module):
    LOW_CHANNELS = 40
    HIGH_CHANNELS = 960

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = mobilenet_v3_large(weights=weights)
        # MobileNetV3 features: 0~16 blocks
        # 표준 LR-ASPP 매핑: low = features[3] (40ch, H/8), high = features[16] (960ch)
        # OS=16 위해 features[7..16]에 dilation 2 적용
        for i in range(7, len(net.features)):
            for m in net.features[i].modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2) and m.kernel_size != (1, 1):
                    m.stride = (1, 1)
                    m.dilation = (2, 2)
                    pad = m.kernel_size[0] // 2 * 2
                    m.padding = (pad, pad)
        self.features = net.features
        self.low_idx = 6   # features[0..6] → ~H/8, 40ch
        self.high_idx = 16

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        low = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.low_idx:
                low = x
        return low, x
```

- [ ] **Step 2: Smoke**

Run: `cd p1 && uv run python -c "
import torch
from src.models.backbones.mobilenet import MobileNetV3LargeBackbone
m = MobileNetV3LargeBackbone(pretrained=False).eval()
x = torch.zeros(1, 3, 480, 640)
with torch.no_grad():
    low, high = m(x)
print('low:', tuple(low.shape), 'high:', tuple(high.shape))
"`
Expected: low/high shape 출력 (정확한 ch는 MobileNetV3 구현에 따라 약간 다를 수 있음 — 실제 채널 수 확인 후 LOW_CHANNELS 조정)

- [ ] **Step 3: Commit**

```bash
git add p1/src/models/backbones/mobilenet.py
git commit -m "p1: add MobileNetV3-Large backbone (dilated, OS=16)"
```

---

### Task 36: `src/models/necks/lr_aspp.py`

**Files:** Create `p1/src/models/necks/lr_aspp.py`

- [ ] **Step 1: Implement**

```python
"""LR-ASPP head (Searching for MobileNetV3 paper).

high (B, 960, H/16, W/16) → 1×1 conv 128 + GAP→sigmoid attention 곱
low (B, 40, H/8, W/8) → 1×1 → num_classes
combine + upsample.
이건 nn.Identity neck + 직접 head 통합 구조: SegmentationModel(backbone, Identity, LRASPPHead).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


class LRASPPHead(nn.Module):
    def __init__(self, low_in: int, high_in: int, num_classes: int, mid: int = 128) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_in, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_in, mid, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.high_classifier = nn.Conv2d(mid, num_classes, kernel_size=1)
        self.low_classifier = nn.Conv2d(low_in, num_classes, kernel_size=1)

    def forward(self, aspp_in_unused: Tensor, low_level: Tensor, output_size: tuple) -> Tensor:
        # aspp_in_unused는 builder에서 전달되지만 LR-ASPP는 별도 path 사용 안 함
        # → SegmentationModel을 통해 backbone (low, high) 받아야 함. 별도 forward 시그니처 필요
        raise NotImplementedError("LRASPPHead is wired through SegmentationModel.lraspp_forward")


class LRASPPModel(nn.Module):
    """LR-ASPP는 backbone (low, high) 둘 다 사용 → 별도 model class."""

    def __init__(self, backbone: nn.Module, low_in: int, high_in: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = LRASPPHead(low_in, high_in, num_classes)
        self._export = False

    def export_mode(self):
        self._export = True
        return self

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        low, high = self.backbone(x)
        feat = self.head.cbr(high) * self.head.scale(high)
        high_cls = self.head.high_classifier(feat)
        low_cls = self.head.low_classifier(low)
        # combine: high upsample to low → add
        high_up = interpolate(high_cls, size=low.shape[2:], mode="bilinear", align_corners=False)
        out = high_up + low_cls
        return interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
```

builder.py 업데이트: `head=="lraspp"` 분기에서 LRASPPModel 직접 반환:
```python
    if head_name == "lraspp":
        from src.models.necks.lr_aspp import LRASPPModel
        return LRASPPModel(backbone, low_ch, high_ch, num_classes)
```

- [ ] **Step 2: Update builder.py and smoke**

Modify `src/models/builder.py` `build_model` 끝에 lraspp 분기 추가 (위 코드).

Run: `cd p1 && uv run python -c "
import torch
cfg = {'model': {'backbone':'mobilenet_v3_large','head':'lraspp','num_classes':21,'pretrained':False,'use_aux':False}}
from src.models.builder import build_model
m = build_model(cfg).eval()
x = torch.zeros(1,3,480,640)
with torch.no_grad(): y = m(x)
print(tuple(y.shape))
"`
Expected: `(1, 21, 480, 640)`

- [ ] **Step 3: Commit**

```bash
git add p1/src/models/necks/lr_aspp.py p1/src/models/builder.py
git commit -m "p1: add LR-ASPP model (backup A)"
```

---

### Task 37: `src/config/light.yaml` + ONNX FLOPs check

**Files:** Create `p1/src/config/light.yaml`

- [ ] **Step 1: Write light.yaml**

`default.yaml` 복사 후 다음만 변경:
```yaml
model:
  backbone: mobilenet_v3_large
  head: lraspp
  num_classes: 21
  output_stride: 16
  pretrained: true
  use_aux: false
```

- [ ] **Step 2: ONNX export + FLOPs check**

Run: `cd p1 && uv run python -c "
import torch, onnx
from src.models.builder import build_model
cfg = {'model':{'backbone':'mobilenet_v3_large','head':'lraspp','num_classes':21,'pretrained':False,'use_aux':False}}
m = build_model(cfg).eval()
torch.onnx.export(m, torch.zeros(1,3,480,640), '/tmp/light.onnx',
    input_names=['input'], output_names=['logits'], opset_version=17, dynamic_axes=None)
from src.utils.flops import count_onnx_flops
total, br = count_onnx_flops('/tmp/light.onnx', (1,3,480,640))
print(f'A FLOPs: {total/1e9:.2f} GFLOPs')
"`
Expected: ~2-3 GFLOPs

- [ ] **Step 3: Commit**

```bash
git add p1/src/config/light.yaml
git commit -m "p1: add light.yaml (A: mobilenet + lraspp)"
```

---

## Phase 9 — README + Final (Tasks 38-39)

### Task 38: `p1/README.md`

**Files:** Create `p1/README.md`

- [ ] **Step 1: Write README**

```markdown
# Project 1 — Semantic Segmentation (학번: 2020314315)

Pascal-VOC 21-class semantic segmentation with ResNet-50 + DeepLabV3+ (OS=16). 자세한 설계는 `../docs/superpowers/specs/2026-04-24-p1-segmentation-design.md` 참조.

## Setup

```bash
cd p1
uv sync                                        # PyTorch + CUDA 12.8 binary
uv run python -m src.data.download             # VOC + SBD + COCO 다운 (최초 1회, ~30-60분)
```

## Training

```bash
# Stage 1 → Stage 2 자동 (resume 자동)
uv run python -m src.train --config src/config/default.yaml
```

중간 ckpt: `checkpoints/training_state_stage{1,2}.pth` (resume용), `best.pth` (val mIoU 갱신 시), `model.pth` (학습 종료, EMA only).

## Evaluation (val mIoU)

```bash
uv run python -m src.eval --config src/config/default.yaml --ckpt checkpoints/model.pth
```

## Inference (TTA)

```bash
# test_public → output/pred_<TAG>
uv run python -m src.infer --config src/config/default.yaml \
    --ckpt checkpoints/model.pth \
    --input input/test_public --output output/pred_FINAL

# 학교 reproduce (submit/img → submit/pred)
uv run python -m src.infer --config src/config/default.yaml \
    --ckpt checkpoints/model.pth \
    --input submit/img --output submit/pred
```

TTA: multi-scale [0.5, 0.75, 1.0, 1.25, 1.5] × hflip = 10× forward. `--no-tta`로 비활성화 가능.

## FLOPs Measurement (채점 기준)

```bash
# 1) ONNX export (가중치 제거, 10MB 이하)
uv run python -m src.export_onnx --config src/config/default.yaml \
    --ckpt checkpoints/model.pth --out model_structure.onnx

# 2) FLOPs 측정 (입력 [1, 3, 480, 640])
uv run python -m src.measure_flops --onnx model_structure.onnx
# → "[ONNX] model_structure.onnx: ~45 GFLOPs"

# 3) (optional) PyTorch sanity check
uv run python -m src.measure_flops --config src/config/default.yaml --ckpt checkpoints/model.pth
```

## Submission

3 채널:

1. **학교 사이트 (코드베이스 zip)**
   ```bash
   cd ..
   zip -r p1/2020314315_project01.zip \
       p1/src p1/checkpoints/model.pth p1/submit \
       p1/2020314315_project01_report.pdf p1/pyproject.toml p1/README.md \
       -x '**/__pycache__/*'
   ```

2. **채점 사이트 — PNG zip**
   ```bash
   uv run python -m src.package_submission \
       --pred output/pred_FINAL \
       --out submission_pred.zip
   # 검증: 1000 PNG, 000-999.png, [0,20], <500MB
   ```

3. **채점 사이트 — ONNX**
   `model_structure.onnx` 그대로 업로드 (≤10MB, 입력 [1,3,480,640]).

## Reproducibility

- 데스크탑(5070 Ti)에서 탐색 → Colab T4/L4에서 처음부터 전체 파이프라인 재실행이 제출물
- WandB Overview에 T4/L4 GPU 증거 자동 기록
- `checkpoints/training_state_stage{1,2}.pth`로 세션 끊겨도 resume

## Library Notes

- **torch>=2.7 + cu128 binary** (5070 Ti Blackwell sm_120 호환)
- **HuggingFace, Albumentations 미사용** (PDF 정책 준수)
- **`onnx`**: ONNX 제출 형식 처리. modeling/training 라이브러리 아님
- **AI 도구**: Claude Code (설계, 코드 작성, 리뷰) — report에 사용 내역 기재
```

- [ ] **Step 2: Commit**

```bash
git add p1/README.md
git commit -m "p1: add README with setup/train/infer/FLOPs/submit"
```

---

### Task 39: Reproduce flow validation

**Files:** No new code. Manual verification.

- [ ] **Step 1: Place a few test images in submit/img/**

```bash
cd p1
cp input/test_public/000.jpg input/test_public/100.jpg submit/img/
```

- [ ] **Step 2: Run reproduce flow per README**

```bash
uv run python -m src.infer --config src/config/default.yaml \
    --ckpt checkpoints/model.pth \
    --input submit/img --output submit/pred
```

Expected: `submit/pred/000.png`, `submit/pred/100.png` 생성, 파일명 jpg와 매칭, mode L uint8 [0..20]

- [ ] **Step 3: Verify outputs**

```bash
uv run python -c "
from PIL import Image
import numpy as np
for n in ['000', '100']:
    arr = np.array(Image.open(f'submit/pred/{n}.png'))
    print(f'{n}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min()}, {arr.max()}]')
"
```
Expected: shape이 원본 jpg와 동일, dtype uint8, [0, 20] 범위

- [ ] **Step 4: Cleanup test inputs (제출 전 submit/img/ 비우기)**

```bash
rm submit/img/*.jpg
```

- [ ] **Step 5: Final commit (no changes — verification only)**

---

## Self-Review Checklist

이 plan을 spec과 대조해서 누락 확인:

| Spec 요구 | Plan task |
|---|---|
| Project structure §4 | Task 2 |
| seed.py | Task 3 |
| metrics.py (mIoU) | Task 4 |
| flops.py (PyTorch + ONNX) | Tasks 5-6 |
| checkpoint.py | Task 7 |
| transforms.py | Task 8 |
| voc.py | Task 9 |
| coco.py | Task 10 |
| download.py | Task 11 |
| data/builder.py + isolation guard | Task 12 |
| backbones/resnet.py | Task 13 |
| necks/aspp.py | Task 14 |
| heads/deeplabv3plus.py | Task 15 |
| aux/fcn_head.py | Task 16 |
| seg_model.py + export_mode | Task 17 |
| models/builder.py | Task 18 |
| ONNX export 검증 | Task 19 |
| losses/seg_loss.py | Task 20 |
| config/default.yaml | Task 21 |
| train.py 점진 구축 | Tasks 22-27 |
| Smoke test | Task 28 |
| eval.py | Task 29 |
| infer.py (TTA) | Task 30 |
| export_onnx.py | Task 31 |
| measure_flops.py | Task 32 |
| package_submission.py | Task 33 |
| End-to-end smoke | Task 34 |
| backbones/mobilenet.py + LR-ASPP (A) | Tasks 35-36 |
| config/light.yaml | Task 37 |
| README.md | Task 38 |
| Reproduce verification | Task 39 |

**누락 없음.** spec의 모든 결정사항에 대응 task 존재.

---

## Execution Choice

Plan complete and saved to `docs/superpowers/plans/2026-04-24-p1-segmentation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. agent-team-design.md Phase 매핑 활용 (data-aug-engineer, model-architect, training-strategist, debugger-competitor, etc.)

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
