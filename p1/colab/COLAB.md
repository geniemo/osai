# Colab Reproduction Guide — Project 1

> **목적**: 데스크탑(5070 Ti) 학습 결과를 Colab T4/L4에서 처음부터 재실행 → 공식 제출물 생성. PDF 명시: "Colab L4 또는 T4: 확정된 솔루션을 전체 파이프라인 처음부터 재실행 → 이 결과가 제출물". WandB Overview에 T4/L4 GPU evidence 자동 기록.

## 사전 준비 (사용자 액션, 1회)

1. **Colab Pro+ 활성화**
2. **Google Drive에 다음 폴더 구조 생성**:
   ```
   /MyDrive/osai-p1/
   ├── img.zip              ← 데스크탑에서 업로드 (test images, ~114MB)
   └── checkpoints/         ← Colab이 자동 채움
   ```
3. **데스크탑에서 Drive로 업로드**:
   - `p1/img.zip` (114MB) → `/MyDrive/osai-p1/img.zip`
   - 또는 `p1/input/test_public/` 통째로 zip해서 업로드

4. **WandB API key 준비**: https://wandb.ai/authorize 에서 복사

5. **Colab 새 노트북 열기**, **런타임 → 런타임 유형 변경 → GPU (T4 또는 L4)**

## Cell 단위 실행

각 셀을 순서대로 복사해서 Colab notebook에 붙여넣고 실행. 셀 간에 변수 공유됨.

---

### Cell 1: Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive')
```

→ 인증 팝업 → 허용. `/content/drive/MyDrive/osai-p1/` 폴더 보이면 성공.

---

### Cell 2: 저장소 clone

```bash
%cd /content
!git clone https://github.com/geniemo/osai.git
%cd osai/p1
```

→ `~/content/osai/p1/` 작업 디렉토리.

---

### Cell 3: uv 설치 + 의존성 sync

```bash
!pip install -q uv
!uv sync
```

→ `.venv/` 생성, PyTorch + CUDA 12.8 binary 설치 (~3-5분, Colab 빠른 인터넷).

**주의**: 첫 `uv sync`는 bash가 새 venv 만드므로 시간 걸림. 후속 셀들은 `!uv run ...` 형식으로 venv 활용.

---

### Cell 4: GPU 확인

```bash
!uv run python -c "
import torch
print('Device:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
print('PyTorch:', torch.__version__)
"
```

**기대**:
- T4: `Tesla T4`, capability `(7, 5)`, ~15.8 GB
- L4: `NVIDIA L4`, capability `(8, 9)`, ~22.5 GB

만약 다르면 런타임 유형 재확인. T4/L4 외 GPU(예: A100)는 spec 위반.

---

### Cell 5: WandB 로그인

```bash
!uv run wandb login
```

→ API key 붙여넣기. 또는 환경변수로:

```python
import os
os.environ['WANDB_API_KEY'] = '여기에_key_붙여넣기'
```

---

### Cell 6: Test images 복사 (Drive → Colab)

```bash
!mkdir -p input
!cp /content/drive/MyDrive/osai-p1/img.zip input/
!cd input && unzip -q img.zip -d test_public
!ls input/test_public | wc -l   # → 1000 expected
```

만약 zip 구조가 다르면 (예: 폴더 한 단계 더 깊이) 압축 풀고 폴더 위치 조정 필요.

---

### Cell 7: 데이터 다운로드 (VOC + COCO)

```bash
!uv run python -m src.data.download --voc-root data/voc --coco-root data/coco
```

→ VOC ~2GB (~5분), COCO ~25GB (~10-30분 Colab 인터넷). 압축 해제 자동.

**시간**: Colab 인터넷이 빠르면 ~15-20분, 느리면 ~30-45분.

---

### Cell 8: COCO mask cache 사전 생성

```bash
!uv run python -m src.build_coco_masks --coco-root data/coco --num-workers 4
```

→ 95K mask 생성, Colab CPU 4-8 코어로 ~5-15분.

---

### Cell 9: 학습 (Stage 1 + Stage 2 자동)

```bash
!uv run python -m src.train --config src/config/colab.yaml
```

**예상 시간** (데스크탑 5070 Ti 기준 ~6h, Colab 환산):
- T4: **~10-12시간** (단일 세션 한계 ~24h, 가능)
- L4: **~7-9시간**

WandB run 자동 생성, GPU evidence 기록됨. 실시간 모니터링:
- https://wandb.ai/g1nie-sungkyunkwan-university/osai-p1-local

**중간 세션 끊김 대비**: ckpt가 Drive에 저장됨 → 새 세션에서 Cell 1-7 다시 + Cell 9 다시 실행 → 자동 resume.

---

### Cell 10: Validation mIoU 측정

```bash
!uv run python -m src.eval \
    --config src/config/colab.yaml \
    --ckpt /content/drive/MyDrive/osai-p1/checkpoints/model.pth
```

→ val mIoU + 21 class별 IoU 출력.

---

### Cell 11: 추론 (TTA) — test_public 1000장

```bash
!mkdir -p output
!uv run python -m src.infer \
    --config src/config/colab.yaml \
    --ckpt /content/drive/MyDrive/osai-p1/checkpoints/model.pth \
    --input input/test_public \
    --output output/pred_FINAL
```

→ 1000 PNG 생성, multi-scale TTA. T4 기준 ~30-40분.

**`--no-tta`** 옵션으로 빠른 추론 가능 (TTA 없이, mIoU -1~3%).

---

### Cell 12: ONNX export (가중치 제거, 채점 ONNX)

```bash
!uv run python -m src.export_onnx \
    --config src/config/colab.yaml \
    --ckpt /content/drive/MyDrive/osai-p1/checkpoints/model.pth \
    --out /content/drive/MyDrive/osai-p1/model_structure.onnx
```

→ `~0.3 MB` 파일 생성.

---

### Cell 13: FLOPs 측정 확인

```bash
!uv run python -m src.measure_flops --onnx /content/drive/MyDrive/osai-p1/model_structure.onnx
```

→ `[ONNX] ...: 81.12 GFLOPs` 기대.

---

### Cell 14: PNG zip 패키징 (채점 사이트 제출용)

```bash
!uv run python -m src.package_submission \
    --pred output/pred_FINAL \
    --out /content/drive/MyDrive/osai-p1/submission_pred.zip
```

→ 검증 통과 시 `[ok] /content/drive/MyDrive/osai-p1/submission_pred.zip`. 1000 PNG (000-999), 압축해제 ≤500MB.

---

### Cell 15: 결과 확인

```bash
!ls -la /content/drive/MyDrive/osai-p1/
!ls -la /content/drive/MyDrive/osai-p1/checkpoints/
```

**최종 산출물 (Drive에 저장됨):**
- `model_structure.onnx` (~0.3 MB) → 채점 사이트 ONNX 업로드
- `submission_pred.zip` (~10-30 MB) → 채점 사이트 PNG 업로드
- `checkpoints/model.pth` (~180 MB) → 학교 사이트 코드베이스 zip 포함
- `checkpoints/training_state_stage{1,2}.pth` → resume 기록 (제출 zip 미포함)

## 채점 사이트 업로드

1. **PNG zip**: `submission_pred.zip` 그대로 업로드
2. **ONNX**: `model_structure.onnx` 그대로 업로드

## 학교 사이트 zip (코드베이스) — 데스크탑에서 생성

Colab의 model.pth (180MB)를 데스크탑으로 다운로드 → 데스크탑에서 다음 실행:

```bash
cd /home/park/workspace/osai
# 1) Colab의 model.pth를 데스크탑 p1/checkpoints/model.pth에 덮어쓰기
# 2) 리포트 PDF p1/2020314315_project01_report.pdf 준비
# 3) zip 생성
zip -r p1/2020314315_project01.zip \
    p1/src p1/checkpoints/model.pth p1/submit \
    p1/2020314315_project01_report.pdf p1/pyproject.toml p1/README.md \
    -x "**/__pycache__/*" "**/.venv/*"
```

## WandB Overview 캡처

학습 완료 후 WandB run 페이지에서:
- Overview 탭의 GPU info (`Tesla T4` 또는 `NVIDIA L4`) 캡처
- Loss/mIoU curve 캡처
- Config 캡처

리포트에 첨부.

## 트러블슈팅

| 문제 | 원인 | 해결 |
|---|---|---|
| `uv sync` 실패 (CUDA wheel 오류) | T4 driver 호환 | torch 2.7+ cu128 binary가 T4 (sm_75) 호환됨. 재시도. |
| OOM (GPU 메모리 부족) | batch_size 16 너무 큰 경우 | colab.yaml에서 `batch_size: 12`로 |
| WandB run 생성 실패 | API key 미입력 | Cell 5 다시 |
| Drive 마운트 실패 | 인증 만료 | 새 세션에서 다시 마운트 |
| 세션 timeout (24h 후) | Pro+ 한계 | 새 세션에서 Cell 1-7 + Cell 9 → 자동 resume |

## 시간 예산

T4 기준 보수적 예산:

| 단계 | 시간 |
|---|---|
| Cell 1-7 셋업 + 데이터 다운 | ~30-45분 |
| Cell 8 mask cache | ~10-15분 |
| Cell 9 학습 (Stage 1+2) | **~10-12시간** |
| Cell 10-14 평가 + 제출물 | ~40-50분 |
| **총** | **~12-14시간** |

→ T4 단일 세션(24h)에 충분. L4면 ~9-11시간.

5/5 23:59 마감까지 여유 있게.
