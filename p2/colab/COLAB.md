# P2 D-256 Colab Operations Guide (Drive-based)

작업 디렉토리는 **`MyDrive/osai/`** (사용자가 repo를 Drive에 그대로 보관하는 워크플로우).

## Pre-flight (one-time)

### 1. Drive에 osai repo 두기

이미 `MyDrive/osai/`에 repo가 있으면 OK. 없으면 노트북 셀 3번이 자동 clone.

### 2. 데이터 zip 위치

```
MyDrive/osai/p2/train_50k_256.zip   ← 학습 데이터 (필수, 1.65GB)
MyDrive/osai/p2/valid_10k_256.zip   ← FID 자기측정용 (선택, 0.33GB)
```

데스크탑에서 이미 `p2/` 폴더에 zip이 있다면, 그 파일을 통째로 Drive `MyDrive/osai/p2/` 로 업로드.

### 3. Colab 노트북 열기

- Colab → **파일 → 노트북 열기 → Google Drive 탭** → `MyDrive/osai/p2/colab/colab_p2_d256.ipynb` 선택
- **런타임 → 런타임 유형 변경 → A100 GPU**

## I/O 핵심 — 왜 데이터를 Colab 로컬로 cp 하나

`ZipImageDataset`은 매 sample마다 zip의 random entry를 읽음. Drive over FUSE는 **random access I/O가 10~30배 느림** → 학습 throughput 치명적 손실.

해결: **학습 데이터 zip만 `/content/p2_data/`로 cp** (Colab 로컬 SSD). 노트북 셀 5번이 자동.

코드와 ckpt는 Drive에 두어도 OK:
- 코드 import: Python module loader는 cold start 후 캐시됨
- ckpt 저장: async + 200k images마다 1회 → Drive write 시간이 학습 멈추지 않음
- ckpt가 Drive 직접 저장이라 **별도 backup 불필요** (disconnect 시 그대로 보존)

## 학습 launch (first session)

노트북 셀 1~6 순차 실행:
1. nvidia-smi (A100 확인)
2. Drive mount
3. Repo sync (clone or git pull)
4. pip install
5. **데이터 zip Drive → Colab 로컬 cp + p2/data/에 symlink**
6. WandB login

그 다음 **"Launch — first session"** 셀:
```python
%cd /content/drive/MyDrive/osai
!PYTHONPATH=. python p2/train.py --config p2/configs/d256.yaml 2>&1 | tee -a p2/runs/d256_main/train.log
```

- 학습 진행 로그는 `MyDrive/osai/p2/runs/d256_main/train.log`에 그대로 누적
- ckpt는 `MyDrive/osai/p2/runs/d256_main/ckpt_*.pt`에 자동 저장

## Resume (after disconnect)

1. 노트북 다시 열기 → A100 재할당
2. 셀 2 (Drive mount), 3 (git pull), 4 (pip), 5 (데이터 cp — Colab 로컬은 disconnect 시 wipe되므로 다시 cp 필요), 6 (wandb login)
3. **"Resume" 셀** 실행 — 최신 ckpt 자동 탐색 후 `--resume` 인자로 재개

`--resume`은 G/D/G_ema/optG/optD/RNG/pl_mean/wandb_run_id 모두 복원 → bit-for-bit 재개.

## FID 자기측정

처음 1회 valid set 통계 캐시 + 매 측정:
```python
# 노트북의 'Self-measure FID' 두 셀 순차 실행
```

이미 cache가 만들어졌으면 두 번째 셀만 매번 실행.

## ABORT GATE 모니터링

WandB(`ffhqgen-skku-p2/d256-stylegan2-scratch`)에서:

| 시점 | 확인 | 기준 |
|---|---|---|
| 1h | throughput | < 100 img/s @ bs32 mixed → abort |
| 6h (~3M images) | FID 자기측정 | > 60 → abort |
| 12h (~7M images) | FID | > 30 → abort |
| 언제든 | loss/grad | NaN/Inf, 발산, OOM 반복 |

abort 시 학습 중단 + 결과 보고 → 옵션 C(baseline 확장) 별도 구체화 결정.

## ONNX export (final)

```python
# 'ONNX export' 셀 실행
%cd /content/drive/MyDrive/osai
!PYTHONPATH=. python p2/export_onnx.py \
    --ckpt p2/runs/d256_main/final.pt \
    --out p2/checkpoints/model.onnx
```

`model.onnx`는 `MyDrive/osai/p2/checkpoints/`에 저장됨 → Drive UI에서 다운로드 → 리더보드 제출.

## 컴퓨트 유닛 관리

A100 ~12 units/h, 952 units 가용 → 약 **79h** A100. 14M images 학습은 ~22h 예상이므로 충분.

학습 종료 또는 abort 시점에 unit 잔량 확인 (Colab UI 우측 상단). 50 units 이하 남으면 ONNX export + 최종 작업에 우선 사용.
