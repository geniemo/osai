# P2 D-256 Colab Operations Guide

## Pre-flight (one-time)

1. **Drive 폴더 만들기**:
   - `MyDrive/p2-data/` — 데이터 zip 보관
   - `MyDrive/p2-runs/d256_main/` — ckpt + log 백업

2. **데이터 업로드 to Drive**:
   - `train_50k_256.zip` (1.65 GB) → `MyDrive/p2-data/train_50k_256.zip`
   - (optional) `valid_10k_256.zip` (~330 MB) → `MyDrive/p2-data/valid_10k_256.zip`

3. **Colab 노트북 열기**:
   - `colab_p2_d256.ipynb`를 Colab에 업로드 또는 GitHub에서 "Open in Colab"
   - Runtime → Change runtime type → **A100 GPU**

## 학습 launch (first session)

노트북 셀 1~7 순차 실행:
1. nvidia-smi (A100 확인)
2. Drive mount
3. Repo clone (`improve` 브랜치)
4. pip install
5. Drive에서 zip 복사 → `p2/data/`
6. WandB login (API key 입력)
7. Drive backup 디렉토리 생성

그 다음 **"Launch — first run"** 셀:
```python
%cd /content/osai
!PYTHONPATH=. python p2/train.py --config p2/configs/d256.yaml 2>&1 | tee -a /content/drive/MyDrive/p2-runs/d256_main/train.log
```

이 명령은 14M images 학습. 한 세션에서 24h 도달 후 disconnect 발생 가능 → resume 절차.

## Backup (periodic)

학습 도중 별도 셀에서 backup loop 실행 권장:
```python
import time, subprocess
while True:
    subprocess.run(['rsync', '-av', '--update',
                    '/content/osai/p2/runs/d256_main/',
                    '/content/drive/MyDrive/p2-runs/d256_main/'])
    time.sleep(600)  # 10분
```

이걸 같은 노트북의 별도 셀로 띄워두면 disconnect 시점에 최대 10분 손실.

## Resume (after disconnect)

1. 노트북 다시 열기 → A100 재할당 (Pro+이면 가용)
2. 셀 1~7 재실행 (mount, clone, install, copy data, wandb)
3. **Drive에서 마지막 ckpt 복원**:
   ```python
   !mkdir -p /content/osai/p2/runs/d256_main
   !cp /content/drive/MyDrive/p2-runs/d256_main/ckpt_*.pt /content/osai/p2/runs/d256_main/
   ```
4. **Resume launch 셀**:
   ```python
   import glob
   latest = sorted(glob.glob('/content/osai/p2/runs/d256_main/ckpt_*.pt'))[-1]
   !PYTHONPATH=. python p2/train.py --config p2/configs/d256.yaml --resume {latest}
   ```

`--resume`은 G/D/G_ema/optG/optD/RNG/pl_mean/wandb_run_id 모두 복원 → bit-for-bit 재개.

## FID 자기측정 (between sessions)

처음 1회 valid set 통계 캐시:
```python
!cp /content/drive/MyDrive/p2-data/valid_10k_256.zip /content/osai/p2/data/
!mkdir -p /content/osai/p2/data/valid_10k_256_dir
!cd /content/osai/p2/data/valid_10k_256_dir && unzip -q -o ../valid_10k_256.zip
!cd /content/osai && python -m pytorch_fid p2/data/valid_10k_256_dir --save-stats p2/checkpoints/fid_stats_256.npz
```

ckpt FID 측정:
```python
latest = sorted(glob.glob('/content/osai/p2/runs/d256_main/ckpt_*.pt'))[-1]
!cd /content/osai && PYTHONPATH=. python p2/eval_fid.py --ckpt {latest} --stats p2/checkpoints/fid_stats_256.npz --n 8000 --batch 32
```

## ABORT GATE 모니터링

WandB에서 다음 시점에 확인:
- **1h (~thr 측정)**: throughput < 100 img/s @ bs32 mixed → abort
- **6h (~3M images)**: FID > 60 → abort
- **12h (~7M images)**: FID > 30 → abort

abort 시 학습 중단, 결과 보고 → 옵션 C(baseline 확장) 별도 구체화.

## ONNX export (final)

```python
!cd /content/osai && PYTHONPATH=. python p2/export_onnx.py \
    --ckpt p2/runs/d256_main/final.pt \
    --out p2/checkpoints/model.onnx
!cp /content/osai/p2/checkpoints/model.onnx /content/drive/MyDrive/p2-runs/d256_main/
```

이후 Drive에서 `model.onnx` 다운로드하여 리더보드 제출.

## 컴퓨트 유닛 관리

A100 ~12 units/h, 952 units → 약 79h. 학습 종료 또는 abort 시점에 unit 잔량 확인:
- Colab UI 우측 상단의 컴퓨트 유닛 표시 확인
- 50 units 이하 남으면 ONNX export + 최종 작업에 우선 사용

학습이 다음 schedule을 따라 진행되도록:
- D2: ~7h 학습 (1~2 sessions)
- D3~D5: 누적 ~60h
- D6: 종료, FID + ONNX
- D7~D8: 리포트 + 제출 패키징
- D9: 최종 제출
