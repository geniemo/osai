# Project 1 — Pascal-VOC Semantic Segmentation (학번: 2020314315)

ResNet-50 + DeepLabV3+ (Output Stride 16). COCO+VOC 2-stage 학습 + Copy-Paste augmentation + Boundary-weighted CE loss.

## 의존성 설치

```bash
cd p1
uv sync
uv run python -m src.data.download \
    --voc-root data/voc --coco-root data/coco
uv run python -m src.build_coco_masks \
    --coco-root data/coco --num-workers 4
```

## 학습

```bash
# Stage 1: COCO+VOC mixed, 160K iter
uv run python -m src.train --config src/config/colab_v2_final_s1.yaml --stage 1

# Stage 2: VOC only, 8K iter, Copy-Paste + Boundary loss (Stage 1 best ckpt에서 bootstrap)
uv run python -m src.train --config src/config/colab_v2_final_s2.yaml --stage 2
```

## 추론 / 재현 (`submit/img/*.jpg` → `submit/pred/*.png`)

```bash
uv run python -m src.infer \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints/model.pth \
    --input submit/img --output submit/pred
```

동일 파일명으로 출력 (`0001.jpg` → `0001.png`). PNG mode L, pixel ∈ [0, 20]. TTA (multi-scale + hflip) 기본 활성화.

## FLOPs 측정

```bash
# 1) ONNX export
uv run python -m src.export_onnx \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints/model.pth \
    --out model_structure.onnx

# 2) FLOPs 측정 (입력 [1, 3, 480, 640])
uv run python -m src.measure_flops --onnx model_structure.onnx
# → "[ONNX] model_structure.onnx: 162.24 GFLOPs"
```
