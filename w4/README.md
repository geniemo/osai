# lab04

Practice project: implement deep learning models (starting with ResNet-50) and measure parameter counts and FLOPs.

## Structure
- `main.py`: build models and compute parameters/FLOPs
- `models/resnet50.py`: ResNet-50 implementation (complete)
- `models/*.py` (except `resnet50.py`): skeletons for students to complete
- `utils/param_utils.py`: parameter counter
- `utils/compute_utils.py`: FLOPs counter (Conv2d/Linear only, bias ignored)

## Run
```bash
python main.py
```

Options:
- `--model` 
- `--num-classes` (default 1000)
- `--batch-size` (default 1)
- `--height` (default 224)
- `--width` (default 224)
- `--device` (default cpu)

Example:
```bash
python main.py --model resnet50 --height 224 --width 224 --device cpu
```

## FLOPs Recording Guide
The tables below record FLOPs by input size for each model. Above each table, write `model name: parameter count`.

resnet50: 25,557,032
| 224x224 | 192x192 | 160x160 | 128x128 |
| --- | --- | --- | --- |
| 4,089,184,256 | 3,004,841,984 | 2,087,321,600 | 1,336,623,104 |

MobileNet v2: 3,504,872
| 224x224 | 192x192 | 160x160 | 128x128 |
| --- | --- | --- | --- |
| 300,774,272 | 221,316,608 | 154,083,200 | 99,074,048 |

ResNet50 + FPN-style: 28,899,368
| 640x640 | 800x800 | 1024x1024 | 800x1333 |
| --- | --- | --- | --- |
| 56,564,121,600 | 88,381,440,000 | 144,804,151,296 | 147,898,393,600 |

MonoDepth2-like U-net: 13,391,360
| 192x640 | 256x832 | 320x1024 | 384x1280 |
| --- | --- | --- | --- |
| 57,975,767,040 | 100,491,329,536 | 154,602,045,440 | 231,903,068,160 |

DeepLabv3-style model: 41,097,512
| 321x321 | 513x513 | 641x641 | 769x769 |
| --- | --- | --- | --- |
| 10,916,053,440 | 26,891,000,256 | 41,477,927,360 | 59,214,424,512 |
