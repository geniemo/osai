"""DeepLabv3-style segmentation model skeleton for practice."""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from models.resnet50 import ResNet50


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def conv3x3(in_ch: int, out_ch: int, dilation: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class ResNetBackbone(ResNet50):
    """ResNet50 backbone for segmentation: GAP+FC 제거, 최종 feature map만 반환.

    FPN backbone와 달리 중간 feature는 불필요 — 최종 출력만 ASPP에 전달.
    """

    def forward_features(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: return final feature map

        # Stem: (3, H, W) → (64, H/4, W/4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4개 Stage: ResNet50.forward()와 동일하지만 GAP+FC를 제거
        x = self.layer1(x)  # (256, H/4, W/4)
        x = self.layer2(x)  # (512, H/8, W/8)
        x = self.layer3(x)  # (1024, H/16, W/16)
        x = self.layer4(x)  # (2048, H/32, W/32) ← 이것만 반환
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: 여러 dilation rate로 다양한 스케일 정보 캡처.

    5개 병렬 branch:
      1) 1×1 conv              — 한 픽셀만 봄 (좁은 시야)
      2) 3×3 conv, dilation=6  — 13×13 영역을 봄
      3) 3×3 conv, dilation=12 — 25×25 영역을 봄
      4) 3×3 conv, dilation=18 — 37×37 영역을 봄
      5) Global Average Pool   — 전체 이미지를 봄 (가장 넓은 시야)

    → 5개 결과를 concat (256×5 = 1280ch) → 1×1 conv로 256ch로 축소
    """

    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
        super().__init__()
        # TODO: build ASPP branches and projection
        self.branches = nn.ModuleList()

        # Branch 1: 1×1 conv (dilation 없음, 가장 좁은 시야)
        self.branches.append(nn.Sequential(
            conv1x1(in_ch, out_ch),          # 2048 → 256
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ))
        # Branch 2~4: 3×3 conv with dilation=6, 12, 18
        # dilation이 클수록 같은 3×3 커널로 더 넓은 영역을 봄
        for rate in rates:
            self.branches.append(nn.Sequential(
                conv3x3(in_ch, out_ch, dilation=rate),  # 2048 → 256
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

        # Branch 5: Global Average Pooling (전체 이미지의 맥락 정보)
        # feature map 전체를 1×1로 압축 → 1×1 conv → 원래 크기로 upsample
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # (2048, H, W) → (2048, 1, 1)
            conv1x1(in_ch, out_ch),          # 2048 → 256
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # 5개 branch 결과를 concat 후 채널 축소
        # 256 × (1 + len(rates) + 1) = 256 × 5 = 1280 → 256
        self.project = nn.Sequential(
            conv1x1(out_ch * (len(rates) + 2), out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)  예: (1, 2048, 11, 11)
        # TODO: implement ASPP forward
        h, w = x.shape[2:]
        # 1×1 + dilation 6,12,18 branch 실행
        outs = [branch(x) for branch in self.branches]  # 각각 (256, 11, 11)
        # Global pool branch: (2048, 11, 11) → (256, 1, 1) → upsample → (256, 11, 11)
        outs.append(nn.functional.interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False))
        # concat: 256×5 = 1280ch → project: 1280 → 256ch
        return self.project(torch.cat(outs, dim=1))


class DeepLabV3(nn.Module):
    """DeepLabV3-style segmentation: Backbone + ASPP + pixel-wise 분류.

    전체 흐름 (321×321 입력, 21 클래스 기준):
      Backbone:  (3, 321, 321) → (2048, 11, 11)    특징 추출
      ASPP:      (2048, 11, 11) → (256, 11, 11)     다중 스케일 정보 융합
      Head:      (256, 11, 11) → (21, 11, 11)       클래스별 점수
      Upsample:  (21, 11, 11) → (21, 321, 321)      입력 크기로 복원
    """

    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        # TODO: create backbone, ASPP, and head
        self.backbone = ResNetBackbone([3, 4, 6, 3])
        self.aspp = ASPP(2048, 256, rates=[6, 12, 18])
        self.head = conv1x1(256, num_classes)  # 픽셀별 분류: 256 → num_classes

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward with resize to input size
        input_size = x.shape[2:]                        # 원본 크기 저장
        features = self.backbone.forward_features(x)    # (2048, H/32, W/32)
        features = self.aspp(features)                  # (256, H/32, W/32)
        out = self.head(features)                       # (num_classes, H/32, W/32)
        # bilinear upsample로 입력 크기 복원 (U-Net과 달리 한 번에 확대)
        return nn.functional.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def deeplab_v3(num_classes: int = 21) -> DeepLabV3:
    return DeepLabV3(num_classes=num_classes)
