"""ResNet-50 + FPN skeleton for practice."""

from typing import List, Tuple

# import torch
import torch.nn as nn
from torch import Tensor

from models.resnet50 import ResNet50


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


class ResNetBackbone(ResNet50):
    """ResNet50을 backbone으로 사용: forward 대신 forward_features로 중간 feature map을 추출.

    ResNet50.forward()는 최종 분류 결과만 반환하지만,
    FPN은 각 Stage의 출력(C2~C5)이 전부 필요하므로 별도 메서드로 분리.
    """

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # x: (N, 3, H, W)
        # TODO: return C2, C3, C4, C5 feature maps

        # Stem (ResNet50에서 상속): 해상도를 1/4로 축소
        x = self.conv1(x)    # (3, H, W) → (64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # → (64, H/4, W/4)

        # 각 Stage 출력을 저장 (ResNet50.forward()는 마지막만 사용했지만, 여기선 전부 보존)
        c2 = self.layer1(x)   # (256, H/4, W/4)   — 기초 패턴
        c3 = self.layer2(c2)  # (512, H/8, W/8)   — 형태
        c4 = self.layer3(c3)  # (1024, H/16, W/16) — 부품
        c5 = self.layer4(c4)  # (2048, H/32, W/32) — 물체
        return c2, c3, c4, c5


class FPN(nn.Module):
    """Feature Pyramid Network: 깊은 층의 의미 정보를 얕은 층에 전달.

    3단계 연산:
      1) lateral (1×1): 각 C의 채널 수를 out_channels(256)로 통일
      2) top-down: 깊은 층을 upsample하여 얕은 층에 더함 (의미 정보 전파)
      3) smooth (3×3): upsample 후 격자 무늬(aliasing)를 완화
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        super().__init__()
        # TODO: build lateral and smoothing layers
        self.lateral = nn.ModuleList()  # 채널 통일용 1×1 conv
        self.smooth = nn.ModuleList()   # aliasing 완화용 3×3 conv
        for ch in in_channels:
            # C2(256), C3(512), C4(1024), C5(2048) → 모두 256ch로 통일
            self.lateral.append(conv1x1(ch, out_channels))
            self.smooth.append(conv3x3(out_channels, out_channels))

    def forward(self, feats: List[Tensor]) -> List[Tensor]:
        # feats: [c2, c3, c4, c5]
        # TODO: build top-down FPN

        # 1) Lateral: 각 feature map의 채널을 256으로 통일
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        # laterals: [l2(256, H/4), l3(256, H/8), l4(256, H/16), l5(256, H/32)]

        # 2) Top-down: 깊은 층(l5) → 얕은 층(l2) 방향으로 upsample + 덧셈
        #   l5를 2배 확대 → l4에 더함 → l4를 2배 확대 → l3에 더함 → ...
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]  # 상위 층의 해상도에 맞춤
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(laterals[i], size=(h, w), mode="nearest")

        # 3) Smooth: 3×3 conv로 upsample 아티팩트 정리
        return [s(l) for s, l in zip(self.smooth, laterals)]


class ResNet50FPN(nn.Module):
    """ResNet50 + FPN: 다중 해상도 feature map을 출력하는 detection용 backbone.

    출력: [P2, P3, P4, P5] — 모두 256ch, 해상도는 입력의 1/4 ~ 1/32
      P2 (256, H/4, W/4)   — 작은 물체 감지에 유리 (고해상도)
      P3 (256, H/8, W/8)
      P4 (256, H/16, W/16)
      P5 (256, H/32, W/32) — 큰 물체 감지에 유리 (넓은 시야)
    """

    def __init__(self, fpn_channels: int = 256) -> None:
        super().__init__()
        # TODO: create backbone + FPN
        self.backbone = ResNetBackbone([3, 4, 6, 3])  # ResNet-50 표준 구성
        # C2~C5의 채널: [256, 512, 1024, 2048] → FPN에서 모두 fpn_channels(256)로 통일
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=fpn_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        # x: (N, 3, H, W)
        # TODO: return [p2, p3, p4, p5]
        feats = self.backbone.forward_features(x)  # → (C2, C3, C4, C5)
        return self.fpn(list(feats))                # → [P2, P3, P4, P5]


def resnet50_fpn() -> ResNet50FPN:
    return ResNet50FPN()
