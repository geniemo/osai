"""MonoDepth2-like U-Net skeleton for practice."""

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


class ConvBlock(nn.Module):
    """Conv 기본 블록: [3×3 conv + BN + ReLU] × 2. Encoder/Decoder 공통 사용."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MonoDepthUNet(nn.Module):
    """MonoDepth2-like U-Net: 이미지 → 같은 크기의 depth map 예측.

    U자 구조:
      Encoder (왼쪽 아래로): 해상도↓ 채널↑ (무엇인지 이해)
      Decoder (오른쪽 위로): 해상도↑ 채널↓ (원래 크기로 복원)
      Skip connection: Encoder의 세밀한 위치 정보를 Decoder에 concat으로 전달

    192×640 입력 기준:
      enc1: (3, 192, 640)   → (64, 192, 640)  ─── skip1 ──→ dec1 입력
      enc2: (64, 96, 320)   → (128, 96, 320)  ─── skip2 ──→ dec2 입력
      enc3: (128, 48, 160)  → (256, 48, 160)  ─── skip3 ──→ dec3 입력
      enc4: (256, 24, 80)   → (512, 24, 80)   ─── skip4 ──→ dec4 입력
      bottleneck: (512, 12, 40) → (512, 12, 40)
      dec4: cat(up+skip4) (1024, 24, 80) → (256, 24, 80)
      dec3: cat(up+skip3) (512, 48, 160) → (128, 48, 160)
      dec2: cat(up+skip2) (256, 96, 320) → (64, 96, 320)
      dec1: cat(up+skip1) (128, 192, 640) → (64, 192, 640)
      head: (64, 192, 640) → (1, 192, 640)  ← depth map
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        # TODO: implement encoder-decoder with skip connections
        b = base_channels  # 64

        # ── Encoder: 해상도를 절반씩 줄이며 특징 추출 ──
        self.enc1 = ConvBlock(in_channels, b)      # 3 → 64
        self.enc2 = ConvBlock(b, b * 2)             # 64 → 128
        self.enc3 = ConvBlock(b * 2, b * 4)         # 128 → 256
        self.enc4 = ConvBlock(b * 4, b * 8)         # 256 → 512
        self.pool = nn.MaxPool2d(2)                  # 각 enc 사이에서 해상도 절반

        # ── Bottleneck: 가장 압축된 지점 ──
        self.bottleneck = ConvBlock(b * 8, b * 8)   # 512 → 512

        # ── Decoder: upsample + skip concat 후 채널 축소 ──
        # 입력 채널 = upsample(이전 dec) + skip(대응 enc) → concat이므로 채널이 2배
        self.dec4 = ConvBlock(b * 8 + b * 8, b * 4)  # 512+512=1024 → 256
        self.dec3 = ConvBlock(b * 4 + b * 4, b * 2)  # 256+256=512 → 128
        self.dec2 = ConvBlock(b * 2 + b * 2, b)      # 128+128=256 → 64
        self.dec1 = ConvBlock(b + b, b)               # 64+64=128 → 64

        # ── Head: 최종 depth 예측 ──
        self.head = conv1x1(b, out_channels)          # 64 → 1 (depth)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward

        # ── Encoder: 각 단계의 출력을 skip connection용으로 보존 ──
        e1 = self.enc1(x)              # (64, H, W)
        e2 = self.enc2(self.pool(e1))  # (128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (512, H/8, W/8)

        # ── Bottleneck ──
        bn = self.bottleneck(self.pool(e4))  # (512, H/16, W/16)

        # ── Decoder: upsample → skip과 concat → ConvBlock으로 채널 축소 ──
        # upsample(bn)을 e4 크기로 맞추고, 채널 방향으로 이어붙임(cat)
        d4 = self.dec4(torch.cat([nn.functional.interpolate(bn, size=e4.shape[2:], mode="bilinear", align_corners=False), e4], dim=1))
        d3 = self.dec3(torch.cat([nn.functional.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([nn.functional.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], dim=1))

        return self.head(d1)  # (1, H, W) — 입력과 같은 해상도의 depth map


def monodepth_unet() -> MonoDepthUNet:
    return MonoDepthUNet()
