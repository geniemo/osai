"""ResNet-50 implementation for practice."""

from typing import List

# import torch
import torch.nn as nn
from torch import Tensor


# padding=1이면 입출력 해상도가 동일하게 유지됨 (stride=1일 때)
def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


# 1x1 conv = 각 픽셀 위치에서 채널 방향으로만 Linear 변환 (공간은 안 건드림)
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """Bottleneck 블록: 1x1 → 3x3 → 1x1 구조로 채널을 좁혔다 넓힘.

    예) channels=64일 때:
        입력 256ch → [1x1: 64ch] → [3x3: 64ch] → [1x1: 256ch] → 출력 256ch
        병목(64ch)에서 3x3 conv을 수행하므로 파라미터가 훨씬 적음.
        (256ch에서 바로 3x3 하면 ~59만, 병목 쓰면 ~7만)
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        mid_channels = channels  # 병목 채널 수 (예: 64, 128, 256, 512)

        # 채널 축소: in_channels → mid_channels (예: 256 → 64)
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 공간 패턴 학습: mid_channels → mid_channels (예: 64 → 64)
        # stride=2이면 여기서 해상도가 절반으로 줄어듦
        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 채널 복원: mid_channels → channels*4 (예: 64 → 256, expansion=4)
        self.conv3 = conv1x1(mid_channels, channels * 4)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU()

        # skip connection에서 F(x) + x를 하려면 shape이 같아야 함.
        # 채널 수나 해상도가 달라지는 경우, x의 shape을 맞춰주는 변환이 필요.
        self.downsample = None
        if stride != 1 or in_channels != channels * 4:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, channels * 4, stride),  # 채널 + 해상도 맞춤
                nn.BatchNorm2d(channels * 4),
            )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        identity = x  # skip connection용 원본 보존

        # 1x1 → BN → ReLU: 채널 축소
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 → BN → ReLU: 공간 패턴 학습 (핵심 연산)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 → BN: 채널 복원 (여기선 ReLU 안 함 — 더한 후에 적용)
        out = self.conv3(out)
        out = self.bn3(out)

        # shape이 다르면 downsample로 맞춰줌
        if self.downsample is not None:
            identity = self.downsample(x)

        # skip connection: F(x) + x — ResNet의 핵심
        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50: 이미지를 단계적으로 추상화하여 분류하는 모델.

    전체 흐름 (224x224 입력 기준):
      Stem:    (3, 224, 224) → (64, 56, 56)      빠르게 크기 줄이기
      Stage 1: (64, 56, 56)  → (256, 56, 56)     기초 패턴 (선, 질감)
      Stage 2: (256, 56, 56) → (512, 28, 28)     형태, 패턴
      Stage 3: (512, 28, 28) → (1024, 14, 14)    부품 (눈, 바퀴)
      Stage 4: (1024,14, 14) → (2048, 7, 7)      물체 수준 이해
      Head:    (2048, 7, 7)  → (1000,)            클래스 점수

    패턴: 해상도는 절반씩 줄이고, 채널은 2배씩 늘림
          → 위치 정보를 압축하면서 의미 정보를 풍부하게 만듦
    """

    def __init__(
        self,
        layers: List[int],  # 각 Stage의 Bottleneck 반복 횟수. ResNet-50은 [3, 4, 6, 3]
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        self.in_channels = 64  # Stem 출력 채널 수. _make_layer에서 갱신됨
        self.dilation = 1

        # ── Stem: 이미지를 다루기 좋은 크기로 빠르게 줄이기 ──
        # 7x7 큰 커널로 넓은 영역을 보면서 224→112로 축소
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112→56

        # ── 4개 Stage: 점점 깊은 의미를 추출 ──
        # _make_layer(병목채널, 블록수, stride)
        #   출력 채널 = 병목채널 × 4 (expansion)
        self.layer1 = self._make_layer(64, layers[0])            # → 256ch, 56×56
        self.layer2 = self._make_layer(128, layers[1], stride=2)  # → 512ch, 28×28
        self.layer3 = self._make_layer(256, layers[2], stride=2)  # → 1024ch, 14×14
        self.layer4 = self._make_layer(512, layers[3], stride=2)  # → 2048ch, 7×7

        # ── Head: 최종 분류 ──
        # 512*4 = 2048 → num_classes
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(
        self,
        channels: int,  # 병목 채널 수 (출력은 channels*4)
        blocks: int,    # Bottleneck 반복 횟수
        stride: int = 1,
    ) -> nn.Sequential:
        layers = []
        # 첫 번째 블록: stride 적용 + downsample 필요 (채널/해상도 변화)
        layers.append(
            Bottleneck(
                self.in_channels,
                channels,
                stride=stride,
            )
        )
        self.in_channels = channels * 4  # 이후 블록의 입력 채널 갱신
        # 나머지 블록: stride=1, 입출력 shape 동일 → downsample 불필요
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.in_channels,
                    channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)

        # ── Stem: (3, 224, 224) → (64, 56, 56) ──
        x = self.conv1(x)   # (3, 224, 224) → (64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (64, 112, 112) → (64, 56, 56)

        # ── Stages: 점점 채널↑ 해상도↓ ──
        x = self.layer1(x)  # (64, 56, 56) → (256, 56, 56)   기초 패턴
        x = self.layer2(x)  # (256, 56, 56) → (512, 28, 28)  형태
        x = self.layer3(x)  # (512, 28, 28) → (1024, 14, 14) 부품
        x = self.layer4(x)  # (1024, 14, 14) → (2048, 7, 7)  물체

        # ── Head ──
        # Global Average Pooling: 각 채널의 H×W 값을 평균 → 숫자 하나
        x = x.mean(dim=(2, 3))  # (B, 2048, 7, 7) → (B, 2048)
        x = self.fc(x)          # (B, 2048) → (B, num_classes)
        # x: (N, num_classes)
        return x


def resnet50(num_classes: int = 1000) -> ResNet50:
    return ResNet50([3, 4, 6, 3], num_classes=num_classes)  # ResNet-50 표준 구성
