"""MobileNetV2 skeleton for practice."""

# import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


class InvertedResidual(nn.Module):
    """Inverted Residual 블록: 좁→넓→좁 구조 (ResNet Bottleneck의 반대).

    예) in_ch=32, out_ch=32, expand_ratio=6 일 때:
        32ch → [1x1: 192ch 확장] → [3x3 depthwise: 192ch] → [1x1: 32ch 축소]
        depthwise conv는 채널별 독립 처리(groups=hidden_ch)로 파라미터가 극적으로 적음.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        # TODO: implement inverted residual block
        hidden_ch = in_ch * expand_ratio  # 확장된 채널 수 (예: 32 × 6 = 192)
        # skip connection 조건: 해상도와 채널 수가 모두 같을 때만
        # (경량 모델이라 ResNet처럼 downsample 변환을 추가하지 않음)
        self.use_res = stride == 1 and in_ch == out_ch

        layers = []
        # ── Phase 1: Expand (1×1) ── 채널 확장 (좁은→넓은)
        # expand_ratio=1인 첫 번째 설정(32→16)에서는 확장 불필요 → 생략
        if expand_ratio != 1:
            layers.extend([
                conv1x1(in_ch, hidden_ch),         # 예: 32 → 192
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
            ])
        # ── Phase 2: Depthwise (3×3) ── 채널별 독립적으로 공간 패턴 학습
        # groups=hidden_ch → 각 채널이 자기만의 3×3 커널로 독립 conv
        # 파라미터: hidden_ch × 1 × 3 × 3 (일반 conv의 1/hidden_ch배)
        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
        ])
        # ── Phase 3: Project (1×1) ── 채널 축소 (넓은→좁은)
        # 활성화 함수 없음! (좁은 채널에서 ReLU를 쓰면 정보 손실이 크기 때문)
        layers.extend([
            conv1x1(hidden_ch, out_ch),            # 예: 192 → 32
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # TODO: implement forward
        if self.use_res:
            return x + self.conv(x)  # skip connection: F(x) + x
        return self.conv(x)          # shape이 다르면 skip 없이 통과


class MobileNetV2(nn.Module):
    """MobileNetV2: depthwise separable conv로 경량화한 분류 모델.

    전체 흐름 (224×224 입력 기준):
      Stem:       (3, 224, 224) → (32, 112, 112)      초기 특징 추출
      Blocks ×17: (32, 112, 112) → (320, 7, 7)        InvertedResidual 반복
      Last Conv:  (320, 7, 7) → (1280, 7, 7)          고차원으로 확장
      Head:       (1280, 7, 7) → (1000,)               분류

    ResNet-50 대비: 파라미터 ~7배↓, FLOPs ~13배↓
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()
        # TODO: implement MobileNetV2 architecture

        # 논문 Table 2의 구성
        # t: expand_ratio, c: 출력 채널, n: 블록 반복 횟수, s: 첫 블록의 stride
        settings = [
            # t, c, n, s
            [1, 16, 1, 1],    # (32, 112, 112) → (16, 112, 112)
            [6, 24, 2, 2],    # (16, 112, 112) → (24, 56, 56)
            [6, 32, 3, 2],    # (24, 56, 56) → (32, 28, 28)
            [6, 64, 4, 2],    # (32, 28, 28) → (64, 14, 14)
            [6, 96, 3, 1],    # (64, 14, 14) → (96, 14, 14)   ← 해상도 유지
            [6, 160, 3, 2],   # (96, 14, 14) → (160, 7, 7)
            [6, 320, 1, 1],   # (160, 7, 7) → (320, 7, 7)     ← 해상도 유지
        ]

        input_ch = int(32 * width_mult)  # width_mult=1.0이면 32

        # ── Stem: 이미지 → 초기 feature map ──
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_ch, 3, stride=2, padding=1, bias=False),  # 224→112
            nn.BatchNorm2d(input_ch),
            nn.ReLU6(inplace=True),
        )

        # ── InvertedResidual 블록들 (총 17개) ──
        # 각 설정의 첫 블록만 stride=s, 나머지는 stride=1
        blocks = []
        for t, c, n, s in settings:
            output_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1  # 첫 블록만 stride 적용 (해상도 축소)
                blocks.append(InvertedResidual(input_ch, output_ch, stride, expand_ratio=t))
                input_ch = output_ch
        self.blocks = nn.Sequential(*blocks)

        # ── Last Conv: 채널을 1280으로 확장 ── (분류 직전 고차원 표현)
        self.last_conv = nn.Sequential(
            conv1x1(input_ch, 1280),  # 320 → 1280
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        # Output projection
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward
        x = self.stem(x)       # (3, 224, 224) → (32, 112, 112)
        x = self.blocks(x)     # (32, 112, 112) → (320, 7, 7)
        x = self.last_conv(x)  # (320, 7, 7) → (1280, 7, 7)
        x = x.mean(dim=(2, 3))  # Global Average Pooling → (1280,)
        x = self.fc(x)          # (1280,) → (num_classes,)
        return x


def mobilenet_v2(num_classes: int = 1000) -> MobileNetV2:
    return MobileNetV2(num_classes=num_classes)