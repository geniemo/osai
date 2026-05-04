"""build_model(cfg) — backbone/neck/head swap by yaml key."""
from __future__ import annotations

import torch.nn as nn

from src.models.backbones.resnet import ResNet50Backbone
from src.models.heads.deeplabv3plus import DeepLabV3PlusHead
from src.models.necks.aspp import ASPP
from src.models.aux.fcn_head import FCNHead
from src.models.seg_model import SegmentationModel


def build_model(cfg: dict):
    mc = cfg["model"]
    bb_name = mc["backbone"]
    head_name = mc["head"]
    num_classes = mc["num_classes"]
    output_stride = mc.get("output_stride", 16)
    pretrained = mc.get("pretrained", True)
    use_aux = mc.get("use_aux", True)

    if bb_name == "resnet50":
        backbone = ResNet50Backbone(output_stride=output_stride, pretrained=pretrained)
        low_ch, high_ch = ResNet50Backbone.LOW_CHANNELS, ResNet50Backbone.HIGH_CHANNELS
    elif bb_name == "mobilenet_v3_large":
        # implemented in Phase 8
        from src.models.backbones.mobilenet import MobileNetV3LargeBackbone
        backbone = MobileNetV3LargeBackbone(pretrained=pretrained)
        low_ch, high_ch = MobileNetV3LargeBackbone.LOW_CHANNELS, MobileNetV3LargeBackbone.HIGH_CHANNELS
    else:
        raise ValueError(f"Unknown backbone: {bb_name}")

    if head_name == "deeplabv3plus":
        rates = mc.get("aspp_rates", [6, 12, 18])
        neck = ASPP(in_channels=high_ch, out_channels=256, rates=tuple(rates))
        head = DeepLabV3PlusHead(low_in_channels=low_ch, aspp_out_channels=256, num_classes=num_classes)
    elif head_name == "lraspp":
        # LR-ASPP는 backbone (low+high) 둘 다 사용하는 별도 구조 → SegmentationModel 우회
        from src.models.necks.lr_aspp import LRASPPModel
        mid = mc.get("lraspp_mid", 128)
        return LRASPPModel(backbone, low_ch, high_ch, num_classes, mid=mid)
    else:
        raise ValueError(f"Unknown head: {head_name}")

    aux_head = FCNHead(in_channels=high_ch, num_classes=num_classes) if use_aux else None
    return SegmentationModel(backbone, neck, head, aux_head)
