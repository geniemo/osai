"""Training entrypoint. 처음 commit은 단일 stage 기본 loop;
후속 task에서 AMP/EMA/resume/WandB/2-stage 추가."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import LambdaLR

from src.data.builder import build_dataloaders
from src.losses.seg_loss import SegLoss
from src.models.builder import build_model
from src.utils.metrics import SegMetric
from src.utils.seed import set_seed


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    base_lr = cfg["optimizer"]["base_lr"]
    bb_mult = cfg["optimizer"]["backbone_lr_mult"]
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return torch.optim.SGD(
        [
            {"params": backbone_params, "lr": base_lr * bb_mult},
            {"params": other_params, "lr": base_lr},
        ],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )


def make_poly_scheduler(opt: torch.optim.Optimizer, total_iters: int, warmup_iters: int, power: float = 0.9) -> LambdaLR:
    def lam(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return (1.0 - progress) ** power
    return LambdaLR(opt, lr_lambda=lam)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, num_classes=21) -> tuple[float, list]:
    metric = SegMetric(num_classes=num_classes, ignore_index=255)
    model.eval()
    for img, mask in loader:
        img = img.to(device); mask = mask.to(device)
        out = model(img)
        if isinstance(out, tuple): out = out[0]
        pred = out.argmax(dim=1)
        metric.update(pred, mask)
    model.train()
    return metric.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()
    ema_decay = cfg["training"]["ema_decay"]
    optimizer = make_optimizer(model, cfg)
    total_iters = cfg["training"]["stage1_iters"]
    scheduler = make_poly_scheduler(optimizer, total_iters, cfg["scheduler"]["warmup_iters"], cfg["scheduler"]["power"])
    criterion = SegLoss(
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["loss"]["ignore_index"],
        dice_weight=cfg["loss"]["dice_weight"],
        aux_weight=cfg["loss"]["aux_weight"],
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["training"]["amp_dtype"] == "fp16"))
    amp_dtype = torch.float16 if cfg["training"]["amp_dtype"] == "fp16" else torch.bfloat16

    iter_count = 0
    data_iter = iter(train_loader)
    model.train()
    log_interval = cfg["training"]["log_interval"]
    while iter_count < total_iters:
        try:
            img, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img, mask = next(data_iter)
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out = model(img)
            main_logits, aux_logits = (out if isinstance(out, tuple) else (out, None))
            loss = criterion(main_logits, aux_logits, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, ema_decay)
        iter_count += 1
        if iter_count % log_interval == 0:
            print(f"iter {iter_count}/{total_iters} loss={loss.item():.4f} lr={optimizer.param_groups[1]['lr']:.5f}")


if __name__ == "__main__":
    main()
