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
from src.utils.checkpoint import save_full, load_full, save_model_only
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


def run_stage(cfg: dict, stage: int) -> None:
    """단일 stage 학습. stage 2는 base_lr/iters/warmup이 다르고, stage 1 best ckpt를 자동 로드."""
    cfg = {**cfg, "data": {**cfg["data"], "stage": stage}}
    if stage == 2:
        cfg["optimizer"] = {**cfg["optimizer"], "base_lr": cfg["training"]["stage2_base_lr"]}
        cfg["scheduler"] = {**cfg["scheduler"], "warmup_iters": cfg["training"]["stage2_warmup"]}
        total_iters = cfg["training"]["stage2_iters"]
    else:
        total_iters = cfg["training"]["stage1_iters"]

    # stage 별 ckpt path
    cfg = {**cfg, "paths": {**cfg["paths"], "training_state": cfg["paths"]["training_state"].replace(".pth", f"_stage{stage}.pth")}}

    # === 이하 기존 main 로직 ===
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    training_state_path = Path(cfg["paths"]["training_state"])
    wandb_run_id = None
    if training_state_path.exists():
        ckpt_peek = torch.load(str(training_state_path), map_location="cpu", weights_only=False)
        wandb_run_id = ckpt_peek.get("wandb_run_id")

    import wandb
    from datetime import datetime
    run_name = f"{cfg['wandb']['run_name_prefix']}_{datetime.now().strftime('%m%d')}_{cfg['model']['backbone']}_stage{stage}"
    run = wandb.init(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        name=run_name,
        config=cfg,
        id=wandb_run_id,
        resume="allow",
        tags=cfg["wandb"]["tags"] + [f"stage{stage}"],
    )
    wandb_run_id = run.id

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_poly_scheduler(optimizer, total_iters, cfg["scheduler"]["warmup_iters"], cfg["scheduler"]["power"])
    criterion = SegLoss(
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["loss"]["ignore_index"],
        dice_weight=cfg["loss"]["dice_weight"],
        aux_weight=cfg["loss"]["aux_weight"],
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["training"]["amp_dtype"] == "fp16"))
    amp_dtype = torch.float16 if cfg["training"]["amp_dtype"] == "fp16" else torch.bfloat16

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()
    ema_decay = cfg["training"]["ema_decay"]

    run.summary["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    n_params = sum(p.numel() for p in model.parameters())
    run.summary["params/total"] = n_params

    ckpt_dir = Path(cfg["paths"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = Path(cfg["paths"]["best_ckpt"])
    final_ckpt_path = Path(cfg["paths"]["final_ckpt"])
    iter_count = 0
    best_miou = 0.0
    if training_state_path.exists():
        meta = load_full(
            path=str(training_state_path),
            model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )
        iter_count = meta["iter"]
        best_miou = meta["best_miou"]
        print(f"[resume] from iter {iter_count}, best_miou={best_miou:.4f}")
    elif stage == 2 and Path(cfg["paths"]["best_ckpt"]).exists():
        bc = torch.load(cfg["paths"]["best_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(bc["ema_state"])
        ema_model.load_state_dict(bc["ema_state"])
        print(f"[stage2] loaded best ckpt from stage1 (mIoU={bc['miou']:.4f})")

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
            wandb.log({
                "step": iter_count,
                "train/loss": loss.item(),
                "lr/backbone": optimizer.param_groups[0]["lr"],
                "lr/head": optimizer.param_groups[1]["lr"],
            }, step=iter_count)
            print(f"iter {iter_count}/{total_iters} loss={loss.item():.4f} lr={optimizer.param_groups[1]['lr']:.5f}")

        if iter_count % cfg["training"]["ckpt_interval"] == 0:
            save_full(
                path=str(training_state_path), iter_count=iter_count, stage=stage,
                model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_miou=best_miou, wandb_run_id=wandb_run_id, config=cfg,
            )

        val_interval = cfg["training"][f"val_interval_stage{stage}"]
        if iter_count % val_interval == 0:
            miou_ema, _ = evaluate(ema_model, val_loader, device)
            wandb.log({"val/mIoU_ema": miou_ema, "val/best_mIoU": best_miou}, step=iter_count)
            print(f"  [val] mIoU_ema={miou_ema:.4f}")
            if miou_ema > best_miou:
                best_miou = miou_ema
                torch.save({"ema_state": ema_model.state_dict(), "iter": iter_count, "miou": miou_ema}, best_ckpt_path)

    save_model_only(str(final_ckpt_path), ema_model)
    print(f"[done] stage {stage} final ckpt → {final_ckpt_path}, best_miou={best_miou:.4f}")
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, default=None, help="1 or 2; 생략 시 둘 다 순차 실행")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage is None:
        run_stage(cfg, 1)
        run_stage(cfg, 2)
    else:
        run_stage(cfg, args.stage)


if __name__ == "__main__":
    main()
