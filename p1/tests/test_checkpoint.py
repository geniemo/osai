import torch
import torch.nn as nn

from src.utils.checkpoint import save_full, load_full, save_model_only


def _build_state():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 0.5 ** s)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    for _ in range(2):
        opt.zero_grad()
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        sched.step()
    return model, opt, sched, scaler


def test_save_load_roundtrip(tmp_path):
    model, opt, sched, scaler = _build_state()
    ema = nn.Linear(4, 2); ema.load_state_dict(model.state_dict())
    path = tmp_path / "ckpt.pth"
    save_full(
        path=str(path), iter_count=100, stage=1,
        model=model, ema_model=ema, optimizer=opt, scheduler=sched, scaler=scaler,
        best_miou=0.5, wandb_run_id="abc", config={"foo": 1},
    )

    new_model = nn.Linear(4, 2)
    new_ema = nn.Linear(4, 2)
    new_opt = torch.optim.SGD(new_model.parameters(), lr=999.0)
    new_sched = torch.optim.lr_scheduler.LambdaLR(new_opt, lr_lambda=lambda s: 1.0)
    new_scaler = torch.amp.GradScaler('cuda', enabled=False)
    meta = load_full(
        path=str(path), model=new_model, ema_model=new_ema,
        optimizer=new_opt, scheduler=new_sched, scaler=new_scaler,
    )
    assert meta["iter"] == 100
    assert meta["stage"] == 1
    assert meta["best_miou"] == 0.5
    assert meta["wandb_run_id"] == "abc"
    assert meta["config"] == {"foo": 1}
    assert "momentum_buffer" in new_opt.state[list(new_opt.state.keys())[0]]


def test_save_model_only_strips_extras(tmp_path):
    model = nn.Linear(4, 2)
    path = tmp_path / "model.pth"
    save_model_only(str(path), model)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    assert "weight" in state and "bias" in state
