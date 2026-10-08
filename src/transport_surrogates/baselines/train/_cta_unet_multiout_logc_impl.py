from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.transport_surrogates.baselines.models.temporal_attn_unet import CTAUNet
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
from src.shared.data.io import flatten_param_folders, load_param_split, load_train_stats
from src.shared.utils.seed import set_seed


def make_weight_log_linear(y_log: torch.Tensor, alpha: float, y_bg: float, y_cap: float) -> torch.Tensor:
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0 else 1.0
    s = torch.clamp((y_log - y_bg) / denom, 0.0, 1.0)
    return 1.0 + alpha * s


def timestep_weights(T: int, mode: str, w_min: float, w_max: float) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, steps=T)
    s = t * t if mode == "quad" else t
    return w_min + (w_max - w_min) * s


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    hub = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    return (weight * hub).mean()


def joint_flip(K: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        K = torch.flip(K, dims=[-1])
        Y = torch.flip(Y, dims=[-1])
    if random.random() < 0.5:
        K = torch.flip(K, dims=[-2])
        Y = torch.flip(Y, dims=[-2])
    return K, Y


@dataclass
class CFG:
    data_root: str = "./data/T25_TSTEP_OVERRIDE_FINAL"
    split_json: str = "splits/param_split_fixed.json"
    stats_json: str = "stats/train_stats.json"
    out_root: str = "runs"
    run_name: str = ""
    seed: int = 0

    patch_size: int = 320
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_logK: bool = True
    eps_k: float = 1e-6
    eps_c: float = 1e-12

    base: int = 64
    t_embed_dim: int = 25
    t_heads: int = 5
    pool_stride: int = 4

    alpha: float = 3.0
    y_bg: float = -12.0
    y_cap: float = -2.0
    huber_delta: float = 1.0
    pred_clamp_min: float = -12.0
    pred_clamp_max: float = 2.0

    t_weight_mode: str = "quad"
    t_w_min: float = 0.5
    t_w_max: float = 1.5
    lambda_temp: float = 0.05
    lambda_mass: float = 0.0

    use_flip: bool = False
    use_warmup: bool = False


def parse_args() -> CFG:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=CFG.data_root)
    ap.add_argument("--split_json", type=str, default=CFG.split_json)
    ap.add_argument("--stats_json", type=str, default=CFG.stats_json)
    ap.add_argument("--out_root", type=str, default=CFG.out_root)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--patch_size", type=int, default=CFG.patch_size)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    ap.add_argument("--grad_clip_norm", type=float, default=CFG.grad_clip_norm)
    ap.add_argument("--num_workers", type=int, default=CFG.num_workers)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--use_logK", dest="use_logK", action="store_true", default=True)
    ap.add_argument("--no_logK", dest="use_logK", action="store_false")
    ap.add_argument("--eps_k", type=float, default=CFG.eps_k)
    ap.add_argument("--eps_c", type=float, default=CFG.eps_c)
    ap.add_argument("--base", type=int, default=CFG.base)
    ap.add_argument("--t_embed_dim", type=int, default=CFG.t_embed_dim)
    ap.add_argument("--t_heads", type=int, default=CFG.t_heads)
    ap.add_argument("--pool_stride", type=int, default=CFG.pool_stride)
    ap.add_argument("--alpha", type=float, default=CFG.alpha)
    ap.add_argument("--y_bg", type=float, default=CFG.y_bg)
    ap.add_argument("--y_cap", type=float, default=CFG.y_cap)
    ap.add_argument("--huber_delta", type=float, default=CFG.huber_delta)
    ap.add_argument("--pred_clamp_min", type=float, default=CFG.pred_clamp_min)
    ap.add_argument("--pred_clamp_max", type=float, default=CFG.pred_clamp_max)
    ap.add_argument("--t_weight_mode", type=str, default=CFG.t_weight_mode, choices=["linear", "quad"])
    ap.add_argument("--t_w_min", type=float, default=CFG.t_w_min)
    ap.add_argument("--t_w_max", type=float, default=CFG.t_w_max)
    ap.add_argument("--lambda_temp", type=float, default=CFG.lambda_temp)
    ap.add_argument("--lambda_mass", type=float, default=CFG.lambda_mass)
    ap.add_argument("--use_flip", action="store_true")
    ap.add_argument("--use_warmup", action="store_true")

    args = ap.parse_args()
    cfg = CFG()
    for field_name in cfg.__dataclass_fields__:
        if hasattr(args, field_name):
            setattr(cfg, field_name, getattr(args, field_name))
    cfg.amp = True
    if args.no_amp:
        cfg.amp = False
    cfg.use_logK = True
    return cfg


def compute_losses(pred: torch.Tensor, target: torch.Tensor, cfg: CFG) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
    target = target.float()
    predL = torch.clamp(pred, cfg.pred_clamp_min, cfg.pred_clamp_max)

    w_y = make_weight_log_linear(target, cfg.alpha, cfg.y_bg, cfg.y_cap)
    T = target.shape[1]
    w_t = timestep_weights(T, cfg.t_weight_mode, cfg.t_w_min, cfg.t_w_max).to(target.device).view(1, T, 1, 1)
    w = w_y * w_t

    loss_main = weighted_huber(predL, target, w, delta=cfg.huber_delta)
    d1 = predL[:, 1:] - predL[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    loss_temp = d2.abs().mean()

    loss_mass = torch.tensor(0.0, device=target.device)
    if cfg.lambda_mass > 0.0:
        pred_phys = torch.clamp(10.0 ** predL - cfg.eps_c, min=0.0)
        true_phys = torch.clamp(10.0 ** target - cfg.eps_c, min=0.0)
        mass_pred = pred_phys.sum(dim=(-2, -1))
        mass_true = true_phys.sum(dim=(-2, -1))
        loss_mass = (torch.abs(mass_pred - mass_true) / (mass_true + 1.0)).mean()

    loss = loss_main + cfg.lambda_temp * loss_temp + cfg.lambda_mass * loss_mass
    return loss, loss_main, loss_temp, loss_mass


def train_one_epoch(model: CTAUNet, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, cfg: CFG) -> dict:
    model.train()
    tloss = tmain = ttemp = 0.0
    n = n_skip = 0
    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)
        if cfg.use_flip:
            K, Y = joint_flip(K, Y)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(K)

        loss, lm, lt, _ = compute_losses(pred, Y, cfg)
        if not torch.isfinite(loss):
            n_skip += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                n_skip += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            optimizer.step()

        tloss += float(loss.item())
        tmain += float(lm.item())
        ttemp += float(lt.item())
        n += 1

    return {"loss": tloss / max(n, 1), "main": tmain / max(n, 1), "temp": ttemp / max(n, 1), "skipped": n_skip}


@torch.no_grad()
def eval_one_epoch(model: CTAUNet, loader: DataLoader, cfg: CFG) -> dict:
    model.eval()
    tloss = tmain = ttemp = 0.0
    n = 0
    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)
        pred = model(K)
        loss, lm, lt, _ = compute_losses(pred, Y, cfg)
        if not torch.isfinite(loss):
            continue
        tloss += float(loss.item())
        tmain += float(lm.item())
        ttemp += float(lt.item())
        n += 1
    return {"loss": tloss / max(n, 1), "main": tmain / max(n, 1), "temp": ttemp / max(n, 1)}


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    if not cfg.run_name:
        cfg.run_name = (
            f"transport_ctaunet_multiout_logc_p{cfg.patch_size}"
            f"_base{cfg.base}_ted{cfg.t_embed_dim}_th{cfg.t_heads}_ps{cfg.pool_stride}"
            f"_twquad{cfg.t_w_min:g}-{cfg.t_w_max:g}"
            f"_a{cfg.alpha:g}_ltemp{cfg.lambda_temp:g}_lmass{cfg.lambda_mass:g}"
            f"_lr{cfg.lr:g}_seed{cfg.seed}"
        )

    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg.__dict__, handle, indent=2)

    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files = flatten_param_folders(cfg.data_root, train_params)
    val_files = flatten_param_folders(cfg.data_root, val_params)
    stats = load_train_stats(cfg.stats_json)

    train_ds = GroundwaterPatchDatasetMultiOut(
        train_files,
        stats,
        patch_size=cfg.patch_size,
        use_logK=cfg.use_logK,
        eps_k=cfg.eps_k,
        eps_c=cfg.eps_c,
    )
    val_ds = GroundwaterPatchDatasetMultiOut(
        val_files,
        stats,
        patch_size=cfg.patch_size,
        use_logK=cfg.use_logK,
        eps_k=cfg.eps_k,
        eps_c=cfg.eps_c,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = CTAUNet(
        in_ch=1,
        out_ch=25,
        base=cfg.base,
        t_embed_dim=cfg.t_embed_dim,
        t_heads=cfg.t_heads,
        pool_stride=cfg.pool_stride,
    ).to(cfg.device)

    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"[CTA-UNet] Params: {n_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "train_main",
                "val_main",
                "train_temp",
                "val_temp",
                "lr",
                "skipped",
            ])

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "stats": stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if cfg.amp else None,
            "best_val": best_val,
        }
        torch.save(ckpt, last_path)

        improved = ev["loss"] < best_val
        if improved:
            best_val = ev["loss"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__, "stats": stats, "best_val": best_val}, best_path)

        with open(log_csv, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch,
                tr["loss"],
                ev["loss"],
                tr["main"],
                ev["main"],
                tr["temp"],
                ev["temp"],
                lr_now,
                tr["skipped"],
            ])

        skip_str = f" | skipped {tr['skipped']}" if tr["skipped"] > 0 else ""
        print(f"Epoch {epoch:03d} | train {tr['loss']:.6f}  val {ev['loss']:.6f} | lr {lr_now:.2e} | best {best_val:.6f} {'*' if improved else ''}{skip_str}")

    print(f"Best: {best_path}")


if __name__ == "__main__":
    main()
