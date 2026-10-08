"""Train Time-Conditioned U-Net with transport-parameter conditioning (manuscript benchmark).

Architecture  : UNet2D_TimeCond_MultiHead (in_ch=4: K + alpha_L + alpha_T/alpha_L + phi_t)
Training loop : per-timestep sampling with mid-bias (matches timecond_baseline)
Loss          : weighted Huber + mass-head auxiliary + mass-conservation penalty
Protocol      : fairness rules , AdamW lr=1e-4, CosineAnnealingLR, 200 epochs
AMP           : disabled by default (matches baseline timecond convention)
Seeds         : run once per seed via --seed argument
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.transport_surrogates.baselines.unet2d_timecond_multihead import UNet2D_TimeCond_MultiHead
from src.shared.data.io import flatten_param_folders, load_param_split, load_train_stats
from src.shared.data.transport_conditioning import (
    GroundwaterPatchDatasetMultiOutTransportConditioned,
    compute_conditioning_stats,
    load_transport_metadata,
    parse_condition_params,
)
from src.shared.utils.seed import set_seed

_N_TIMESTEPS = 25


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class CFG:
    data_root:  str = "./data/T25_TSTEP_OVERRIDE_FINAL"
    split_json: str = "splits/param_split_fixed.json"
    stats_json: str = "configs/train_stats.json"
    out_root:   str = "runs/timecond_transport_conditioned"
    run_name:   str = ""
    seed:       int = 0

    patch_size:   int   = 320
    batch_size:   int   = 16
    epochs:       int   = 200
    lr:           float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    num_workers:  int   = 0

    amp:      bool  = True
    use_logK: bool  = True
    eps_k:    float = 1e-6
    eps_c:    float = 1e-12

    # Time sampling (mid-bias toward middle timesteps)
    mid_bias_sigma: float = 0.18

    # Loss
    alpha:             float = 3.0
    y_bg:              float = -12.0
    y_cap:             float = -2.0
    huber_delta:       float = 1.0
    lambda_mass_head:  float = 0.05
    lambda_mass_cons:  float = 0.001

    # Architecture
    base_channels:   int = 64
    mass_mlp_hidden: int = 256

    resume: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train Time-Conditioned U-Net with transport conditioning (manuscript)."
    )
    ap.add_argument("--data_root",  type=str,   default=CFG.data_root)
    ap.add_argument("--split_json", type=str,   default=CFG.split_json)
    ap.add_argument("--stats_json", type=str,   default=CFG.stats_json)
    ap.add_argument("--out_root",   type=str,
                    default="runs/timecond_transport_conditioned")
    ap.add_argument("--run_name",   type=str,   default="")
    ap.add_argument("--seed",       type=int,   default=0)

    ap.add_argument("--patch_size",    type=int,   default=CFG.patch_size)
    ap.add_argument("--batch_size",    type=int,   default=CFG.batch_size)
    ap.add_argument("--epochs",        type=int,   default=CFG.epochs)
    ap.add_argument("--lr",            type=float, default=CFG.lr)
    ap.add_argument("--weight_decay",  type=float, default=CFG.weight_decay)
    ap.add_argument("--grad_clip_norm", type=float, default=CFG.grad_clip_norm)
    ap.add_argument("--num_workers",   type=int,   default=CFG.num_workers)
    ap.add_argument("--amp",     action="store_true", default=True)
    ap.add_argument("--no_amp",  action="store_true")
    ap.add_argument("--use_logK", dest="use_logK", action="store_true", default=True)
    ap.add_argument("--no_logK", dest="use_logK", action="store_false")
    ap.add_argument("--eps_k",   type=float, default=CFG.eps_k)
    ap.add_argument("--eps_c",   type=float, default=CFG.eps_c)

    ap.add_argument("--mid_bias_sigma",    type=float, default=CFG.mid_bias_sigma)
    ap.add_argument("--alpha",             type=float, default=CFG.alpha)
    ap.add_argument("--lambda_mass_head",  type=float, default=CFG.lambda_mass_head)
    ap.add_argument("--lambda_mass_cons",  type=float, default=CFG.lambda_mass_cons)

    ap.add_argument("--base_channels",   type=int, default=CFG.base_channels)
    ap.add_argument("--mass_mlp_hidden", type=int, default=CFG.mass_mlp_hidden)

    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--condition_params", type=str, default="alpha_L,alpha_T_ratio")
    ap.add_argument("--transport_metadata_csv", type=str, default="")
    ap.add_argument("--alpha_L_transform", type=str, default="log10",
                    choices=["identity", "log10"])
    ap.add_argument("--alpha_T_ratio_transform", type=str, default="log10",
                    choices=["identity", "log10"])
    return ap.parse_args()


def args_to_cfg(args: argparse.Namespace) -> CFG:
    cfg = CFG()
    for field_name in cfg.__dataclass_fields__:
        if hasattr(args, field_name):
            setattr(cfg, field_name, getattr(args, field_name))
    cfg.resume = args.resume
    cfg.amp = True
    if args.no_amp:
        cfg.amp = False
    elif args.amp:
        cfg.amp = True
    return cfg


# ----------------------------------------------------------------------
# Loss helpers
# ----------------------------------------------------------------------

def make_pixel_weight(
    Y_log: torch.Tensor, alpha: float, y_bg: float, y_cap: float
) -> torch.Tensor:
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0.0 else 1.0
    s = (Y_log - y_bg) / denom
    s = torch.clamp(s, 0.0, 1.0)
    return 1.0 + alpha * s


def weighted_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    hub = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    return (weight * hub).mean()


def log10_to_physC(Y_log: torch.Tensor, eps_c: float) -> torch.Tensor:
    C = (10.0 ** Y_log) - eps_c
    return torch.clamp(C, min=0.0)


_TIMESTEP_RNG = np.random.default_rng(0)


def seed_timestep_sampler(seed: int) -> None:
    """Seed the timestep sampler from the run seed (historical runs used an unseeded generator)."""
    global _TIMESTEP_RNG
    _TIMESTEP_RNG = np.random.default_rng(seed)


def sample_timestep_mid_bias(batch_size: int, sigma: float) -> int:
    """Sample a single timestep index with mid-bias (Gaussian centred at 0.5)."""
    rng = _TIMESTEP_RNG
    frac = rng.normal(0.5, sigma)
    frac = float(np.clip(frac, 0.0, 1.0))
    return int(round(frac * (_N_TIMESTEPS - 1)))


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------

def train_one_epoch(
    model: UNet2D_TimeCond_MultiHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: CFG,
) -> dict:
    model.train()
    s_total = s_main = s_mass_h = 0.0
    n = n_skip = 0

    for batch in loader:
        # K: (B, 3, H, W)  [K_norm + alpha_L_norm + alpha_T_ratio_norm]
        # C_log: (B, 25, H, W)
        K_3ch = batch["K"].to(cfg.device, non_blocking=True)
        C_log = batch["C_log"].to(cfg.device, non_blocking=True)

        B, _, H, W = K_3ch.shape

        # Sample a random timestep index with mid-bias
        t = sample_timestep_mid_bias(B, cfg.mid_bias_sigma)
        phi_t = torch.full((B, 1, H, W), t / (_N_TIMESTEPS - 1.0),
                           dtype=K_3ch.dtype, device=cfg.device)

        # Build 4-channel input: [K_norm, alpha_L_norm, alpha_T_ratio_norm, phi_t]
        X = torch.cat([K_3ch, phi_t], dim=1)                     # (B, 4, H, W)
        Y = C_log[:, t, :, :].unsqueeze(1).float()               # (B, 1, H, W)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred_map_raw, pred_mass_raw = model(X)

        pred_map  = pred_map_raw.float()
        pred_map  = torch.nan_to_num(pred_map, nan=0.0, posinf=2.0, neginf=-12.0)
        pred_mass = pred_mass_raw.float()
        Y_f = Y.float()

        w         = make_pixel_weight(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        loss_main = weighted_huber(pred_map, Y_f, w, delta=cfg.huber_delta)

        C_pred    = log10_to_physC(pred_map, cfg.eps_c)
        C_true    = log10_to_physC(Y_f,      cfg.eps_c)
        mass_true = C_true.sum(dim=(-1, -2), keepdim=False).unsqueeze(1)
        mass_pred = C_pred.sum(dim=(-1, -2), keepdim=False).unsqueeze(1)

        loss_mass_head = F.l1_loss(pred_mass, mass_true)
        loss_mass_cons = F.l1_loss(mass_pred, mass_true)

        loss = (
            loss_main
            + cfg.lambda_mass_head * loss_mass_head
            + cfg.lambda_mass_cons * loss_mass_cons
        )

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.update()
            n_skip += 1
            print(f"[WARN] Non-finite loss. Skipping batch.")
            continue

        optimizer.zero_grad(set_to_none=True)
        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False
            )
            if not torch.isfinite(gn):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                n_skip += 1
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False
            )
            if not torch.isfinite(gn):
                optimizer.zero_grad(set_to_none=True)
                n_skip += 1
                continue
            optimizer.step()

        s_total  += float(loss.item())
        s_main   += float(loss_main.item())
        s_mass_h += float(loss_mass_head.item())
        n += 1

    return {
        "loss":      s_total  / max(n, 1),
        "main":      s_main   / max(n, 1),
        "mass_head": s_mass_h / max(n, 1),
        "skipped":   n_skip,
    }


@torch.no_grad()
def eval_one_epoch(
    model: UNet2D_TimeCond_MultiHead,
    loader: DataLoader,
    cfg: CFG,
) -> dict:
    model.eval()
    s_total = s_main = s_mass_h = 0.0
    n = 0

    for batch in loader:
        K_3ch = batch["K"].to(cfg.device, non_blocking=True)
        C_log = batch["C_log"].to(cfg.device, non_blocking=True)

        B, _, H, W = K_3ch.shape

        # Evaluate uniformly across timesteps (unbiased validation)
        for t in range(_N_TIMESTEPS):
            phi_t = torch.full((B, 1, H, W), t / (_N_TIMESTEPS - 1.0),
                               dtype=K_3ch.dtype, device=cfg.device)
            X = torch.cat([K_3ch, phi_t], dim=1)
            Y = C_log[:, t, :, :].unsqueeze(1).float()

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                pred_map_raw, pred_mass_raw = model(X)

            pred_map  = pred_map_raw.float()
            pred_map  = torch.nan_to_num(pred_map, nan=0.0, posinf=2.0, neginf=-12.0)
            pred_mass = pred_mass_raw.float()
            Y_f = Y.float()

            w         = make_pixel_weight(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
            loss_main = weighted_huber(pred_map, Y_f, w, delta=cfg.huber_delta)

            C_true    = log10_to_physC(Y_f, cfg.eps_c)
            mass_true = C_true.sum(dim=(-1, -2), keepdim=False).unsqueeze(1)
            loss_mass_head = F.l1_loss(pred_mass, mass_true)
            loss = loss_main + cfg.lambda_mass_head * loss_mass_head

            if not torch.isfinite(loss):
                continue

            s_total  += float(loss.item())
            s_main   += float(loss_main.item())
            s_mass_h += float(loss_mass_head.item())
            n += 1

    return {
        "loss":      s_total  / max(n, 1),
        "main":      s_main   / max(n, 1),
        "mass_head": s_mass_h / max(n, 1),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def write_conditioning_metadata(
    out_dir: str,
    condition_params: list,
    transforms: dict,
    conditioning_stats: dict,
    metadata_source: str,
) -> None:
    path = os.path.join(out_dir, "transport_conditioning.json")
    payload = {
        "use_transport_conditioning": True,
        "condition_params": list(condition_params),
        "transforms": dict(transforms),
        "normalization": conditioning_stats,
        "normalization_split": "train parameter folders only",
        "metadata_source": metadata_source or "paper full Cartesian design order",
        "input_channels": 1 + len(condition_params) + 1,  # K + cond + phi_t
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    cfg = args_to_cfg(args)
    set_seed(cfg.seed)
    seed_timestep_sampler(cfg.seed)
    torch.backends.cudnn.benchmark = True

    condition_params = parse_condition_params(args.condition_params)
    if not condition_params:
        raise SystemExit("No condition params specified.")

    transforms = {
        "alpha_L": args.alpha_L_transform,
        "alpha_T_ratio": args.alpha_T_ratio_transform,
    }

    if not cfg.run_name:
        cfg.run_name = f"seed{cfg.seed}"

    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files = flatten_param_folders(cfg.data_root, train_params)
    val_files   = flatten_param_folders(cfg.data_root, val_params)
    stats = load_train_stats(cfg.stats_json)
    metadata = load_transport_metadata(args.transport_metadata_csv)
    conditioning_stats = compute_conditioning_stats(
        train_files, metadata, condition_params, transforms
    )

    n_cond = len(condition_params)
    in_ch_base = 1 + n_cond   # K + conditioning channels (phi_t added at batch time)
    in_ch_model = in_ch_base + 1  # +1 for phi_t

    cfg_dict = dict(cfg.__dict__)
    cfg_dict.update({
        "model_family": "timecond_transport_conditioned",
        "use_transport_conditioning": True,
        "condition_params": condition_params,
        "transport_metadata_csv": args.transport_metadata_csv,
        "transport_conditioning_transforms": transforms,
        "transport_conditioning_stats": conditioning_stats,
        "input_channels": in_ch_model,
        "n_timesteps": _N_TIMESTEPS,
    })
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg_dict, handle, indent=2)
    write_conditioning_metadata(
        out_dir, condition_params, transforms, conditioning_stats,
        args.transport_metadata_csv,
    )

    train_ds = GroundwaterPatchDatasetMultiOutTransportConditioned(
        train_files, stats,
        patch_size=cfg.patch_size,
        use_logK=cfg.use_logK,
        eps_k=cfg.eps_k,
        eps_c=cfg.eps_c,
        metadata=metadata,
        condition_params=condition_params,
        conditioning_stats=conditioning_stats,
    )
    val_ds = GroundwaterPatchDatasetMultiOutTransportConditioned(
        val_files, stats,
        patch_size=cfg.patch_size,
        use_logK=cfg.use_logK,
        eps_k=cfg.eps_k,
        eps_c=cfg.eps_c,
        metadata=metadata,
        condition_params=condition_params,
        conditioning_stats=conditioning_stats,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # in_ch=4: K_norm + alpha_L_norm + alpha_T_ratio_norm + phi_t
    model = UNet2D_TimeCond_MultiHead(
        in_ch=in_ch_model,
        base=cfg.base_channels,
        mass_mlp_hidden=cfg.mass_mlp_hidden,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = float("inf")
    start_epoch = 1
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    log_csv   = os.path.join(out_dir, "train_log.csv")

    if cfg.resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError(
                f"--resume set but no checkpoint at {last_path}"
            )
        ckpt = torch.load(last_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if cfg.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val    = float(ckpt.get("best_val", float("inf")))

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                "epoch", "train_loss", "val_loss",
                "train_main", "val_main",
                "train_mass_head", "val_mass_head",
                "lr", "skipped",
            ])

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "cfg":       cfg_dict,
            "stats":     stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict() if cfg.amp else None,
            "best_val":  best_val,
        }, last_path)

        # Checkpoint by val_main (reconstruction quality, not combined loss)
        if ev["main"] < best_val:
            best_val = ev["main"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg":   cfg_dict,
                "stats": stats,
                "best_val": best_val,
            }, best_path)

        skipped = tr.get("skipped", 0)
        with open(log_csv, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch,
                tr["loss"], ev["loss"],
                tr["main"], ev["main"],
                tr["mass_head"], ev["mass_head"],
                lr_now, skipped,
            ])

        print(
            f"Epoch {epoch:03d}"
            f" | train {tr['loss']:.6f}  val {ev['loss']:.6f}"
            f" | main(val) {ev['main']:.6f}"
            f" | lr {lr_now:.2e}"
            f" | best {best_val:.6f}"
        )

    print("Best:", best_path)


if __name__ == "__main__":
    main()
