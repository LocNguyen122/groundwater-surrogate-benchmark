import os
import csv
import json
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.shared.utils.seed import set_seed
from src.shared.data.io import load_param_split, flatten_param_folders, load_train_stats
from src.transport_surrogates.baselines.unet2d_timecond_multihead import UNet2D_TimeCond_MultiHead

# Dataset returns:
#   X:     (B, 2, P, P)  , channel 0 = normalised log-K,
#                          channel 1 = time encoding (linear or sinusoidal)
#   Y_log: (B, 1, P, P)  , log10(C + eps) for the sampled timestep
from src.shared.data.dataset_timecond_patch import GroundwaterTimeCondPatchDatasetLogC


# ----------------------------------------------------------------------
# Loss helpers
# ----------------------------------------------------------------------

def make_pixel_weight(Y_log: torch.Tensor, alpha: float, y_bg: float, y_cap: float) -> torch.Tensor:
    """
    Concentration-aware pixel weight in [1, 1+alpha].
    Identical formula to Option A / B / Pix2Pix for fair comparison.
    alpha=3.0 (was 20.0 , caused NaN under AMP and unfair comparison).
    """
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0.0 else 1.0
    s = (Y_log - y_bg) / denom
    s = torch.clamp(s, 0.0, 1.0)
    return 1.0 + alpha * s


def weighted_huber(pred: torch.Tensor, target: torch.Tensor,
                   weight: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    hub = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    return (weight * hub).mean()


def log10_to_physC(Y_log: torch.Tensor, eps_c: float) -> torch.Tensor:
    C = (10.0 ** Y_log) - eps_c
    return torch.clamp(C, min=0.0)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class CFG:
    data_root:  str = "./data/T25_TSTEP_OVERRIDE_FINAL"
    split_json: str = "splits/param_split_fixed.json"
    stats_json: str = "configs/train_stats.json"
    out_root:   str = "runs"
    run_name:   str = ""
    seed:       int = 0

    patch_size:   int   = 320
    batch_size:   int   = 16
    epochs:       int   = 200
    lr:           float = 1e-4
    weight_decay: float = 0.0
    num_workers:  int   = 0

    amp:            bool  = True
    grad_clip_norm: float = 1.0         # matches A/B/Pix2Pix

    # Normalisation
    use_logK: bool  = True
    eps_k:    float = 1e-6
    eps_c:    float = 1e-12

    # Time sampling
    t_index_mode:   str   = "mid_bias"  # training: biased toward mid/late timesteps
    mid_bias_sigma: float = 0.18

    # Loss
    # alpha=3.0 to match A/B (was 20.0 , unfair comparison + NaN risk)
    alpha:     float = 3.0
    y_bg:      float = -12.0
    y_cap:     float = -2.0
    huber_delta: float = 1.0

    lambda_mass_head: float = 0.05     # auxiliary mass-head weight
    lambda_mass_cons: float = 0.001    # was hardcoded , now configurable

    # Model
    base_channels:   int = 64
    mass_mlp_hidden: int = 256

    # Resume interrupted training
    resume: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> CFG:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--run_name",   type=str,   default="")
    ap.add_argument("--out_root",   type=str,   default="runs")          # configuration note
    ap.add_argument("--patch_size", type=int,   default=320)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--epochs",     type=int,   default=200)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--amp",        action="store_true", default=True)
    ap.add_argument("--no_amp",     action="store_true")
    ap.add_argument("--use_logK",   action="store_true", default=True)   # configuration note
    ap.add_argument("--no_logK",    action="store_true")

    # alpha in argparse with correct default=3.0
    ap.add_argument("--alpha",      type=float, default=CFG.alpha,
                    help="Pixel weight scale. Must match A/B/Pix2Pix for fair comparison. Default=3.0")
    ap.add_argument("--y_bg",       type=float, default=CFG.y_bg)
    ap.add_argument("--y_cap",      type=float, default=CFG.y_cap)
    ap.add_argument("--mid_bias_sigma", type=float, default=CFG.mid_bias_sigma)

    ap.add_argument("--lambda_mass_head", type=float, default=CFG.lambda_mass_head)
    ap.add_argument("--lambda_mass_cons", type=float, default=CFG.lambda_mass_cons)  # FIX T6
    ap.add_argument("--grad_clip_norm",   type=float, default=CFG.grad_clip_norm)

    # Paths (allow override for non-standard setups)
    ap.add_argument("--data_root",  type=str,   default=CFG.data_root)
    ap.add_argument("--split_json", type=str,   default=CFG.split_json)
    ap.add_argument("--stats_json", type=str,   default=CFG.stats_json)

    # Architecture
    ap.add_argument("--base_channels",   type=int,   default=CFG.base_channels)
    ap.add_argument("--mass_mlp_hidden", type=int,   default=CFG.mass_mlp_hidden)

    # Loss / training details
    ap.add_argument("--weight_decay",  type=float, default=CFG.weight_decay)
    ap.add_argument("--num_workers",   type=int,   default=CFG.num_workers)
    ap.add_argument("--huber_delta",   type=float, default=CFG.huber_delta)
    ap.add_argument("--eps_k",         type=float, default=CFG.eps_k)
    ap.add_argument("--eps_c",         type=float, default=CFG.eps_c)
    ap.add_argument("--t_index_mode",  type=str,   default=CFG.t_index_mode,
                    choices=["uniform", "mid_bias"])

    # Resume
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last.pt in out_root/run_name/")

    args = ap.parse_args()
    cfg = CFG()
    cfg.seed            = args.seed
    cfg.run_name        = args.run_name
    cfg.out_root        = args.out_root
    cfg.patch_size      = args.patch_size
    cfg.batch_size      = args.batch_size
    cfg.epochs          = args.epochs
    cfg.lr              = args.lr
    cfg.alpha           = args.alpha
    cfg.y_bg            = args.y_bg
    cfg.y_cap           = args.y_cap
    cfg.mid_bias_sigma  = args.mid_bias_sigma
    cfg.lambda_mass_head = args.lambda_mass_head
    cfg.lambda_mass_cons = args.lambda_mass_cons
    cfg.grad_clip_norm   = args.grad_clip_norm

    # Wire new args
    cfg.data_root        = args.data_root
    cfg.split_json       = args.split_json
    cfg.stats_json       = args.stats_json
    cfg.base_channels    = args.base_channels
    cfg.mass_mlp_hidden  = args.mass_mlp_hidden
    cfg.weight_decay     = args.weight_decay
    cfg.num_workers      = args.num_workers
    cfg.huber_delta      = args.huber_delta
    cfg.eps_k            = args.eps_k
    cfg.eps_c            = args.eps_c
    cfg.t_index_mode     = args.t_index_mode
    cfg.resume           = args.resume

    if args.no_amp:   cfg.amp = False
    elif args.amp:    cfg.amp = True
    if args.no_logK:  cfg.use_logK = False

    return cfg


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, cfg: CFG) -> dict:
    model.train()
    s_total = s_main = s_mass = 0.0
    n = n_skip = 0

    for batch in loader:
        X = batch["X"].to(cfg.device, non_blocking=True)      # (B,2,P,P)
        Y = batch["Y_log"].to(cfg.device, non_blocking=True)  # (B,1,P,P)

        # -- Forward (fp16) ----------------------------------------------------
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred_map_raw, pred_mass_raw = model(X)

        # FIX T1 + T4: nan_to_num and fp32 cast OUTSIDE autocast
        pred_map  = pred_map_raw.float()
        pred_map  = torch.nan_to_num(pred_map, nan=0.0, posinf=2.0, neginf=-12.0)
        pred_mass = pred_mass_raw.float()

        Y_f = Y.float()

        # -- Loss (fp32 throughout) ---------------------------------------------
        w          = make_pixel_weight(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        loss_main  = weighted_huber(pred_map, Y_f, w, delta=cfg.huber_delta)

        C_pred     = log10_to_physC(pred_map, cfg.eps_c)
        C_true     = log10_to_physC(Y_f,      cfg.eps_c)
        mass_true  = C_true.sum(dim=(-1, -2), keepdim=False).unsqueeze(1)   # (B,1)
        mass_pred  = C_pred.sum(dim=(-1, -2), keepdim=False).unsqueeze(1)

        loss_mass_head = F.l1_loss(pred_mass, mass_true)   # aux head regression
        loss_mass_cons = F.l1_loss(mass_pred, mass_true)   # pixel-level mass penalty

        loss = (loss_main
                + cfg.lambda_mass_head * loss_mass_head
                + cfg.lambda_mass_cons * loss_mass_cons)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.update()
            n_skip += 1
            print(f"[WARN] Non-finite loss ({loss.item():.4f}). Skipping batch.")
            continue

        # -- Backward + step ---------------------------------------------------
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
                print(f"[WARN] Non-finite grad norm. Skipping step.")
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

        s_total += float(loss.item())
        s_main  += float(loss_main.item())
        s_mass  += float(loss_mass_head.item())
        n += 1

    return {
        "loss":      s_total / max(n, 1),
        "main":      s_main  / max(n, 1),
        "mass_head": s_mass  / max(n, 1),
        "n_skip":    n_skip,
    }


@torch.no_grad()
def eval_one_epoch(model, loader, cfg: CFG) -> dict:
    model.eval()
    s_total = s_main = s_mass = 0.0
    n = 0

    for batch in loader:
        X = batch["X"].to(cfg.device, non_blocking=True)
        Y = batch["Y_log"].to(cfg.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred_map_raw, pred_mass_raw = model(X)

        # FIX T1 + T4: fp32 + nan_to_num outside autocast
        pred_map  = pred_map_raw.float()
        pred_map  = torch.nan_to_num(pred_map, nan=0.0, posinf=2.0, neginf=-12.0)
        pred_mass = pred_mass_raw.float()

        Y_f = Y.float()
        w   = make_pixel_weight(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)

        loss_main      = weighted_huber(pred_map, Y_f, w, delta=cfg.huber_delta)
        C_true         = log10_to_physC(Y_f, cfg.eps_c)
        mass_true      = C_true.sum(dim=(-1, -2), keepdim=False).unsqueeze(1)
        loss_mass_head = F.l1_loss(pred_mass, mass_true)
        loss           = loss_main + cfg.lambda_mass_head * loss_mass_head

        if not torch.isfinite(loss):
            continue

        s_total += float(loss.item())
        s_main  += float(loss_main.item())
        s_mass  += float(loss_mass_head.item())
        n += 1

    return {
        "loss":      s_total / max(n, 1),
        "main":      s_main  / max(n, 1),
        "mass_head": s_mass  / max(n, 1),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    if not cfg.run_name:
        cfg.run_name = (
            f"transport_timecond_multihead"
            f"_p{cfg.patch_size}"
            f"_a{cfg.alpha:g}"
            f"_sig{cfg.mid_bias_sigma:g}"
            f"_lmh{cfg.lambda_mass_head:g}"
            f"_lr{cfg.lr:g}"
            f"_seed{cfg.seed}"
        )

    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # -- Data -----------------------------------------------------------------
    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files = flatten_param_folders(cfg.data_root, train_params)
    val_files   = flatten_param_folders(cfg.data_root, val_params)
    stats       = load_train_stats(cfg.stats_json)

    train_ds = GroundwaterTimeCondPatchDatasetLogC(
        train_files, stats,
        patch_size=cfg.patch_size,
        use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c,
        seed=cfg.seed,
        t_index_mode=cfg.t_index_mode,
        mid_bias_sigma=cfg.mid_bias_sigma,
    )
    val_ds = GroundwaterTimeCondPatchDatasetLogC(
        val_files, stats,
        patch_size=cfg.patch_size,
        use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c,
        seed=cfg.seed + 999,
        t_index_mode="uniform",     # unbiased validation
        mid_bias_sigma=cfg.mid_bias_sigma,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    # -- Model -----------------------------------------------------------------
    model = UNet2D_TimeCond_MultiHead(
        in_ch=2, base=cfg.base_channels, mass_mlp_hidden=cfg.mass_mlp_hidden
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # removed CosineAnnealingLR , no LR scheduler matches A/B/Pix2Pix baseline
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val    = float("inf")
    start_epoch = 1
    best_path   = os.path.join(out_dir, "best.pt")
    last_path   = os.path.join(out_dir, "last.pt")
    log_csv     = os.path.join(out_dir, "train_log.csv")

    # -- Resume from last.pt --------------------------------------------------
    if cfg.resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError(
                f"--resume requested but no checkpoint found at {last_path}\n"
                f"Check that --run_name matches the interrupted run folder name."
            )
        ckpt = torch.load(last_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if cfg.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"[RESUME] Loaded {last_path}")
        print(f"[RESUME] Completed epoch {ckpt['epoch']}, best_val(main) {best_val:.6f}")
        print(f"[RESUME] Continuing from epoch {start_epoch} / {cfg.epochs}")
    # ----------------------------------------------------------------------

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_main", "val_main",
                "train_mass_head", "val_mass_head",
                "n_skip", "best_val",
            ])

    # -- Training loop ---------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "cfg":       cfg.__dict__,
            "stats":     stats,
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict() if cfg.amp else None,
            "best_val":  best_val,            # FIX: needed for correct resume
        }, last_path)

        # FIX T5: checkpoint by val_main (reconstruction quality only),
        #         NOT combined loss which includes auxiliary mass head.
        improved = ev["main"] < best_val
        if improved:
            best_val = ev["main"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg":   cfg.__dict__,
                "stats": stats,
            }, best_path)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                tr["loss"], ev["loss"],
                tr["main"], ev["main"],
                tr["mass_head"], ev["mass_head"],
                tr["n_skip"], best_val,
            ])

        print(
            f"Epoch {epoch:03d}"
            f" | train {tr['loss']:.6f} val {ev['loss']:.6f}"
            f" | main(val) {ev['main']:.6f}"
            f" | massH(val) {ev['mass_head']:.6f}"
            f" | skip={tr['n_skip']}"
            f" | best_main {best_val:.6f} {'*' if improved else ''}"
        )

    print("Best:", best_path)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# Run commands
# ----------------------------------------------------------------------
# Standard run (seeds 0/1/2):
#   python -m src.train.train_unet_timecond_multihead_logc ^
#     --seed 0 --patch_size 320 --batch_size 16 --epochs 200 ^
#     --lr 0.0001 --amp --use_logK --alpha 3.0 --lambda_mass_head 0.05
#
# Smoke test (1 epoch, CPU):
#   python -m src.train.train_unet_timecond_multihead_logc ^
#     --seed 0 --patch_size 320 --batch_size 4 --epochs 1 --no_amp
