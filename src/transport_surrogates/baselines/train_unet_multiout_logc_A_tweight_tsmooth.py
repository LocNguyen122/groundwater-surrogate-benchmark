import os, json, csv, argparse, math, random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.shared.utils.seed import set_seed
from src.shared.data.io import load_param_split, flatten_param_folders, load_train_stats
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
from src.transport_surrogates.baselines.unet2d import UNet2D


# ===========================================================================
# Loss helpers
# ===========================================================================


def make_weight_log_linear(y_log, alpha, y_bg, y_cap):
    """
    Per-pixel weight in [1, 1+alpha] based on log10-concentration magnitude.
    Background pixels (y ~ y_bg) get weight 1.0.
    Plume-core pixels (y ~ y_cap) get weight 1+alpha.
    This prevents the 97% background pixels from dominating gradients at
    early timesteps when the plume is small.
    """
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0 else 1.0
    s = (y_log - y_bg) / denom
    s = torch.clamp(s, 0.0, 1.0)
    return 1.0 + alpha * s


def timestep_weights(T: int, mode: str = "quad",
                     w_min: float = 0.5, w_max: float = 1.5) -> torch.Tensor:
    """
    Returns (T,) weights increasing toward late times.
    mode="quad": quadratic ramp , emphasises late-time more strongly.
    mode="linear": linear ramp.
    Default: w_min=0.5, w_max=1.5.
    Combined with alpha=3: max weight = (1+3)*1.5 = 6.0 (safe in fp16 max ~65504).
    """
    t = torch.linspace(0.0, 1.0, steps=T)
    s = t * t if mode == "quad" else t
    return w_min + (w_max - w_min) * s


def weighted_huber(pred, target, weight, delta=1.0):
    """Pixel-wise weighted SmoothL1 (Huber) loss."""
    hub = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    return (weight * hub).mean()


def log10_to_physC(y_log10: torch.Tensor, eps_c: float) -> torch.Tensor:
    return torch.clamp((10.0 ** y_log10) - eps_c, min=0.0)


# ===========================================================================
# Augmentation + scheduler
# ACADEMIC NOTE: use_flip and use_warmup default to False.
# These are engineering improvements, NOT architectural contributions.
# Apply them to ALL models or to NONE. Never only to A/B , that would
# make any improvement unattributable to the architecture.
# ===========================================================================

def joint_flip(K: torch.Tensor, Y: torch.Tensor):
    """
    Jointly flip K (B,1,P,P) and Y (B,T,P,P) along the same spatial axes.
    Physically valid: groundwater transport under isotropic K is symmetric
    under H/V reflection. MUST apply same flip to both K and Y , flipping
    only one corrupts the K->C mapping.
    """
    if random.random() < 0.5:
        K = torch.flip(K, dims=[-1])
        Y = torch.flip(Y, dims=[-1])
    if random.random() < 0.5:
        K = torch.flip(K, dims=[-2])
        Y = torch.flip(Y, dims=[-2])
    return K, Y


def warmup_cosine_scheduler(optimizer, warmup_epochs: int,
                             total_epochs: int) -> LambdaLR:
    """
    Linear warmup 0->lr over warmup_epochs, then cosine decay to 0.
    Drop-in replacement for CosineAnnealingLR.
    """
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class CFG:
    # Paths
    data_root:  str = "./data/T25_TSTEP_OVERRIDE_FINAL"
    split_json: str = "splits/param_split_fixed.json"
    stats_json: str = "configs/train_stats.json"

    # Run
    out_root: str = "runs"
    run_name: str = ""
    seed:     int = 0

    # Training
    patch_size:     int   = 320
    batch_size:     int   = 16
    epochs:         int   = 200
    lr:             float = 1e-4
    weight_decay:   float = 0.0
    grad_clip_norm: float = 1.0
    num_workers:    int   = 0
    amp:            bool  = True
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu"

    # Input normalisation
    use_logK: bool  = True
    eps_k:    float = 1e-6
    eps_c:    float = 1e-12

    # Architecture
    base_channels: int = 64

    # Loss: spatial pixel weighting (Option A contribution)
    # alpha=3.0 -> plume pixels up to 4x; combined with t_w_max=1.5 -> max 6x (fp16 safe)
    alpha:       float = 3.0
    y_bg:        float = -12.0
    y_cap:       float = -2.0
    huber_delta: float = 1.0

    # Prediction clamping (log10-space)
    pred_clamp_min: float = -12.0
    pred_clamp_max: float =   2.0

    # Loss: timestep weighting (Option A contribution)
    t_weight_mode: str   = "quad"
    t_w_min:       float = 0.5
    t_w_max:       float = 1.5

    # Loss: 2nd-order temporal smoothness (Option A contribution)
    lambda_temp: float = 0.05
    lambda_mass: float = 0.0

    # Engineering options (must be equal across ALL models for fair comparison)
    use_flip:      bool = False
    use_warmup:    bool = False
    warmup_epochs: int  = 10


# ===========================================================================
# Argument parsing
# IMPORTANT: Every ap.add_argument default matches its CFG field.
# Bug in original: --alpha default=20.0 silently overrode CFG.alpha=3.0,
# and --t_w_max default=2.0 overrode CFG.t_w_max=1.5, every run.
# ===========================================================================

def parse_args() -> CFG:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root",  type=str, default=CFG.data_root)
    ap.add_argument("--split_json", type=str, default=CFG.split_json)
    ap.add_argument("--stats_json", type=str, default=CFG.stats_json)
    ap.add_argument("--out_root",   type=str, default=CFG.out_root)
    ap.add_argument("--run_name",   type=str, default="")
    ap.add_argument("--seed",       type=int, default=0)

    ap.add_argument("--patch_size",     type=int,   default=CFG.patch_size)
    ap.add_argument("--batch_size",     type=int,   default=CFG.batch_size)
    ap.add_argument("--epochs",         type=int,   default=CFG.epochs)
    ap.add_argument("--lr",             type=float, default=CFG.lr)
    ap.add_argument("--weight_decay",   type=float, default=CFG.weight_decay)
    ap.add_argument("--grad_clip_norm", type=float, default=CFG.grad_clip_norm)
    ap.add_argument("--num_workers",    type=int,   default=CFG.num_workers)
    ap.add_argument("--amp",    action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--base_channels", type=int, default=CFG.base_channels)

    ap.add_argument("--use_logK", dest="use_logK", action="store_true", default=True)

    ap.add_argument("--no_logK", dest="use_logK", action="store_false")
    ap.add_argument("--eps_k", type=float, default=CFG.eps_k)
    ap.add_argument("--eps_c", type=float, default=CFG.eps_c)

    ap.add_argument("--alpha",       type=float, default=CFG.alpha)       # was 20.0 , corrected
    ap.add_argument("--y_bg",        type=float, default=CFG.y_bg)
    ap.add_argument("--y_cap",       type=float, default=CFG.y_cap)
    ap.add_argument("--huber_delta", type=float, default=CFG.huber_delta)

    ap.add_argument("--pred_clamp_min", type=float, default=CFG.pred_clamp_min)
    ap.add_argument("--pred_clamp_max", type=float, default=CFG.pred_clamp_max)

    ap.add_argument("--t_weight_mode", type=str,   default=CFG.t_weight_mode,
                    choices=["linear", "quad"])
    ap.add_argument("--t_w_min",       type=float, default=CFG.t_w_min)
    ap.add_argument("--t_w_max",       type=float, default=CFG.t_w_max)   # was 2.0 , corrected

    ap.add_argument("--lambda_temp", type=float, default=CFG.lambda_temp)
    ap.add_argument("--lambda_mass", type=float, default=CFG.lambda_mass)

    ap.add_argument("--use_flip",      action="store_true")
    ap.add_argument("--use_warmup",    action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=CFG.warmup_epochs)

    args = ap.parse_args()

    cfg = CFG()
    for field in cfg.__dataclass_fields__:
        if hasattr(args, field):
            setattr(cfg, field, getattr(args, field))

    # AMP: --amp/--no_amp are flags, not stored as CFG fields
    cfg.amp = True
    if args.no_amp:
        cfg.amp = False
    elif args.amp:
        cfg.amp = True

    return cfg


# ===========================================================================
# Training loop
# ===========================================================================

def train_one_epoch(model, loader, optimizer, scaler, cfg: CFG):
    model.train()
    tloss = tmain = ttemp = tmass = 0.0
    n = 0
    n_skipped = 0

    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)      # (B, 1, P, P)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)  # (B, 25, P, P)

        # Augmentation: train only, never eval
        if cfg.use_flip:
            K, Y = joint_flip(K, Y)

        # Forward in fp16 (speed), loss in fp32 (stability)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(K)  # (B, 25, P, P) fp16 under AMP

        pred = pred.float()
        Y_f  = Y.float()

        # nan_to_num: torch.clamp does NOT fix NaN.
        # GroupNorm can produce NaN with zero-variance groups (random init + AMP).
        # Replace NaN with 0.0 (log10-space background) for a real gradient signal.
        pred  = torch.nan_to_num(pred, nan=0.0,
                                 posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
        predL = torch.clamp(pred, cfg.pred_clamp_min, cfg.pred_clamp_max)

        # Spatial pixel weight: plume pixels up to (1+alpha)x over background
        w_y = make_weight_log_linear(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        T   = Y_f.shape[1]
        w_t = timestep_weights(T, cfg.t_weight_mode,
                               cfg.t_w_min, cfg.t_w_max).to(Y_f.device).view(1, T, 1, 1)
        w   = w_y * w_t

        loss_main = weighted_huber(predL, Y_f, w, delta=cfg.huber_delta)

        # 2nd-order temporal smoothness: penalises curvature, not plume growth.
        # 1st-order difference penalises growth which is physically wrong.
        d1 = predL[:, 1:] - predL[:, :-1]   # (B, T-1, P, P)
        d2 = d1[:, 1:] - d1[:, :-1]         # (B, T-2, P, P)
        loss_temp = d2.abs().mean()

        loss_mass = torch.tensor(0.0, device=cfg.device)
        if cfg.lambda_mass > 0.0:
            Cp = log10_to_physC(predL, cfg.eps_c)
            Ct = log10_to_physC(
                torch.clamp(Y_f, cfg.pred_clamp_min, cfg.pred_clamp_max), cfg.eps_c)
            loss_mass = F.l1_loss(Cp.sum(dim=(-1, -2)), Ct.sum(dim=(-1, -2)))

        loss = loss_main + cfg.lambda_temp * loss_temp + cfg.lambda_mass * loss_mass

        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            if not torch.isfinite(gnorm):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            if not torch.isfinite(gnorm):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        tloss += float(loss.item())
        tmain += float(loss_main.item())
        ttemp += float(loss_temp.item())
        tmass += float(loss_mass.item()) if cfg.lambda_mass > 0.0 else 0.0
        n += 1

    if n_skipped > 0:
        print(f"  [WARNING] {n_skipped}/{n + n_skipped} batches skipped. "
              f"Epoch-1 spikes are normal. If persistent past epoch 5, "
              f"reduce --alpha (now {cfg.alpha}) or --t_w_max (now {cfg.t_w_max}).")

    return {
        "loss": tloss / max(n, 1), "main": tmain / max(n, 1),
        "temp": ttemp / max(n, 1), "mass": tmass / max(n, 1),
        "skipped": n_skipped,
    }


# ===========================================================================
# Evaluation loop
# ===========================================================================

@torch.no_grad()
def eval_one_epoch(model, loader, cfg: CFG):
    model.eval()
    tloss = tmain = ttemp = tmass = 0.0
    n = 0

    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)

        pred  = model(K)
        pred  = torch.nan_to_num(pred.float(), nan=0.0,
                                 posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
        Y_f   = Y.float()
        predL = torch.clamp(pred, cfg.pred_clamp_min, cfg.pred_clamp_max)

        w_y = make_weight_log_linear(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        T   = Y_f.shape[1]
        w_t = timestep_weights(T, cfg.t_weight_mode,
                               cfg.t_w_min, cfg.t_w_max).to(Y_f.device).view(1, T, 1, 1)
        w   = w_y * w_t

        loss_main = weighted_huber(predL, Y_f, w, delta=cfg.huber_delta)
        d1 = predL[:, 1:] - predL[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        loss_temp = d2.abs().mean()

        loss_mass = torch.tensor(0.0, device=cfg.device)
        if cfg.lambda_mass > 0.0:
            Cp = log10_to_physC(predL, cfg.eps_c)
            Ct = log10_to_physC(
                torch.clamp(Y_f, cfg.pred_clamp_min, cfg.pred_clamp_max), cfg.eps_c)
            loss_mass = F.l1_loss(Cp.sum(dim=(-1, -2)), Ct.sum(dim=(-1, -2)))

        loss = loss_main + cfg.lambda_temp * loss_temp + cfg.lambda_mass * loss_mass
        if not torch.isfinite(loss):
            continue

        tloss += float(loss.item())
        tmain += float(loss_main.item())
        ttemp += float(loss_temp.item())
        tmass += float(loss_mass.item()) if cfg.lambda_mass > 0.0 else 0.0
        n += 1

    return {
        "loss": tloss / max(n, 1), "main": tmain / max(n, 1),
        "temp": ttemp / max(n, 1), "mass": tmass / max(n, 1),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # makes the real error visible

    if not cfg.run_name:
        cfg.run_name = (
            f"transport_A_unet_multiout_logc_p{cfg.patch_size}"
            f"_tw{cfg.t_weight_mode}{cfg.t_w_min:g}-{cfg.t_w_max:g}"
            f"_a{cfg.alpha:g}_ltemp{cfg.lambda_temp:g}"
            f"_lr{cfg.lr:g}_seed{cfg.seed}"
        )

    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files = flatten_param_folders(cfg.data_root, train_params)
    val_files   = flatten_param_folders(cfg.data_root, val_params)
    stats       = load_train_stats(cfg.stats_json)

    train_ds = GroundwaterPatchDatasetMultiOut(
        train_files, stats, patch_size=cfg.patch_size,
        use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c)
    val_ds = GroundwaterPatchDatasetMultiOut(
        val_files, stats, patch_size=cfg.patch_size,
        use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model     = UNet2D(in_ch=1, out_ch=25, base=cfg.base_channels).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    if cfg.use_warmup:
        scheduler = warmup_cosine_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val  = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    log_csv   = os.path.join(out_dir, "train_log.csv")

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_main", "val_main", "train_temp", "val_temp",
                "lr", "skipped",
            ])

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "cfg": cfg.__dict__, "stats": stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if cfg.amp else None,
            "best_val": best_val,
        }, last_path)

        improved = ev["loss"] < best_val
        if improved:
            best_val = ev["loss"]
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "cfg": cfg.__dict__, "stats": stats, "best_val": best_val,
            }, best_path)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, tr["loss"], ev["loss"],
                tr["main"], ev["main"], tr["temp"], ev["temp"],
                lr_now, tr["skipped"],
            ])

        skip_str = f" | skipped {tr['skipped']}" if tr["skipped"] > 0 else ""
        print(f"Epoch {epoch:03d} | train {tr['loss']:.6f}  val {ev['loss']:.6f}"
              f" | lr {lr_now:.2e} | best {best_val:.6f} {'*' if improved else ''}{skip_str}")

    print("Best:", best_path)


if __name__ == "__main__":
    main()


# ===========================================================================
# Run commands
# ===========================================================================
# Standard (engineering options OFF , default, for fair comparison with baseline):
# python -m src.train.train_unet_multiout_logc_A_tweight_tsmooth ^
#   --seed 0 --patch_size 320 --batch_size 16 --epochs 200 ^
#   --lr 0.0001 --amp --use_logK ^
#   --alpha 3.0 --t_weight_mode quad --t_w_min 0.5 --t_w_max 1.5 ^
#   --lambda_temp 0.05
#
# With engineering options (ONLY if also applied to baseline):
# python -m src.train.train_unet_multiout_logc_A_tweight_tsmooth ^
#   --seed 0 --patch_size 320 --batch_size 16 --epochs 200 ^
#   --lr 0.0001 --amp --use_logK ^
#   --alpha 3.0 --t_weight_mode quad --t_w_min 0.5 --t_w_max 1.5 ^
#   --lambda_temp 0.05 --use_flip --use_warmup --warmup_epochs 10
