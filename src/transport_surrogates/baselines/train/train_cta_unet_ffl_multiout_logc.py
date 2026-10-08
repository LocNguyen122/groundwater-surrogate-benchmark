"""
CTA-UNet + focal frequency loss for transport-conditioning benchmark.
"""

import os, json, csv, argparse, math, random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.shared.utils.seed import set_seed
from src.shared.data.io import load_param_split, flatten_param_folders, load_train_stats
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet
from src.transport_surrogates.baselines.focal_frequency_loss import FocalFrequencyLoss


def make_weight_log_linear(y_log, alpha, y_bg, y_cap):
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0 else 1.0
    s = (y_log - y_bg) / denom
    s = torch.clamp(s, 0.0, 1.0)
    return 1.0 + alpha * s


def timestep_weights(T: int, mode: str = "quad", w_min: float = 0.5, w_max: float = 1.5) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, steps=T)
    s = t * t if mode == "quad" else t
    return w_min + (w_max - w_min) * s


def weighted_huber(pred, target, weight, delta=1.0):
    hub = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    return (weight * hub).mean()


def log10_to_physC(y_log10: torch.Tensor, eps_c: float) -> torch.Tensor:
    return torch.clamp((10.0 ** y_log10) - eps_c, min=0.0)


def joint_flip(K: torch.Tensor, Y: torch.Tensor):
    if random.random() < 0.5:
        K = torch.flip(K, dims=[-1]); Y = torch.flip(Y, dims=[-1])
    if random.random() < 0.5:
        K = torch.flip(K, dims=[-2]); Y = torch.flip(Y, dims=[-2])
    return K, Y


def warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int) -> LambdaLR:
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


@dataclass
class CFG:
    data_root: str = "./data/T25_TSTEP_OVERRIDE_FINAL"
    split_json: str = "splits/param_split_fixed.json"
    stats_json: str = "configs/train_stats.json"
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
    model_family: str = "cta_unet"
    loss_variant: str = "focal_frequency"
    base_channels: int = 64
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
    lambda_ffl: float = 0.01
    ffl_alpha: float = 1.0
    lambda_mass: float = 0.0
    use_flip: bool = False
    use_warmup: bool = False
    warmup_epochs: int = 10
    resume: bool = False


def parse_args() -> CFG:
    ap = argparse.ArgumentParser()
    for name, default in [
        ("data_root", CFG.data_root), ("split_json", CFG.split_json), ("stats_json", CFG.stats_json),
        ("out_root", CFG.out_root), ("run_name", ""), ("seed", 0), ("patch_size", CFG.patch_size),
        ("batch_size", CFG.batch_size), ("epochs", CFG.epochs), ("lr", CFG.lr),
        ("weight_decay", CFG.weight_decay), ("grad_clip_norm", CFG.grad_clip_norm), ("num_workers", CFG.num_workers),
        ("base_channels", CFG.base_channels), ("t_embed_dim", CFG.t_embed_dim), ("t_heads", CFG.t_heads),
        ("pool_stride", CFG.pool_stride), ("eps_k", CFG.eps_k), ("eps_c", CFG.eps_c),
        ("alpha", CFG.alpha), ("y_bg", CFG.y_bg), ("y_cap", CFG.y_cap), ("huber_delta", CFG.huber_delta),
        ("pred_clamp_min", CFG.pred_clamp_min), ("pred_clamp_max", CFG.pred_clamp_max),
        ("t_w_min", CFG.t_w_min), ("t_w_max", CFG.t_w_max), ("lambda_temp", CFG.lambda_temp),
        ("lambda_ffl", CFG.lambda_ffl), ("ffl_alpha", CFG.ffl_alpha), ("lambda_mass", CFG.lambda_mass),
        ("warmup_epochs", CFG.warmup_epochs),
    ]:
        ap.add_argument(f"--{name}", type=type(default), default=default)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--use_logK", dest="use_logK", action="store_true", default=True)
    ap.add_argument("--no_logK", dest="use_logK", action="store_false")
    ap.add_argument("--t_weight_mode", type=str, default=CFG.t_weight_mode, choices=["linear", "quad"])
    ap.add_argument("--use_flip", action="store_true")
    ap.add_argument("--use_warmup", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = CFG()
    for field in cfg.__dataclass_fields__:
        if hasattr(args, field):
            setattr(cfg, field, getattr(args, field))
    cfg.amp = not args.no_amp
    return cfg


def train_one_epoch(model, loader, optimizer, scaler, ffl_loss: FocalFrequencyLoss, cfg: CFG):
    model.train()
    tloss = tmain = ttemp = tffl = tmass = 0.0
    n = n_skipped = 0
    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True); Y = batch["C_log"].to(cfg.device, non_blocking=True)
        if cfg.use_flip:
            K, Y = joint_flip(K, Y)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(K)
        pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
        Y_f = Y.float(); predL = torch.clamp(pred, cfg.pred_clamp_min, cfg.pred_clamp_max)
        w_y = make_weight_log_linear(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        timesteps = Y_f.shape[1]
        w_t = timestep_weights(timesteps, cfg.t_weight_mode, cfg.t_w_min, cfg.t_w_max).to(Y_f.device).view(1, timesteps, 1, 1)
        loss_main = weighted_huber(predL, Y_f, w_y * w_t, delta=cfg.huber_delta)
        d1 = predL[:, 1:] - predL[:, :-1]; d2 = d1[:, 1:] - d1[:, :-1]; loss_temp = d2.abs().mean()
        loss_ffl = ffl_loss(predL, Y_f)
        loss_mass = torch.tensor(0.0, device=cfg.device)
        if cfg.lambda_mass > 0.0:
            pred_phys = torch.clamp(10.0 ** predL - cfg.eps_c, min=0.0)
            true_phys = torch.clamp(10.0 ** Y_f - cfg.eps_c, min=0.0)
            mass_pred = pred_phys.sum(dim=(-2, -1))
            mass_true = true_phys.sum(dim=(-2, -1))
            loss_mass = (torch.abs(mass_pred - mass_true) / (mass_true + 1.0)).mean()
        loss = loss_main + cfg.lambda_temp * loss_temp + cfg.lambda_ffl * loss_ffl + cfg.lambda_mass * loss_mass
        if not torch.isfinite(loss):
            n_skipped += 1; optimizer.zero_grad(set_to_none=True); continue
        optimizer.zero_grad(set_to_none=True)
        if cfg.amp:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            if not torch.isfinite(gnorm):
                n_skipped += 1; optimizer.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            if not torch.isfinite(gnorm):
                n_skipped += 1; optimizer.zero_grad(set_to_none=True); continue
            optimizer.step()
        tloss += float(loss.item()); tmain += float(loss_main.item()); ttemp += float(loss_temp.item()); tffl += float(loss_ffl.item()); tmass += float(loss_mass.item()) if cfg.lambda_mass > 0.0 else 0.0; n += 1
    return {"loss": tloss / max(n, 1), "main": tmain / max(n, 1), "temp": ttemp / max(n, 1), "ffl": tffl / max(n, 1), "mass": tmass / max(n, 1), "skipped": n_skipped}


@torch.no_grad()
def eval_one_epoch(model, loader, ffl_loss: FocalFrequencyLoss, cfg: CFG):
    model.eval()
    tloss = tmain = ttemp = tffl = tmass = 0.0
    n = 0
    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True); Y = batch["C_log"].to(cfg.device, non_blocking=True)
        pred = model(K)
        pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
        Y_f = Y.float(); predL = torch.clamp(pred, cfg.pred_clamp_min, cfg.pred_clamp_max)
        w_y = make_weight_log_linear(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        timesteps = Y_f.shape[1]
        w_t = timestep_weights(timesteps, cfg.t_weight_mode, cfg.t_w_min, cfg.t_w_max).to(Y_f.device).view(1, timesteps, 1, 1)
        loss_main = weighted_huber(predL, Y_f, w_y * w_t, delta=cfg.huber_delta)
        d1 = predL[:, 1:] - predL[:, :-1]; d2 = d1[:, 1:] - d1[:, :-1]; loss_temp = d2.abs().mean()
        loss_ffl = ffl_loss(predL, Y_f)
        loss_mass = torch.tensor(0.0, device=cfg.device)
        if cfg.lambda_mass > 0.0:
            pred_phys = torch.clamp(10.0 ** predL - cfg.eps_c, min=0.0)
            true_phys = torch.clamp(10.0 ** Y_f - cfg.eps_c, min=0.0)
            mass_pred = pred_phys.sum(dim=(-2, -1))
            mass_true = true_phys.sum(dim=(-2, -1))
            loss_mass = (torch.abs(mass_pred - mass_true) / (mass_true + 1.0)).mean()
        loss = loss_main + cfg.lambda_temp * loss_temp + cfg.lambda_ffl * loss_ffl + cfg.lambda_mass * loss_mass
        if not torch.isfinite(loss):
            continue
        tloss += float(loss.item()); tmain += float(loss_main.item()); ttemp += float(loss_temp.item()); tffl += float(loss_ffl.item()); tmass += float(loss_mass.item()) if cfg.lambda_mass > 0.0 else 0.0; n += 1
    return {"loss": tloss / max(n, 1), "main": tmain / max(n, 1), "temp": ttemp / max(n, 1), "ffl": tffl / max(n, 1), "mass": tmass / max(n, 1)}


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    if not cfg.run_name:
        cfg.run_name = (
            f"transport_cta_unet_ffl_multiout_logc_p{cfg.patch_size}_ffl{cfg.lambda_ffl:g}_ffa{cfg.ffl_alpha:g}"
            f"_tw{cfg.t_weight_mode}{cfg.t_w_min:g}-{cfg.t_w_max:g}_a{cfg.alpha:g}_ltemp{cfg.lambda_temp:g}_lr{cfg.lr:g}_seed{cfg.seed}"
        )
    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files = flatten_param_folders(cfg.data_root, train_params); val_files = flatten_param_folders(cfg.data_root, val_params); stats = load_train_stats(cfg.stats_json)
    train_ds = GroundwaterPatchDatasetMultiOut(train_files, stats, patch_size=cfg.patch_size, use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c)
    val_ds = GroundwaterPatchDatasetMultiOut(val_files, stats, patch_size=cfg.patch_size, use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    model = CTAUNet(in_ch=1, out_ch=25, base=cfg.base_channels, t_embed_dim=cfg.t_embed_dim, t_heads=cfg.t_heads, pool_stride=cfg.pool_stride).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs) if cfg.use_warmup else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    ffl_loss = FocalFrequencyLoss(alpha=cfg.ffl_alpha).to(cfg.device)
    best_val = float("inf"); start_epoch = 1
    best_path = os.path.join(out_dir, "best.pt"); last_path = os.path.join(out_dir, "last.pt"); log_csv = os.path.join(out_dir, "train_log.csv")
    if cfg.resume:
        ckpt = torch.load(last_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])
        if cfg.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1; best_val = ckpt.get("best_val", float("inf"))
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_main", "val_main", "train_temp", "val_temp", "train_ffl", "val_ffl", "train_mass", "val_mass", "lr", "skipped"])
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, ffl_loss, cfg)
        ev = eval_one_epoch(model, val_loader, ffl_loss, cfg)
        scheduler.step(); lr_now = scheduler.get_last_lr()[0]
        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__, "stats": stats, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if cfg.amp else None, "best_val": best_val}, last_path)
        improved = ev["loss"] < best_val
        if improved:
            best_val = ev["loss"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__, "stats": stats, "best_val": best_val}, best_path)
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, tr["loss"], ev["loss"], tr["main"], ev["main"], tr["temp"], ev["temp"], tr["ffl"], ev["ffl"], tr["mass"], ev["mass"], lr_now, tr["skipped"]])
        print(f"Epoch {epoch:03d} | train {tr['loss']:.6f}  val {ev['loss']:.6f} | lr {lr_now:.2e} | best {best_val:.6f} {'*' if improved else ''}")
    print("Best:", best_path)


if __name__ == "__main__":
    main()
