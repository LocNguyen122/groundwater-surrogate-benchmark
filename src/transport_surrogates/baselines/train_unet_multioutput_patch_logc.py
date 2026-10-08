import os
import csv
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.transport_surrogates.baselines.unet2d import UNet2D
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
from src.shared.data.io import load_param_split, flatten_param_folders, load_train_stats
from src.shared.utils.seed import set_seed


def parse_timestep_spec(spec: str) -> Optional[List[int]]:
    """
    Parse timestep selection strings.

    Supported formats:
      - "all" (or empty) -> None
      - "0-15" (inclusive range)
      - "0:15" (inclusive range)
      - "0,1,2,3"
    """
    s = (spec or "").strip().lower()
    if s in ("", "all", "full"):
        return None

    if "-" in s:
        a, b = s.split("-", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError(f"Invalid timestep range '{spec}'.")
        return list(range(start, end + 1))

    if ":" in s:
        a, b = s.split(":", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError(f"Invalid timestep range '{spec}'.")
        return list(range(start, end + 1))

    vals = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not vals:
        raise ValueError(f"Invalid timestep spec '{spec}'.")
    return vals


@dataclass
class CFG:
    # Paths
    data_root: str = r"data"
    split_json: str = r"splits/param_split_fixed.json"
    stats_json: str = r"configs/train_stats.json"

    # Run control
    run_name: str = ""  # if empty -> auto
    out_root: str = "runs"
    resume_path: Optional[str] = None

    # Repro
    seed: int = 0

    # Train
    patch_size: int = 320
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    amp: bool = True
    grad_clip_norm: float = 1.0
    num_workers: int = 0

    # Data transforms
    use_logK: bool = True
    eps_k: float = 1e-6
    eps_c: float = 1e-12

    # New: timestep subset control
    train_timesteps: str = "all"
    val_timesteps: str = ""  # empty -> same as train_timesteps
    output_channels: int = 25

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark: bool = True


def parse_args() -> CFG:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--data_root", type=str, default=r"data")
    ap.add_argument("--split_json", type=str, default=r"splits/param_split_fixed.json")
    ap.add_argument("--stats_json", type=str, default=r"configs/train_stats.json")

    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--resume_path", type=str, default="")

    ap.add_argument("--use_logK", dest="use_logK", action="store_true", default=True)

    ap.add_argument("--no_logK", dest="use_logK", action="store_false")
    ap.add_argument("--eps_k", type=float, default=1e-6)
    ap.add_argument("--eps_c", type=float, default=1e-12)

    ap.add_argument("--train_timesteps", type=str, default="all")
    ap.add_argument("--val_timesteps", type=str, default="")

    args = ap.parse_args()

    cfg = CFG()
    cfg.seed = args.seed
    cfg.patch_size = args.patch_size
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip_norm = args.grad_clip_norm
    cfg.num_workers = args.num_workers

    cfg.data_root = args.data_root
    cfg.split_json = args.split_json
    cfg.stats_json = args.stats_json

    cfg.run_name = args.run_name
    cfg.out_root = args.out_root
    cfg.resume_path = args.resume_path if args.resume_path.strip() else None

    # AMP flags
    if args.no_amp:
        cfg.amp = False
    elif args.amp:
        cfg.amp = True

    # If you explicitly set --use_logK, respect it; otherwise keep default True
    if args.use_logK:
        cfg.use_logK = True
    cfg.eps_k = args.eps_k
    cfg.eps_c = args.eps_c

    cfg.train_timesteps = args.train_timesteps
    cfg.val_timesteps = args.val_timesteps

    return cfg


def train_one_epoch(model, loader, optimizer, scaler, cfg: CFG):
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)        # (B,1,P,P)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)    # (B,T,P,P) log10(C+eps)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(K)  # (B,T,P,P)
            loss = F.l1_loss(pred, Y)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            print("[WARN] Non-finite loss. Skipping batch.")
            continue

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False
            )
            if not torch.isfinite(total_norm):
                optimizer.zero_grad(set_to_none=True)
                print("[WARN] Non-finite grad norm. Skipping step.")
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False
            )
            if not torch.isfinite(total_norm):
                optimizer.zero_grad(set_to_none=True)
                print("[WARN] Non-finite grad norm. Skipping step.")
                continue
            optimizer.step()

        total += float(loss.item())
        n += 1

    return {"l1": total / max(n, 1)}


@torch.no_grad()
def eval_one_epoch(model, loader, cfg: CFG):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)
        pred = model(K)
        loss = F.l1_loss(pred, Y)
        if not torch.isfinite(loss):
            continue
        total += float(loss.item())
        n += 1
    return {"l1": total / max(n, 1)}


def main():
    cfg = parse_args()

    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    set_seed(cfg.seed)

    # Run folder
    if not cfg.run_name:
        cfg.run_name = f"transport_multiout_unet_logc_p{cfg.patch_size}_seed{cfg.seed}_lr{cfg.lr:g}"
    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Split
    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files = flatten_param_folders(cfg.data_root, train_params)
    val_files = flatten_param_folders(cfg.data_root, val_params)

    train_t_idx = parse_timestep_spec(cfg.train_timesteps)
    val_spec = cfg.val_timesteps if cfg.val_timesteps.strip() else cfg.train_timesteps
    val_t_idx = parse_timestep_spec(val_spec)

    stats = load_train_stats(cfg.stats_json)

    train_ds = GroundwaterPatchDatasetMultiOut(
        train_files, stats, cfg.patch_size,
        cfg.use_logK, cfg.eps_k, cfg.eps_c,
        timesteps=train_t_idx,
    )
    val_ds = GroundwaterPatchDatasetMultiOut(
        val_files, stats, cfg.patch_size,
        cfg.use_logK, cfg.eps_k, cfg.eps_c,
        timesteps=val_t_idx,
    )

    sample = train_ds[0]
    cfg.output_channels = int(sample["C_log"].shape[0])

    val_sample = val_ds[0]
    val_out_channels = int(val_sample["C_log"].shape[0])
    if val_out_channels != cfg.output_channels:
        raise ValueError(
            "Train and validation timestep selections produce different channel counts. "
            f"train={cfg.output_channels}, val={val_out_channels}."
        )

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    model = UNet2D(in_ch=1, out_ch=cfg.output_channels, base=64).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")

    start_epoch = 1
    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if cfg.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[Resume] {cfg.resume_path} -> epoch {start_epoch}, best_val={best_val:.6f}")

    if start_epoch == 1 and not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_l1", "val_l1", "lr"])

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # Save last (for exact resume)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "cfg": asdict(cfg),
                "stats": stats,
                "best_val": best_val,
            },
            last_path
        )

        improved = ev["l1"] < best_val
        if improved:
            best_val = ev["l1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "stats": stats,
                    "best_val": best_val,
                },
                best_path
            )

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, tr["l1"], ev["l1"], lr_now])

        print(
            f"Epoch {epoch:03d} | train L1 {tr['l1']:.6f} | val L1 {ev['l1']:.6f} "
            f"| lr {lr_now:.2e} | best {best_val:.6f} {'*' if improved else ''}"
        )

    print("[Done]")
    print("Best:", best_path)
    print("Last:", last_path)


if __name__ == "__main__":
    main()

# Default baseline behavior (full 0..24) stays unchanged:
# python -m src.train.train_unet_multioutput_patch_logc --seed 0 --patch_size 320 --batch_size 16 --epochs 200 --lr 1e-4 --amp
# Journal extrapolation setup example:
# python -m src.train.train_unet_multioutput_patch_logc --train_timesteps 0-15 --val_timesteps 0-15
