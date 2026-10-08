"""Train DeepONet2D with transport-parameter conditioning channels (manuscript benchmark).

Architecture  : DeepONet2D (in_ch=3: K + alpha_L + alpha_T/alpha_L, out_ch=25)
Training loss : weighted Huber + temporal smoothness (matches deeponet_baseline in registry)
Protocol      : fairness rules , AdamW lr=1e-4, CosineAnnealingLR, 200 epochs, AMP enabled
Seeds         : run once per seed via --seed argument
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader

from src.transport_surrogates.baselines.deeponet2d import DeepONet2D
from src.transport_surrogates.baselines.train_unet_multiout_logc_A_tweight_tsmooth import (
    CFG,
    eval_one_epoch,
    train_one_epoch,
    warmup_cosine_scheduler,
)
from src.shared.data.io import flatten_param_folders, load_param_split, load_train_stats
from src.shared.data.transport_conditioning import (
    GroundwaterPatchDatasetMultiOutTransportConditioned,
    compute_conditioning_stats,
    load_transport_metadata,
    parse_condition_params,
)
from src.shared.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train DeepONet2D with transport-parameter conditioning (manuscript)."
    )
    ap.add_argument("--data_root", type=str, default=CFG.data_root)
    ap.add_argument("--split_json", type=str, default=CFG.split_json)
    ap.add_argument("--stats_json", type=str, default=CFG.stats_json)
    ap.add_argument("--out_root", type=str,
                    default="runs/deeponet_transport_conditioned")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

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

    # DeepONet architecture (matches deeponet_baseline registry args)
    ap.add_argument("--branch_width", type=int, default=128)
    ap.add_argument("--branch_latent", type=int, default=384)
    ap.add_argument("--trunk_width", type=int, default=384)
    ap.add_argument("--basis_rank", type=int, default=96)

    # Temporal loss (matches deeponet_baseline registry args)
    ap.add_argument("--alpha", type=float, default=CFG.alpha)
    ap.add_argument("--t_weight_mode", type=str, default=CFG.t_weight_mode,
                    choices=["linear", "quad"])
    ap.add_argument("--t_w_min", type=float, default=CFG.t_w_min)
    ap.add_argument("--t_w_max", type=float, default=CFG.t_w_max)
    ap.add_argument("--lambda_temp", type=float, default=CFG.lambda_temp)
    ap.add_argument("--lambda_mass", type=float, default=CFG.lambda_mass)

    ap.add_argument("--use_warmup", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=CFG.warmup_epochs)
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
    for field in cfg.__dataclass_fields__:
        if hasattr(args, field):
            setattr(cfg, field, getattr(args, field))
    cfg.resume = args.resume
    cfg.amp = True
    if args.no_amp:
        cfg.amp = False
    elif args.amp:
        cfg.amp = True
    for attr in ("alpha", "t_weight_mode", "t_w_min", "t_w_max", "lambda_temp", "lambda_mass"):
        if hasattr(args, attr):
            setattr(cfg, attr, getattr(args, attr))
    return cfg


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
        "input_channels": 1 + len(condition_params),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    cfg = args_to_cfg(args)
    set_seed(cfg.seed)
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
    val_files = flatten_param_folders(cfg.data_root, val_params)
    stats = load_train_stats(cfg.stats_json)
    metadata = load_transport_metadata(args.transport_metadata_csv)
    conditioning_stats = compute_conditioning_stats(
        train_files, metadata, condition_params, transforms
    )

    n_cond = len(condition_params)
    in_ch = 1 + n_cond

    cfg_dict = dict(cfg.__dict__)
    cfg_dict.update({
        "model_family": "deeponet_transport_conditioned",
        "use_transport_conditioning": True,
        "condition_params": condition_params,
        "transport_metadata_csv": args.transport_metadata_csv,
        "transport_conditioning_transforms": transforms,
        "transport_conditioning_stats": conditioning_stats,
        "input_channels": in_ch,
        "branch_width": args.branch_width,
        "branch_latent": args.branch_latent,
        "trunk_width": args.trunk_width,
        "basis_rank": args.basis_rank,
    })
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg_dict, handle, indent=2)
    write_conditioning_metadata(
        out_dir, condition_params, transforms, conditioning_stats,
        args.transport_metadata_csv,
    )

    train_ds = GroundwaterPatchDatasetMultiOutTransportConditioned(
        train_files, stats,
        patch_size=cfg.patch_size, use_logK=cfg.use_logK,
        eps_k=cfg.eps_k, eps_c=cfg.eps_c,
        metadata=metadata, condition_params=condition_params,
        conditioning_stats=conditioning_stats,
    )
    val_ds = GroundwaterPatchDatasetMultiOutTransportConditioned(
        val_files, stats,
        patch_size=cfg.patch_size, use_logK=cfg.use_logK,
        eps_k=cfg.eps_k, eps_c=cfg.eps_c,
        metadata=metadata, condition_params=condition_params,
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

    model = DeepONet2D(
        in_ch=in_ch,
        out_ch=25,
        branch_width=args.branch_width,
        branch_latent=args.branch_latent,
        trunk_width=args.trunk_width,
        basis_rank=args.basis_rank,
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
    log_csv = os.path.join(out_dir, "train_log.csv")

    if cfg.resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"--resume set but no checkpoint at {last_path}")
        ckpt = torch.load(last_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if cfg.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", float("inf")))

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["epoch", "train_loss", "val_loss", "train_main", "val_main",
                 "train_temp", "val_temp", "train_mass", "val_mass", "lr", "skipped"]
            )

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": cfg_dict,
            "stats": stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if cfg.amp else None,
            "best_val": best_val,
        }, last_path)

        if ev["loss"] < best_val:
            best_val = ev["loss"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": cfg_dict,
                "stats": stats,
                "best_val": best_val,
            }, best_path)

        skipped = tr.get("skipped", 0) + ev.get("skipped", 0)
        with open(log_csv, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch,
                tr["loss"], ev["loss"],
                tr.get("main", tr["loss"]), ev.get("main", ev["loss"]),
                tr.get("temp", 0.0), ev.get("temp", 0.0),
                tr.get("mass", 0.0), ev.get("mass", 0.0),
                lr_now, skipped,
            ])

        print(
            f"Epoch {epoch:03d} | train {tr['loss']:.6f}  val {ev['loss']:.6f}"
            f" | lr {lr_now:.2e} | best {best_val:.6f}"
        )

    print("Best:", best_path)


if __name__ == "__main__":
    main()
