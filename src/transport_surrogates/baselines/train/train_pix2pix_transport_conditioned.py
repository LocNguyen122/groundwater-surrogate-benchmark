"""Train Multi-Output Pix2Pix with transport-parameter conditioning channels (manuscript).

Architecture  : Pix2PixGenerator (in_ch=3: K + alpha_L + alpha_T/alpha_L, out_ch=25)
              : PatchDiscriminator (in_ch=3+25=28)
Training loss : GAN adversarial + weighted L1 (matches pix2pix_baseline)
Protocol      : fairness rules , AdamW lr=1e-4, CosineAnnealingLR, 200 epochs, AMP enabled
Seeds         : run once per seed via --seed argument

NOTE: Pix2Pix uses a fixed alpha=3.0 (aligned with other benchmark models). The original
baseline Pix2Pix used alpha=20.0 , this transport-conditioned version corrects that to match
the fairness protocol used by all other models.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.shared.data.io import flatten_param_folders, load_param_split, load_train_stats
from src.shared.data.transport_conditioning import (
    GroundwaterPatchDatasetMultiOutTransportConditioned,
    compute_conditioning_stats,
    load_transport_metadata,
    parse_condition_params,
)
from src.shared.utils.seed import set_seed


# -- Model definitions ---------------------------------------------------------

class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=True)
        g = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else (2 if out_ch % 2 == 0 else 1))
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Pix2PixGenerator(nn.Module):
    """UNet-style generator.  Input: (B, in_ch, P, P).  Output: (B, 25, P, P)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 25, base: int = 64):
        super().__init__()
        b = base
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.e2 = ConvGNAct(b, 2 * b)
        self.e3 = ConvGNAct(2 * b, 4 * b)
        self.e4 = ConvGNAct(4 * b, 8 * b)
        self.e5 = ConvGNAct(8 * b, 8 * b)
        self.e6 = ConvGNAct(8 * b, 8 * b)
        self.mid = nn.Sequential(
            nn.Conv2d(8 * b, 8 * b, 3, 1, 1),
            nn.GroupNorm(8 if (8 * b) % 8 == 0 else 1, 8 * b),
            nn.ReLU(inplace=True),
        )

        def up(in_c: int, out_c: int, dropout: bool = False) -> nn.Sequential:
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=True),
                nn.GroupNorm(8 if out_c % 8 == 0 else 1, out_c),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.d6 = up(8 * b, 8 * b, dropout=True)
        self.d5 = up(16 * b, 8 * b, dropout=True)
        self.d4 = up(16 * b, 8 * b)
        self.d3 = up(16 * b, 4 * b)
        self.d2 = up(8 * b, 2 * b)
        self.d1 = up(4 * b, b)
        self.out = nn.Conv2d(2 * b, out_ch, 3, 1, 1)

    @staticmethod
    def _match(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-2:] != b.shape[-2:]:
            a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        return a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        m = self.mid(e6)
        d6 = self.d6(m)
        d5 = self.d5(torch.cat([d6, self._match(e6, d6)], dim=1))
        d4 = self.d4(torch.cat([d5, self._match(e5, d5)], dim=1))
        d3 = self.d3(torch.cat([d4, self._match(e4, d4)], dim=1))
        d2 = self.d2(torch.cat([d3, self._match(e3, d3)], dim=1))
        d1 = self.d1(torch.cat([d2, self._match(e2, d2)], dim=1))
        return self.out(torch.cat([d1, self._match(e1, d1)], dim=1))


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator.  Input: (B, in_ch, P, P).  in_ch = n_cond_ch + 25."""

    def __init__(self, in_ch: int = 26, base: int = 64):
        super().__init__()
        b = base
        g = lambda c: 8 if c % 8 == 0 else 1  # noqa: E731
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            ConvGNAct(b, 2 * b),
            ConvGNAct(2 * b, 4 * b),
            nn.Conv2d(4 * b, 8 * b, 4, 1, 1), nn.GroupNorm(g(8 * b), 8 * b),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * b, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -- Loss helpers --------------------------------------------------------------

def make_weight_log_linear(
    y_log: torch.Tensor, alpha: float, y_bg: float, y_cap: float
) -> torch.Tensor:
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0.0 else 1.0
    s = torch.clamp((y_log - y_bg) / denom, 0.0, 1.0)
    return 1.0 + alpha * s


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * (pred - target).abs()).mean()


# -- Config --------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class CFG:
    data_root: str = "data"
    split_json: str = "splits/param_split_fixed.json"
    stats_json: str = "configs/train_stats.json"
    out_root: str = "runs/pix2pix_transport_conditioned"
    run_name: str = ""
    seed: int = 0

    patch_size: int = 320
    batch_size: int = 16
    epochs: int = 200
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 0
    amp: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    lambda_l1: float = 100.0
    lambda_gan: float = 1.0
    alpha: float = 3.0   # corrected to match fairness protocol (was 20.0 in baseline)
    y_bg: float = -12.0
    y_cap: float = -2.0
    base_channels: int = 64
    use_logK: bool = True
    eps_k: float = 1e-6
    eps_c: float = 1e-12


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train Pix2Pix with transport-parameter conditioning (manuscript)."
    )
    ap.add_argument("--data_root", type=str, default=CFG.data_root)
    ap.add_argument("--split_json", type=str, default=CFG.split_json)
    ap.add_argument("--stats_json", type=str, default=CFG.stats_json)
    ap.add_argument("--out_root", type=str, default=CFG.out_root)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=0)
    amp_group = ap.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    logk_group = ap.add_mutually_exclusive_group()
    logk_group.add_argument("--use_logK", dest="use_logK", action="store_true")
    logk_group.add_argument("--no_logK", dest="use_logK", action="store_false")
    ap.set_defaults(amp=True, use_logK=True)
    ap.add_argument("--eps_k", type=float, default=CFG.eps_k)
    ap.add_argument("--eps_c", type=float, default=CFG.eps_c)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--lambda_l1", type=float, default=100.0)
    ap.add_argument("--lambda_gan", type=float, default=1.0)
    ap.add_argument("--condition_params", type=str, default="alpha_L,alpha_T_ratio")
    ap.add_argument("--transport_metadata_csv", type=str, default="")
    ap.add_argument("--alpha_L_transform", type=str, default="log10",
                    choices=["identity", "log10"])
    ap.add_argument("--alpha_T_ratio_transform", type=str, default="log10",
                    choices=["identity", "log10"])
    return ap.parse_args()


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


# -- Training helpers ----------------------------------------------------------

def train_gan_epoch(G, D, train_loader, opt_g, opt_d, scaler_g, scaler_d, cfg: CFG):
    G.train(); D.train()
    bce = nn.BCEWithLogitsLoss()
    total_g, total_d, n = 0.0, 0.0, 0

    for batch in train_loader:
        X = batch["K"].to(cfg.device, non_blocking=True)    # (B, 3, P, P)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)  # (B, 25, P, P)

        # -- Discriminator step ------------------------------------------------
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            fake_Y = G(X).detach()
            real_in = torch.cat([X, Y], dim=1)
            fake_in = torch.cat([X, fake_Y], dim=1)
            d_real = D(real_in)
            d_fake = D(fake_in)
            loss_d = 0.5 * (
                bce(d_real, torch.ones_like(d_real))
                + bce(d_fake, torch.zeros_like(d_fake))
            )

        opt_d.zero_grad(set_to_none=True)
        if cfg.amp:
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
        else:
            loss_d.backward()
            opt_d.step()

        # -- Generator step ----------------------------------------------------
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            fake_Y = G(X)
            w_y = make_weight_log_linear(Y, cfg.alpha, cfg.y_bg, cfg.y_cap)
            adv_loss = bce(D(torch.cat([X, fake_Y], dim=1)), torch.ones_like(d_real))
            l1_loss = weighted_l1(fake_Y, Y, w_y)
            loss_g = cfg.lambda_gan * adv_loss + cfg.lambda_l1 * l1_loss

        opt_g.zero_grad(set_to_none=True)
        if cfg.amp:
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            loss_g.backward()
            opt_g.step()

        total_g += loss_g.item()
        total_d += loss_d.item()
        n += 1

    return {"loss_g": total_g / max(n, 1), "loss_d": total_d / max(n, 1)}


@torch.no_grad()
def eval_generator(G, val_loader, cfg: CFG) -> dict:
    G.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        X = batch["K"].to(cfg.device, non_blocking=True)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            fake_Y = G(X)
            w_y = make_weight_log_linear(Y, cfg.alpha, cfg.y_bg, cfg.y_cap)
            loss = weighted_l1(fake_Y, Y, w_y)
        total += loss.item()
        n += 1
    return {"val_l1": total / max(n, 1)}


def main() -> None:
    args = parse_args()
    cfg = CFG()
    cfg.data_root = args.data_root
    cfg.split_json = args.split_json
    cfg.stats_json = args.stats_json
    cfg.out_root = args.out_root
    cfg.seed = args.seed
    cfg.run_name = args.run_name
    cfg.patch_size = args.patch_size
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr_g = args.lr
    cfg.lr_d = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.use_logK = args.use_logK
    cfg.eps_k = args.eps_k
    cfg.eps_c = args.eps_c
    cfg.alpha = args.alpha
    cfg.lambda_l1 = args.lambda_l1
    cfg.lambda_gan = args.lambda_gan
    cfg.amp = args.amp

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

    in_ch = 1 + len(condition_params)  # K + alpha_L + alpha_T_ratio = 3

    cfg_dict = dict(cfg.__dict__)
    cfg_dict.update({
        "model_family": "pix2pix_transport_conditioned",
        "use_transport_conditioning": True,
        "condition_params": condition_params,
        "transport_metadata_csv": args.transport_metadata_csv,
        "transport_conditioning_transforms": transforms,
        "transport_conditioning_stats": conditioning_stats,
        "input_channels": in_ch,
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

    G = Pix2PixGenerator(in_ch=in_ch, out_ch=25, base=cfg.base_channels).to(cfg.device)
    D = PatchDiscriminator(in_ch=in_ch + 25, base=cfg.base_channels).to(cfg.device)

    opt_g = torch.optim.AdamW(G.parameters(), lr=cfg.lr_g, weight_decay=cfg.weight_decay,
                               betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(D.parameters(), lr=cfg.lr_d, weight_decay=cfg.weight_decay,
                               betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=cfg.epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=cfg.epochs)
    scaler_g = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val_l1 = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["epoch", "loss_g", "loss_d", "val_l1", "lr_g"]
            )

    for epoch in range(1, cfg.epochs + 1):
        tr = train_gan_epoch(G, D, train_loader, opt_g, opt_d, scaler_g, scaler_d, cfg)
        ev = eval_generator(G, val_loader, cfg)
        sched_g.step()
        sched_d.step()
        lr_now = sched_g.get_last_lr()[0]

        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "sched_g": sched_g.state_dict(),
            "sched_d": sched_d.state_dict(),
            "best_val_l1": best_val_l1,
        }, last_path)

        if ev["val_l1"] < best_val_l1:
            best_val_l1 = ev["val_l1"]
            torch.save({
                "epoch": epoch,
                "model": G.state_dict(),
                "cfg": cfg_dict,
                "stats": stats,
                "best_val_l1": best_val_l1,
            }, best_path)

        with open(log_csv, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch, tr["loss_g"], tr["loss_d"], ev["val_l1"], lr_now]
            )

        print(
            f"Epoch {epoch:03d} | G {tr['loss_g']:.4f}  D {tr['loss_d']:.4f}"
            f"  val_L1 {ev['val_l1']:.6f} | best {best_val_l1:.6f} | lr {lr_now:.2e}"
        )

    print("Best generator checkpoint:", best_path)


if __name__ == "__main__":
    main()
