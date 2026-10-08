import os, json, csv, argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.shared.utils.seed import set_seed
from src.shared.data.io import load_param_split, flatten_param_folders, load_train_stats
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
# Models
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=True)
        g = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else (2 if out_ch % 2 == 0 else 1))
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Pix2PixGenerator(nn.Module):
    """
    UNet-like generator:
      Input:  (B,1,P,P) normalized K
      Output: (B,25,P,P) log10(C+eps)
    """
    def __init__(self, in_ch=1, out_ch=25, base=64):
        super().__init__()
        b = base

        self.e1 = nn.Sequential(nn.Conv2d(in_ch, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))  # no norm
        self.e2 = ConvGNAct(b, 2*b)
        self.e3 = ConvGNAct(2*b, 4*b)
        self.e4 = ConvGNAct(4*b, 8*b)
        self.e5 = ConvGNAct(8*b, 8*b)
        self.e6 = ConvGNAct(8*b, 8*b)

        self.mid = nn.Sequential(
            nn.Conv2d(8*b, 8*b, 3, 1, 1),
            nn.GroupNorm(8 if (8*b)%8==0 else 1, 8*b),
            nn.ReLU(inplace=True),
        )

        def up(in_c, out_c, dropout=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=True),
                      nn.GroupNorm(8 if out_c%8==0 else 1, out_c),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.d6 = up(8*b, 8*b, dropout=True)
        self.d5 = up(16*b, 8*b, dropout=True)
        self.d4 = up(16*b, 8*b)
        self.d3 = up(16*b, 4*b)
        self.d2 = up(8*b, 2*b)
        self.d1 = up(4*b, b)

        self.out = nn.Conv2d(2*b, out_ch, 3, 1, 1)

    def _match(self,a, b):
        if a.shape[-2:] != b.shape[-2:]:
            a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        return a

    def forward(self, x):

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

        y = self.out(torch.cat([d1, self._match(e1, d1)], dim=1))

        return y


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator on concatenated (K, C_log):
      Input: (B, 1+25, P, P)
      Output: (B, 1, pH, pW) logits
    """
    def __init__(self, in_ch=26, base=64):
        super().__init__()
        b = base
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            ConvGNAct(b, 2*b),
            ConvGNAct(2*b, 4*b),
            nn.Conv2d(4*b, 8*b, 4, 1, 1), nn.GroupNorm(8 if (8*b)%8==0 else 1, 8*b), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*b, 1, 4, 1, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)
# Loss helpers
def make_weight_log_linear(y_log, alpha, y_bg, y_cap):
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0 else 1.0
    s = (y_log - y_bg) / denom
    s = torch.clamp(s, 0.0, 1.0)
    return 1.0 + alpha * s


def weighted_l1(pred, target, w):
    return (w * (pred - target).abs()).mean()
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
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    wd: float = 0.0

    num_workers: int = 0
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # losses
    lambda_l1: float = 100.0   # standard pix2pix
    lambda_gan: float = 1.0
    alpha: float = 20.0
    y_bg: float = -12.0
    y_cap: float = -2.0

    base_channels: int = 64
    use_logK: bool = True
    eps_k: float = 1e-6
    eps_c: float = 1e-12


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=CFG.data_root)
    ap.add_argument("--split_json", type=str, default=CFG.split_json)
    ap.add_argument("--stats_json", type=str, default=CFG.stats_json)
    ap.add_argument("--out_root", type=str, default=CFG.out_root)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    logk_group = ap.add_mutually_exclusive_group()
    logk_group.add_argument("--use_logK", dest="use_logK", action="store_true")
    logk_group.add_argument("--no_logK", dest="use_logK", action="store_false")
    ap.add_argument("--eps_k", type=float, default=CFG.eps_k)
    ap.add_argument("--eps_c", type=float, default=CFG.eps_c)
    amp_group = ap.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True, use_logK=True)
    args = ap.parse_args()

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
    cfg.use_logK = args.use_logK
    cfg.eps_k = args.eps_k
    cfg.eps_c = args.eps_c
    cfg.amp = args.amp
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    if not cfg.run_name:
        cfg.run_name = f"transport_pix2pix_multiout_logc_p{cfg.patch_size}_lr{cfg.lr_g:g}_seed{cfg.seed}"

    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files = flatten_param_folders(cfg.data_root, train_params)
    val_files = flatten_param_folders(cfg.data_root, val_params)

    stats = load_train_stats(cfg.stats_json)

    train_ds = GroundwaterPatchDatasetMultiOut(
        train_files, stats, patch_size=cfg.patch_size,
        use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c
    )
    val_ds = GroundwaterPatchDatasetMultiOut(
        val_files, stats, patch_size=cfg.patch_size,
        use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    G = Pix2PixGenerator(in_ch=1, out_ch=25, base=cfg.base_channels).to(cfg.device)
    D = PatchDiscriminator(in_ch=26, base=cfg.base_channels).to(cfg.device)

    optG = torch.optim.AdamW(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999), weight_decay=cfg.wd)
    optD = torch.optim.AdamW(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999), weight_decay=cfg.wd)
    schedG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=cfg.epochs)
    schedD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_G", "train_D", "val_L1", "val_GAN"])

    def val_epoch():
        G.eval()
        l1s = []
        gans = []
        with torch.no_grad():
            for batch in val_loader:
                K = batch["K"].to(cfg.device, non_blocking=True)       # (B,1,P,P)
                Y = batch["C_log"].to(cfg.device, non_blocking=True)   # (B,25,P,P)
                pred = G(K)
                w = make_weight_log_linear(Y, cfg.alpha, cfg.y_bg, cfg.y_cap)
                l1 = weighted_l1(pred, Y, w)
                fake_in = torch.cat([K, pred], dim=1)
                logits = D(fake_in)
                g_gan = bce(logits, torch.ones_like(logits))
                l1s.append(float(l1.item()))
                gans.append(float(g_gan.item()))
        G.train()
        return float(np.mean(l1s)), float(np.mean(gans))

    for epoch in range(1, cfg.epochs + 1):
        G.train(); D.train()
        g_losses, d_losses = [], []

        for batch in train_loader:
            K = batch["K"].to(cfg.device, non_blocking=True)
            Y = batch["C_log"].to(cfg.device, non_blocking=True)
            # Train D
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                with torch.no_grad():
                    pred = G(K)

                real_in = torch.cat([K, Y], dim=1)
                fake_in = torch.cat([K, pred], dim=1)

                real_logits = D(real_in)
                fake_logits = D(fake_in)

                d_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))

            optD.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.scale(d_loss).backward()
                scaler.step(optD)
            else:
                d_loss.backward()
                optD.step()
            # Train G
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                pred = G(K)
                w = make_weight_log_linear(Y, cfg.alpha, cfg.y_bg, cfg.y_cap)
                l1 = weighted_l1(pred, Y, w)

                fake_in = torch.cat([K, pred], dim=1)
                fake_logits = D(fake_in)
                g_gan = bce(fake_logits, torch.ones_like(fake_logits))

                g_loss = cfg.lambda_l1 * l1 + cfg.lambda_gan * g_gan

            optG.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.scale(g_loss).backward()
                scaler.step(optG)
                scaler.update()
            else:
                g_loss.backward()
                optG.step()

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))

        val_l1, val_gan = val_epoch()
        schedG.step()
        schedD.step()

        torch.save({
            "epoch": epoch,
            "gen": G.state_dict(),
            "disc": D.state_dict(),
            "cfg": cfg.__dict__,
            "stats": stats,
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "schedG": schedG.state_dict(),
            "schedD": schedD.state_dict(),
            "scaler": scaler.state_dict() if cfg.amp else None
        }, last_path)

        improved = val_l1 < best_val
        if improved:
            best_val = val_l1
            torch.save({"epoch": epoch, "gen": G.state_dict(), "cfg": cfg.__dict__, "stats": stats}, best_path)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, float(np.mean(g_losses)), float(np.mean(d_losses)), val_l1, val_gan])

        print(f"Epoch {epoch:03d} | trainG {np.mean(g_losses):.4f} trainD {np.mean(d_losses):.4f} "
              f"| valL1 {val_l1:.4f} | best {best_val:.4f} {'*' if improved else ''}")

    print("Best:", best_path)


if __name__ == "__main__":
    main()

# python -m src.train.train_pix2pix_multioutput_patch_logc ^
#   --seed 2 --patch_size 320 --batch_size 16 --epochs 200 --lr 0.0001 --amp
