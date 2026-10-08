import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """
    Lightweight focal frequency loss for dense regression.

    Based on the core ICCV 2021 idea: emphasize spectral regions where the current
    prediction still differs from the target, while down-weighting easy frequencies.
    This implementation is intentionally simple and pure PyTorch so it can run on
    standard PyTorch environments without external dependencies.
    """

    def __init__(self, alpha: float = 1.0, loss_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        batch, timesteps, height, width = pred.shape
        pred = pred.reshape(batch * timesteps, 1, height, width)
        target = target.reshape(batch * timesteps, 1, height, width)

        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        diff = pred_fft - target_fft

        spectrum_gap = torch.sqrt(diff.real.pow(2) + diff.imag.pow(2) + self.eps)
        weight = spectrum_gap.pow(self.alpha)
        weight = weight / (weight.amax(dim=(-2, -1), keepdim=True).detach() + self.eps)
        weight = weight.detach()

        freq_error = diff.real.pow(2) + diff.imag.pow(2)
        return self.loss_weight * (weight * freq_error).mean()
