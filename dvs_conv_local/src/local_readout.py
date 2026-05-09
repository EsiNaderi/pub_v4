"""Local class-pool readout for spiking convolutional maps.

The readout is deliberately simple and local:

    raw_logit_c = mean_{channels in class c, y, x} rho

Cross-entropy credit is derived manually. There is no autograd and no
transported downstream weight matrix.
"""

from __future__ import annotations

import torch


def class_pool_logits(rho: torch.Tensor, class_index: torch.Tensor,
                      classes: int, temperature: float) -> torch.Tensor:
    """Mean-pool convolutional rates into class logits.

    rho: (B, O, H, W)
    class_index: (O,), assigning each output channel to a class pool.
    """
    logits = torch.zeros(rho.shape[0], classes, dtype=rho.dtype, device=rho.device)
    ci = class_index.to(device=rho.device)
    for c in range(classes):
        logits[:, c] = rho[:, ci == c].mean(dim=(1, 2, 3))
    return (logits - logits.mean(dim=1, keepdim=True)) * temperature


def cross_entropy_credit(rho: torch.Tensor, labels: torch.Tensor,
                         class_index: torch.Tensor, classes: int,
                         temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (loss, logits, d loss / d rho) without autograd."""
    B, O, H, W = rho.shape
    logits = class_pool_logits(rho, class_index, classes, temperature)
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=1, keepdim=True).clamp_min(1e-12)

    loss = -torch.log(probs[torch.arange(B, device=labels.device), labels].clamp_min(1e-12)).mean()

    grad_logits = probs
    grad_logits[torch.arange(B, device=labels.device), labels] -= 1.0
    grad_logits = grad_logits * (temperature / float(B))

    credit = torch.zeros_like(rho)
    ci = class_index.to(device=rho.device)
    for c in range(classes):
        mask = ci == c
        denom = float(mask.sum().item() * H * W)
        credit[:, mask] = grad_logits[:, c].view(B, 1, 1, 1) / denom
    return loss.detach(), logits.detach(), credit.detach()
