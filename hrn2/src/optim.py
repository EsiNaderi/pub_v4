"""Per-tensor Adam optimizer for hand-assembled local-rule gradients.

Mirrors pub_v3/experiments/resonant_layer.py's Adam for compatibility.
"""

from __future__ import annotations

import torch


EPS = 1e-12


class Adam:
    def __init__(self, params: list[torch.Tensor], lr: float):
        self.params = params
        self.lr = lr
        self.t = 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def step(self, grads: list[torch.Tensor], clip: float = 0.0) -> None:
        self.t += 1
        for i, (p, g_raw) in enumerate(zip(self.params, grads)):
            g = g_raw.detach()
            if clip > 0:
                norm = g.norm()
                if float(norm.item()) > clip:
                    g = g * (clip / norm.clamp_min(EPS))
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g.square()
            mhat = self.m[i] / (1.0 - 0.9 ** self.t)
            vhat = self.v[i] / (1.0 - 0.999 ** self.t)
            p -= self.lr * mhat / (vhat.sqrt() + 1e-8)
