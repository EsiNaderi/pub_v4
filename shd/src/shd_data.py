"""SHD (Spiking Heidelberg Digits) loader.

Reads the prebinned cache produced from the canonical SHD HDF5 files. The
cache contains:
    x_tr  shape (N_tr, T, F=700)  float32   {0, 1} (binned spike rate per bin)
    y_tr  shape (N_tr,)           int64     class labels in [0, 20)
    x_te  shape (N_te, T, F=700)  float32
    y_te  shape (N_te,)           int64

Default cache used here is T=100 bins of 10 ms each, 4000 train / 1000 test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


CACHE_PATHS = [
    Path("/Users/esi/research/lambda/shd_cache/shd_binned_T100_dt10_tr4000_te1000.npz"),
    Path("/Users/esi/research/pub_v4/shd/data/shd_binned_T100_dt10_tr4000_te1000.npz"),
]

DEFAULT_T = 100
DEFAULT_F = 700
N_CLASSES = 20


def load_shd(
    cache_path: Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (x_tr, y_tr, x_te, y_te) as float32/int64 torch tensors.

    x_*: (N, T, F) where T=100, F=700 in the default cache.
    y_*: (N,) labels in [0, 20).
    """
    if cache_path is None:
        for p in CACHE_PATHS:
            if p.exists():
                cache_path = p
                break
        if cache_path is None:
            raise FileNotFoundError(
                "No SHD cache found at any of: "
                + ", ".join(str(p) for p in CACHE_PATHS)
            )
    d = np.load(cache_path)
    xtr = torch.from_numpy(np.ascontiguousarray(d["x_tr"]))
    ytr = torch.from_numpy(np.ascontiguousarray(d["y_tr"]))
    xte = torch.from_numpy(np.ascontiguousarray(d["x_te"]))
    yte = torch.from_numpy(np.ascontiguousarray(d["y_te"]))
    return xtr, ytr, xte, yte


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_shd()
    print(f"train: x={tuple(xtr.shape)}, y={tuple(ytr.shape)}")
    print(f"test : x={tuple(xte.shape)}, y={tuple(yte.shape)}")
    print(f"label range: train [{int(ytr.min())}, {int(ytr.max())}], test [{int(yte.min())}, {int(yte.max())}]")
    print(f"x range: train [{xtr.min().item():.3f}, {xtr.max().item():.3f}]")
    print(f"x mean (per-channel mean over time, then mean): {xtr.mean().item():.5f}")
    print(f"x sparsity (fraction == 0): {(xtr == 0).float().mean().item():.4f}")
