"""SMNIST data loader.

Loads MNIST images, flattens to 784-element sequences, returns (B, T=784, 1)
single-channel tensors. Intended to be loaded once into RAM and indexed by
batches; SMNIST is small enough for this.

Stores cache to disk as a single .pt with float32 tensors. Subsequent calls
reload from cache.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import torch

MNIST_DIR = Path("/Users/esi/research/data/MNIST/raw")
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _read_idx_images(path: Path) -> torch.Tensor:
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x00000803
        buf = f.read(n * rows * cols)
    arr = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
    return arr.view(n, rows * cols).to(torch.float32) / 255.0


def _read_idx_labels(path: Path) -> torch.Tensor:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x00000801
        buf = f.read(n)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone().to(torch.int64)


def load_smnist(cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (x_train, y_train, x_test, y_test).

    x_*: (N, 784) float32 in [0, 1]
    y_*: (N,) int64 in [0, 10)
    """

    cache_file = CACHE_DIR / "smnist_cache.pt"
    if cache and cache_file.exists():
        d = torch.load(cache_file, weights_only=True)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]

    xtr = _read_idx_images(MNIST_DIR / "train-images-idx3-ubyte")
    ytr = _read_idx_labels(MNIST_DIR / "train-labels-idx1-ubyte")
    xte = _read_idx_images(MNIST_DIR / "t10k-images-idx3-ubyte")
    yte = _read_idx_labels(MNIST_DIR / "t10k-labels-idx1-ubyte")
    if cache:
        torch.save({"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}, cache_file)
    return xtr, ytr, xte, yte


class SMNISTBatcher:
    """Iterates random minibatches as (B, T, 1) single-channel tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch: int, device: str, seed: int = 0):
        self.x = x.to(device)
        self.y = y.to(device)
        self.batch = batch
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self) -> int:
        return (self.x.shape[0] + self.batch - 1) // self.batch

    def shuffle_iter(self):
        order = torch.randperm(self.x.shape[0], generator=self.gen)
        for s in range(0, self.x.shape[0], self.batch):
            idx = order[s : s + self.batch].to(self.device)
            xb = self.x.index_select(0, idx).unsqueeze(-1)  # (B, 784, 1)
            yb = self.y.index_select(0, idx)
            yield xb, yb

    def seq_iter(self):
        for s in range(0, self.x.shape[0], self.batch):
            xb = self.x[s : s + self.batch].unsqueeze(-1)
            yb = self.y[s : s + self.batch]
            yield xb, yb


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_smnist()
    print(f"train: x={tuple(xtr.shape)}, y={tuple(ytr.shape)}")
    print(f"test : x={tuple(xte.shape)}, y={tuple(yte.shape)}")
    print(f"label range: train [{ytr.min().item()}, {ytr.max().item()}], test [{yte.min().item()}, {yte.max().item()}]")
    print(f"pixel range: train [{xtr.min().item():.3f}, {xtr.max().item():.3f}]")
