"""Benchmark forward pass on different devices/batches/lengths."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from hrn import HierarchicalResonantNet, make_default_config


def bench(device: str, batch: int, T: int, repeat: int = 3, backward: bool = False):
    torch.manual_seed(0)
    cfg = make_default_config()
    net = HierarchicalResonantNet(cfg).to(device)
    x = torch.rand(batch, T, 1, device=device)
    y = torch.zeros(batch, dtype=torch.long, device=device)
    crit = torch.nn.CrossEntropyLoss()

    # warmup
    if backward:
        logits, _ = net(x)
        loss = crit(logits, y)
        loss.backward()
    else:
        with torch.no_grad():
            net(x)

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        t0 = time.time()
        if backward:
            for p in net.parameters():
                if p.grad is not None: p.grad = None
            logits, _ = net(x)
            loss = crit(logits, y)
            loss.backward()
        else:
            with torch.no_grad():
                net(x)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    return min(times)


def main():
    print(f"device | batch | T   | fwd (s) | fwd+back (s)", flush=True)
    for device in ["cpu"]:
        for batch in [32, 64, 128]:
            for T in [392, 784]:
                t_fwd = bench(device, batch, T, backward=False, repeat=2)
                t_back = bench(device, batch, T, backward=True, repeat=2)
                print(f"{device:>6} | {batch:>5} | {T:>3} | {t_fwd:>6.3f}  | {t_back:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
