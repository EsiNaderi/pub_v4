"""Analyze a trained HRN checkpoint:
- Per-layer firing rates
- Per-class output pool rates (does each pool fire selectively for its class?)
- Confusion matrix
- Pool diversity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet
from train_bptt import build_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--aux_head", action="store_true")
    p.add_argument("--samples_per_class", type=int, default=20)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    net = build_net(args.arch, aux_head=args.aux_head)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"loaded ckpt: epoch {ckpt.get('epoch', '?')} test_acc {ckpt.get('test_acc', '?')}", flush=True)

    xtr, ytr, xte, yte = load_smnist()

    # Build evaluation set: samples_per_class images per class
    by_class = []
    for c in range(10):
        idx = (yte == c).nonzero().squeeze()[:args.samples_per_class]
        by_class.append(xte[idx])
    x_test = torch.stack(by_class).reshape(-1, 784)
    y_test = torch.arange(10).repeat_interleave(args.samples_per_class)

    with torch.no_grad():
        logits, info = net(x_test.unsqueeze(-1), return_layers=True)

    pred = logits.argmax(dim=1)
    accuracy = float((pred == y_test).float().mean().item())
    print(f"\n=== Test on {x_test.shape[0]} samples ===")
    print(f"accuracy: {accuracy:.4f}")

    # confusion matrix
    cm = torch.zeros(10, 10, dtype=torch.long)
    for i in range(len(y_test)):
        cm[y_test[i], pred[i]] += 1
    print("\nConfusion matrix (rows=true, cols=pred):")
    for r in range(10):
        print(" ".join(f"{cm[r, c].item():>4}" for c in range(10)))

    # per-layer rates
    print("\n=== Layer rates ===")
    for i, s in enumerate(info["layer_spikes"]):
        rate = s.mean().item()
        per_neuron_rate = s.mean(dim=(0, 1))
        print(f"  layer {i}: total rate {rate:.4f} (std across neurons: {per_neuron_rate.std().item():.4f})")

    # per-class output pool rate
    out_seq = info["layer_spikes"][-1]
    B, T, N_out = out_seq.shape
    pool_size = net.cfg.out_pool_size
    n_classes = net.cfg.n_classes
    out_p = out_seq.view(B, T, n_classes, pool_size)
    tail = max(1, int(round(T * net.cfg.tail_fraction)))
    pool_rate = out_p[:, T - tail:].mean(dim=(1, 3))                 # (B, n_classes)
    # group by class
    print("\n=== Per-class output pool firing rates (rows=class, cols=pool) ===")
    n_per_class = args.samples_per_class
    for c in range(10):
        rates = pool_rate[c * n_per_class : (c + 1) * n_per_class].mean(dim=0)
        print(f"  class {c}: " + " ".join(f"{r.item():.3f}" for r in rates))

    # Best pool per class
    print("\n=== Top firing pool per class ===")
    for c in range(10):
        rates = pool_rate[c * n_per_class : (c + 1) * n_per_class].mean(dim=0)
        top_pool = rates.argmax().item()
        if top_pool == c:
            note = "  ✓"
        else:
            note = ""
        print(f"  class {c}: top pool {top_pool}{note}")


if __name__ == "__main__":
    main()
