"""Final report: load latest checkpoint, evaluate on full SMNIST test set,
and produce a comprehensive summary.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from smnist_data import load_smnist
from train_bptt import build_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="results/ckpt_overnight_default_10k.pt")
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--aux_head", action="store_true", default=True)
    p.add_argument("--full_test", action="store_true", default=True)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    print(f"Loading checkpoint {args.ckpt} ...", flush=True)
    ckpt = torch.load(args.ckpt, weights_only=False)
    print(f"  ckpt fields: {list(ckpt.keys())}", flush=True)
    print(f"  ckpt epoch: {ckpt.get('epoch', '?')}, ckpt test_acc: {ckpt.get('test_acc', '?')}", flush=True)

    net = build_net(args.arch, aux_head=args.aux_head)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    print(f"\nLoading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if not args.full_test:
        xte, yte = xte[:1000], yte[:1000]
    print(f"  test set: {xte.shape[0]} samples", flush=True)

    print(f"\nEvaluating on full test set ...", flush=True)
    t0 = time.time()
    n_correct = 0; n_total = 0
    pool_logits_correct = 0
    cm = torch.zeros(10, 10, dtype=torch.long)
    with torch.no_grad():
        for s in range(0, xte.shape[0], args.batch):
            xb = xte[s:s+args.batch].unsqueeze(-1)
            yb = yte[s:s+args.batch]
            logits, info = net(xb)
            pred = logits.argmax(dim=1)
            n_correct += int((pred == yb).sum().item())
            n_total += xb.shape[0]
            if "pool_logits" in info:
                pred_pool = info["pool_logits"].argmax(dim=1)
                pool_logits_correct += int((pred_pool == yb).sum().item())
            for i in range(xb.shape[0]):
                cm[yb[i], pred[i]] += 1
    wall = time.time() - t0
    print(f"  done in {wall:.1f}s")
    print(f"\n=== FINAL RESULTS ===")
    print(f"aux head accuracy: {n_correct/n_total:.4f} ({n_correct}/{n_total})")
    if pool_logits_correct > 0:
        print(f"pool_rate readout (aux readout): {pool_logits_correct/n_total:.4f}")

    print(f"\nConfusion matrix (rows=true, cols=pred):")
    for r in range(10):
        row_total = cm[r].sum().item()
        accuracy_per_class = cm[r, r].item() / max(row_total, 1)
        print(f"  {r}: " + " ".join(f"{cm[r,c].item():>4}" for c in range(10)) + f"  | acc {accuracy_per_class:.3f}")


if __name__ == "__main__":
    main()
