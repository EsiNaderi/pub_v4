"""Plot training trajectory from a CSV log."""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/run_overnight_default_10k.csv")
    p.add_argument("--out", type=str, default="figures/training_trajectory.png")
    args = p.parse_args()

    rows = []
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"No data in {args.csv}")
        return

    # Print as text table
    print(f"\nTraining trajectory ({args.csv}):")
    print(f"{'epoch':>5} {'step':>5} {'wall':>7} {'tr_loss':>8} {'tr_acc':>7} {'te_loss':>8} {'te_acc':>7}")
    for r in rows:
        print(f"{r['epoch']:>5} {r['step']:>5} {r['wall']:>7} "
              f"{r['train_loss']:>8} {r['train_acc']:>7} "
              f"{r['test_loss']:>8} {r['test_acc']:>7}")

    # Try matplotlib plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(r["epoch"]) for r in rows]
        tr_loss = [float(r["train_loss"]) for r in rows]
        tr_acc = [float(r["train_acc"]) for r in rows]
        te_loss = [float(r["test_loss"]) for r in rows]
        te_acc = [float(r["test_acc"]) for r in rows]

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(epochs, tr_loss, label="train", marker="o")
        ax[0].plot(epochs, te_loss, label="test", marker="s")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss (cross-entropy)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].plot(epochs, tr_acc, label="train", marker="o")
        ax[1].plot(epochs, te_acc, label="test", marker="s")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        ax[1].axhline(y=0.95, color="r", linestyle="--", alpha=0.4, label="target 95%")

        plt.suptitle(f"BPTT training: {Path(args.csv).stem}")
        plt.tight_layout()
        plt.savefig(args.out, dpi=120)
        print(f"\nPlot saved to {args.out}")
    except Exception as e:
        print(f"\n(plotting failed: {e})")


if __name__ == "__main__":
    main()
