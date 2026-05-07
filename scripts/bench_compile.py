"""Test torch.compile() speedup on the resonator forward."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from hrn import HRNConfig, LayerSpec, HierarchicalResonantNet, make_default_config


def bench_one(net, x, y, crit, repeat=3, backward=False):
    if backward:
        # warmup
        logits, _ = net(x); loss = crit(logits, y); loss.backward()
        for p in net.parameters():
            if p.grad is not None: p.grad = None
    else:
        with torch.no_grad():
            net(x)
    times = []
    for _ in range(repeat):
        t0 = time.time()
        if backward:
            for p in net.parameters():
                if p.grad is not None: p.grad = None
            logits, _ = net(x); loss = crit(logits, y); loss.backward()
        else:
            with torch.no_grad():
                net(x)
        times.append(time.time() - t0)
    return min(times)


def main():
    torch.manual_seed(0)
    cfg = make_default_config()
    net_eager = HierarchicalResonantNet(cfg)
    net_compiled = HierarchicalResonantNet(cfg)
    net_compiled.load_state_dict(net_eager.state_dict())

    crit = torch.nn.CrossEntropyLoss()
    x = torch.rand(32, 392, 1)
    y = torch.zeros(32, dtype=torch.long)

    print("eager fwd:", f"{bench_one(net_eager, x, y, crit, backward=False):.3f}", flush=True)
    print("eager bwd:", f"{bench_one(net_eager, x, y, crit, backward=True):.3f}", flush=True)

    try:
        net_c = torch.compile(net_compiled, mode="reduce-overhead", dynamic=False)
        # warmup compiles
        with torch.no_grad():
            net_c(x)
        print("compiled fwd:", f"{bench_one(net_c, x, y, crit, backward=False):.3f}", flush=True)
        print("compiled bwd:", f"{bench_one(net_c, x, y, crit, backward=True):.3f}", flush=True)
    except Exception as e:
        print(f"compile failed: {e}", flush=True)


if __name__ == "__main__":
    main()
