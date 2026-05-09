"""DVS Gesture frame cache for local-rule benchmarks."""

from __future__ import annotations

from pathlib import Path

import torch


DEFAULT_DATA_ROOTS = [
    Path("/Users/esi/research/biosuccess/data"),
    Path("/Users/esi/research/biosnn2/data"),
    Path("/Users/esi/research/biosnn3/data"),
    Path("/Users/esi/research/pmnsit/data"),
]


def _find_data_root() -> Path:
    for root in DEFAULT_DATA_ROOTS:
        if (root / "DVSGesture").exists():
            return root
    raise FileNotFoundError(
        "No local DVSGesture folder found. Checked: "
        + ", ".join(str(p / "DVSGesture") for p in DEFAULT_DATA_ROOTS)
    )


def _build_split(train: bool, data_root: Path, time_bins: int, spatial: int):
    import tonic
    import tonic.transforms as transforms

    sensor_size = tonic.datasets.DVSGesture.sensor_size
    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(sensor_size=sensor_size, target_size=(spatial, spatial)),
        transforms.ToFrame(sensor_size=(spatial, spatial, 2), n_time_bins=time_bins),
    ])
    ds = tonic.datasets.DVSGesture(save_to=str(data_root), train=train, transform=transform)
    xs = []
    ys = []
    tag = "train" if train else "test"
    for i in range(len(ds)):
        frames, label = ds[i]
        x = torch.as_tensor(frames, dtype=torch.float32).clamp_(0.0, 1.0)
        xs.append(x)
        ys.append(int(label))
        if i % 100 == 0:
            print(f"  {tag}: cached {i}/{len(ds)}", flush=True)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def _limit_split(x, y, limit, seed):
    if limit <= 0 or limit >= x.shape[0]:
        return x, y
    order = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))
    idx = order[:limit]
    return x[idx].contiguous(), y[idx].contiguous()


def load_dvsgesture(
    cache_dir: str | Path,
    time_bins: int = 12,
    spatial: int = 32,
    train_limit: int = 0,
    test_limit: int = 0,
    seed: int = 20260508,
    data_root: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return binary/clipped frames `(N,T,2,H,W)` and labels.

    The cache is local to this experiment folder. The raw DVS Gesture
    archive is read from an existing local tonic-compatible directory.
    """
    if data_root is None:
        data_root = _find_data_root()
    else:
        data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"dvsgesture_T{time_bins}_S{spatial}_binary.pt"

    if cache_path.exists():
        xtr, ytr, xte, yte = torch.load(cache_path, weights_only=False)
    else:
        print(f"Building DVS Gesture cache at {cache_path}", flush=True)
        xtr, ytr = _build_split(True, data_root, time_bins, spatial)
        xte, yte = _build_split(False, data_root, time_bins, spatial)
        torch.save((xtr, ytr, xte, yte), cache_path)

    xtr, ytr = _limit_split(xtr, ytr, train_limit, seed)
    xte, yte = _limit_split(xte, yte, test_limit, seed + 1)
    return xtr, ytr, xte, yte


NUM_CLASSES = 11
