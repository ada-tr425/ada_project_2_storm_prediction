"""
Small, reusable IO utilities (seed, device, checkpoint).

These are intended to be called from notebooks.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def seed_everything(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch (if installed)."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        # Allow using the helpers without torch (e.g. pure data exploration).
        pass


def get_device(prefer_cuda: bool = True):
    """Return a torch.device if torch exists, else return None."""

    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ModuleNotFoundError:
        return None


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer=None,
    epoch: int | None = None,
    global_step: int | None = None,
    best_metric: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Save a training checkpoint.

    Convention:
    - Save *per task* to avoid overwriting between tasks.
    - Usually write to: outputs/checkpoints/task{k}/best.pt
    """

    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    if global_step is not None:
        ckpt["global_step"] = int(global_step)
    if best_metric is not None:
        ckpt["best_metric"] = float(best_metric)
    if extra:
        ckpt["extra"] = dict(extra)
    torch.save(ckpt, p)
    return p


def load_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer=None,
    map_location: str | None = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint saved by `save_checkpoint` and restore state dicts."""

    import torch

    ckpt = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
