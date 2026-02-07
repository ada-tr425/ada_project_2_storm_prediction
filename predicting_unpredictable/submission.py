"""
Submission helpers aligned with the official Surprise storms checker notebook.

This module does NOT replace the official checker; it helps you generate files
that will pass it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .constants import EVENT_TYPES

TaskTag = Literal["task1", "task2", "task3", "task4"]


def task_tag(task: int | TaskTag) -> TaskTag:
    """Convert task int/tag to standardized TaskTag string."""
    if isinstance(task, str):
        if task not in ("task1", "task2", "task3", "task4"):
            raise ValueError(f"Unknown task tag: {task}")
        return task
    if task == 1:
        return "task1"
    if task == 2:
        return "task2"
    if task == 3:
        return "task3"
    if task == 4:
        return "task4"
    raise ValueError(f"Unknown task: {task}")


def validate_prediction(task: int | TaskTag, arr: np.ndarray) -> None:
    """
    Validate a prediction array against the official checker rules.

    Raises ValueError/AssertionError on mismatch.
    """
    tag = task_tag(task)
    if tag == "task1":
        assert arr.shape == (384, 384, 12), (
            "task1: wrong shape (expected (384, 384, 12))"
        )
        assert arr.dtype == np.float32, "task1: wrong dtype (expected float32)"
        return
    if tag == "task2":
        assert arr.shape == (384, 384, 36), (
            "task2: wrong shape (expected (384, 384, 36))"
        )
        assert arr.dtype == np.float32, "task2: wrong dtype (expected float32)"
        return
    if tag == "task3":
        assert arr.shape == (), "task3: output must be a scalar array"
        # Official checker uses `str(x)` membership.
        assert str(arr) in EVENT_TYPES, "task3: invalid class string"
        return
    if tag == "task4":
        assert arr.ndim == 2 and arr.shape[1] == 3, (
            "task4: wrong shape (expected 2D with 3 columns)"
        )
        assert arr.dtype == np.float32, "task4: wrong dtype (expected float32)"
        return
    raise ValueError(f"Unknown task tag: {tag}")


def save_prediction(
    *,
    out_dir: str | Path,
    team_name: str,
    task: int | TaskTag,
    storm_id: str,
    arr: np.ndarray,
    validate: bool = True,
) -> Path:
    """
    Save a single prediction file using the required naming convention:
      <team-name>-task{k}-<storm-id>.npy
    """
    tag = task_tag(task)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure exact dtype requirements (task3 is special: string/object).
    if tag in ("task1", "task2", "task4"):
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

    if validate:
        validate_prediction(tag, arr)

    path = out_dir_path / f"{team_name}-{tag}-{storm_id}.npy"
    np.save(path, arr)
    return path


def decode_event_type(idx: int) -> str:
    """Decode integer class index to event type string (Task 3)."""
    n_classes = len(EVENT_TYPES)
    if not (0 <= idx < n_classes):
        raise ValueError(
            f"Invalid event class index: {idx} (must be 0-{n_classes - 1})"
        )
    return EVENT_TYPES[idx]


def denormalize_tensor(
    t: np.ndarray | torch.Tensor,
    *,
    norm_type: str,
    mean: float | None = None,
    std: float | None = None,
    data_min: float | None = None,
    data_max: float | None = None,
    target_min: float = 0.0,
    target_max: float = 1.0,
    eps: float = 1e-6,
    log_scale: bool = True
) -> np.ndarray | torch.Tensor:
    """Denormalize tensor (zscore/minmax/log). Compatible with numpy/torch."""
    # Input validation
    valid_types = ["zscore", "minmax", "log"]
    norm_type = norm_type.lower()
    if norm_type not in valid_types:
        raise ValueError(
            f"Invalid norm_type: {norm_type} (choose {valid_types})"
        )

    # Helper: Get exp function (compatible with numpy/torch)
    def exp_func(x):
        return torch.exp(x) if isinstance(x, torch.Tensor) else np.exp(x)

    # Helper: Get clip function (compatible with numpy/torch)
    def clip_func(x, min_val, max_val):
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min=min_val, max=max_val)
        return np.clip(x, a_min=min_val, a_max=max_val)

    # 1. Denormalize Z-SCORE
    if norm_type == "zscore":
        if mean is None or std is None:
            raise ValueError(
                "mean/std required for zscore denormalization"
            )
        return t * float(std) + float(mean)

    # 2. Denormalize MINMAX
    elif norm_type == "minmax":
        if data_min is None or data_max is None:
            raise ValueError(
                "data_min/data_max required for minmax denormalization"
            )
        data_min, data_max = float(data_min), float(data_max)
        if data_max == data_min:
            raise ValueError(
                "data_max cannot equal data_min (division by zero error)"
            )

        scaled = (t - target_min) / (target_max - target_min)
        return scaled * (data_max - data_min) + data_min

    # 3. Denormalize LOG
    elif norm_type == "log":
        if data_min is None or data_max is None:
            raise ValueError(
                "data_min/data_max required for log denormalization"
            )
        data_min, data_max = float(data_min), float(data_max)

        # Reverse log scaling (if applied)
        if log_scale:
            log_min = np.log(data_min + eps)
            log_max = np.log(data_max + eps)
            if log_max == log_min:
                raise ValueError("log_max == log_min (no variance after log)")
            t_log = (
                (t - target_min) / (target_max - target_min)
                * (log_max - log_min)
                + log_min
            )
        else:
            t_log = t

        # Reverse log transformation
        t_denorm = exp_func(t_log) - eps

        # Safety clip to original data range
        return clip_func(t_denorm, min_val=data_min, max_val=data_max)
