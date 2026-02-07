"""
Reusable metrics helpers.

Note: official surprise-storm ranking uses L1 (absolute) error.
"""

from __future__ import annotations

import numpy as np


def mae_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error for numpy arrays."""

    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred={pred.shape} target={target.shape}"
        )
    return float(np.mean(np.abs(pred - target)))


def accuracy_from_logits_numpy(
    logits: np.ndarray, targets: np.ndarray
) -> float:
    """
    Classification accuracy.

    - logits: (B, C)
    - targets: (B,) integer labels
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (B,C). Got {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D (B,). Got {targets.shape}")
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == targets))


def macro_f1_numpy(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    labels: list[int] | None = None,
) -> float:
    """
    Macro-F1 using sklearn (optional but included in repo requirements).

    - targets: (B,) integer labels
    - preds: (B,) integer labels
    """

    from sklearn.metrics import f1_score

    return float(f1_score(targets, preds, average="macro", labels=labels))
