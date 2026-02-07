"""Project constants (preprocess.py/submission.py shared)."""

from __future__ import annotations

# Channel-wise normalization stats (z-score/minmax/log)
# Updated with EDA-derived actual values (replace legacy assumptions)
NORM_STATS: dict[str, dict[str, float]] = {
    # VIS: EDA-derived z-score + minmax (3σ rule)
    "vis": {
        "mean": 5276.38773,
        "std": (11983.0 - 267.0) / 6,  # 3σ: std = (max - min)/6
        "data_min": 267.0,             # EDA actual min
        "data_max": 11983.0,           # EDA actual max
        "target_min": 0.0,
        "target_max": 1.0,
    },
    # IR069: EDA-derived z-score + minmax (3σ rule)
    "ir069": {
        "mean": -3732.29056,
        "std": (-1257.0 - (-7500.0)) / 6,  # 3σ: std = (max - min)/6
        "data_min": -7500.0,               # EDA actual min
        "data_max": -1257.0,               # EDA actual max
        "target_min": 0.0,
        "target_max": 1.0,
    },
    # IR107: EDA-derived z-score + minmax (3σ rule)
    "ir107": {
        "mean": -2539.46477,
        "std": (2431.0 - (-7422.0)) / 6,   # 3σ: std = (max - min)/6
        "data_min": -7422.0,               # EDA actual min
        "data_max": 2431.0,                # EDA actual max
        "target_min": 0.0,
        "target_max": 1.0,
    },
    # VIL: EDA-derived log normalization (uint8)
    "vil": {
        "data_min": 0.0,        # EDA actual min
        "data_max": 211.0,      # EDA actual max (replace 255.0)
        "eps": 1e-6,            # Avoid log(0)
        "log_scale": True,      # Scale log output to [0,1]
        "target_min": 0.0,
        "target_max": 1.0,
    },
}

# Task 2 baseline UNet normalization (linear transform to [0,1])
# Formula: (x+offset)/scale (IR) | x/scale (VIL)
TASK2_NORM_PARAMS: dict[str, dict[str, float]] = {
    "ir069": {"offset": 7500.0, "scale": 6243.0},  # EDA-derived
    "ir107": {"offset": 7422.0, "scale": 9853.0},  # EDA-derived
    "vil": {"scale": 211.0},                       # EDA-derived (no offset)
}

# Task 3 event classification classes (ordered for label mapping)
EVENT_TYPES: list[str] = [
    "Flash Flood",
    "Flood",
    "Funnel Cloud",
    "Hail",
    "Heavy Rain",
    "Lightning",
    "Thunderstorm Wind",
    "Tornado",
]

__all__ = ["EVENT_TYPES", "NORM_STATS", "TASK2_NORM_PARAMS"]
