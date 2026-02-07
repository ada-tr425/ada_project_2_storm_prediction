"""
Design goals:

- Keep the repository structure simple: notebooks contain model/training code.
- This module provides only reusable utilities: download + lightweight IO
  helpers.

Official dataset source: benmoseley/ese-dl-2025-26-group-project (HuggingFace)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

HF_REPO_ID = "benmoseley/ese-dl-2025-26-group-project"

TRAIN_FILES = ("train.h5", "events.csv")
SURPRISE_FILES = (
    "surprise-task1.h5",
    "surprise-task2.h5",
    "surprise-task3.h5",
    "surprise-task4.h5",
    "surprise-events1.csv",
    "surprise-events2.csv",
    "surprise-events3.csv",
    "surprise-events4.csv",
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_files(
    filenames: Iterable[str],
    *,
    local_dir: str | Path = "data",
    repo_id: str = HF_REPO_ID,
) -> list[Path]:
    """
    Download dataset files from HuggingFace Hub into `local_dir`.

    Notes:
    - We intentionally keep downloads out of import-time side effects.
    """

    from huggingface_hub import hf_hub_download

    local_dir_path = ensure_dir(local_dir)
    out: list[Path] = []
    for fn in filenames:
        hf_hub_download(
            repo_id=repo_id,
            filename=fn,
            repo_type="dataset",
            local_dir=str(local_dir_path),
        )
        out.append(local_dir_path / fn)
    return out


def download_training_data(
    *, local_dir: str | Path = "data", repo_id: str = HF_REPO_ID
) -> list[Path]:
    return download_files(TRAIN_FILES, local_dir=local_dir, repo_id=repo_id)


def download_surprise_data(
    *, local_dir: str | Path = "data", repo_id: str = HF_REPO_ID
) -> list[Path]:
    return download_files(SURPRISE_FILES, local_dir=local_dir, repo_id=repo_id)


def read_events_csv(path: str | Path = "data/events.csv"):
    """
    Read `events.csv` with `start_utc` parsed.

    Returns a pandas.DataFrame (imported lazily to keep base deps light).
    """

    import pandas as pd

    return pd.read_csv(path, parse_dates=["start_utc"])


def unique_storm_ids(events_df) -> list[str]:
    """Return unique storm ids from a pandas dataframe with an `id` column."""

    return sorted(events_df["id"].astype(str).unique().tolist())


def open_h5(path: str | Path, mode: str = "r"):
    """Open an HDF5 file using h5py (lazy import)."""

    import h5py

    return h5py.File(str(path), mode)


def load_event_arrays(
    h5_file,
    *,
    storm_id: str,
    img_types: Iterable[str],
) -> dict[str, np.ndarray]:
    """
    Load one storm from an opened h5 file.

    Args:
        h5_file: an opened h5py.File
        storm_id: like 'S778114'
        img_types: iterable of dataset names, e.g.
            ['vis', 'ir069', 'ir107', 'vil', 'lght']
    """

    grp = h5_file[storm_id]
    out: dict[str, np.ndarray] = {}
    for t in img_types:
        out[str(t)] = grp[str(t)][:]
    return out


def to_numpy_mapping(x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Utility: normalize mapping to a regular dict of numpy arrays."""

    return {k: np.asarray(v) for k, v in x.items()}
