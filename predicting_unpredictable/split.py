"""
Storm-wise train/val split utilities.

We split by storm `id` (never by rows) to avoid leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Split:
    version: str
    seed: int
    train_ids: list[str]
    val_ids: list[str]


def save_split(split: Split, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(split), indent=2, sort_keys=True) + "\n")
    return p


def load_split(path: str | Path) -> Split:
    data = json.loads(Path(path).read_text())
    return Split(
        version=str(data["version"]),
        seed=int(data["seed"]),
        train_ids=list(data["train_ids"]),
        val_ids=list(data["val_ids"]),
    )


def assert_disjoint(train_ids: list[str], val_ids: list[str]) -> None:
    inter = set(train_ids) & set(val_ids)
    if inter:
        raise ValueError(
            f"Split leakage: ids appear in both: {sorted(inter)[:10]}"
        )


def make_stormwise_stratified_split(
    *,
    events_csv: str | Path = "data/events.csv",
    val_fraction: float = 0.2,
    seed: int = 20260126,
    version: str = "v1",
) -> Split:
    """
    Create a stratified storm-wise split using `event_type` at the storm level.

    - `events.csv` has 5 rows per storm id (img_type). We take the first row
      per id for the storm-level label.
    """

    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit

    df = pd.read_csv(events_csv, parse_dates=["start_utc"])
    storms = (
        df.groupby("id", as_index=False)
        .first()[["id", "event_type"]]
        .sort_values("id")
        .reset_index(drop=True)
    )
    ids = storms["id"].astype(str).to_numpy()
    y = storms["event_type"].astype(str).to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    (train_idx, val_idx) = next(splitter.split(ids, y))
    train_ids = ids[train_idx].tolist()
    val_ids = ids[val_idx].tolist()
    assert_disjoint(train_ids, val_ids)

    return Split(
        version=version, seed=seed, train_ids=train_ids, val_ids=val_ids
    )
