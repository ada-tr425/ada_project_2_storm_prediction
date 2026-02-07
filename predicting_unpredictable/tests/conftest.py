"""
Shared pytest fixtures for predicting_unpredictable tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def storm_data():
    """
    Mock storm data dictionary with realistic shapes and dtypes.
    Use deterministic, low-entropy arrays for fast and predictable tests.
    """
    h384, w384, t = 384, 384, 36
    h192, w192 = 192, 192

    vil = np.zeros((h384, w384, t), dtype=np.uint8)
    vis = np.zeros((h384, w384, t), dtype=np.int16)
    ir069 = np.zeros((h192, w192, t), dtype=np.int16)
    ir107 = np.zeros((h192, w192, t), dtype=np.int16)

    # Lightning format: [t_seconds, lat, lon, pixel_x, pixel_y]
    lght = np.zeros((0, 5), dtype=np.float32)

    return {
        "vil": vil,
        "vis": vis,
        "ir069": ir069,
        "ir107": ir107,
        "lght": lght,
    }


@pytest.fixture
def tmp_events_csv(tmp_path):
    """
    Create a temporary events.csv file with mock data.
    Returns the path to the CSV file.

    With 4 event types and 25% val_fraction, we need at least 4 storms in val,
    which means at least 16 storms total, with at least 4 per class.
    """
    # Create mock events data with required columns
    # Each storm has 5 rows (one per img_type)
    # 20 storms total: 5 per event type (Tornado, Hail, Lightning, Flood)
    storm_ids = []
    img_types = []
    event_types = []

    events = ["Tornado", "Hail", "Lightning", "Flood"]
    storm_counter = 1

    for event in events:
        for _ in range(5):  # 5 storms per event type
            storm_id = f"S{storm_counter:06d}"
            for img_type in ["vis", "ir069", "ir107", "vil", "lght"]:
                storm_ids.append(storm_id)
                img_types.append(img_type)
                event_types.append(event)
            storm_counter += 1

    data = {
        "id": storm_ids,
        "img_type": img_types,
        "event_type": event_types,
        "start_utc": ["2020-01-01 00:00:00"] * len(storm_ids),
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "events.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_h5_file(tmp_path):
    """
    Create a temporary HDF5 file with mock storm data.
    Returns the path to the HDF5 file.
    """
    import h5py

    h5_path = tmp_path / "test.h5"
    h384, w384, t = 384, 384, 36
    h192, w192 = 192, 192

    with h5py.File(h5_path, "w") as f:
        # Create a storm group
        grp = f.create_group("S000001")
        grp.create_dataset(
            "vil", data=np.zeros((h384, w384, t), dtype=np.uint8)
        )
        grp.create_dataset(
            "vis", data=np.zeros((h384, w384, t), dtype=np.int16)
        )
        grp.create_dataset(
            "ir069", data=np.zeros((h192, w192, t), dtype=np.int16)
        )
        grp.create_dataset(
            "ir107", data=np.zeros((h192, w192, t), dtype=np.int16)
        )
        grp.create_dataset("lght", data=np.zeros((10, 5), dtype=np.float32))

        # Create another storm group
        grp2 = f.create_group("S000002")
        grp2.create_dataset(
            "vil", data=np.ones((h384, w384, t), dtype=np.uint8)
        )
        grp2.create_dataset(
            "vis", data=np.ones((h384, w384, t), dtype=np.int16)
        )
        grp2.create_dataset(
            "ir069", data=np.ones((h192, w192, t), dtype=np.int16)
        )
        grp2.create_dataset(
            "ir107", data=np.ones((h192, w192, t), dtype=np.int16)
        )
        grp2.create_dataset("lght", data=np.ones((5, 5), dtype=np.float32))

    return h5_path
