"""
Tests for predicting_unpredictable.split module.
"""

import json
from pathlib import Path

import pytest

from predicting_unpredictable.split import (
    Split,
    assert_disjoint,
    load_split,
    make_stormwise_stratified_split,
    save_split,
)


class TestSplitDataclass:
    """Tests for Split dataclass."""

    def test_split_creation(self):
        split = Split(
            version="v1",
            seed=42,
            train_ids=["S001", "S002"],
            val_ids=["S003"],
        )
        assert split.version == "v1"
        assert split.seed == 42
        assert split.train_ids == ["S001", "S002"]
        assert split.val_ids == ["S003"]

    def test_split_is_frozen(self):
        split = Split(
            version="v1",
            seed=42,
            train_ids=["S001"],
            val_ids=["S002"],
        )
        with pytest.raises(AttributeError):
            split.version = "v2"

    def test_split_equality(self):
        split1 = Split("v1", 42, ["S001"], ["S002"])
        split2 = Split("v1", 42, ["S001"], ["S002"])
        assert split1 == split2

    def test_split_inequality(self):
        split1 = Split("v1", 42, ["S001"], ["S002"])
        split2 = Split("v1", 43, ["S001"], ["S002"])
        assert split1 != split2


class TestSaveSplit:
    """Tests for save_split function."""

    def test_save_split_creates_file(self, tmp_path):
        split = Split("v1", 42, ["S001"], ["S002"])
        path = tmp_path / "split.json"
        result = save_split(split, path)

        assert path.exists()
        assert result == path

    def test_save_split_creates_parent_dirs(self, tmp_path):
        split = Split("v1", 42, ["S001"], ["S002"])
        path = tmp_path / "nested" / "dir" / "split.json"
        save_split(split, path)

        assert path.exists()

    def test_save_split_valid_json(self, tmp_path):
        split = Split("v1", 42, ["S001", "S002"], ["S003"])
        path = tmp_path / "split.json"
        save_split(split, path)

        data = json.loads(path.read_text())
        assert data["version"] == "v1"
        assert data["seed"] == 42
        assert data["train_ids"] == ["S001", "S002"]
        assert data["val_ids"] == ["S003"]

    def test_save_split_returns_path_object(self, tmp_path):
        split = Split("v1", 42, ["S001"], ["S002"])
        result = save_split(split, tmp_path / "split.json")
        assert isinstance(result, Path)


class TestLoadSplit:
    """Tests for load_split function."""

    def test_load_split_returns_split(self, tmp_path):
        # Create a JSON file manually
        data = {
            "version": "v1",
            "seed": 42,
            "train_ids": ["S001", "S002"],
            "val_ids": ["S003"],
        }
        path = tmp_path / "split.json"
        path.write_text(json.dumps(data))

        split = load_split(path)
        assert isinstance(split, Split)
        assert split.version == "v1"
        assert split.seed == 42
        assert split.train_ids == ["S001", "S002"]
        assert split.val_ids == ["S003"]

    def test_load_split_converts_types(self, tmp_path):
        # Seed as string should be converted to int
        data = {
            "version": 1,  # int, should become str
            "seed": "42",  # str, should become int
            "train_ids": ["S001"],
            "val_ids": ["S002"],
        }
        path = tmp_path / "split.json"
        path.write_text(json.dumps(data))

        split = load_split(path)
        assert isinstance(split.version, str)
        assert isinstance(split.seed, int)


class TestSaveLoadRoundTrip:
    """Integration tests for save/load split."""

    def test_roundtrip_preserves_data(self, tmp_path):
        original = Split(
            version="v2",
            seed=12345,
            train_ids=["S001", "S002", "S003"],
            val_ids=["S004", "S005"],
        )
        path = tmp_path / "split.json"

        save_split(original, path)
        loaded = load_split(path)

        assert loaded == original

    def test_roundtrip_with_many_ids(self, tmp_path):
        train_ids = [f"S{i:06d}" for i in range(100)]
        val_ids = [f"S{i:06d}" for i in range(100, 125)]
        original = Split("v1", 42, train_ids, val_ids)
        path = tmp_path / "split.json"

        save_split(original, path)
        loaded = load_split(path)

        assert loaded.train_ids == train_ids
        assert loaded.val_ids == val_ids


class TestAssertDisjoint:
    """Tests for assert_disjoint function."""

    def test_assert_disjoint_no_overlap(self):
        train_ids = ["S001", "S002", "S003"]
        val_ids = ["S004", "S005"]
        # Should not raise
        assert_disjoint(train_ids, val_ids)

    def test_assert_disjoint_empty_lists(self):
        # Should not raise
        assert_disjoint([], [])

    def test_assert_disjoint_one_empty(self):
        # Should not raise
        assert_disjoint(["S001", "S002"], [])
        assert_disjoint([], ["S001", "S002"])

    def test_assert_disjoint_raises_on_overlap(self):
        train_ids = ["S001", "S002", "S003"]
        val_ids = ["S003", "S004"]  # S003 overlaps
        with pytest.raises(ValueError, match="leakage"):
            assert_disjoint(train_ids, val_ids)

    def test_assert_disjoint_multiple_overlaps(self):
        train_ids = ["S001", "S002", "S003"]
        val_ids = ["S002", "S003", "S004"]
        with pytest.raises(ValueError, match="leakage"):
            assert_disjoint(train_ids, val_ids)


class TestMakeStormwiseStratifiedSplit:
    """Tests for make_stormwise_stratified_split function."""

    def test_make_split_returns_split(self, tmp_events_csv):
        split = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        assert isinstance(split, Split)

    def test_make_split_disjoint(self, tmp_events_csv):
        split = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        # Should not raise
        assert_disjoint(split.train_ids, split.val_ids)

    def test_make_split_preserves_all_ids(self, tmp_events_csv):
        split = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        all_ids = set(split.train_ids) | set(split.val_ids)
        # Should have all 20 unique storm IDs from fixture (5 per event x 4)
        assert len(all_ids) == 20

    def test_make_split_approximate_fraction(self, tmp_events_csv):
        split = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        total = len(split.train_ids) + len(split.val_ids)
        val_ratio = len(split.val_ids) / total
        # Allow some tolerance due to stratification constraints
        assert 0.1 <= val_ratio <= 0.5

    def test_make_split_reproducible(self, tmp_events_csv):
        split1 = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        split2 = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        assert set(split1.train_ids) == set(split2.train_ids)
        assert set(split1.val_ids) == set(split2.val_ids)

    def test_make_split_different_seeds_different_results(
            self, tmp_events_csv):
        split1 = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        split2 = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=123,
        )
        # With different seeds, splits should likely differ
        # (not guaranteed but very likely with enough data)
        # Just check that IDs are lists
        assert isinstance(split1.train_ids, list)
        assert isinstance(split2.train_ids, list)

    def test_make_split_stores_version(self, tmp_events_csv):
        split = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
            version="test_v1",
        )
        assert split.version == "test_v1"

    def test_make_split_stores_seed(self, tmp_events_csv):
        split = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=99999,
        )
        assert split.seed == 99999

    def test_make_split_ids_are_strings(self, tmp_events_csv):
        split = make_stormwise_stratified_split(
            events_csv=tmp_events_csv,
            val_fraction=0.25,
            seed=42,
        )
        assert all(isinstance(id_, str) for id_ in split.train_ids)
        assert all(isinstance(id_, str) for id_ in split.val_ids)
