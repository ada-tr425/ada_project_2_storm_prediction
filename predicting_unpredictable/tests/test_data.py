"""
Tests for predicting_unpredictable.data module.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from predicting_unpredictable.data import (
    HF_REPO_ID,
    SURPRISE_FILES,
    TRAIN_FILES,
    download_files,
    download_surprise_data,
    download_training_data,
    ensure_dir,
    load_event_arrays,
    open_h5,
    read_events_csv,
    to_numpy_mapping,
    unique_storm_ids,
)


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_folder"
        assert not new_dir.exists()
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_creates_nested_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        result = ensure_dir(nested)
        assert nested.exists()
        assert result == nested

    def test_ensure_dir_existing_directory(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_dir(existing)
        assert existing.exists()
        assert result == existing

    def test_ensure_dir_returns_path_object(self, tmp_path):
        result = ensure_dir(str(tmp_path / "test"))
        assert isinstance(result, Path)


class TestDownloadFiles:
    """Tests for download_files function with mocked HuggingFace Hub."""

    @patch("huggingface_hub.hf_hub_download")
    def test_download_files_calls_hf_hub(self, mock_download, tmp_path):
        filenames = ["file1.h5", "file2.csv"]
        download_files(filenames, local_dir=tmp_path)

        assert mock_download.call_count == 2
        for call, fn in zip(mock_download.call_args_list, filenames):
            assert call.kwargs["filename"] == fn
            assert call.kwargs["repo_id"] == HF_REPO_ID
            assert call.kwargs["repo_type"] == "dataset"

    @patch("huggingface_hub.hf_hub_download")
    def test_download_files_returns_paths(self, mock_download, tmp_path):
        filenames = ["file1.h5", "file2.csv"]
        result = download_files(filenames, local_dir=tmp_path)

        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)
        assert result[0] == tmp_path / "file1.h5"
        assert result[1] == tmp_path / "file2.csv"

    @patch("huggingface_hub.hf_hub_download")
    def test_download_files_custom_repo_id(self, mock_download, tmp_path):
        custom_repo = "custom/repo"
        download_files(["test.h5"], local_dir=tmp_path, repo_id=custom_repo)

        mock_download.assert_called_once()
        assert mock_download.call_args.kwargs["repo_id"] == custom_repo


class TestDownloadTrainingData:
    """Tests for download_training_data function."""

    @patch("predicting_unpredictable.data.download_files")
    def test_download_training_data_calls_download_files(self, mock_download):
        download_training_data(local_dir="test_dir")
        mock_download.assert_called_once_with(
            TRAIN_FILES, local_dir="test_dir", repo_id=HF_REPO_ID
        )

    def test_train_files_contains_expected_files(self):
        assert "train.h5" in TRAIN_FILES
        assert "events.csv" in TRAIN_FILES


class TestDownloadSurpriseData:
    """Tests for download_surprise_data function."""

    @patch("predicting_unpredictable.data.download_files")
    def test_download_surprise_data_calls_download_files(self, mock_download):
        download_surprise_data(local_dir="test_dir")
        mock_download.assert_called_once_with(
            SURPRISE_FILES, local_dir="test_dir", repo_id=HF_REPO_ID
        )

    def test_surprise_files_contains_expected_files(self):
        # Check for surprise h5 files
        for i in range(1, 5):
            assert f"surprise-task{i}.h5" in SURPRISE_FILES
            assert f"surprise-events{i}.csv" in SURPRISE_FILES


class TestReadEventsCsv:
    """Tests for read_events_csv function."""

    def test_read_events_csv_returns_dataframe(self, tmp_events_csv):
        df = read_events_csv(tmp_events_csv)
        assert isinstance(df, pd.DataFrame)

    def test_read_events_csv_has_expected_columns(self, tmp_events_csv):
        df = read_events_csv(tmp_events_csv)
        assert "id" in df.columns
        assert "event_type" in df.columns
        assert "start_utc" in df.columns

    def test_read_events_csv_parses_dates(self, tmp_events_csv):
        df = read_events_csv(tmp_events_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["start_utc"])


class TestUniqueStormIds:
    """Tests for unique_storm_ids function."""

    def test_unique_storm_ids_returns_sorted_list(self):
        df = pd.DataFrame({
            "id": ["S003", "S001", "S002", "S001", "S003"]
        })
        result = unique_storm_ids(df)
        assert result == ["S001", "S002", "S003"]

    def test_unique_storm_ids_returns_list(self):
        df = pd.DataFrame({"id": ["S001", "S002"]})
        result = unique_storm_ids(df)
        assert isinstance(result, list)

    def test_unique_storm_ids_converts_to_string(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = unique_storm_ids(df)
        assert all(isinstance(s, str) for s in result)

    def test_unique_storm_ids_empty_dataframe(self):
        df = pd.DataFrame({"id": []})
        result = unique_storm_ids(df)
        assert result == []


class TestOpenH5:
    """Tests for open_h5 function."""

    def test_open_h5_returns_file_object(self, mock_h5_file):
        import h5py

        with open_h5(mock_h5_file) as f:
            assert isinstance(f, h5py.File)

    def test_open_h5_default_mode_is_read(self, mock_h5_file):
        with open_h5(mock_h5_file) as f:
            assert f.mode == "r"

    def test_open_h5_can_read_groups(self, mock_h5_file):
        with open_h5(mock_h5_file) as f:
            assert "S000001" in f
            assert "S000002" in f

    def test_open_h5_accepts_string_path(self, mock_h5_file):
        with open_h5(str(mock_h5_file)) as f:
            assert "S000001" in f


class TestLoadEventArrays:
    """Tests for load_event_arrays function."""

    def test_load_event_arrays_returns_dict(self, mock_h5_file):
        with open_h5(mock_h5_file) as f:
            result = load_event_arrays(
                f, storm_id="S000001", img_types=["vil"]
            )
        assert isinstance(result, dict)

    def test_load_event_arrays_loads_requested_types(self, mock_h5_file):
        with open_h5(mock_h5_file) as f:
            result = load_event_arrays(
                f, storm_id="S000001", img_types=["vil", "vis", "ir069"]
            )
        assert set(result.keys()) == {"vil", "vis", "ir069"}

    def test_load_event_arrays_returns_numpy_arrays(self, mock_h5_file):
        with open_h5(mock_h5_file) as f:
            result = load_event_arrays(
                f, storm_id="S000001", img_types=["vil", "lght"]
            )
        assert all(isinstance(v, np.ndarray) for v in result.values())

    def test_load_event_arrays_correct_shapes(self, mock_h5_file):
        with open_h5(mock_h5_file) as f:
            result = load_event_arrays(
                f,
                storm_id="S000001",
                img_types=["vil", "vis", "ir069", "ir107", "lght"],
            )
        assert result["vil"].shape == (384, 384, 36)
        assert result["vis"].shape == (384, 384, 36)
        assert result["ir069"].shape == (192, 192, 36)
        assert result["ir107"].shape == (192, 192, 36)
        assert result["lght"].shape == (10, 5)

    def test_load_event_arrays_different_storm(self, mock_h5_file):
        with open_h5(mock_h5_file) as f:
            result = load_event_arrays(
                f, storm_id="S000002", img_types=["vil"]
            )
        # S000002 was created with np.ones
        assert result["vil"][0, 0, 0] == 1


class TestToNumpyMapping:
    """Tests for to_numpy_mapping function."""

    def test_to_numpy_mapping_returns_dict(self):
        input_dict = {"a": np.array([1, 2, 3])}
        result = to_numpy_mapping(input_dict)
        assert isinstance(result, dict)

    def test_to_numpy_mapping_converts_values(self):
        input_dict = {"a": [1, 2, 3], "b": (4, 5, 6)}
        result = to_numpy_mapping(input_dict)
        assert isinstance(result["a"], np.ndarray)
        assert isinstance(result["b"], np.ndarray)

    def test_to_numpy_mapping_preserves_numpy_arrays(self):
        arr = np.array([1, 2, 3])
        input_dict = {"a": arr}
        result = to_numpy_mapping(input_dict)
        np.testing.assert_array_equal(result["a"], arr)

    def test_to_numpy_mapping_preserves_keys(self):
        input_dict = {"key1": [1], "key2": [2], "key3": [3]}
        result = to_numpy_mapping(input_dict)
        assert set(result.keys()) == {"key1", "key2", "key3"}

    def test_to_numpy_mapping_empty_dict(self):
        result = to_numpy_mapping({})
        assert result == {}
