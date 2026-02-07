"""
Tests for predicting_unpredictable.submission module.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from predicting_unpredictable.constants import EVENT_TYPES
from predicting_unpredictable.submission import (
    decode_event_type,
    denormalize_tensor,
    save_prediction,
    task_tag,
    validate_prediction,
)


class TestTaskTag:
    """Tests for task_tag function."""

    @pytest.mark.parametrize("task_int,expected", [
        (1, "task1"),
        (2, "task2"),
        (3, "task3"),
        (4, "task4"),
    ])
    def test_task_tag_from_int(self, task_int, expected):
        assert task_tag(task_int) == expected

    @pytest.mark.parametrize("task_str", ["task1", "task2", "task3", "task4"])
    def test_task_tag_from_string(self, task_str):
        assert task_tag(task_str) == task_str

    def test_task_tag_invalid_int_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            task_tag(5)

    def test_task_tag_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown task tag"):
            task_tag("task5")

    def test_task_tag_zero_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            task_tag(0)


class TestValidatePrediction:
    """Tests for validate_prediction function."""

    # Task 1 tests
    def test_validate_task1_valid(self):
        arr = np.zeros((384, 384, 12), dtype=np.float32)
        validate_prediction(1, arr)

    def test_validate_task1_wrong_shape_raises(self):
        arr = np.zeros((384, 384, 24), dtype=np.float32)
        with pytest.raises(AssertionError, match="wrong shape"):
            validate_prediction(1, arr)

    def test_validate_task1_wrong_dtype_raises(self):
        arr = np.zeros((384, 384, 12), dtype=np.float64)
        with pytest.raises(AssertionError, match="wrong dtype"):
            validate_prediction(1, arr)

    # Task 2 tests
    def test_validate_task2_valid(self):
        arr = np.zeros((384, 384, 36), dtype=np.float32)
        validate_prediction(2, arr)

    def test_validate_task2_wrong_shape_raises(self):
        arr = np.zeros((384, 384, 12), dtype=np.float32)
        with pytest.raises(AssertionError, match="wrong shape"):
            validate_prediction(2, arr)

    def test_validate_task2_wrong_dtype_raises(self):
        arr = np.zeros((384, 384, 36), dtype=np.int32)
        with pytest.raises(AssertionError, match="wrong dtype"):
            validate_prediction(2, arr)

    # Task 3 tests
    def test_validate_task3_valid(self):
        for event_type in EVENT_TYPES:
            arr = np.array(event_type)
            validate_prediction(3, arr)

    def test_validate_task3_wrong_shape_raises(self):
        arr = np.array(["Tornado", "Hail"])  # 1D array, not scalar
        with pytest.raises(AssertionError, match="scalar"):
            validate_prediction(3, arr)

    def test_validate_task3_invalid_class_raises(self):
        arr = np.array("InvalidEvent")
        with pytest.raises(AssertionError, match="invalid class"):
            validate_prediction(3, arr)

    # Task 4 tests
    def test_validate_task4_valid(self):
        arr = np.zeros((10, 3), dtype=np.float32)
        validate_prediction(4, arr)

    def test_validate_task4_empty_valid(self):
        arr = np.zeros((0, 3), dtype=np.float32)
        validate_prediction(4, arr)

    def test_validate_task4_wrong_columns_raises(self):
        arr = np.zeros((10, 5), dtype=np.float32)
        with pytest.raises(AssertionError, match="wrong shape"):
            validate_prediction(4, arr)

    def test_validate_task4_wrong_dtype_raises(self):
        arr = np.zeros((10, 3), dtype=np.float64)
        with pytest.raises(AssertionError, match="wrong dtype"):
            validate_prediction(4, arr)

    # String task tag tests
    def test_validate_with_string_tag(self):
        arr = np.zeros((384, 384, 12), dtype=np.float32)
        validate_prediction("task1", arr)


class TestSavePrediction:
    """Tests for save_prediction function."""

    def test_save_prediction_creates_file(self, tmp_path):
        arr = np.zeros((384, 384, 12), dtype=np.float32)
        path = save_prediction(
            out_dir=tmp_path,
            team_name="testteam",
            task=1,
            storm_id="S123456",
            arr=arr,
        )
        assert path.exists()

    def test_save_prediction_correct_filename(self, tmp_path):
        arr = np.zeros((384, 384, 12), dtype=np.float32)
        path = save_prediction(
            out_dir=tmp_path,
            team_name="sally",
            task=1,
            storm_id="S851230",
            arr=arr,
        )
        assert path.name == "sally-task1-S851230.npy"

    def test_save_prediction_creates_output_dir(self, tmp_path):
        arr = np.zeros((384, 384, 12), dtype=np.float32)
        out_dir = tmp_path / "nested" / "predictions"
        save_prediction(
            out_dir=out_dir,
            team_name="team",
            task=1,
            storm_id="S000001",
            arr=arr,
        )
        assert out_dir.exists()

    def test_save_prediction_converts_dtype(self, tmp_path):
        # Float64 should be converted to float32
        arr = np.zeros((384, 384, 12), dtype=np.float64)
        path = save_prediction(
            out_dir=tmp_path,
            team_name="team",
            task=1,
            storm_id="S000001",
            arr=arr,
        )
        loaded = np.load(path)
        assert loaded.dtype == np.float32

    def test_save_prediction_returns_path(self, tmp_path):
        arr = np.zeros((384, 384, 12), dtype=np.float32)
        result = save_prediction(
            out_dir=tmp_path,
            team_name="team",
            task=1,
            storm_id="S000001",
            arr=arr,
        )
        assert isinstance(result, Path)

    def test_save_prediction_loadable(self, tmp_path):
        arr = np.random.randn(384, 384, 12).astype(np.float32)
        path = save_prediction(
            out_dir=tmp_path,
            team_name="team",
            task=1,
            storm_id="S000001",
            arr=arr,
        )
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, arr)

    def test_save_prediction_skip_validation(self, tmp_path):
        # Invalid shape but validation disabled
        arr = np.zeros((100, 100, 5), dtype=np.float32)
        save_prediction(
            out_dir=tmp_path,
            team_name="team",
            task=1,
            storm_id="S000001",
            arr=arr,
            validate=False,
        )

    @pytest.mark.parametrize("task", [1, 2, 3, 4])
    def test_save_prediction_all_tasks(self, tmp_path, task):
        if task == 1:
            arr = np.zeros((384, 384, 12), dtype=np.float32)
        elif task == 2:
            arr = np.zeros((384, 384, 36), dtype=np.float32)
        elif task == 3:
            arr = np.array("Tornado")
        else:  # task 4
            arr = np.zeros((10, 3), dtype=np.float32)

        path = save_prediction(
            out_dir=tmp_path,
            team_name="sally",
            task=task,
            storm_id="S000001",
            arr=arr,
        )
        assert path.exists()
        assert f"task{task}" in path.name


class TestDenormalizeTensor:
    """Tests for denormalize_tensor function."""

    # Common test params
    TEST_MEAN = 1000.0
    TEST_STD = 500.0
    TEST_MIN = 0.0
    TEST_MAX = 255.0
    TEST_LOG_EPS = 1e-6

    def test_denormalize_zscore_basic(self):
        """Test zscore denormalization (numpy array)."""
        normalized = np.array([0.0, 1.0, -1.0])
        result = denormalize_tensor(
            normalized,
            norm_type="zscore",
            mean=self.TEST_MEAN,
            std=self.TEST_STD
        )
        expected = np.array([1000.0, 1500.0, 500.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_denormalize_zscore_torch(self):
        """Test zscore denormalization (torch tensor)."""
        normalized = torch.tensor([0.0, 1.0, -1.0])
        result = denormalize_tensor(
            normalized,
            norm_type="zscore",
            mean=self.TEST_MEAN,
            std=self.TEST_STD
        )
        expected = torch.tensor([1000.0, 1500.0, 500.0])
        torch.testing.assert_close(result, expected)

    def test_denormalize_zscore_missing_params_raises(self):
        """Test zscore missing mean/std raises error."""
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="mean/std required"):
            denormalize_tensor(arr, norm_type="zscore")

    def test_denormalize_minmax_basic(self):
        """Test minmax denormalization (numpy array)."""
        normalized = np.array([0.0, 0.5, 1.0])
        result = denormalize_tensor(
            normalized,
            norm_type="minmax",
            data_min=self.TEST_MIN,
            data_max=self.TEST_MAX
        )
        expected = np.array([0.0, 127.5, 255.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_denormalize_minmax_torch(self):
        """Test minmax denormalization (torch tensor)."""
        normalized = torch.tensor([0.0, 0.5, 1.0])
        result = denormalize_tensor(
            normalized,
            norm_type="minmax",
            data_min=self.TEST_MIN,
            data_max=self.TEST_MAX
        )
        expected = torch.tensor([0.0, 127.5, 255.0])
        torch.testing.assert_close(result, expected)

    def test_denormalize_minmax_missing_params_raises(self):
        """Test minmax missing data_min/data_max raises error."""
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="data_min/data_max required"):
            denormalize_tensor(arr, norm_type="minmax")

    def test_denormalize_minmax_equal_minmax_raises(self):
        """Test minmax with equal data_min/data_max raises error."""
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="data_max cannot equal data_min"):
            denormalize_tensor(
                arr,
                norm_type="minmax",
                data_min=10.0,
                data_max=10.0
            )

    def test_denormalize_log_basic(self):
        """Test log denormalization (numpy array, log_scale=True)."""
        # Normalized values (0-1 range)
        normalized = np.array([0.0, 1.0])

        result = denormalize_tensor(
            normalized,
            norm_type="log",
            data_min=self.TEST_MIN,
            data_max=self.TEST_MAX,
            eps=self.TEST_LOG_EPS,
            log_scale=True
        )
        # Expected: exp(log_min) - eps = 0, exp(log_max) - eps = 255
        expected = np.array([self.TEST_MIN, self.TEST_MAX])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_denormalize_log_no_log_scale(self):
        """Test log denormalization with log_scale=False."""
        normalized = np.array([np.log(1.0 + self.TEST_LOG_EPS)])
        result = denormalize_tensor(
            normalized,
            norm_type="log",
            data_min=0.0,
            data_max=10.0,
            eps=self.TEST_LOG_EPS,
            log_scale=False
        )
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_denormalize_log_clip(self):
        """Test log denormalization clips to original data range."""
        # Normalized value that would exceed max when denormalized
        normalized = np.array([2.0])  # Way above 1.0
        result = denormalize_tensor(
            normalized,
            norm_type="log",
            data_min=self.TEST_MIN,
            data_max=self.TEST_MAX,
            eps=self.TEST_LOG_EPS
        )
        assert result[0] == self.TEST_MAX  # Clipped to max

    def test_denormalize_log_missing_params_raises(self):
        """Test log missing data_min/data_max raises error."""
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="data_min/data_max required"):
            denormalize_tensor(arr, norm_type="log")

    def test_denormalize_invalid_norm_type_raises(self):
        """Test invalid norm_type raises error."""
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="Invalid norm_type"):
            denormalize_tensor(arr, norm_type="invalid_type")

    def test_denormalize_preserves_shape(self):
        """Test denormalization preserves input shape (3D array)."""
        arr = np.zeros((384, 384, 12))
        result = denormalize_tensor(
            arr,
            norm_type="minmax",
            data_min=self.TEST_MIN,
            data_max=self.TEST_MAX
        )
        assert result.shape == arr.shape


class TestDecodeEventType:
    """Tests for decode_event_type function."""

    def test_decode_all_event_types(self):
        for idx, expected in enumerate(EVENT_TYPES):
            assert decode_event_type(idx) == expected

    def test_decode_event_type_returns_string(self):
        result = decode_event_type(0)
        assert isinstance(result, str)

    def test_decode_event_type_negative_raises(self):
        with pytest.raises(ValueError, match="Invalid event class index"):
            decode_event_type(-1)

    def test_decode_event_type_too_large_raises(self):
        with pytest.raises(ValueError, match="Invalid event class index"):
            decode_event_type(len(EVENT_TYPES))

    def test_decode_event_type_specific_values(self):
        assert decode_event_type(EVENT_TYPES.index("Tornado")) == "Tornado"
        assert decode_event_type(EVENT_TYPES.index("Hail")) == "Hail"
        assert decode_event_type(EVENT_TYPES.index("Lightning")) == "Lightning"
