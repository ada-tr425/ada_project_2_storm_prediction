import numpy as np
import pytest
import torch

from predicting_unpredictable.constants import EVENT_TYPES
from predicting_unpredictable.preprocess import (
    PreprocessTask1,
    PreprocessTask2,
    PreprocessTask3,
    PreprocessTask4,
    int16_image_to_tchw,
    rasterize_lightning,
    stack_channels_tchw,
    normalize_tensor,
    upsample_192_to_384,
    vil_to_tchw,
)


@pytest.fixture(scope="module")
def storm_data():
    # Deterministic low-entropy test arrays
    h384, w384, t = 384, 384, 36
    h192, w192 = 192, 192

    vil = np.zeros((h384, w384, t), dtype=np.uint8)
    vis = np.zeros((h384, w384, t), dtype=np.int16)
    ir069 = np.zeros((h192, w192, t), dtype=np.int16)
    ir107 = np.zeros((h192, w192, t), dtype=np.int16)
    lght = np.zeros((0, 5), dtype=np.float32)  # Empty lightning data

    return {
        "vil": vil,
        "vis": vis,
        "ir069": ir069,
        "ir107": ir107,
        "lght": lght,
    }


def test_vil_to_tchw_shape_dtype_range():
    """Test vil_to_tchw output shape/dtype/range."""
    x = vil_to_tchw(np.zeros((384, 384, 36), dtype=np.uint8))
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert tuple(x.shape) == (36, 1, 384, 384)
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0


def test_vil_to_tchw_raises_on_bad_shape():
    """Test ValueError for invalid vil_to_tchw input shape."""
    with pytest.raises(ValueError):
        vil_to_tchw(np.zeros((384, 384), dtype=np.uint8))


def test_int16_image_to_tchw_shape_dtype():
    """Test int16_image_to_tchw output shape/dtype."""
    x = int16_image_to_tchw(np.zeros((384, 384, 36), dtype=np.int16))
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert tuple(x.shape) == (36, 1, 384, 384)


def test_int16_image_to_tchw_raises_on_bad_shape():
    """Test ValueError for invalid int16_image_to_tchw input shape."""
    with pytest.raises(ValueError):
        int16_image_to_tchw(np.zeros((384, 384), dtype=np.int16))


def test_upsample_192_to_384_happy_path():
    """Test upsample_192_to_384 valid input."""
    x = torch.zeros((36, 1, 192, 192), dtype=torch.float32)
    y = upsample_192_to_384(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.float32
    assert tuple(y.shape) == (36, 1, 384, 384)


def test_upsample_192_to_384_type_and_shape_errors():
    """Test upsample_192_to_384 type/shape errors."""
    with pytest.raises(TypeError):
        upsample_192_to_384(np.zeros((36, 1, 192, 192), dtype=np.float32))
    with pytest.raises(ValueError):
        upsample_192_to_384(torch.zeros((192, 192), dtype=torch.float32))
    bad_shape = torch.zeros((36, 1, 193, 192), dtype=torch.float32)
    with pytest.raises(ValueError):
        upsample_192_to_384(bad_shape)


def test_normalize_tensor_zscore_basic():
    """Test normalize_tensor zscore mode."""
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    z = normalize_tensor(t, norm_type="zscore", mean=1.0, std=1.0)
    assert torch.allclose(z, torch.tensor([0.0, 1.0], dtype=torch.float32))


def test_normalize_tensor_minmax_basic():
    """Test normalize_tensor minmax mode (0-1 range)."""
    t = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float32)
    scaled = normalize_tensor(
        t,
        norm_type="minmax",
        data_min=0.0,
        data_max=10.0
    )
    expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    assert torch.allclose(scaled, expected)
    assert isinstance(scaled, torch.Tensor)
    assert scaled.dtype == torch.float32


def test_normalize_tensor_minmax_custom_range():
    """Test normalize_tensor minmax mode (custom range)."""
    t = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
    scaled = normalize_tensor(
        t,
        norm_type="minmax",
        data_min=100.0,
        data_max=300.0,
        target_min=-1.0,
        target_max=1.0
    )
    expected = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(scaled, expected)


def test_normalize_tensor_minmax_division_by_zero():
    """Test normalize_tensor minmax division by zero error."""
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="data_max cannot equal data_min"):
        normalize_tensor(t, norm_type="minmax", data_min=5.0, data_max=5.0)


def test_normalize_tensor_minmax_4d_tensor():
    """Test normalize_tensor minmax on 4D tensor."""
    t = torch.rand((36, 1, 384, 384), dtype=torch.float32) * 100
    scaled = normalize_tensor(
        t,
        norm_type="minmax",
        data_min=0.0,
        data_max=100.0
    )
    assert tuple(scaled.shape) == (36, 1, 384, 384)
    assert float(scaled.min()) >= 0.0
    assert float(scaled.max()) <= 1.0


def test_normalize_tensor_raises_on_non_tensor():
    """Test normalize_tensor non-tensor input error."""
    with pytest.raises(TypeError):
        normalize_tensor(
            np.array([1.0, 2.0]),
            norm_type="zscore",
            mean=0.0,
            std=1.0
        )
    with pytest.raises(TypeError):
        normalize_tensor(
            np.array([1.0, 2.0]),
            norm_type="minmax",
            data_min=0.0,
            data_max=10.0
        )
    with pytest.raises(TypeError):
        normalize_tensor(
            np.array([1.0, 2.0]),
            norm_type="log",
            data_min=0.0,
            data_max=10.0
        )


def test_normalize_tensor_raises_on_invalid_type():
    """Test normalize_tensor invalid norm_type error."""
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="Invalid norm_type"):
        normalize_tensor(t, norm_type="invalid_type", mean=0.0, std=1.0)


def test_normalize_tensor_raises_on_missing_params():
    """Test normalize_tensor missing required params error."""
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="mean/std required for zscore"):
        normalize_tensor(t, norm_type="zscore", mean=1.0)
    msg = "data_min/data_max required for minmax"
    with pytest.raises(ValueError, match=msg):
        normalize_tensor(t, norm_type="minmax", data_min=0.0)
    with pytest.raises(ValueError, match="data_min/data_max required for log"):
        normalize_tensor(t, norm_type="log", data_min=0.0)


def test_normalize_tensor_log_basic():
    """Test normalize_tensor log mode (with scaling)."""
    t = torch.tensor(
        [0.0, 1.0, np.e - 1e-6],
        dtype=torch.float32
    )
    scaled = normalize_tensor(
        t,
        norm_type="log",
        data_min=0.0,
        data_max=np.e - 1e-6,
        eps=1e-6,
        log_scale=True
    )
    assert float(scaled.min()) >= 0.0
    assert float(scaled.max()) <= 1.0
    assert torch.allclose(scaled[-1], torch.tensor(1.0))


def test_normalize_tensor_log_no_scale():
    """Test normalize_tensor log mode (no scaling)."""
    t = torch.tensor([1e-6, 1.0 + 1e-6], dtype=torch.float32)
    val_0 = t[0].item()
    val_1 = t[1].item()

    scalar_0 = torch.tensor(val_0 + 1e-6, dtype=torch.float32)
    scalar_1 = torch.tensor(val_1 + 1e-6, dtype=torch.float32)
    expected_0 = torch.log(scalar_0)
    expected_1 = torch.log(scalar_1)

    log_result = normalize_tensor(
        t,
        norm_type="log",
        data_min=1e-6,
        data_max=1.0 + 1e-6,
        eps=1e-6,
        log_scale=False
    )
    assert torch.allclose(log_result[0], expected_0)
    assert torch.allclose(log_result[1], expected_1)


def test_normalize_tensor_log_negative_data():
    """Test normalize_tensor log negative data error."""
    t = torch.tensor([-1.0, 2.0], dtype=torch.float32)
    msg = "log only supports non-negative data"
    with pytest.raises(ValueError, match=msg):
        normalize_tensor(
            t,
            norm_type="log",
            data_min=-1.0,
            data_max=2.0
        )


def test_stack_channels_tchw_happy_path():
    """Test stack_channels_tchw valid input."""
    a = torch.zeros((36, 1, 384, 384), dtype=torch.float32)
    b = torch.ones((36, 1, 384, 384), dtype=torch.float32)
    x = stack_channels_tchw([a, b])
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == (36, 2, 384, 384)


def test_stack_channels_tchw_errors():
    """Test stack_channels_tchw empty/invalid input errors."""
    with pytest.raises(ValueError):
        stack_channels_tchw([])
    with pytest.raises(TypeError):
        stack_channels_tchw([torch.zeros((1, 1, 1, 1)), "not a tensor"])


def test_rasterize_lightning_empty():
    """Test rasterize_lightning empty input."""
    counts, heatmaps = rasterize_lightning(np.zeros((0, 5), dtype=np.float32))
    assert isinstance(counts, torch.Tensor)
    assert isinstance(heatmaps, torch.Tensor)
    assert tuple(counts.shape) == (36,)
    assert tuple(heatmaps.shape) == (36, 384, 384)
    assert float(counts.sum()) == 0.0
    assert float(heatmaps.sum()) == 0.0


def test_rasterize_lightning_binning_and_bounds():
    """Test rasterize_lightning binning/bounds filtering."""
    lght = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 20.0],
            [299.9, 0.0, 0.0, 10.0, 20.0],
            [300.0, 0.0, 0.0, 10.0, 20.0],
            [900.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 10.0, 20.0],
            [0.0, 0.0, 0.0, 999.0, 20.0],
        ],
        dtype=np.float32,
    )
    counts, heatmaps = rasterize_lightning(
        lght,
        t_bins=36,
        frame_seconds=300.0
    )

    assert int(counts[0].item()) == 2
    assert int(counts[1].item()) == 1
    assert int(counts[3].item()) == 1
    assert int(counts.sum().item()) == 4
    assert int(heatmaps[0, 20, 10].item()) == 2
    assert int(heatmaps[1, 20, 10].item()) == 1
    assert int(heatmaps[3, 0, 0].item()) == 1


def test_preprocess_task1_shapes(storm_data):
    """Test PreprocessTask1 output shapes (VIL log norm)."""
    task = PreprocessTask1()
    x, y = task(storm_data)
    assert tuple(x.shape) == (12, 1, 384, 384)
    assert tuple(y.shape) == (12, 1, 384, 384)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_preprocess_task2_shapes(storm_data):
    """Test PreprocessTask2 output shapes (minmax/log norm)."""
    task = PreprocessTask2()
    x, y = task(storm_data)
    assert tuple(x.shape) == (36, 3, 384, 384)
    assert tuple(y.shape) == (36, 1, 384, 384)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_preprocess_task3_label_encoding(storm_data):
    """Test PreprocessTask3 label encoding logic."""
    task = PreprocessTask3()
    x, y = task(storm_data, event_type_str="Tornado")
    assert tuple(x.shape) == (36, 4, 384, 384)
    assert y.dtype == torch.long
    assert int(y.item()) == EVENT_TYPES.index("Tornado")

    _, y_unknown = task(storm_data, event_type_str="Unknown Event")
    assert int(y_unknown.item()) == -1


def test_preprocess_task4_shapes_and_targets(storm_data):
    """Test PreprocessTask4 output shapes/targets."""
    task = PreprocessTask4()
    x, y_counts, y_heatmaps = task(storm_data)
    assert tuple(x.shape) == (36, 4, 384, 384)
    assert tuple(y_counts.shape) == (36,)
    assert tuple(y_heatmaps.shape) == (36, 384, 384)
