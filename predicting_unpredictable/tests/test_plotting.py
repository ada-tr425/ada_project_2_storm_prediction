"""
Tests for predicting_unpredictable.plotting module.
"""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from pathlib import Path  # noqa: E402

from predicting_unpredictable.plotting import (  # noqa: E402
    DEFAULT_CMAPS,
    plot_single_image,
    plot_image_grid,
    plot_frame_sequence,
    plot_storm_overview,
    plot_lightning_scatter,
    plot_lightning_heatmap,
    plot_prediction_comparison,
    plot_training_curves,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_metric_comparison,
    make_gif,
    plot_event,
    create_animation,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_image():
    """Create a sample 2D image."""
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def sample_frames_hwt():
    """Create sample frames in (H, W, T) format."""
    return np.random.rand(64, 64, 12).astype(np.float32)


@pytest.fixture
def sample_frames_thw():
    """Create sample frames in (T, H, W) format."""
    return np.random.rand(12, 64, 64).astype(np.float32)


@pytest.fixture
def sample_frames_tchw():
    """Create sample frames in (T, C, H, W) format."""
    return np.random.rand(12, 1, 64, 64).astype(np.float32)


@pytest.fixture
def sample_lightning():
    """Create sample lightning data."""
    n = 50
    return np.column_stack([
        np.random.rand(n) * 10800,  # time in seconds (0-3h)
        np.random.rand(n) * 384,    # x pixel
        np.random.rand(n) * 384,    # y pixel
        np.random.rand(n) * 384,    # col 3 (x in different coord)
        np.random.rand(n) * 384,    # col 4 (y in different coord)
    ]).astype(np.float32)


@pytest.fixture
def empty_lightning():
    """Create empty lightning data."""
    return np.zeros((0, 5), dtype=np.float32)


class TestDefaultCmaps:
    """Tests for DEFAULT_CMAPS constant."""

    def test_default_cmaps_contains_expected_keys(self):
        expected_keys = {"vis", "ir069", "ir107", "vil", "lght"}
        assert expected_keys == set(DEFAULT_CMAPS.keys())

    def test_default_cmaps_values_are_strings(self):
        for key, value in DEFAULT_CMAPS.items():
            assert isinstance(value, str)


class TestPlotSingleImage:
    """Tests for plot_single_image function."""

    def test_returns_axes(self, sample_image):
        ax = plot_single_image(sample_image)
        assert isinstance(ax, plt.Axes)

    def test_with_title(self, sample_image):
        ax = plot_single_image(sample_image, title="Test Title")
        assert ax.get_title() == "Test Title"

    def test_with_custom_cmap(self, sample_image):
        ax = plot_single_image(sample_image, cmap="hot")
        assert isinstance(ax, plt.Axes)

    def test_without_colorbar(self, sample_image):
        ax = plot_single_image(sample_image, colorbar=False)
        assert isinstance(ax, plt.Axes)

    def test_with_existing_axes(self, sample_image):
        fig, ax = plt.subplots()
        result_ax = plot_single_image(sample_image, ax=ax)
        assert result_ax is ax

    def test_with_vmin_vmax(self, sample_image):
        ax = plot_single_image(sample_image, vmin=0.2, vmax=0.8)
        assert isinstance(ax, plt.Axes)

    def test_custom_figsize(self, sample_image):
        ax = plot_single_image(sample_image, figsize=(10, 10))
        fig = ax.get_figure()
        assert fig.get_size_inches()[0] == pytest.approx(10, abs=0.1)


class TestPlotImageGrid:
    """Tests for plot_image_grid function."""

    def test_returns_figure(self, sample_image):
        images = [sample_image for _ in range(4)]
        fig = plot_image_grid(images)
        assert isinstance(fig, plt.Figure)

    def test_with_titles(self, sample_image):
        images = [sample_image for _ in range(4)]
        titles = ["A", "B", "C", "D"]
        fig = plot_image_grid(images, titles=titles)
        assert isinstance(fig, plt.Figure)

    def test_with_custom_grid(self, sample_image):
        images = [sample_image for _ in range(6)]
        fig = plot_image_grid(images, nrows=2, ncols=3)
        axes = fig.get_axes()
        # Should have at least 6 axes (may have colorbars too)
        assert len(axes) >= 6

    def test_with_nrows_only(self, sample_image):
        images = [sample_image for _ in range(6)]
        fig = plot_image_grid(images, nrows=2)
        assert isinstance(fig, plt.Figure)

    def test_with_ncols_only(self, sample_image):
        images = [sample_image for _ in range(6)]
        fig = plot_image_grid(images, ncols=3)
        assert isinstance(fig, plt.Figure)

    def test_with_suptitle(self, sample_image):
        images = [sample_image for _ in range(4)]
        fig = plot_image_grid(images, suptitle="Grid Title")
        assert fig._suptitle.get_text() == "Grid Title"

    def test_with_colorbar(self, sample_image):
        images = [sample_image for _ in range(4)]
        fig = plot_image_grid(images, colorbar=True)
        assert isinstance(fig, plt.Figure)

    def test_single_image(self, sample_image):
        fig = plot_image_grid([sample_image])
        assert isinstance(fig, plt.Figure)


class TestPlotFrameSequence:
    """Tests for plot_frame_sequence function."""

    def test_returns_figure_hwt(self, sample_frames_hwt):
        fig = plot_frame_sequence(sample_frames_hwt)
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_thw(self, sample_frames_thw):
        fig = plot_frame_sequence(sample_frames_thw)
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_tchw(self, sample_frames_tchw):
        fig = plot_frame_sequence(sample_frames_tchw)
        assert isinstance(fig, plt.Figure)

    def test_with_frame_indices(self, sample_frames_hwt):
        fig = plot_frame_sequence(sample_frames_hwt, frame_indices=[0, 5, 10])
        assert isinstance(fig, plt.Figure)

    def test_with_custom_img_type(self, sample_frames_hwt):
        fig = plot_frame_sequence(sample_frames_hwt, img_type="vil")
        assert isinstance(fig, plt.Figure)

    def test_with_custom_ncols(self, sample_frames_hwt):
        fig = plot_frame_sequence(sample_frames_hwt, ncols=4)
        assert isinstance(fig, plt.Figure)

    def test_with_suptitle(self, sample_frames_hwt):
        fig = plot_frame_sequence(sample_frames_hwt, suptitle="Custom Title")
        assert fig._suptitle.get_text() == "Custom Title"


class TestPlotStormOverview:
    """Tests for plot_storm_overview function."""

    def test_returns_figure(self, storm_data):
        fig = plot_storm_overview(storm_data)
        assert isinstance(fig, plt.Figure)

    def test_with_different_frame_idx(self, storm_data):
        fig = plot_storm_overview(storm_data, frame_idx=17)
        assert isinstance(fig, plt.Figure)

    def test_with_subset_of_data(self):
        # Only VIL data
        data = {"vil": np.zeros((384, 384, 36), dtype=np.uint8)}
        fig = plot_storm_overview(data)
        assert isinstance(fig, plt.Figure)

    def test_with_custom_figsize(self, storm_data):
        fig = plot_storm_overview(storm_data, figsize=(20, 5))
        assert isinstance(fig, plt.Figure)


class TestPlotLightningScatter:
    """Tests for plot_lightning_scatter function."""

    def test_returns_axes(self, sample_lightning):
        ax = plot_lightning_scatter(sample_lightning)
        assert isinstance(ax, plt.Axes)

    def test_with_empty_lightning(self, empty_lightning):
        ax = plot_lightning_scatter(empty_lightning)
        assert isinstance(ax, plt.Axes)

    def test_with_title(self, sample_lightning):
        ax = plot_lightning_scatter(sample_lightning, title="Lightning")
        assert ax.get_title() == "Lightning"

    def test_without_color_by_time(self, sample_lightning):
        ax = plot_lightning_scatter(sample_lightning, color_by_time=False)
        assert isinstance(ax, plt.Axes)

    def test_with_existing_axes(self, sample_lightning):
        fig, ax = plt.subplots()
        result_ax = plot_lightning_scatter(sample_lightning, ax=ax)
        assert result_ax is ax

    def test_custom_marker_size(self, sample_lightning):
        ax = plot_lightning_scatter(sample_lightning, s=20, alpha=0.8)
        assert isinstance(ax, plt.Axes)


class TestPlotLightningHeatmap:
    """Tests for plot_lightning_heatmap function."""

    def test_returns_axes(self, sample_lightning):
        ax = plot_lightning_heatmap(sample_lightning)
        assert isinstance(ax, plt.Axes)

    def test_with_empty_lightning(self, empty_lightning):
        ax = plot_lightning_heatmap(empty_lightning)
        assert isinstance(ax, plt.Axes)

    def test_with_custom_resolution(self, sample_lightning):
        ax = plot_lightning_heatmap(sample_lightning, resolution=(192, 192))
        assert isinstance(ax, plt.Axes)

    def test_with_custom_sigma(self, sample_lightning):
        ax = plot_lightning_heatmap(sample_lightning, sigma=10.0)
        assert isinstance(ax, plt.Axes)

    def test_with_title(self, sample_lightning):
        ax = plot_lightning_heatmap(sample_lightning, title="Heatmap")
        assert ax.get_title() == "Heatmap"

    def test_with_custom_cmap(self, sample_lightning):
        ax = plot_lightning_heatmap(sample_lightning, cmap="viridis")
        assert isinstance(ax, plt.Axes)


class TestPlotPredictionComparison:
    """Tests for plot_prediction_comparison function."""

    def test_returns_figure_hwt(self):
        pred = np.random.rand(64, 64, 12).astype(np.float32)
        target = np.random.rand(64, 64, 12).astype(np.float32)
        fig = plot_prediction_comparison(pred, target)
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_thw(self):
        pred = np.random.rand(12, 64, 64).astype(np.float32)
        target = np.random.rand(12, 64, 64).astype(np.float32)
        fig = plot_prediction_comparison(pred, target)
        assert isinstance(fig, plt.Figure)

    def test_with_frame_indices(self):
        pred = np.random.rand(12, 64, 64).astype(np.float32)
        target = np.random.rand(12, 64, 64).astype(np.float32)
        fig = plot_prediction_comparison(
            pred, target, frame_indices=[0, 5, 10]
        )
        assert isinstance(fig, plt.Figure)

    def test_with_suptitle(self):
        pred = np.random.rand(12, 64, 64).astype(np.float32)
        target = np.random.rand(12, 64, 64).astype(np.float32)
        fig = plot_prediction_comparison(pred, target, suptitle="Comparison")
        assert fig._suptitle.get_text() == "Comparison"

    def test_with_custom_cmap(self):
        pred = np.random.rand(12, 64, 64).astype(np.float32)
        target = np.random.rand(12, 64, 64).astype(np.float32)
        fig = plot_prediction_comparison(pred, target, cmap="hot")
        assert isinstance(fig, plt.Figure)

    def test_single_frame(self):
        pred = np.random.rand(1, 64, 64).astype(np.float32)
        target = np.random.rand(1, 64, 64).astype(np.float32)
        fig = plot_prediction_comparison(pred, target)
        assert isinstance(fig, plt.Figure)


class TestPlotTrainingCurves:
    """Tests for plot_training_curves function."""

    def test_returns_figure_loss_only(self):
        history = {"train_loss": [1.0, 0.5, 0.3], "val_loss": [1.2, 0.6, 0.4]}
        fig = plot_training_curves(history)
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_acc_only(self):
        history = {"train_acc": [0.5, 0.7, 0.9], "val_acc": [0.4, 0.6, 0.8]}
        fig = plot_training_curves(history)
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_mixed(self):
        history = {
            "train_loss": [1.0, 0.5],
            "val_loss": [1.2, 0.6],
            "train_acc": [0.5, 0.9],
            "val_acc": [0.4, 0.8],
        }
        fig = plot_training_curves(history)
        assert isinstance(fig, plt.Figure)

    def test_with_other_metrics(self):
        history = {"f1_score": [0.5, 0.7, 0.9], "mae": [1.0, 0.5, 0.3]}
        fig = plot_training_curves(history)
        assert isinstance(fig, plt.Figure)

    def test_with_title(self):
        history = {"train_loss": [1.0, 0.5, 0.3]}
        fig = plot_training_curves(history, title="Training History")
        assert fig._suptitle.get_text() == "Training History"

    def test_empty_history(self):
        # Empty history should still work (edge case)
        history = {}
        fig = plot_training_curves(history)
        assert isinstance(fig, plt.Figure)


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_returns_figure(self):
        cm = np.array([[10, 2], [3, 15]])
        fig = plot_confusion_matrix(cm)
        assert isinstance(fig, plt.Figure)

    def test_with_class_names(self):
        cm = np.array([[10, 2], [3, 15]])
        fig = plot_confusion_matrix(cm, class_names=["A", "B"])
        assert isinstance(fig, plt.Figure)

    def test_with_normalization(self):
        cm = np.array([[10, 2], [3, 15]])
        fig = plot_confusion_matrix(cm, normalize=True)
        assert isinstance(fig, plt.Figure)

    def test_with_title(self):
        cm = np.array([[10, 2], [3, 15]])
        fig = plot_confusion_matrix(cm, title="Confusion Matrix")
        axes = fig.get_axes()
        assert any(ax.get_title() == "Confusion Matrix" for ax in axes)

    def test_multiclass(self):
        cm = np.array([
            [10, 2, 1],
            [3, 15, 2],
            [1, 1, 12]
        ])
        fig = plot_confusion_matrix(cm)
        assert isinstance(fig, plt.Figure)

    def test_with_custom_cmap(self):
        cm = np.array([[10, 2], [3, 15]])
        fig = plot_confusion_matrix(cm, cmap="Reds")
        assert isinstance(fig, plt.Figure)


class TestPlotClassDistribution:
    """Tests for plot_class_distribution function."""

    def test_returns_figure_integer_labels(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 3])
        fig = plot_class_distribution(labels, class_names=["A", "B", "C", "D"])
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_string_labels(self):
        labels = ["Tornado", "Hail", "Tornado", "Lightning"]
        class_names = ["Tornado", "Hail", "Lightning"]
        fig = plot_class_distribution(labels, class_names=class_names)
        assert isinstance(fig, plt.Figure)

    def test_with_title(self):
        labels = np.array([0, 0, 1, 1])
        fig = plot_class_distribution(
            labels, class_names=["A", "B"], title="Distribution"
        )
        axes = fig.get_axes()
        assert any(ax.get_title() == "Distribution" for ax in axes)

    def test_with_default_class_names(self):
        # Should use EVENT_TYPES by default
        labels = np.array([0, 1, 2, 3])
        fig = plot_class_distribution(labels)
        assert isinstance(fig, plt.Figure)


class TestPlotMetricComparison:
    """Tests for plot_metric_comparison function."""

    def test_returns_figure(self):
        metrics = {"Model A": 0.85, "Model B": 0.92, "Model C": 0.78}
        fig = plot_metric_comparison(metrics)
        assert isinstance(fig, plt.Figure)

    def test_with_title(self):
        metrics = {"A": 0.5, "B": 0.7}
        fig = plot_metric_comparison(metrics, title="Model Comparison")
        axes = fig.get_axes()
        assert any(ax.get_title() == "Model Comparison" for ax in axes)

    def test_with_custom_ylabel(self):
        metrics = {"A": 0.5, "B": 0.7}
        fig = plot_metric_comparison(metrics, ylabel="Accuracy")
        assert isinstance(fig, plt.Figure)

    def test_single_metric(self):
        metrics = {"Single": 0.95}
        fig = plot_metric_comparison(metrics)
        assert isinstance(fig, plt.Figure)


class TestMakeGif:
    """Tests for make_gif function."""

    def test_creates_gif(self, tmp_path, sample_image):
        # Create temp image files
        files = []
        for i in range(3):
            fig, ax = plt.subplots()
            ax.imshow(sample_image * (i + 1))
            file_path = tmp_path / f"frame_{i}.png"
            fig.savefig(file_path)
            plt.close(fig)
            files.append(file_path)

        outfile = tmp_path / "test.gif"
        result = make_gif(outfile, files)

        assert result == outfile
        assert outfile.exists()

    def test_with_custom_fps(self, tmp_path, sample_image):
        files = []
        for i in range(2):
            fig, ax = plt.subplots()
            ax.imshow(sample_image)
            file_path = tmp_path / f"frame_{i}.png"
            fig.savefig(file_path)
            plt.close(fig)
            files.append(file_path)

        outfile = tmp_path / "test.gif"
        make_gif(outfile, files, fps=5)
        assert outfile.exists()


class TestPlotEvent:
    """Tests for plot_event function."""

    def test_returns_figure_static(self, storm_data):
        fig = plot_event(storm_data, "TEST001", frame_indices=[0], show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_custom_frame_indices(self, storm_data):
        fig = plot_event(
            storm_data, "TEST001", frame_indices=[0, 10, 20], show=False
        )
        assert isinstance(fig, plt.Figure)

    def test_with_lightning_data(self):
        """Test with actual lightning data."""
        storm_data = {
            "vil": np.zeros((384, 384, 36), dtype=np.uint8),
            "vis": np.zeros((384, 384, 36), dtype=np.int16),
            "ir069": np.zeros((192, 192, 36), dtype=np.int16),
            "ir107": np.zeros((192, 192, 36), dtype=np.int16),
            "lght": np.array([
                [300, 100, 100, 100, 100],  # Frame ~1
                [900, 200, 200, 200, 200],  # Frame ~3
            ], dtype=np.float32),
        }
        fig = plot_event(
            storm_data, "TEST001", frame_indices=[0, 1, 3], show=False
        )
        assert isinstance(fig, plt.Figure)

    def test_output_gif(self, storm_data, tmp_path):
        """Test GIF output."""
        # Use smaller data for faster test
        small_data = {
            "vil": np.zeros((64, 64, 36), dtype=np.uint8),
            "vis": np.zeros((64, 64, 36), dtype=np.int16),
            "ir069": np.zeros((64, 64, 36), dtype=np.int16),
            "ir107": np.zeros((64, 64, 36), dtype=np.int16),
            "lght": np.zeros((0, 5), dtype=np.float32),
        }
        result = plot_event(
            small_data,
            "TEST001",
            output_gif=True,
            save_gif=True,
            gif_dir=tmp_path,
        )
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".gif"


class TestCreateAnimation:
    """Tests for create_animation function."""

    def test_creates_gif_thw(self, tmp_path):
        frames = np.random.rand(5, 32, 32).astype(np.float32)
        outfile = tmp_path / "animation.gif"
        result = create_animation(frames, outfile=outfile)
        assert result == outfile
        assert outfile.exists()

    def test_creates_gif_hwt(self, tmp_path):
        frames = np.random.rand(32, 32, 5).astype(np.float32)
        outfile = tmp_path / "animation.gif"
        create_animation(frames, outfile=outfile)
        assert outfile.exists()

    def test_with_custom_cmap(self, tmp_path):
        frames = np.random.rand(5, 32, 32).astype(np.float32)
        outfile = tmp_path / "animation.gif"
        create_animation(frames, outfile=outfile, cmap="hot")
        assert outfile.exists()

    def test_with_title_func(self, tmp_path):
        frames = np.random.rand(5, 32, 32).astype(np.float32)
        outfile = tmp_path / "animation.gif"
        create_animation(
            frames,
            outfile=outfile,
            title_func=lambda i: f"Time: {i * 5} min"
        )
        assert outfile.exists()

    def test_with_vmin_vmax(self, tmp_path):
        frames = np.random.rand(5, 32, 32).astype(np.float32)
        outfile = tmp_path / "animation.gif"
        create_animation(
            frames, outfile=outfile, vmin=0.2, vmax=0.8
        )
        assert outfile.exists()
