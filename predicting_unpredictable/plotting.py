"""
Visualization utilities for the storm prediction project.

This module provides plotting functions for:
- Individual image types (VIS, IR069, IR107, VIL)
- Frame sequences and animations
- Lightning event data
- Model predictions vs ground truth comparisons
- Training curves and metrics
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .constants import EVENT_TYPES


# Default colormaps for different image types
DEFAULT_CMAPS = {
    "vis": "gray",
    "ir069": "coolwarm",
    "ir107": "coolwarm",
    "vil": "viridis",
    "lght": "hot",
}


def plot_single_image(
    img: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> plt.Axes:
    """
    Plot a single 2D image.

    Parameters
    ----------
    img : np.ndarray
        2D array (H, W) to display.
    title : str, optional
        Title for the plot.
    cmap : str
        Matplotlib colormap name.
    colorbar : bool
        Whether to add a colorbar.
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    vmin, vmax : float, optional
        Color scale limits.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    plt.Axes
        The axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


def plot_image_grid(
    images: Sequence[np.ndarray],
    *,
    titles: Sequence[str] | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """
    Plot multiple images in a grid layout.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List of 2D arrays to display.
    titles : Sequence[str], optional
        Titles for each subplot.
    nrows, ncols : int, optional
        Grid dimensions. If not specified, auto-calculated.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    suptitle : str, optional
        Super title for the entire figure.
    colorbar : bool
        Whether to add colorbars.
    vmin, vmax : float, optional
        Color scale limits (shared across all images).

    Returns
    -------
    plt.Figure
        The figure object.
    """
    n = len(images)
    if nrows is None and ncols is None:
        ncols = min(6, n)
        nrows = (n + ncols - 1) // ncols
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows

    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def plot_frame_sequence(
    frames: np.ndarray,
    *,
    img_type: str = "vil",
    frame_indices: Sequence[int] | None = None,
    ncols: int = 6,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """
    Plot a sequence of frames from a storm.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (H, W, T) or (T, H, W) or (T, C, H, W).
    img_type : str
        Image type for colormap selection.
    frame_indices : Sequence[int], optional
        Which frame indices to plot. If None, plots all.
    ncols : int
        Number of columns in the grid.
    figsize : tuple, optional
        Figure size.
    suptitle : str, optional
        Super title.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    # Handle different input shapes
    if frames.ndim == 3:
        if frames.shape[2] <= 36:
            # Shape (H, W, T)
            frames = np.transpose(frames, (2, 0, 1))
        # else: Shape (T, H, W) - already correct
    elif frames.ndim == 4:
        # Shape (T, C, H, W) -> (T, H, W) take first channel
        frames = frames[:, 0, :, :]

    n_frames = frames.shape[0]
    if frame_indices is None:
        frame_indices = list(range(n_frames))

    images = [frames[i] for i in frame_indices]
    titles = [f"Frame {i}" for i in frame_indices]
    cmap = DEFAULT_CMAPS.get(img_type, "viridis")

    return plot_image_grid(
        images,
        titles=titles,
        ncols=ncols,
        cmap=cmap,
        figsize=figsize,
        suptitle=suptitle or f"{img_type.upper()} Frame Sequence",
    )


def plot_storm_overview(
    storm_data: dict[str, np.ndarray],
    *,
    frame_idx: int = 0,
    figsize: tuple[float, float] = (16, 4),
) -> plt.Figure:
    """
    Plot all available image types for a single frame of a storm.

    Parameters
    ----------
    storm_data : dict[str, np.ndarray]
        Dictionary with keys like 'vis', 'ir069', 'ir107', 'vil'.
    frame_idx : int
        Which frame index to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    keys_order = ["vis", "ir069", "ir107", "vil"]
    img_types = [k for k in keys_order if k in storm_data]
    n = len(img_types)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img_type in zip(axes, img_types):
        data = storm_data[img_type]
        # Handle (H, W, T) shape
        if data.ndim == 3:
            frame = data[:, :, frame_idx]
        else:
            frame = data
        cmap = DEFAULT_CMAPS.get(img_type, "viridis")
        im = ax.imshow(frame, cmap=cmap)
        ax.set_title(f"{img_type.upper()}", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Storm Overview - Frame {frame_idx}", fontsize=14)
    fig.tight_layout()
    return fig


def plot_lightning_scatter(
    lght: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    s: float = 10,
    alpha: float = 0.6,
    color_by_time: bool = True,
) -> plt.Axes:
    """
    Scatter plot of lightning events.

    Parameters
    ----------
    lght : np.ndarray
        Lightning data of shape (N, 5) with columns
        (time, x, y, ...).
    ax : plt.Axes, optional
        Existing axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    s : float
        Marker size.
    alpha : float
        Marker transparency.
    color_by_time : bool
        Whether to color points by time.

    Returns
    -------
    plt.Axes
        The axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if lght.shape[0] == 0:
        ax.text(
            0.5,
            0.5,
            "No lightning events",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 384)
        ax.set_ylim(384, 0)
    else:
        times = lght[:, 0]
        x = lght[:, 1]
        y = lght[:, 2]

        if color_by_time:
            scatter = ax.scatter(
                x, y, c=times, s=s, alpha=alpha, cmap="plasma"
            )
            plt.colorbar(scatter, ax=ax, label="Time (s)")
        else:
            ax.scatter(x, y, s=s, alpha=alpha, color="red")

        ax.set_xlim(0, 384)
        ax.set_ylim(384, 0)

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_lightning_heatmap(
    lght: np.ndarray,
    *,
    resolution: tuple[int, int] = (384, 384),
    sigma: float = 5.0,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    cmap: str = "hot",
) -> plt.Axes:
    """
    Plot lightning events as a density heatmap.

    Parameters
    ----------
    lght : np.ndarray
        Lightning data of shape (N, 5) or (N, 3).
    resolution : tuple
        Output heatmap resolution (H, W).
    sigma : float
        Gaussian blur sigma for smoothing.
    ax : plt.Axes, optional
        Existing axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    cmap : str
        Colormap for the heatmap.

    Returns
    -------
    plt.Axes
        The axes object.
    """
    from scipy.ndimage import gaussian_filter

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    H, W = resolution
    heatmap = np.zeros((H, W), dtype=np.float32)

    if lght.shape[0] > 0:
        x = lght[:, 1].astype(int)
        y = lght[:, 2].astype(int)
        # Clip to valid range
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        np.add.at(heatmap, (y, x), 1)
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    im = ax.imshow(heatmap, cmap=cmap)
    plt.colorbar(im, ax=ax, label="Density")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def plot_prediction_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    frame_indices: Sequence[int] | None = None,
    ncols: int = 4,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """
    Compare predicted frames with ground truth.

    Parameters
    ----------
    pred : np.ndarray
        Predicted frames (H, W, T) or (T, H, W).
    target : np.ndarray
        Ground truth frames (H, W, T) or (T, H, W).
    frame_indices : Sequence[int], optional
        Which frames to compare. If None, selects evenly spaced frames.
    ncols : int
        Number of columns (frames) to show.
    cmap : str
        Colormap.
    figsize : tuple, optional
        Figure size.
    suptitle : str, optional
        Super title.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    # Normalize shapes to (T, H, W)
    if pred.ndim == 3 and pred.shape[2] <= 36:
        pred = np.transpose(pred, (2, 0, 1))
    if target.ndim == 3 and target.shape[2] <= 36:
        target = np.transpose(target, (2, 0, 1))

    n_frames = pred.shape[0]
    if frame_indices is None:
        step = max(1, n_frames // ncols)
        frame_indices = list(range(0, n_frames, step))[:ncols]

    n = len(frame_indices)
    if figsize is None:
        figsize = (4 * n, 12)

    fig, axes = plt.subplots(3, n, figsize=figsize)
    if n == 1:
        axes = axes.reshape(3, 1)

    # Compute shared vmin/vmax
    vmin = min(pred.min(), target.min())
    vmax = max(pred.max(), target.max())

    for col, frame_idx in enumerate(frame_indices):
        # Prediction
        axes[0, col].imshow(pred[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"Pred t={frame_idx}")
        axes[0, col].axis("off")

        # Ground truth
        axes[1, col].imshow(target[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"GT t={frame_idx}")
        axes[1, col].axis("off")

        # Difference
        diff = pred[frame_idx] - target[frame_idx]
        diff_max = np.abs(diff).max() or 1
        axes[2, col].imshow(diff, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
        axes[2, col].set_title(f"Diff t={frame_idx}")
        axes[2, col].axis("off")

    # Add row labels
    axes[0, 0].set_ylabel("Prediction", fontsize=12)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=12)
    axes[2, 0].set_ylabel("Difference", fontsize=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    *,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
) -> plt.Figure:
    """
    Plot training history curves.

    Parameters
    ----------
    history : dict[str, list[float]]
        Dictionary with metric names as keys and lists of values.
        Common keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    figsize : tuple
        Figure size.
    title : str, optional
        Super title.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    # Group metrics by type
    loss_keys = [k for k in history if "loss" in k.lower()]
    acc_keys = [k for k in history if "acc" in k.lower()]
    other_keys = [k for k in history if k not in loss_keys + acc_keys]

    n_plots = sum([len(loss_keys) > 0, len(acc_keys) > 0, len(other_keys) > 0])
    n_plots = max(1, n_plots)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot losses
    if loss_keys:
        ax = axes[plot_idx]
        for key in loss_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot accuracies
    if acc_keys:
        ax = axes[plot_idx]
        for key in acc_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot other metrics
    if other_keys:
        ax = axes[plot_idx]
        for key in other_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Other Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    cmap: str = "Blues",
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    class_names : list[str], optional
        Names for each class. Defaults to EVENT_TYPES.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    cmap : str
        Colormap.
    normalize : bool
        Whether to normalize rows to sum to 1.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    if class_names is None:
        class_names = EVENT_TYPES[: cm.shape[0]]

    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_class_distribution(
    labels: np.ndarray | list,
    *,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> plt.Figure:
    """
    Plot distribution of class labels.

    Parameters
    ----------
    labels : np.ndarray or list
        Array of integer labels or string labels.
    class_names : list[str], optional
        Names for each class. Defaults to EVENT_TYPES.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    if class_names is None:
        class_names = EVENT_TYPES

    labels = np.array(labels)

    # Handle both integer and string labels
    if labels.dtype.kind in ("U", "S", "O"):
        # String labels
        unique, counts = np.unique(labels, return_counts=True)
        label_to_count = dict(zip(unique, counts))
        counts = [label_to_count.get(name, 0) for name in class_names]
    else:
        # Integer labels
        counts = [np.sum(labels == i) for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(class_names)), counts, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Count")
    ax.set_title(title or "Class Distribution")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_metric_comparison(
    metrics: dict[str, float],
    *,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    ylabel: str = "Score",
) -> plt.Figure:
    """
    Bar plot comparing different metrics or models.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary mapping metric/model names to values.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    ylabel : str
        Y-axis label.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values, color="steelblue")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Metric Comparison")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def make_gif(
    outfile: str | Path,
    files: Sequence[str | Path],
    *,
    fps: int = 10,
    loop: int = 0,
    display: bool = False,
) -> Path:
    """
    Create an animated GIF from a sequence of image files.

    Parameters
    ----------
    outfile : str or Path
        Output GIF file path.
    files : Sequence[str or Path]
        List of image file paths to combine.
    fps : int
        Frames per second.
    loop : int
        Number of loops (0 = infinite).
    display : bool
        Whether to display the GIF in Jupyter notebook.

    Returns
    -------
    Path
        Path to the created GIF file.
    """
    import PIL.Image

    outfile = Path(outfile)
    imgs = [PIL.Image.open(str(f)) for f in files]
    imgs[0].save(
        fp=str(outfile),
        format="gif",
        append_images=imgs[1:],
        save_all=True,
        duration=int(1000 / fps),
        loop=loop,
    )

    if display:
        try:
            import IPython.display

            im = IPython.display.Image(filename=str(outfile))
            im.reload()
            IPython.display.display(im)
        except ImportError:
            pass

    return outfile


def plot_event(
    storm_data: dict[str, np.ndarray],
    storm_id: str = "",
    *,
    output_gif: bool = False,
    save_gif: bool = False,
    gif_dir: str | Path = ".",
    frame_indices: Sequence[int] | None = None,
    figsize: tuple[float, float] = (16, 4),
    dpi: int = 150,
    show: bool = True,
) -> plt.Figure | Path | None:
    """
    Plot a storm event with all image types and lightning overlay.

    This function can either display static frames or create an animated GIF
    showing all 36 frames of the storm.

    Parameters
    ----------
    storm_data : dict[str, np.ndarray]
        Dictionary containing storm data with keys:
        'vis', 'ir069', 'ir107', 'vil', 'lght'.
        Each image array should be shape (H, W, T).
    storm_id : str
        Storm ID for labeling (e.g., 'S778114').
    output_gif : bool
        If True, creates an animated GIF instead of static plots.
    save_gif : bool
        If True and output_gif=True, keeps the GIF file after display.
    gif_dir : str or Path
        Directory to save the GIF file.
    frame_indices : Sequence[int], optional
        Which frames to plot (only used when output_gif=False).
        Defaults to [0, 17, 34].
    figsize : tuple
        Figure size for each frame.
    dpi : int
        DPI for saved images.
    show : bool
        If True, calls plt.show() for each frame (set False for testing
        or non-interactive backends).

    Returns
    -------
    plt.Figure or Path or None
        - If output_gif=False: returns the last Figure
        - If output_gif=True and save_gif=True: returns Path to GIF
        - Otherwise: None
    """
    # Get lightning data
    lght = storm_data.get("lght", np.array([]).reshape(0, 5))
    t = lght[:, 0] if lght.shape[0] > 0 else np.array([])

    def plot_frame(ti: int, save_path: str | None = None) -> plt.Figure:
        """Plot a single frame."""
        # Find lightning strikes in current frame (±2.5 min window)
        if len(t) > 0:
            f = (t >= ti * 5 * 60 - 2.5 * 60) & (t < ti * 5 * 60 + 2.5 * 60)
        else:
            f = np.array([], dtype=bool)

        fig, axs = plt.subplots(1, 4, figsize=figsize)
        fig.suptitle(f"Event: {storm_id}, Frame: {ti}, Time: {ti * 5} min")

        # VIS
        if "vis" in storm_data:
            axs[0].imshow(
                storm_data["vis"][:, :, ti], vmin=0, vmax=10000, cmap="gray"
            )
        axs[0].set_title("Visible")
        axs[0].axis("off")

        # IR069
        if "ir069" in storm_data:
            axs[1].imshow(
                storm_data["ir069"][:, :, ti],
                vmin=-8000,
                vmax=-1000,
                cmap="viridis",
            )
        axs[1].set_title("Infrared (Water Vapor)")
        axs[1].axis("off")

        # IR107
        if "ir107" in storm_data:
            axs[2].imshow(
                storm_data["ir107"][:, :, ti],
                vmin=-7000,
                vmax=2000,
                cmap="inferno",
            )
        axs[2].set_title("Infrared (Cloud/Surface Temp)")
        axs[2].axis("off")

        # VIL with lightning overlay
        if "vil" in storm_data:
            axs[3].imshow(
                storm_data["vil"][:, :, ti], vmin=0, vmax=255, cmap="turbo"
            )
        axs[3].set_title("Radar (VIL)")
        axs[3].axis("off")

        # Add lightning markers
        if lght.shape[0] > 0 and np.any(f):
            axs[3].scatter(
                lght[f, 3], lght[f, 4], marker="x", s=30, c="tab:red"
            )
        axs[3].set_xlim(0, 384)
        axs[3].set_ylim(384, 0)

        if save_path:
            fig.savefig(
                save_path,
                bbox_inches="tight",
                dpi=dpi,
                pad_inches=0.02,
                facecolor="white",
            )
            plt.close(fig)

        return fig

    gif_dir = Path(gif_dir)

    if output_gif:
        # Create GIF from all 36 frames
        temp_files = []
        for ti in range(36):
            temp_file = gif_dir / f"_temp_{storm_id}_{ti}.png"
            plot_frame(ti, save_path=str(temp_file))
            temp_files.append(temp_file)

        gif_path = gif_dir / f"{storm_id}.gif"
        make_gif(gif_path, temp_files, display=True)

        # Clean up temp files
        for temp_file in temp_files:
            os.remove(temp_file)

        if not save_gif:
            os.remove(gif_path)
            return None
        return gif_path
    else:
        # Plot static frames
        if frame_indices is None:
            frame_indices = [0, 17, 34]

        fig = None
        for ti in frame_indices:
            fig = plot_frame(ti)
            if show:
                plt.show()
        return fig


def plot_event_from_file(
    h5_path: str | Path,
    storm_id: str,
    *,
    output_gif: bool = False,
    save_gif: bool = False,
    gif_dir: str | Path = ".",
    frame_indices: Sequence[int] | None = None,
) -> plt.Figure | Path | None:
    """
    Load and plot a storm event directly from an HDF5 file.

    This is a convenience wrapper around plot_event that handles data loading.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file (e.g., 'data/train.h5').
    storm_id : str
        Storm ID to plot (e.g., 'S778114').
    output_gif : bool
        If True, creates an animated GIF.
    save_gif : bool
        If True, keeps the GIF file.
    gif_dir : str or Path
        Directory for GIF output.
    frame_indices : Sequence[int], optional
        Which frames to plot (when not creating GIF).

    Returns
    -------
    plt.Figure or Path or None
        See plot_event for return value details.
    """
    from .data import load_event_arrays, open_h5

    with open_h5(h5_path) as f:
        storm_data = load_event_arrays(
            f,
            storm_id=storm_id,
            img_types=["vis", "ir069", "ir107", "vil", "lght"],
        )

    return plot_event(
        storm_data,
        storm_id=storm_id,
        output_gif=output_gif,
        save_gif=save_gif,
        gif_dir=gif_dir,
        frame_indices=frame_indices,
    )


def create_animation(
    frames: np.ndarray,
    *,
    outfile: str | Path = "animation.gif",
    fps: int = 10,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title_func: Callable[[int], str] | None = None,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 100,
    cleanup: bool = True,
) -> Path:
    """
    Create an animated GIF from a sequence of frames.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (T, H, W) or (H, W, T).
    outfile : str or Path
        Output GIF file path.
    fps : int
        Frames per second.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float, optional
        Color scale limits.
    title_func : Callable[[int], str], optional
        Function that takes frame index and returns title string.
    figsize : tuple
        Figure size.
    dpi : int
        Image DPI.
    cleanup : bool
        Whether to remove temporary frame images.

    Returns
    -------
    Path
        Path to the created GIF file.
    """
    outfile = Path(outfile)

    # Handle different input shapes
    if frames.ndim == 3 and frames.shape[2] <= 36:
        frames = np.transpose(frames, (2, 0, 1))

    n_frames = frames.shape[0]
    if vmin is None:
        vmin = frames.min()
    if vmax is None:
        vmax = frames.max()

    temp_files = []
    for i in range(n_frames):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frames[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        if title_func:
            ax.set_title(title_func(i))
        else:
            ax.set_title(f"Frame {i}")

        temp_file = outfile.parent / f"_temp_frame_{i}.png"
        fig.savefig(temp_file, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        temp_files.append(temp_file)

    make_gif(outfile, temp_files, fps=fps)

    if cleanup:
        for f in temp_files:
            os.remove(f)

    return outfile


def plot_task3_training_history(
    history: dict[str, list[float]],
    *,
    figsize: tuple[float, float] = (16, 5),
    title: str | None = None,
) -> plt.Figure:
    """
    Plot Training/Validation Loss and Validation Macro F1 side-by-side.

    Parameters
    ----------
    history : dict
        Dictionary with keys 'train_loss', 'val_loss', 'val_f1'.
    figsize : tuple
        Total figure size (W, H).
    title : str, optional
        Prefix for sub-titles.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Training & Validation Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r--", label="Validation Loss")
    ax1.set_title(f"{title + ': ' if title else ''}Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Right: Validation Macro F1-Score
    if "val_f1" in history:
        ax2.plot(epochs, history["val_f1"], "g-o", label="Validation Macro F1")
        ax2.set_title(
            f"{title + ': ' if title else ''}Validation Macro F1-Score"
        )
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Macro F1 Score")
        ax2.legend()
        ax2.grid(True)

    fig.tight_layout()
    return fig


def plot_cm_from_preds(
    targets: np.ndarray | list,
    preds: np.ndarray | list,
    *,
    class_names: list[str] | None = None,
    title: str | None = "Confusion Matrix",
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Helper to calculate and plot a confusion matrix
    directly from label vectors.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, preds)
    return plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=title,
        figsize=figsize,
        cmap=cmap,
        normalize=False,
    )


__all__ = [
    "DEFAULT_CMAPS",
    "plot_single_image",
    "plot_image_grid",
    "plot_frame_sequence",
    "plot_storm_overview",
    "plot_lightning_scatter",
    "plot_lightning_heatmap",
    "plot_prediction_comparison",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_class_distribution",
    "plot_metric_comparison",
    "make_gif",
    "plot_event",
    "plot_event_from_file",
    "create_animation",
    "plot_task3_training_history",
    "plot_cm_from_preds",
]
