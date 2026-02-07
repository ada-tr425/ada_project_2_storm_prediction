"""
Reusable preprocessing helpers.

Guiding principle: keep these helpers small and notebook-friendly.
They operate primarily on numpy arrays from the H5 files and return torch
tensors in the canonical (T, C, H, W) layout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import NORM_STATS, EVENT_TYPES


def vil_to_tchw(vil_uint8_hwt: np.ndarray) -> torch.Tensor:
    """Convert VIL (H,W,T) uint8 [0,255] -> torch float32 (T,1,H,W) [0,1]."""
    if vil_uint8_hwt.ndim != 3:
        raise ValueError(f"Expected (H,W,T), got {vil_uint8_hwt.shape}")
    x = torch.from_numpy(vil_uint8_hwt.astype(np.float32, copy=False))
    x = x.permute(2, 0, 1).unsqueeze(1)  # (T,1,H,W)
    return x / 255.0


def int16_image_to_tchw(img_int16_hwt: np.ndarray) -> torch.Tensor:
    """Convert int16 image (H,W,T) -> float32 (T,1,H,W) (no norm)."""
    if img_int16_hwt.ndim != 3:
        raise ValueError(f"Expected (H,W,T), got {img_int16_hwt.shape}")
    x = torch.from_numpy(img_int16_hwt.astype(np.float32, copy=False))
    return x.permute(2, 0, 1).unsqueeze(1)


def upsample_192_to_384(tchw: torch.Tensor) -> torch.Tensor:
    """Upsample tensor 192x192 -> 384x384 (bilinear interpolation)."""
    if not isinstance(tchw, torch.Tensor):
        raise TypeError("upsample_192_to_384 expects a torch.Tensor")
    if tchw.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(tchw.shape)}")
    h, w = int(tchw.shape[-2]), int(tchw.shape[-1])
    if (h, w) != (192, 192):
        raise ValueError(f"Expected 192x192, got {(h, w)}")
    return F.interpolate(
        tchw, size=(384, 384), mode="bilinear", align_corners=False
    )


def normalize_tensor(
    t: torch.Tensor,
    *,
    norm_type: str,
    mean: float | None = None,
    std: float | None = None,
    data_min: float | None = None,
    data_max: float | None = None,
    target_min: float = 0.0,
    target_max: float = 1.0,
    eps: float = 1e-6,
    log_scale: bool = True
) -> torch.Tensor:
    """Normalize tensor with zscore/minmax/log (unified interface)."""
    if not isinstance(t, torch.Tensor):
        raise TypeError("normalize_tensor expects a torch.Tensor")

    norm_type = norm_type.lower()
    valid_types = ["zscore", "minmax", "log"]
    if norm_type not in valid_types:
        raise ValueError(
            f"Invalid norm_type: {norm_type} (choose {valid_types})"
        )

    if norm_type == "zscore":
        if mean is None or std is None:
            raise ValueError("mean/std required for zscore")
        return (t - float(mean)) / float(std)

    elif norm_type == "minmax":
        if data_min is None or data_max is None:
            raise ValueError("data_min/data_max required for minmax")
        data_min, data_max = float(data_min), float(data_max)
        if data_max == data_min:
            raise ValueError(
                "data_max cannot equal data_min (division by zero error)"
            )
        scaled = (t - data_min) / (data_max - data_min)
        return scaled * (target_max - target_min) + target_min

    elif norm_type == "log":
        if data_min is None or data_max is None:
            raise ValueError("data_min/data_max required for log")
        if torch.any(t < 0):
            raise ValueError("log only supports non-negative data")

        t_log = torch.log(t + eps)
        if log_scale:
            log_min = np.log(data_min + eps)
            log_max = np.log(data_max + eps)
            if log_max == log_min:
                raise ValueError("log_max == log_min (no variance after log)")
            scaled = (t_log - log_min) / (log_max - log_min)
            return scaled * (target_max - target_min) + target_min
        return t_log


def stack_channels_tchw(channels: list[torch.Tensor]) -> torch.Tensor:
    """Stack (T,1,H,W) tensors -> (T,C,H,W)."""
    if not channels:
        raise ValueError("channels must be non-empty")
    if not all(isinstance(c, torch.Tensor) for c in channels):
        raise TypeError("all channels must be torch.Tensors")
    return torch.cat(channels, dim=1)


def rasterize_lightning(
    lght_n5: np.ndarray,
    t_bins: int = 36,
    frame_seconds: float = 300.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert lightning (N,5) -> per-frame counts (T,) + heatmaps."""
    counts = np.zeros(t_bins, dtype=np.float32)
    heatmaps = np.zeros((t_bins, 384, 384), dtype=np.float32)

    if lght_n5.shape[0] == 0:
        return torch.from_numpy(counts), torch.from_numpy(heatmaps)

    t_sec = lght_n5[:, 0]
    px = lght_n5[:, 3]
    py = lght_n5[:, 4]
    t_idx = np.floor(t_sec / frame_seconds).astype(int)

    mask = (
        (t_idx >= 0)
        & (t_idx < t_bins)
        & (px >= 0)
        & (px < 384)
        & (py >= 0)
        & (py < 384)
    )

    t_valid = t_idx[mask]
    x_valid = px[mask].astype(int)
    y_valid = py[mask].astype(int)

    np.add.at(counts, t_valid, 1)
    np.add.at(heatmaps, (t_valid, y_valid, x_valid), 1)

    return torch.from_numpy(counts), torch.from_numpy(heatmaps)


class PreprocessTask1:
    """
    Task1: VIL prediction (12->12)
    Input: vil
    Output: x(12), y(12)
    """

    def __call__(
        self, storm_data: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vil = vil_to_tchw(storm_data["vil"])
        x = vil[0:12]
        y = vil[12:24]
        return x, y


class PreprocessTask2:
    """
    Task2: VIL reconstruction.

    Input: vis / ir069 / ir107 / vil
    Output: x, y
    """

    def __call__(
        self, storm_data: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Target VIL: log normalization + clamp to [0,1]
        vil_raw = vil_to_tchw(storm_data["vil"])
        vil = normalize_tensor(
            vil_raw,
            norm_type="log",
            **NORM_STATS["vil"]
        )
        vil = torch.clamp(vil, 0.0, 1.0)

        # Input channels: zscore normalization
        vis = int16_image_to_tchw(storm_data["vis"])
        vis = normalize_tensor(vis, norm_type="zscore", **NORM_STATS["vis"])

        ir069 = int16_image_to_tchw(storm_data["ir069"])
        ir069 = normalize_tensor(
                ir069, norm_type="zscore", **NORM_STATS["ir069"]
            )
        ir069 = upsample_192_to_384(ir069)

        ir107 = int16_image_to_tchw(storm_data["ir107"])
        ir107 = normalize_tensor(
                ir107, norm_type="zscore", **NORM_STATS["ir107"]
                )
        ir107 = upsample_192_to_384(ir107)

        x = stack_channels_tchw([vis, ir069, ir107])
        return x, vil


class PreprocessTask3:
    """
    Task3: Event Classification.

    Input: images + event_type
    Output: x, y
    """
    def __call__(
        self,
        storm_data: dict[str, np.ndarray],
        event_type_str: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # VIL: log normalization (input channel 3)
        vil_raw = vil_to_tchw(storm_data["vil"])
        vil = normalize_tensor(
            vil_raw,
            norm_type="log",
            **NORM_STATS["vil"]
        )

        # Other channels: minmax normalization
        vis = int16_image_to_tchw(storm_data["vis"])
        vis = normalize_tensor(vis, norm_type="minmax", **NORM_STATS["vis"])

        ir069 = int16_image_to_tchw(storm_data["ir069"])
        ir069 = normalize_tensor(
            ir069, norm_type="minmax", **NORM_STATS["ir069"]
        )
        ir069 = upsample_192_to_384(ir069)

        ir107 = int16_image_to_tchw(storm_data["ir107"])
        ir107 = normalize_tensor(
            ir107, norm_type="minmax", **NORM_STATS["ir107"]
        )
        ir107 = upsample_192_to_384(ir107)

        # Stack all 4 channels (VIS, IR069, IR107, VIL)
        x = stack_channels_tchw([vis, ir069, ir107, vil])

        # Encode label
        try:
            label_idx = EVENT_TYPES.index(event_type_str)
        except ValueError:
            label_idx = -1
        y = torch.tensor(label_idx, dtype=torch.long)

        return x, y


class PreprocessTask4:
    """Task4: Lightning Prediction | Input: images + lght | Output: x, etc."""

    def __call__(
        self, storm_data: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reuse Task3 normalization logic (VIL log + others minmax)
        vil_raw = vil_to_tchw(storm_data["vil"])
        vil = normalize_tensor(
            vil_raw,
            norm_type="log",
            **NORM_STATS["vil"]
        )

        vis = int16_image_to_tchw(storm_data["vis"])
        vis = normalize_tensor(vis, norm_type="minmax", **NORM_STATS["vis"])

        ir069 = int16_image_to_tchw(storm_data["ir069"])
        ir069 = normalize_tensor(
            ir069, norm_type="minmax", **NORM_STATS["ir069"]
        )
        ir069 = upsample_192_to_384(ir069)

        ir107 = int16_image_to_tchw(storm_data["ir107"])
        ir107 = normalize_tensor(
            ir107, norm_type="minmax", **NORM_STATS["ir107"]
        )
        ir107 = upsample_192_to_384(ir107)

        x = stack_channels_tchw([vis, ir069, ir107, vil])

        # Process lightning targets
        lght = storm_data["lght"]
        y_counts, y_heatmaps = rasterize_lightning(lght)

        return x, y_counts, y_heatmaps
