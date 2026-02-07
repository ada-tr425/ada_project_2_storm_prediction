"""
baseline_task4.py

Task 4 baseline:
- Use official PreprocessTask4 to build lightning volume labels.
- Model: UNet3D_ResSE (predicting_unpredictable.models.task4).
- Loss: FocalLoss on [B, T, 1, H, W] lightning probability.
"""

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from sklearn.metrics import accuracy_score, f1_score, recall_score

# Adjust these imports to your package layout
from predicting_unpredictable import data
from predicting_unpredictable.models.task4 import UNet3D_ResSE
from predicting_unpredictable.preprocess import PreprocessTask4

import h5py
import cv2

import torch.nn.functional as F
from predicting_unpredictable.models.task4 import UNet3D_TimeBucket

# ========================= Dataset & DataModule =========================


class Task4BaselineDataset(Dataset):
    """
    Baseline dataset for Task 4.
    Uses official PreprocessTask4.
    x: [T=36, C=4, H=384, W=384]
    y: [T=36, 1, H, W]  (binary lightning volume)
    """

    def __init__(
        self,
        storm_ids: List[str],
        events_df: pd.DataFrame,
        h5_path: str = "data/train.h5",
    ):
        self.storm_ids = list(storm_ids)
        self.h5_path = h5_path

        # events_df must contain at least: id, event_type
        self.events_df = events_df.set_index("id")

        self.preprocess = PreprocessTask4()

    def __len__(self) -> int:
        return len(self.storm_ids)

    def __getitem__(self, idx: int):
        storm_id = self.storm_ids[idx]
        event_type_str = self.events_df.loc[storm_id, "event_type"]

        # 1. Load raw arrays from HDF5
        with data.open_h5(self.h5_path) as f:
            storm_data = data.load_event_arrays(
                f,
                storm_id=storm_id,
                img_types=["vis", "ir069", "ir107", "vil"],
            )

        # 2. Apply preprocessing
        # x: (36, 4, 384, 384), y: (36, 1, 384, 384)
        x, y, _ = self.preprocess(storm_data, event_type_str)

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)

        return x, y, {"storm_id": storm_id}


class Task4BaselineDataModule:
    """
    Simple data module for train/val loaders.
    """

    def __init__(
        self,
        events_df: pd.DataFrame,
        h5_path: str = "data/train.h5",
        batch_size: int = 2,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        with data.open_h5(h5_path) as f:
            all_ids = list(f.keys())

        rng = random.Random(seed)
        rng.shuffle(all_ids)

        n_total = len(all_ids)
        n_train = int(train_ratio * n_total)
        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train:]

        self.train_dataset = Task4BaselineDataset(
            storm_ids=train_ids,
            events_df=events_df,
            h5_path=h5_path,
        )
        self.val_dataset = Task4BaselineDataset(
            storm_ids=val_ids,
            events_df=events_df,
            h5_path=h5_path,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


# ========================= Focal Loss =========================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.
                Tensor) -> torch.Tensor:
        """
        pred: probabilities in [0,1], same shape as target
        target: 0/1 float tensor
        """
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        pt = pred * target + (1 - pred) * (1 - target)  # p if y=1 else 1-p
        w = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -w * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()


# ========================= Train / Validate =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0

    for x, y, meta in loader:
        x = x.to(device)  # [B,T,C,H,W]
        y = y.to(device)  # [B,T,1,H,W]

        optimizer.zero_grad()
        pred = model(x)   # [B,T,1,H,W]
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.3,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0

    all_preds = []
    all_tgts = []

    for x, y, meta in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)  # [B,T,1,H,W], probabilities

        loss = criterion(pred, y)
        total_loss += loss.item()
        n_batches += 1

        preds_bin = (pred >= threshold).to(torch.int64)
        tgts_bin = y.to(torch.int64)

        all_preds.append(preds_bin.cpu().view(-1).numpy())
        all_tgts.append(tgts_bin.cpu().view(-1).numpy())

    if len(all_preds) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    all_preds = np.concatenate(all_preds)
    all_tgts = np.concatenate(all_tgts)

    assert set(np.unique(all_preds)).issubset({0, 1})
    assert set(np.unique(all_tgts)).issubset({0, 1})

    avg_loss = total_loss / max(1, n_batches)
    acc = accuracy_score(all_tgts, all_preds)
    f1 = f1_score(all_tgts, all_preds, average="binary")
    rec = recall_score(all_tgts, all_preds, average="binary")

    return avg_loss, acc, f1, rec


# ========================= Main train function =========================

def train_baseline_task4(
    train_events_csv: str = "data/train-events.csv",
    h5_path: str = "data/train.h5",
    num_epochs: int = 6,
    batch_size: int = 2,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    threshold_val: float = 0.05,
    save_path: str = "best_unet3d_resse_task4.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    events_df = pd.read_csv(train_events_csv)

    dm = Task4BaselineDataModule(
        events_df=events_df,
        h5_path=h5_path,
        batch_size=batch_size,
        train_ratio=0.8,
        seed=42,
    )
    train_loader = dm.train_loader
    val_loader = dm.val_loader

    model = UNet3D_ResSE(in_ch=4, base_ch=16).to(device)
    criterion = FocalLoss(alpha=0.9, gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_state = None
    best_rec = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion)
        val_loss, val_acc, val_f1, val_rec = validate(
            model, val_loader, device, criterion, threshold=threshold_val
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"val_f1={val_f1:.4f}, "
            f"val_recall={val_rec:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_rec = val_rec
            best_state = model.state_dict()

    if best_state is not None:
        torch.save(best_state, save_path)
        print("Saved best baseline model to:", save_path)
        print("Best val F1:", best_val_f1)
        print("Best val Recall:", best_rec)


if __name__ == "__main__":
    train_baseline_task4()

"""
improved_timebucket_task4.py

Improved Task 4 model:
- Labels: time-bucket one-hot + no-flash, shape [T, C_out, H, W].
- Model: UNet3D_TimeBucket (3D U-Net with Res+SE blocks).
- Loss: weighted soft cross-entropy + auxiliary flash vs no-flash focal BCE.
"""

# ========================= Constants =========================

T = 36                    # number of frames
FRAME_DT = 300.0          # seconds per frame
BUCKET_DT = 10.0
W_BUCKETS = int(FRAME_DT // BUCKET_DT)  # number of time buckets per frame
C_OUT = W_BUCKETS + 1     # time buckets + no-flash

H_SMALL, W_SMALL = 384, 384
H_ORIG,  W_ORIG = 384, 384

H5_PATH = "data/train.h5"


# ========================= Data loading & preprocessing =====================

def load_event(storm_id: str):
    """
    Load one event from train.h5.

    Returns a dict with keys:

    - "vis":   (384,384,36)
    - "ir069": (192,192,36)
    - "ir107": (192,192,36)
    - "vil":   (384,384,36)
    - "lght":  (N,5)  [t, lat, lon, x, y]
    """
    with h5py.File(H5_PATH, "r") as f:
        grp = f[storm_id]
        event = {
            "vis":   grp["vis"][:].astype(np.float32),
            "ir069": grp["ir069"][:].astype(np.float32),
            "ir107": grp["ir107"][:].astype(np.float32),
            "vil":   grp["vil"][:].astype(np.float32),
            "lght":  grp["lght"][:].astype(np.float32),
        }
    return event


def upsample_to_384(x_192_hwT: np.ndarray) -> np.ndarray:
    """
    Upsample (192,192,T) to (384,384,T) using bilinear interpolation.
    """
    H, W, T_ = x_192_hwT.shape
    out = np.zeros((384, 384, T_), dtype=np.float32)
    for t in range(T_):
        out[:, :, t] = cv2.resize(
            x_192_hwT[:, :, t],
            (384, 384),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


def build_x_36_from_event(event, H_small=H_SMALL, W_small=W_SMALL):
    """
    Build input x: [T=36, C=4, H_small, W_small].
    Channels: vis, ir069(upsampled to 384), ir107(upsampled to 384), vil.
    """
    vis = event["vis"]        # (384,384,36)
    vil = event["vil"]
    ir069 = upsample_to_384(event["ir069"])        # (384,384,36)
    ir107 = upsample_to_384(event["ir107"])        # (384,384,36)

    H, W, T_ = vis.shape
    assert T_ == T, f"expected T={T}, got {T_}"

    # (H,W,4,T)
    stack_hwct = np.stack([vis, ir069, ir107, vil], axis=2)
    # (T,4,H,W)
    x_tchw = np.transpose(stack_hwct, (3, 2, 0, 1))

    x_small = np.zeros((T, 4, H_small, W_small), dtype=np.float32)
    for t_idx in range(T):
        for c in range(4):
            x_small[t_idx, c] = cv2.resize(
                x_tchw[t_idx, c],
                (W_small, H_small),
                interpolation=cv2.INTER_AREA
            )
    return x_small


def build_time_bucket_onehot(
    event_lght: np.ndarray,  # [N,5] (t, lat, lon, x, y), t in seconds
    H_small=H_SMALL,
    W_small=W_SMALL,
    frame_dt=FRAME_DT,
    bucket_dt=BUCKET_DT,
    H_orig=H_ORIG,
    W_orig=W_ORIG,
    T_=T,
    use_earliest=True
):
    """
    Build time-bucket one-hot tensor.

    Returns:
        y_onehot: [T, C_out, H_small, W_small]
      - channels 0..W_BUCKETS-1: time buckets
      - channel W_BUCKETS: no-flash
    """
    w = int(frame_dt // bucket_dt)
    C = w + 1
    y_onehot = np.zeros((T_, C, H_small, W_small), dtype=np.float32)
    y_onehot[:, w, :, :] = 1.0  # default no-flash = 1

    if event_lght.size == 0:
        return y_onehot

    t_arr = event_lght[:, 0].astype(np.float32)  # seconds
    x_arr = event_lght[:, 3].astype(np.float32)
    y_arr = event_lght[:, 4].astype(np.float32)

    x_small = np.clip(
        (x_arr / W_orig * W_small).round().astype(int),
        0, W_small - 1
    )
    y_small = np.clip(
        (y_arr / H_orig * H_small).round().astype(int),
        0, H_small - 1
    )

    filled = np.zeros((T_, H_small, W_small), dtype=np.uint8)

    for t_sec, xs, ys in zip(t_arr, x_small, y_small):
        if t_sec < 0 or t_sec >= T_ * frame_dt:
            continue
        k = int(t_sec // frame_dt)          # frame index
        rel_t = t_sec - k * frame_dt
        bucket_idx = int(rel_t // bucket_dt)  # 0..w-1

        if use_earliest:
            if filled[k, ys, xs] == 0:
                y_onehot[k, w, ys, xs] = 0.0
                y_onehot[k, bucket_idx, ys, xs] = 1.0
                filled[k, ys, xs] = 1
        else:
            # same as earliest here; can be extended to accumulate counts
            if filled[k, ys, xs] == 0:
                y_onehot[k, w, ys, xs] = 0.0
                y_onehot[k, bucket_idx, ys, xs] = 1.0
                filled[k, ys, xs] = 1

    return y_onehot


# ========================= Dataset & DataLoader =========================

class TimeBucketDatasetOneHot(Dataset):
    """
    Dataset that returns:
      x: [T,4,H_small,W_small]
      y: [T,C_out,H_small,W_small] (time-bucket one-hot)
      meta: { "storm_id":... }
    """

    def __init__(
        self,
        ids: List[str],
        H_small: int = H_SMALL,
        W_small: int = W_SMALL,
        frame_dt: float = FRAME_DT,
        bucket_dt: float = BUCKET_DT,
    ):
        self.ids = list(ids)
        self.H_small = H_small
        self.W_small = W_small
        self.frame_dt = frame_dt
        self.bucket_dt = bucket_dt

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        storm_id = self.ids[idx]
        event = load_event(storm_id)

        x_small = build_x_36_from_event(
            event, H_small=self.H_small, W_small=self.W_small
        )  # [T,4,H_small,W_small]

        y_onehot = build_time_bucket_onehot(
            event["lght"],
            H_small=self.H_small, W_small=self.W_small,
            frame_dt=self.frame_dt, bucket_dt=self.bucket_dt,
            H_orig=H_ORIG, W_orig=W_ORIG, T_=T, use_earliest=True
        )  # [T,C_out,H_small,W_small]

        x = torch.from_numpy(x_small)     # float32
        y = torch.from_numpy(y_onehot)    # float32
        meta = {"storm_id": storm_id}
        return x, y, meta


def build_timebucket_dataloaders(
    h5_path: str = H5_PATH,
    batch_size: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Build train/val DataLoaders for time-bucket model.
    """
    with h5py.File(h5_path, "r") as f:
        all_ids = list(f.keys())

    rng = random.Random(seed)
    rng.shuffle(all_ids)

    n_total = len(all_ids)
    n_train = int(train_ratio * n_total)
    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:]

    ds_train = TimeBucketDatasetOneHot(
        train_ids, H_small=H_SMALL, W_small=W_SMALL)
    ds_val = TimeBucketDatasetOneHot(
        val_ids,   H_small=H_SMALL, W_small=W_SMALL)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader

# ========================= Loss functions & metrics =========================


def soft_ce_weighted(
    logits: torch.Tensor,         # [B,T,C,H,W]
    target_probs: torch.Tensor,   # [B,T,C,H,W], one-hot / soft labels
    class_weights: torch.Tensor = None,  # [C]
    pixel_mask: torch.Tensor = None,     # [B,T,H,W] or [B,T,1,H,W]
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Soft cross-entropy with optional class weights and pixel weights.
    """
    if label_smoothing > 0:
        C = logits.shape[2]
        smooth = label_smoothing / max(1, C - 1)
        target_probs = (1 - label_smoothing) * target_probs + smooth

    log_probs = F.log_softmax(logits, dim=2)  # [B,T,C,H,W]

    if class_weights is not None:
        cw = class_weights.view(1, 1, -1, 1, 1)
        target_w = target_probs * cw
    else:
        target_w = target_probs

    loss_map = -(target_w * log_probs).sum(dim=2)  # [B,T,H,W]

    if pixel_mask is not None:
        if pixel_mask.dim() == 5:
            pixel_mask = pixel_mask.squeeze(2)      # [B,T,H,W]
        loss_map = loss_map * pixel_mask
        denom = torch.clamp(pixel_mask.sum(), min=1.0)
    else:
        denom = loss_map.numel()

    loss = loss_map.sum() / denom
    return loss


def focal_bce(p: torch.Tensor, y: torch.Tensor,
              alpha: float = 0.85, gamma: float = 1.0,
              eps: float = 1e-7) -> torch.Tensor:
    """
    Focal loss for probabilities p against binary targets y.
    """
    p = torch.clamp(p, eps, 1 - eps)
    loss_pos = -alpha * (1 - p) ** gamma * torch.log(p) * y
    loss_neg = -(1 - alpha) * p ** gamma * torch.log(1 - p) * (1 - y)
    return (loss_pos + loss_neg).mean()


def compute_binary_metrics(pred_flash: torch.Tensor, gt_flash: torch.Tensor):
    """
    Compute accuracy/precision/recall/f1 for binary flash vs no-flash.
    """
    pred = pred_flash.to(torch.bool).view(-1)
    gt = gt_flash.to(torch.bool).view(-1)

    TP = torch.sum(pred & gt).item()
    FP = torch.sum(pred & ~gt).item()
    FN = torch.sum(~pred & gt).item()
    TN = torch.sum(~pred & ~gt).item()

    total = TP + FP + FN + TN
    accuracy = (TP + TN) / max(1, total)
    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }

# ========================= Validation =========================


@torch.no_grad()
def validate_fixed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    threshold: float = 0.5,
    label_smoothing: float = 0.05
):
    model.eval()
    total_loss, n = 0.0, 0
    all_pred_flash, all_gt_flash = [], []

    for x, y_onehot, _ in loader:
        x = x.to(device)          # [B,T,4,H,W]
        y = y_onehot.to(device)   # [B,T,C_out,H,W]

        logits = model(x)         # [B,T,C_out,H,W]

        gt_flash = (1.0 - y[:, :, -1, :, :])  # [B,T,H,W]
        mask = gt_flash.unsqueeze(2)          # [B,T,1,H,W]

        # 3x3 dilation for pixel weights
        mask_dil = F.max_pool2d(
            mask.view(-1, 1, H_SMALL, W_SMALL),
            kernel_size=3, stride=1, padding=1
        ).view(mask.shape)

        pixel_mask = torch.where(
            mask_dil > 0,
            torch.tensor(2.0, device=device),
            torch.tensor(1.0, device=device)
        )

        loss_main = soft_ce_weighted(
            logits, y,
            class_weights=class_weights,
            pixel_mask=pixel_mask,
            label_smoothing=label_smoothing
        )

        probs = torch.softmax(logits, dim=2)
        p_flash = probs[:, :, :W_BUCKETS].sum(dim=2)  # [B,T,H,W]
        loss_aux = focal_bce(p_flash, gt_flash, alpha=0.6, gamma=0.5)

        lambda_aux = 0.2
        loss = loss_main + lambda_aux * loss_aux

        total_loss += loss.item()
        n += 1

        pred_flash = (p_flash >= threshold)
        gt_bin = (gt_flash > 0.5)

        all_pred_flash.append(pred_flash.cpu())
        all_gt_flash.append(gt_bin.cpu())

    pred_cat = torch.cat(all_pred_flash)
    gt_cat = torch.cat(all_gt_flash)

    metrics = compute_binary_metrics(pred_cat, gt_cat)
    avg_loss = total_loss / max(1, n)
    return avg_loss, metrics


# ========================= One training epoch =========================

def train_one_epoch_timebucket(
    model,
    loader,
    optimizer,
    device,
    class_weights,
    label_smoothing=0.05,
):
    model.train()
    total_loss, n = 0.0, 0

    for x, y_onehot, _ in loader:
        x = x.to(device)
        y = y_onehot.to(device)

        optimizer.zero_grad()

        logits = model(x)  # [B,T,C_out,H,W]

        gt_flash = (1.0 - y[:, :, -1, :, :])  # [B,T,H,W]
        mask = gt_flash.unsqueeze(2)

        mask_dil = F.max_pool2d(
            mask.view(-1, 1, H_SMALL, W_SMALL),
            kernel_size=3, stride=1, padding=1
        ).view(mask.shape)

        pixel_mask = torch.where(
            mask_dil > 0,
            torch.tensor(2.0, device=device),
            torch.tensor(1.0, device=device)
        )

        loss_main = soft_ce_weighted(
            logits, y,
            class_weights=class_weights,
            pixel_mask=pixel_mask,
            label_smoothing=label_smoothing
        )

        probs = torch.softmax(logits, dim=2)
        p_flash = probs[:, :, :W_BUCKETS].sum(dim=2)  # [B,T,H,W]
        loss_aux = focal_bce(p_flash, gt_flash, alpha=0.5, gamma=1.0)

        lambda_aux = 0.1
        loss = loss_main + lambda_aux * loss_aux

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(1, n)

# ========================= Main train function =========================


def train_timebucket_task4(
    h5_path: str = H5_PATH,
    num_epochs: int = 6,
    batch_size: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    threshold_val: float = 0.5,
    save_path: str = "best_time_bucket_model_onehot_focal.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = build_timebucket_dataloaders(
        h5_path=h5_path,
        batch_size=batch_size,
        train_ratio=0.8,
        seed=42,
    )

    model = UNet3D_TimeBucket(
        in_ch=4, base_ch=8, T=T, C_out=C_OUT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # class weights: emphasize flash buckets slightly more than no-flash
    class_weights = torch.ones(C_OUT, dtype=torch.float32, device=device)
    class_weights[:W_BUCKETS] = 2.0   # flash buckets
    class_weights[-1] = 1.0           # no-flash

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch_timebucket(
            model, train_loader, optimizer, device,
            class_weights, label_smoothing=0.1
        )

        val_loss, metrics = validate_fixed(
            model, val_loader, device,
            class_weights, threshold=threshold_val,
            label_smoothing=0.1
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}, "
            f"val_recall={metrics['recall']:.4f}, "
            f"val_precision={metrics['precision']:.4f}, "
            f"val_f1={metrics['f1']:.4f}"
        )

        # Use validation loss as early-stopping criterion
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        torch.save(best_state, save_path)
        print("Saved best time-bucket model to:", save_path)
        print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    # Run training when the script is invoked directly.
    train_timebucket_task4()
