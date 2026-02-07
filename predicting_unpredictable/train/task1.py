# predicting_unpredictable/train/task1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data import open_h5
from ..io import save_checkpoint
from ..models.task1 import BottleneckConvLSTMUNet, Task1CNNBaseline
from ..preprocess import vil_to_tchw


def build_index(
    h5_file,
    ids,
    in_len=12,
    out_len=12,
    stride: int = 1,
    offset: int = 0,
):

    assert stride >= 1
    idx = []
    win = in_len + out_len
    for sid in ids:
        T = int(h5_file[sid]["vil"].shape[-1])  # usually 36
        last = T - win
        if last < 0:
            continue
        for t in range(offset, last + 1, stride):
            idx.append((str(sid), int(t)))
    return idx


class VilWindowDatasetCNN(Dataset):

    def __init__(self, h5_path, index, in_len=12, out_len=12):
        self.h5_path = h5_path
        self.index = index
        self.in_len = in_len
        self.out_len = out_len
        self.f = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.f is None:
            self.f = open_h5(self.h5_path, "r")

        sid, t = self.index[i]
        dset = self.f[sid]["vil"]
        clip_uint8 = dset[:, :, t:t + self.in_len + self.out_len]

        clip = vil_to_tchw(clip_uint8)
        x = clip[:self.in_len].squeeze(1)
        y = clip[self.in_len:self.in_len + self.out_len].squeeze(1)
        return x, y


@torch.no_grad()
def evaluate_l1_mae(model, loader, device, clamp_pred=True):
    model.eval()
    l1 = nn.L1Loss()
    total_loss, total_mae, n = 0.0, 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred = model(xb)
        if clamp_pred:
            pred = pred.clamp(0.0, 1.0)

        loss = l1(pred, yb)
        mae = (pred - yb).abs().mean()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_mae += float(mae.item()) * bs
        n += bs

    return {"loss": total_loss / n, "mae": total_mae / n}


def fit_task1_cnn(
    *,
    h5_path: str,
    train_ids: list[str],
    val_ids: list[str],
    device,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_first_window_only: bool = True,
    clamp_pred: bool = True,
    save_best_path: str | None = None,
    train_stride: int = 1,
):
    with open_h5(h5_path, "r") as f:
        train_index = build_index(f, train_ids, 12, 12, stride=train_stride)
        if val_first_window_only:
            val_index = [(sid, 0) for sid in val_ids]
        else:
            val_index = build_index(f, val_ids, 12, 12)
    train_ds = VilWindowDatasetCNN(h5_path, train_index)
    val_ds = VilWindowDatasetCNN(h5_path, val_index)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = Task1CNNBaseline().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    l1 = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        print(f"[epoch {epoch}] start...")
        train_loss_sum, train_mae_sum, n = 0.0, 0.0, 0

        for it, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            if clamp_pred:
                pred = pred.clamp(0.0, 1.0)

            loss = l1(pred, yb)
            mae = (pred - yb).abs().mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            train_mae_sum += mae.item() * bs
            n += bs

            if it % 50 == 0:
                print(
                    f"Epoch {epoch} iter {it}/{len(train_loader)} "
                    f"loss {loss.item():.5f}"
                )

        train_metrics = {"loss": train_loss_sum / n, "mae": train_mae_sum / n}

        val_metrics = evaluate_l1_mae(
            model, val_loader, device, clamp_pred=clamp_pred
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_metrics['loss']:.5f} "
            f"mae {train_metrics['mae']:.5f} || "
            f"val loss {val_metrics['loss']:.5f} mae {val_metrics['mae']:.5f}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "val_mae": val_metrics["mae"],
            }
            if save_best_path is not None:
                torch.save(best_state, save_best_path)
                print(f"saved best -> {save_best_path}")

    return model, best_state


class VilSeqDataset(Dataset):
    """
    return:
      x: (T_in, 1, H, W)  float32 in [0,1]
      y: (T_out, H, W)    float32 in [0,1]
    """

    def __init__(self, h5_path, index, in_len=12, out_len=12):
        self.h5_path = h5_path
        self.index = index
        self.in_len = in_len
        self.out_len = out_len
        self.f = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.f is None:
            self.f = open_h5(self.h5_path, "r")

        sid, t = self.index[i]
        dset = self.f[sid]["vil"]  # (H,W,T) uint8
        clip_uint8 = dset[:, :, t:t + self.in_len + self.out_len]
        # (H,W,24) uint8

        clip = vil_to_tchw(clip_uint8)  # (24,1,H,W) in [0,1]
        x = clip[:self.in_len]  # (12,1,H,W)
        y = clip[self.in_len:self.in_len + self.out_len].squeeze(1)  # (12,H,W)
        return x, y


def weighted_lead_l1(pred, target, w_pos=5.0, thr=20.0 / 255.0, power=0.5):
    w_pix = torch.ones_like(target)
    w_pix = w_pix + (target > thr).float() * (w_pos - 1.0)

    T = target.shape[1]
    w_lead = (
        torch.arange(1, T + 1, device=target.device, dtype=target.dtype)
        ** power
    )
    w_lead = w_lead / w_lead.mean()
    w_lead = w_lead.view(1, T, 1, 1)

    w = w_pix * w_lead
    return (w * (pred - target).abs()).mean()


def intensity_weight(target):
    w = torch.ones_like(target)
    w = w + (target > 20.0 / 255.0).float() * 2.0
    w = w + (target > 80.0 / 255.0).float() * 3.0
    w = w + (target > 160.0 / 255.0).float() * 5.0
    return w


def tv_loss(pred):
    # pred: (B,T,H,W)
    dx = (pred[..., :, 1:] - pred[..., :, :-1]).abs().mean()
    dy = (pred[..., 1:, :] - pred[..., :-1, :]).abs().mean()
    return dx + dy


def _gaussian_kernel(window_size, sigma, device, dtype):
    coords = (
        torch.arange(window_size, device=device, dtype=dtype)
        - window_size // 2
    )
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-8)
    k2d = g[:, None] * g[None, :]
    return k2d.view(1, 1, window_size, window_size)


def ssim_loss(
    pred,
    target,
    window_size=11,
    sigma=1.5,
    C1=0.01 ** 2,
    C2=0.03 ** 2,
):
    """
    pred/target: (B,T,H,W) or (B,1,H,W). The input should be in [0,1] range.
    Return: 1-SSIM(Smaller is better)
    """
    if pred.dim() == 4:  # (B,T,H,W)
        B, T, H, W = pred.shape
        pred = pred.reshape(B * T, 1, H, W)
        target = target.reshape(B * T, 1, H, W)
    elif pred.dim() == 3:
        raise ValueError("pred shape should be (B,T,H,W) or (B,1,H,W)")

    device, dtype = pred.device, pred.dtype
    k = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    pad = window_size // 2

    mu_x = F.conv2d(pred, k, padding=pad, groups=1)
    mu_y = F.conv2d(target, k, padding=pad, groups=1)

    pred_sq = pred * pred
    target_sq = target * target
    pred_target = pred * target

    sigma_x = F.conv2d(pred_sq, k, padding=pad, groups=1) - mu_x * mu_x
    sigma_y = (
        F.conv2d(target_sq, k, padding=pad, groups=1) - mu_y * mu_y
    )
    sigma_xy = (
        F.conv2d(pred_target, k, padding=pad, groups=1) - mu_x * mu_y
    )

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-8
    )
    ssim = numerator / denominator
    return (1.0 - ssim).mean()


@torch.no_grad()
def evaluate_epoch(model, loader, device, clamp_pred=True):
    model.eval()
    l1 = nn.L1Loss()
    total_loss, n = 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)  # (B,12,1,H,W)
        yb = yb.to(device, non_blocking=True)  # (B,12,H,W)

        pred = model(xb)  # (B,12,H,W)
        if clamp_pred:
            pred = pred.clamp(0, 1)

        loss = l1(pred, yb)
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / n


def fit_task1_unet(
    *,
    h5_path: str,
    train_ids: list[str],
    val_ids: list[str],
    device,
    model: nn.Module | None = None,  # advanced model
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_first_window_only: bool = True,
    clamp_pred: bool = True,
    save_best_path: str | None = None,
    train_stride: int = 1,
    mix_w: float = 0.2,
    int_w: float = 0.12,
    tv_w: float = 0.0025,
    range_w: float = 0.01,
    w_pos: float = 5.0,
    thr: float = 20.0 / 255.0,
    lead_power: float = 0.5,
    log_every: int = 100,
):
    # 1) index
    with open_h5(h5_path, "r") as f:
        train_index = build_index(f, train_ids, 12, 12, stride=train_stride)
        if val_first_window_only:
            val_index = [(sid, 0) for sid in val_ids]
        else:
            val_index = build_index(f, val_ids, 12, 12)

    train_ds = VilSeqDataset(h5_path, train_index)
    val_ds = VilSeqDataset(h5_path, val_index)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 2) model
    if model is None:
        model = BottleneckConvLSTMUNet(in_channels=1, out_channels=12, base=8)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
    }

    @torch.no_grad()
    def eval_l1_mae(loader):
        model.eval()
        total_loss, total_mae, n = 0.0, 0.0, 0

        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)  # (B,12,1,H,W)
            yb = yb.to(device, non_blocking=True)  # (B,12,H,W)

            pred = model(xb)  # (B,12,H,W)
            if clamp_pred:
                pred = pred.clamp(0.0, 1.0)

            l1 = (pred - yb).abs().mean()

            bs = xb.size(0)
            total_loss += float(l1.item()) * bs
            total_mae += float(l1.item()) * bs
            n += bs
            if i % 20 == 0:
                print(f"  Validation iter {i}/{len(loader)}", flush=True)

        return {"loss": total_loss / n, "mae": total_mae / n}

    # 3) train loop
    for epoch in range(1, epochs + 1):
        model.train()
        print(f"[epoch {epoch}] start...", flush=True)

        for it, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)  # (B,12,1,H,W)
            yb = yb.to(device, non_blocking=True)  # (B,12,H,W)

            pred_raw = model(xb)  # (B,12,H,W)

            l1 = (pred_raw - yb).abs().mean()
            wloss = weighted_lead_l1(
                pred_raw, yb, w_pos=w_pos, thr=thr, power=lead_power
            )

            iw = intensity_weight(yb)
            lint = (iw * (pred_raw - yb).abs()).mean()

            tv = tv_loss(pred_raw)
            range_pen = (
                F.relu(-pred_raw).mean()
                + F.relu(pred_raw - 1.0).mean()
            )

            loss = (
                (1 - mix_w) * l1
                + mix_w * wloss
                + int_w * lint
                + tv_w * tv
                + range_w * range_pen
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if it % log_every == 0:
                print(
                    f"Epoch {epoch} iter {it}/{len(train_loader)} "
                    f"loss {loss.item():.5f}",
                    flush=True,
                )

        # 4) epoch metrics
        train_metrics = eval_l1_mae(train_loader)
        val_metrics = eval_l1_mae(val_loader)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_metrics['loss']:.5f} "
            f"mae {train_metrics['mae']:.5f} || "
            f"val loss {val_metrics['loss']:.5f} mae {val_metrics['mae']:.5f}",
            flush=True
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "epoch": epoch,
                "val_loss": best_val,
                "val_mae": val_metrics["mae"],
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
            }

            if save_best_path is not None:
                save_checkpoint(
                    save_best_path,
                    model=model,
                    optimizer=opt,
                    epoch=epoch,
                    best_metric=best_val,
                    extra={
                        "task": "task1_unet",
                        "metrics": best_state,
                        "loss_weights": {
                            "mix_w": mix_w,
                            "int_w": int_w,
                            "tv_w": tv_w,
                            "range_w": range_w,
                            "w_pos": w_pos,
                            "thr": thr,
                            "lead_power": lead_power,
                        },
                    },
                )
                print(f"saved best -> {save_best_path}", flush=True)

    return model, {"best": best_state, "history": history}
