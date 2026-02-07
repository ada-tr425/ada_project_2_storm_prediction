import torch
import torch.nn.functional as F

# --- Loss Functions ---


def simple_hi_under(logits, y, t=0.6, lam=0.08):
    pred = torch.sigmoid(logits)
    l1 = (pred - y).abs().mean()

    hi = (y > t).float()
    hi_under = (F.relu(y - pred) * hi).sum() / hi.sum().clamp_min(1.0)

    return l1 + lam * hi_under


# --- Core Loops ---


def train_epoch(model, optimizer, criterion, loader, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_loss_draw = 0.0
    num_samples = 0

    for i, (x, y) in enumerate(loader):
        # limit the number of batches to run per epoch
        if (max_batches is not None) and (i >= max_batches):
            break

        x, y = x.to(device), y.to(device)

        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        y = y.reshape(B*T, 1, H, W)

        optimizer.zero_grad()
        logits = model(x)
        loss = simple_hi_under(logits, y, t=0.6, lam=0.08)
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            loss_draw = criterion(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * (B * T)
        total_loss_draw += loss_draw.item() * (B * T)
        num_samples += B * T

        # clear memory
        # del x, y, pred, loss
        # torch.cuda.empty_cache()

    return total_loss / num_samples, total_loss_draw / num_samples


@torch.no_grad()
def valid_epoch(model, criterion, loader, device,
                max_batches=None, clamp01=False):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if (max_batches is not None) and (i >= max_batches):
                break

            x, y = x.to(device), y.to(device)

            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)
            y = y.reshape(B*T, 1, H, W)

            logits = model(x)
            pred = torch.sigmoid(logits)
            # if clamp01:
            #     pred = pred.clamp(0, 1)

            loss = criterion(pred, y)

            total_loss += loss.item() * (B * T)

            num_samples += B * T

    return total_loss / num_samples


@torch.no_grad()
def evaluate_mae(model, loader, device, max_batches=None):
    """
    Compute Mean Absolute Error over a dataset.
    """
    model.eval()
    total_abs_error = 0.0
    total_pixels = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if (max_batches is not None) and (i >= max_batches):
                break

            x, y = x.to(device), y.to(device)

            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)
            y = y.reshape(B*T, 1, H, W)

            pred = torch.sigmoid(model(x))

            # absolute error
            abs_err = (pred - y).abs()

            total_abs_error += abs_err.sum().item()
            total_pixels += abs_err.numel()

    mae = total_abs_error / total_pixels
    return mae
