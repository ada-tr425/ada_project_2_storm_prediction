"""
Task 3 Training Pipeline.

Includes:

- **Task3Dataset**: Handles data loading, preprocessing, and augmentation
  (Optimized for worker-safe lazy HDF5 loading).
- **train_task3**: Main entry point for training with support for
  Baseline configuration, Extension 1 (Weighted Loss), Extension 2
  (Data Augmentation), Extension 3 (Learning Rate Scheduler),
  and Optimization (AMP Mixed Precision & Early Stopping).
"""

import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from predicting_unpredictable import data, io, metrics, split
from predicting_unpredictable.constants import EVENT_TYPES
from predicting_unpredictable.models.task3 import CNN3d
from predicting_unpredictable.preprocess import PreprocessTask3


class Task3Dataset(Dataset):
    """
    Dataset for Task 3 Storm Classification.

    Loads raw H5 data using worker-safe lazy loading to improve performance
    and stability in multi-process DataLoaders.
    """

    def __init__(
        self,
        storm_ids: List[str],
        events_df: pd.DataFrame,
        h5_path: str = "data/train.h5",
        augment: bool = False,
    ):
        """
        Args:
            storm_ids: List of storm IDs to include in this split.
            events_df: DataFrame containing event metadata (id, event_type).
            h5_path: Path to the HDF5 training file.
            augment: If True, applies random rotations and flips (Extension 2).
        """
        self.storm_ids = storm_ids
        self.h5_path = h5_path
        self.augment = augment

        # Create a lookup map: ID -> Event Type String
        # Grouping by ID to handle the multiple rows per storm structure in csv
        self.meta_map = events_df.groupby("id")["event_type"].first()

        # Instantiate the shared preprocessing pipeline
        self.preprocess = PreprocessTask3()

        # Worker-safe lazy loading initialization
        self.h5_file = None

    def __len__(self):
        return len(self.storm_ids)

    def __getitem__(self, idx):
        # WORKER-SAFE FILE OPENING
        # Opening inside __getitem__ once per worker prevents
        # multi-threading errors and significantly speeds up
        # access by avoiding repeated open/close calls.
        if self.h5_file is None:
            import h5py

            self.h5_file = h5py.File(self.h5_path, "r")

        storm_id = self.storm_ids[idx]
        event_type_str = self.meta_map[storm_id]

        # Use the already open handle 'self.h5_file'
        storm_data = data.load_event_arrays(
            self.h5_file,
            storm_id=storm_id,
            img_types=["vis", "ir069", "ir107", "vil"],
        )

        # 2. Apply preprocessing (Upsample -> Normalize -> Stack)
        # Returns x: (36, 4, 384, 384), y: scalar tensor
        x, y = self.preprocess(storm_data, event_type_str)

        # 3. Data Augmentation (Extension 2)
        if self.augment:
            # Random Rotation (0, 90, 180, 270 degrees)
            k = random.randint(0, 3)
            if k > 0:
                # Rotate on the last two dimensions (H, W)
                x = torch.rot90(x, k, dims=[-2, -1])

            # Random Horizontal Flip
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])

            # Random Vertical Flip
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-2])

        return x, y


def train_task3(
    events_csv: str = "data/events.csv",
    h5_path: str = "data/train.h5",
    val_fraction: float = 0.2,
    seed: int = 42,
    batch_size: int = 8,
    lr: float = 1e-3,
    epochs: int = 20,  # <--- Default increased to 20
    use_weighted_loss: bool = False,
    augment_data: bool = False,
    use_scheduler: bool = False,
    use_amp: bool = True,  # <--- New: Mixed Precision
    early_stopping_patience: int = 5,  # <--- New: Early Stopping
    save_dir: str = "checkpoints/task3",
):
    """
    Main training loop for Task 3.

    Args:
        use_weighted_loss: Enable Extension 1 (Class Balancing Weights).
        augment_data: Enable Extension 2 (Data Augmentation for Training).
        use_scheduler: Enable Extension 3 (ReduceLROnPlateau Scheduler).
        use_amp: Enable Automatic Mixed Precision for faster training.
        early_stopping_patience: Stop training if val_f1 doesn't improve
           for N epochs.
    """

    # 1. Setup Environment
    io.seed_everything(seed)
    device = io.get_device()
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Preparing data splits...")
    # Load metadata
    df = data.read_events_csv(events_csv)

    # Create Stratified Split
    splits = split.make_stormwise_stratified_split(
        events_csv=events_csv, val_fraction=val_fraction, seed=seed
    )

    print(f"Train samples: {len(splits.train_ids)}")
    print(f"Val samples:   {len(splits.val_ids)}")

    # Create Datasets
    train_dataset = Task3Dataset(
        splits.train_ids, df, h5_path=h5_path, augment=augment_data
    )
    val_dataset = Task3Dataset(
        splits.val_ids, df, h5_path=h5_path, augment=False
    )

    # Create DataLoaders with optimizations for AMP/GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,  # Faster CPU->GPU transfer
        persistent_workers=True,  # Avoid re-spawning workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # 3. Extension 1: Weighted Loss Calculation
    criterion_args = {}
    if use_weighted_loss:
        print("Extension 1 Enabled: Calculating class weights...")
        train_df = df[df["id"].isin(splits.train_ids)]
        unique_storms = train_df.groupby("id")[["event_type"]].first()
        counts = unique_storms["event_type"].value_counts()
        sorted_counts = [counts.get(event, 0) for event in EVENT_TYPES]

        max_count = max(sorted_counts)
        weights = [max_count / (c + 1e-6) for c in sorted_counts]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion_args["weight"] = class_weights
        print(f"Class Weights: {weights}")

    # 4. Model & Optimizer Setup
    print("Initializing Model...")
    model = CNN3d(num_classes=len(EVENT_TYPES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(**criterion_args)

    # Extension 3: Scheduler
    scheduler = None
    if use_scheduler:
        print("Extension 3 Enabled: Initializing ReduceLROnPlateau...")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, verbose=True
        )

    # Initialize AMP Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5. Training Loop
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    patience_counter = 0

    print(
        f"Starting training for {epochs} epochs "
        f"(Aug={augment_data}, Weighted={use_weighted_loss}, "
        f"Sched={use_scheduler}, AMP={use_amp}, "
        f"EarlyStop={early_stopping_patience})..."
    )

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        )
        for x_batch, y_batch in train_bar:
            # non_blocking=True allows async data transfer with pin_memory
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # AMP Context
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            # Scaled Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                # AMP Context for inference (speeds up fp16 ops)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # --- METRICS ---
        val_acc = np.mean(np.array(all_preds) == np.array(all_targets))
        val_f1 = metrics.macro_f1_numpy(
            np.array(all_targets), np.array(all_preds)
        )

        # Step Scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Update History
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # --- CHECKPOINTING & EARLY STOPPING ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0  # Reset counter
            save_path = f"{save_dir}/best_model.pt"
            io.save_checkpoint(
                path=save_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_metric=best_val_f1,
                extra=(
                    {"class_weights": weights} if use_weighted_loss else None
                ),
            )
            print(f"--> New Best Model saved! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(
                f"--> No improvement. Patience:"
                f"{patience_counter}/{early_stopping_patience}"
            )

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered. Training finished.")
                break

    print("\nTraining Complete.")
    return history, best_val_f1
