"""
Tests for predicting_unpredictable.io module.
"""

import os
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from predicting_unpredictable.io import (
    get_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


class SimpleModel(nn.Module):
    """Simple model for testing checkpoint save/load."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestSeedEverything:
    """Tests for seed_everything function."""

    def test_seed_everything_sets_python_random(self):
        seed_everything(42)
        val1 = random.random()
        seed_everything(42)
        val2 = random.random()
        assert val1 == val2

    def test_seed_everything_sets_numpy_random(self):
        seed_everything(42)
        val1 = np.random.rand()
        seed_everything(42)
        val2 = np.random.rand()
        assert val1 == val2

    def test_seed_everything_sets_torch_random(self):
        seed_everything(42)
        val1 = torch.rand(1).item()
        seed_everything(42)
        val2 = torch.rand(1).item()
        assert val1 == val2

    def test_seed_everything_sets_env_var(self):
        seed_everything(12345)
        assert os.environ["PYTHONHASHSEED"] == "12345"

    def test_seed_everything_different_seeds_produce_different_values(self):
        seed_everything(1)
        val1 = random.random()
        seed_everything(2)
        val2 = random.random()
        assert val1 != val2


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_cpu_when_prefer_cuda_false(self):
        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_cpu_when_cuda_unavailable(self, mock_cuda):
        device = get_device(prefer_cuda=True)
        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_device_cuda_when_available_and_preferred(self, mock_cuda):
        device = get_device(prefer_cuda=True)
        assert device.type == "cuda"


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        result = save_checkpoint(path, model=model)

        assert path.exists()
        assert result == path

    def test_save_checkpoint_creates_parent_dirs(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "nested" / "dir" / "checkpoint.pt"
        save_checkpoint(path, model=model)

        assert path.exists()

    def test_save_checkpoint_contains_model_state(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model)

        ckpt = torch.load(path)
        assert "model_state_dict" in ckpt

    def test_save_checkpoint_with_optimizer(self, tmp_path):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model, optimizer=optimizer)

        ckpt = torch.load(path)
        assert "optimizer_state_dict" in ckpt

    def test_save_checkpoint_with_epoch(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model, epoch=10)

        ckpt = torch.load(path)
        assert ckpt["epoch"] == 10

    def test_save_checkpoint_with_global_step(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model, global_step=1000)

        ckpt = torch.load(path)
        assert ckpt["global_step"] == 1000

    def test_save_checkpoint_with_best_metric(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model, best_metric=0.95)

        ckpt = torch.load(path)
        assert ckpt["best_metric"] == 0.95

    def test_save_checkpoint_with_extra(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        extra = {"config": {"lr": 0.001}, "history": [1, 2, 3]}
        save_checkpoint(path, model=model, extra=extra)

        ckpt = torch.load(path)
        assert ckpt["extra"] == extra

    def test_save_checkpoint_returns_path_object(self, tmp_path):
        model = SimpleModel()
        result = save_checkpoint(tmp_path / "ckpt.pt", model=model)
        assert isinstance(result, Path)


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_restores_model_state(self, tmp_path):
        # Create and save model
        model1 = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model1)

        # Create new model and load
        model2 = SimpleModel()
        load_checkpoint(path, model=model2)

        # Compare state dicts
        for key in model1.state_dict():
            torch.testing.assert_close(
                model1.state_dict()[key],
                model2.state_dict()[key],
            )

    def test_load_checkpoint_restores_optimizer_state(self, tmp_path):
        model = SimpleModel()
        optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
        # Take a step to change optimizer state
        x = torch.randn(1, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer1.step()

        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model, optimizer=optimizer1)

        # Load into new optimizer
        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        load_checkpoint(path, model=model2, optimizer=optimizer2)

        # Check state matches
        assert optimizer1.state_dict()["param_groups"] == \
            optimizer2.state_dict()["param_groups"]

    def test_load_checkpoint_returns_full_dict(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            path, model=model, epoch=5, global_step=100, best_metric=0.8
        )

        model2 = SimpleModel()
        ckpt = load_checkpoint(path, model=model2)

        assert ckpt["epoch"] == 5
        assert ckpt["global_step"] == 100
        assert ckpt["best_metric"] == 0.8

    def test_load_checkpoint_map_location(self, tmp_path):
        model = SimpleModel()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model)

        model2 = SimpleModel()
        # Should not raise even if originally saved on different device
        load_checkpoint(path, model=model2, map_location="cpu")

    def test_load_checkpoint_without_optimizer(self, tmp_path):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model=model, optimizer=optimizer)

        model2 = SimpleModel()
        # Should work without passing optimizer
        ckpt = load_checkpoint(path, model=model2)
        assert "optimizer_state_dict" in ckpt


class TestCheckpointRoundTrip:
    """Integration tests for save/load checkpoint."""

    def test_full_checkpoint_roundtrip(self, tmp_path):
        # Setup
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters())
        extra_data = {"config": {"batch_size": 32}}

        # Save
        path = tmp_path / "full_ckpt.pt"
        save_checkpoint(
            path,
            model=model1,
            optimizer=optimizer1,
            epoch=10,
            global_step=500,
            best_metric=0.92,
            extra=extra_data,
        )

        # Load
        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        ckpt = load_checkpoint(path, model=model2, optimizer=optimizer2)

        # Verify
        assert ckpt["epoch"] == 10
        assert ckpt["global_step"] == 500
        assert ckpt["best_metric"] == 0.92
        assert ckpt["extra"] == extra_data

        # Model weights match
        for key in model1.state_dict():
            torch.testing.assert_close(
                model1.state_dict()[key],
                model2.state_dict()[key],
            )
