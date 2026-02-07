"""
Training utilities for each task.

Design goals:

- Keep training loops and loss functions in task-specific modules.
- Notebooks import from here for cleaner code.
- Common utilities (e.g., EarlyStopping) can be added at package level.

Usage::

    from predicting_unpredectable.train.task2 import (
        train_epoch,
        valid_epoch,
        compute_total_loss,
    )
"""

from __future__ import annotations

__all__ = ["task1", "task2", "task3", "task4"]
