"""

Design goals:
- Keep model definitions in standalone .py files (one model per file).
- Group by task for clarity: task1/, task2/, task3/, task4/.
- Keep this package lightweight: no training loops here
"""

from __future__ import annotations

__all__ = ["task1", "task2", "task3", "task4"]
