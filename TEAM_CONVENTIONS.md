# TEAM_CONVENTIONS.md

## Team Sally

- Fuchs, Ava R
- HU, Wenzhi
- LI, Zhifeng
- Pejic, Milica
- RAO, Tingyu
- Takatani, Masahiko
- WU, Jiayu
- Yu, Zirui
- ZHANG, Wanyu

## Installation

```bash
# Basic installation
pip install -e .

# With development dependencies (pytest, flake8)
pip install -e ".[dev]"

# With documentation dependencies (sphinx)
pip install -e ".[docs]"
```

## Deliverables

- **One report notebook per task** in `notebooks-final/`:
  - `Task-1-report.ipynb`
  - `Task-2-report.ipynb`
  - `Task-3-report.ipynb`
  - `Task-4-report.ipynb`
- **Surprise storms submission files**: 40 `.npy` files (4 tasks × 10 storms) generated to `outputs/predictions/` and checked using the **official checker notebook**.

## Data Access

- Training data (HuggingFace):
  - `train.h5`
  - `events.csv`
- Surprise data (HuggingFace):
  - `surprise-task1.h5` … `surprise-task4.h5`
  - `surprise-events1.csv` … `surprise-events4.csv`

Download helpers are in `predicting_unpredictable/data.py`.

## Submission File Contract (must match official checker)

Filename: `<team-name>-task{k}-<storm-id>.npy`

Shapes and dtypes:
- **Task 1**: `(384, 384, 12)` `float32`
- **Task 2**: `(384, 384, 36)` `float32`
- **Task 3**: scalar string array: `np.array('<category>')`, where category is one of:
  - `'Flash Flood','Flood','Funnel Cloud','Hail','Heavy Rain','Lightning','Thunderstorm Wind','Tornado'`
- **Task 4**: `(N, 3)` `float32` with columns `(time_seconds, vil_pixel_x, vil_pixel_y)`

Helper functions to save/validate are in `predicting_unpredictable/submission.py` (but you still must run the official checker notebook before uploading).

## Package Structure

`predicting_unpredictable/` contains:

- **Core modules**: `data.py`, `preprocess.py`, `metrics.py`, `submission.py`, `io.py`, `split.py`, `constants.py`, `plotting.py`
- **`models/`**: Model definitions (task1.py - task4.py)
- **`train/`**: Training utilities (task1.py - task4.py)
- **`tests/`**: Unit tests

## Where Code Should Live

- **Notebooks**: model architectures, training loops, experiment tracking, best-checkpoint selection.
- **Package** (`predicting_unpredictable/`): small reusable utilities only.

## Code Quality Requirements

All code must pass CI checks before merging:

1. **Flake8**: Run `flake8 predicting_unpredictable/` to check for linting errors
2. **Pytest**: Run `pytest` to ensure all tests pass
3. **Docs**: Documentation must build without errors

## CI/CD Pipeline

GitHub Actions automatically runs on push/PR to `main`, `dev`, `feature-preprocess`:

- **python-ci** job:
  - Flake8 linting
  - Byte-compile syntax check
  - Import smoke tests
  - Pytest unit tests
  - Dependency health check

- **docs** job:
  - Sphinx API documentation generation
  - Documentation build verification

## Building Documentation Locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Generate API docs and build HTML
sphinx-apidoc -f -o docs/api predicting_unpredictable
cd docs && make html
# Windows: docs\make.bat html

# View: open docs/_build/html/index.html
```

## Outputs (do not commit)

- `data/` (downloaded datasets)
- `outputs/` (checkpoints and predictions)
- `docs/_build/` (generated documentation)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
