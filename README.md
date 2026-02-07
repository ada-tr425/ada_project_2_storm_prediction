# Predicting the Unpredictable

A machine learning project for lightning storm prediction, developed as part of the FEMA challenge.

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

## Project Overview

This project tackles four prediction tasks using the HuggingFace dataset `benmoseley/ese-dl-2025-26-group-project` containing 800 example storms:

1. **Task 1: VIL Frame Prediction** - Predict the next 12 VIL frames from 12 input frames
2. **Task 2: VIL Reconstruction** - Reconstruct VIL frames from satellite imagery
3. **Task 3: Event Type Classification** - Classify storms into 8 event types
4. **Task 4: Lightning Prediction** - Predict lightning flash locations and times

## Installation

```bash
# Clone the repository
git clone https://github.com/ada-tr425/storm_prediction.git
cd storm_prediction

# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Documentation

API documentation is automatically generated from docstrings using Sphinx. The CI pipeline verifies that documentation builds successfully.

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Generate API documentation
sphinx-apidoc -f -o docs/api predicting_unpredictable

# Build HTML documentation
cd docs && make html
# Windows: docs\make.bat html

# Open docs/_build/html/index.html in your browser
```

## Useful Resources

- [Project introduction slides](https://imperiallondon-my.sharepoint.com/:b:/g/personal/bm1417_ic_ac_uk/IQDh5erhwXmATaT5ANnvqAFEAaUeRS_WN1DHDecpt3FPDrc)
- [Colab notebook: Example data downloading and exploration](https://colab.research.google.com/drive/15Rw4zi3V8S9g4h89KJ28LPzZakDJAT0r?usp=sharing)
- [Colab notebook: Surprise storms - description and submission instructions](https://colab.research.google.com/drive/19OuZsdBfTslpJdY60T8BOhbRKOtSP-XX?usp=sharing)

## Final Notebooks

Please place your final Jupyter notebooks in the `notebooks-final/` folder using the [template](notebooks-final/Task-%5Bnumber%5D-report.ipynb) provided.
