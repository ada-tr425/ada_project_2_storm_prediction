# Predicting the Unpredictable

> **文档结构说明**：以下**前半部分为中文概述**（便于快速阅读）；**后半部分为英文原文**，与课程/仓库最初表述一致，便于对照与引用。

---

## 中文概述

本项目面向**雷暴与对流天气相关预测**，作为 FEMA 挑战的一部分，使用机器学习完成多项任务。

### 项目概览

数据来自 Hugging Face 数据集 `benmoseley/ese-dl-2025-26-group-project`，包含 800 个示例风暴样本。项目包含四项预测任务：

1. **任务 1：VIL 帧预测** — 用前 12 帧 VIL 预测后续 12 帧  
2. **任务 2：VIL 重建** — 由卫星影像重建 VIL 帧  
3. **任务 3：事件类型分类** — 将风暴分为 8 类事件类型  
4. **任务 4：闪电预测** — 预测闪电发生的位置与时间  

### 安装

```bash
# Clone the repository
git clone https://github.com/ada-tr425/storm_prediction.git
cd storm_prediction

# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### 文档

API 文档由 Sphinx 根据 docstring 自动生成；CI 会校验文档能否成功构建。

**本地构建文档：**

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

### 相关资源

- [项目介绍幻灯片](https://imperiallondon-my.sharepoint.com/:b:/g/personal/bm1417_ic_ac_uk/IQDh5erhwXmATaT5ANnvqAFEAaUeRS_WN1DHDecpt3FPDrc)
- [Colab：数据下载与探索示例](https://colab.research.google.com/drive/15Rw4zi3V8S9g4h89KJ28LPzZakDJAT0r?usp=sharing)
- [Colab：Surprise storms 说明与提交指引](https://colab.research.google.com/drive/19OuZsdBfTslpJdY60T8BOhbRKOtSP-XX?usp=sharing)

### 最终报告笔记本

请将最终 Jupyter 笔记本放在 `notebooks-final/` 目录，并使用提供的[模板](notebooks-final/Task-%5Bnumber%5D-report.ipynb)。

---

## English (original)

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
