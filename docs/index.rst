Predicting the Unpredictable
============================

A machine learning project for lightning storm prediction, developed as part of the FEMA challenge.

.. note::

   This project uses the HuggingFace dataset ``benmoseley/ese-dl-2025-26-group-project``
   containing 800 example storms across the entire US.

Project Overview
----------------

This project tackles four prediction tasks:

1. **Task 1: VIL Frame Prediction** - Predict the next 12 VIL frames from 12 input frames
2. **Task 2: VIL Reconstruction** - Reconstruct VIL frames from satellite imagery (vis, ir069, ir107)
3. **Task 3: Event Type Classification** - Classify storms into 8 event types
4. **Task 4: Lightning Prediction** - Predict lightning flash locations and times

Team Sally
----------

* Fuchs, Ava R
* HU, Wenzhi
* LI, Zhifeng
* Pejic, Milica
* RAO, Tingyu
* Takatani, Masahiko
* WU, Jiayu
* Yu, Zirui
* ZHANG, Wanyu

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ada-tr425/storm_prediction.git
   cd storm_prediction

   # Install the package in editable mode
   pip install -e .

   # Or with documentation dependencies
   pip install -e ".[docs]"

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
