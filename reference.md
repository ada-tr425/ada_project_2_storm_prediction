# Reference
## Task1

1. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. (https://arxiv.org/abs/1506.04214)
- Journal: Advances in Neural Information Processing Systems (NeurIPS).
- Authors: Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C. (2015).

2. U-Net: Convolutional Networks for Biomedical Image Segmentation. (https://arxiv.org/abs/1505.04597)
- Journal: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).
- Authors: Ronneberger, O., Fischer, P., & Brox, T. (2015).

3. Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM). (https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
- Journal: IEEE Transactions on Image Processing, 13(4), 600–612.
- Author: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).

4. Precipitation nowcasting based on ConvLSTM-UNet deep spatiotemporal network.
- Journal: Journal of Electronic Imaging, 33(1).
- Authors: Zheng, Y., Qin, Z., Chen, Y., Wang, J., & Shi, X. (2024).

5. End-to-end modeling plan: Baseline → ConvLSTM‑UNet
- Model: ChatGPT5.2
- Prompts:
  1) I’m doing VIL 12→12 forecasting: each storm is a (384, 384, 36) uint8 sequence; I use the first 12 frames to predict the next 12. Give me an end-to-end technical roadmap from a minimal baseline to a stronger model (ConvLSTM‑UNet). For each step: what problem it addresses, why it should help, and the smallest experiment to validate it.
  2) Design a strong sanity-check baseline: treat the 12 input frames as channels and use a 2D CNN to output 12 future frames. Specify input/output tensor shapes, a suggested layer stack, output-range handling (clamp vs sigmoid), training hyperparameters (lr/batch/epochs), and a minimal evaluation plan (overall MAE + lead1/lead12).
  3) I want explicit temporal modeling. Compare ConvLSTM, 3D CNN, and temporal attention for this task (384×384, sparse high-intensity cores, 12→12). Explain pros/cons and why ConvLSTM‑UNet is a reasonable tradeoff (performance vs compute/memory).
  4) Provide a ConvLSTM‑UNet architecture sketch (encoder/decoder/skip connections, where ConvLSTM sits, and how dimensions flow). List 3 common implementation pitfalls (dimension order, hidden-state handling, one-shot output vs autoregressive) and give concrete debugging checks for each.

6. Data pipeline & splitting: avoid leakage, windowing, stride tradeoffs
- Model: ChatGPT5.2
- Prompts:
  1) Design a data pipeline for storm sequences with (H,W,T)=(384,384,36): HDF5 loading, normalization to [0,1], conversion to (T,C,H,W), and generating 12→12 supervised pairs. Provide pseudocode and explicit shape comments at each step.
  2) My windowing is: input t..t+11, target t+12..t+23. How many training windows does one storm produce with stride=1 vs stride=4, and how does stride affect sample correlation, training speed, and generalization?
  3) I plan a storm-wise split by id (optionally stratified) to avoid leakage. Explain why frame/window-level random splitting is invalid here, and provide a clear, implementable split logic with key caveats.
  4) For validation I only use the t=0 window per storm to save compute. Assess the bias/representativeness risk, then propose 2 more reliable but still cheap validation alternatives (e.g., K windows sampled per storm, or stratified sampling over start times), with the cost/benefit tradeoffs.

7. Loss design: long-tail intensities, sparse cores, blurry long lead times
- Model: ChatGPT5.2
- Prompts:
  1) EDA suggests VIL is long-tailed and intense storm cores are rare; plain L1/MAE tends to underestimate peaks and produce darker/blurry later frames. Explain the optimization reason for this, and propose 3 practical loss-design directions (intensity reweighting, robust losses like Huber/quantile, structure-aware losses like SSIM/gradient).
  2) Turn the following composite loss into formulas + PyTorch implementation notes: base L1; lead-time weighting (later leads higher weight); intensity weighting (higher weight for pixels with target y>τ, start with τ=20/255); TV/gradient regularization; range penalty for values outside [0,1] (before clamping).
  3) I’m worried intensity weighting increases false positives (hallucinated cores). Give a tuning strategy: how to choose τ, how to cap weights, and how to monitor the FP/FN tradeoff using simple diagnostics (e.g., MAE restricted to y>τ region + predicted thresholded area curves).
  4) Design a minimal ablation study: start from pure L1 and add terms one by one (lead-weight → intensity-weight → TV → range → optional SSIM). For each step, state expected effects on overall MAE, lead12 MAE, visual sharpness, and training stability.

8. Training stability & debugging: dark outputs, over-smoothing, compute limits
- Model: ChatGPT5.2
- Prompts:
  1) My predictions become darker over lead time and peaks disappear. Provide a layered debugging flow: check normalization (and any inverse scaling), then output activation (sigmoid/clamp), then loss-term weights, then regularizers (TV/range). For each layer: what to inspect, what plots to make, and what change to try.
  2) I train on CPU/MPS with tight memory/throughput constraints at 384×384 and 12→12. Give an optimization checklist: batch size strategy, mixed precision feasibility, crop/patch training tradeoffs, gradient checkpointing, reducing base channels, and choosing stride.
  3) ConvLSTM training is unstable (loss spikes/gradients explode). Suggest stabilization steps: gradient clipping, initialization, learning-rate schedule/warmup, normalization choices (GroupNorm/LayerNorm placement), and how to confirm the issue is in ConvLSTM rather than data or loss scaling.
  4) How can I quickly test whether the model learned mostly ‘persistence’ vs ‘motion’? Propose 2 diagnostic experiments (e.g., time-shuffling inputs, reversing time, using only the last frame) and explain how to interpret outcomes.

---
## Task2

1. GPT was used to support reasoning about loss function design and its effect on prediction 
behaviour, particularly underestimation of high-intensity regions.
- Model: GPT5.1
- Prompt: Asking about differences between L1, MSE, and combined loss functions, and how different loss choices influence smoothness, sharpness, and intensity bias in regression models.

2. PT was used to explore ideas for emphasising strong signal regions during training without 
changing the core model architecture.
- Model: GPT5.1
- Prompt: Discussing the use of masked or threshold-based losses (e.g. focusing on high-value regions) and asking how such approaches affect learning dynamics and visual results.

3. GPT was used to help interpret training and validation behaviour during experiments, including 
loss oscillations and signs of overfitting when training on limited data.
- Model: GPT5.1
- Prompt: Asking diagnostic questions about why validation loss or MAE may fluctuate, how small dataset size affects generalisation, and how to decide when to stop tuning.

4. GPT was used for clarification on visualisation and qualitative analysis of model outputs.
- Model: GPT5.1
- Prompt: Asking how to visualise multiple prediction frames, compare predicted and target fields, and interpret qualitative differences between model outputs. The final 3-layer UNet architecture was derived from course lecture material.

5. ChatGPT was used to assist with understanding the meteorological context relevant to the task.
- Model:GPT-5.1
- Prompt: Clarifying the physical relationships between infrared satellite measurements (IR069 water vapor channel and IR107 cloud-top temperature) and Vertically Integrated Liquid (VIL) radar observations.

6. ChatGPT was used to assist with exploratory data analysis.
- Model: GPT-5.1
- Prompt: Generating exploratory data analysis code for VIS brightness distribution analysis.

7. ChatGPT was used to assist with improving technical writing and report organisation.
- Model: GPT-5.1
- Prompt: Providing suggestions on report structure, clarity, and technical wording.

---
## Task3
1. Data Normalization with Min-Max Scaling
- Website: https://www.statology.org/normalize-data-between-0-and-1/
- Description: Explains how to normalize data to the [0, 1] range using the Min-Max Scaling formula $\frac{x - x_{min}}{x_{max} - x_{min}}$, including Python code examples using Pandas and Scikit-Learn.

2. AdaptiveAvgPool3d Layer Documentation
- Website: https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html
- Description: Official PyTorch documentation for the `AdaptiveAvgPool3d` layer. It explains how to define a fixed output size regardless of input dimensions, which is crucial for processing video or 3D medical data before feeding it into fully connected layers.

3. Weighted Loss for Imbalanced Classification
- Website: https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
- Description: Discusses methods for handling data imbalance in classification tasks. It details how to assign class weights within PyTorch's `CrossEntropyLoss` to force the model to pay more attention to minority classes during training.

4. Weight Decay vs. L2 Regularization with Adam
- Website: https://www.codegenes.net/blog/how-does-weight-decay-work-adam-pytorch/
- Description: Provides an in-depth analysis of how Weight Decay interacts with the Adam optimizer in PyTorch. It highlights the difference between L2 regularization and weight decay, pointing out issues in standard Adam implementations and recommending `AdamW`.

5. ReduceLROnPlateau Scheduler Documentation
- Website: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
- Description: Official PyTorch documentation for the `ReduceLROnPlateau` learning rate scheduler. It explains how to automatically reduce the learning rate when a monitored metric (e.g., validation loss) stops improving to help the model converge.

6. Implementing Focal Loss for Class Imbalance
- Website: https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
- Description: Explains the theory behind Focal Loss (originally for RetinaNet) and provides a specific PyTorch implementation (`nn.Module`). It is used to solve class imbalance by down-weighting easy examples and focusing training on hard negatives.

7. Beginner’s Guide to ReLU Activation
- Website: https://www.unitxlabs.com/rectified-linear-unit-machine-vision-system-beginners-guide/
- Description: A beginner's guide to the Rectified Linear Unit (ReLU) activation function for machine vision. It covers the mathematical definition $f(x) = max(0, x)$, how it mitigates the vanishing gradient problem, and its efficiency benefits.

8. Cross-Entropy Loss Function Explained
- Website: https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/
- Description: Detailed explanation of the Cross-Entropy Loss function, the standard loss for classification problems. It includes the mathematical derivation and explains how it measures the difference between the predicted probability distribution and the actual labels.

9. Matplotlib Fill Between Lines Demo
- Website: https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html
- Description: Official Matplotlib gallery example demonstrating the `fill_between` function. It shows how to shade the area between two curves, which is essential for visualizing confidence intervals or error margins in plots.

10. Drawing Vertical Lines in Matplotlib
- Website: https://stackoverflow.com/questions/24988448/how-to-draw-vertical-lines-on-a-given-plot
- Description: A StackOverflow thread summarizing various methods to draw vertical lines on Matplotlib plots. It covers the usage of `axvline` (for lines spanning the full plot height) and `vlines` (for lines with specific start/end points).

11. Accuracy Paradox in Imbalanced Datasets
- Website: https://mljourney.com/why-accuracy-falls-short-for-evaluating-imbalanced-datasets/1
- Description: Discusses the limitations of using accuracy as a metric for imbalanced datasets, often referred to as the Accuracy Paradox." It explains how a model can achieve deceptively high accuracy by simply predicting the majority class (e.g., 99% accuracy in a 99:1 dataset) while completely failing to identify the minority class.

12. PyTorch-NumPy Type Mismatch and NaN Handling
- Model: Gemini
- Prompt: Fix TypeError: can't assign a numpy.float32 to a torch. FloatTensor (Handling NaNs and Type conversion)

13. Handling Overrepresented Classes (Data Dropout vs. WeightedRandomSampler)
- Model: Gemini
- Prompt: I want to add dropout for overrepresented classes

14. Preprocessing Storm Data for Faster Training
- Model: Gemini
- Prompt: Optimizing training speed by pre-loading the dataset into RAM (Caching) to eliminate disk I/O bottlenecks, and implementing a custom PyTorch Dataset to handle temporal indexing for storm events.

---
## Task4

1. Using the gpt to ask how to improve the U-net model,but built the model all by myself
- Model: ChatGPT5.1
- Prompt: giving the old model and asking which model we can conbined with U-net

2. Using the gpt to find some solution about how to change the loss function to inprove the recall and keep high accuracy,but chosing all of the hyper-pharemeter to improve the performance.
- Model: ChatGPT5.1
- Prompt: giving the previous function and the training result.

---
## Packaging & Workflow

1. Project framework & repository structure
- Model: ChatGPT5.2
- Prompt:
  1) Given this repo goal (4 storm-prediction tasks) and the existing package name predicting_unpredictable, propose a clean project structure that separates: reusable library code (predicting_unpredictable/), task-specific models (models/task{k}.py), training utilities (train/task{k}.py), notebooks (notebooks-final/), and outputs (outputs/). Include a short rationale and naming conventions.
  2) Define clear boundaries: what code should live in notebooks vs what must live in the Python package. Use the constraints: deliverables are final notebooks per task, but utilities should be reusable and tested. Provide concrete examples (e.g., ‘data loading helper belongs in package’, ‘experiment hyperparameter sweep belongs in notebook’).
  3) Create a lightweight ‘architecture decision checklist’ for adding a new module to predicting_unpredictable/: required public API, docstring expectations, unit tests, and how it should be imported to pass CI smoke tests.
  4) Design a minimal CLI or entrypoint strategy (optional) for running common actions (download data, run unit tests, build docs, validate submissions) without bloating the package. Explain the simplest approach compatible with pyproject.toml + CI.

2. Packaging & dependency management
- Model: ChatGPT5.2
- Prompt:
  1) Audit my packaging approach: I have pyproject.toml using setuptools and a separate requirements.txt used by CI. Propose the best practice to keep dependencies consistent (single source of truth vs dual files) and explain tradeoffs for a student ML project.
  2) Given pyproject.toml already defines project.dependencies and optional extras dev and docs, propose an installation matrix for contributors (base/dev/docs) and the exact commands they should run (editable install included).
  3) Recommend how to pin versions (or not) for ML libraries (torch, numpy, etc.) in a reproducible way, while keeping CI stable. Provide a policy: what to pin strictly, what to allow as ranges, and why.
  4) Explain how to package the project for distribution (building a wheel/sdist) and how to validate the install in a clean environment. Provide a step-by-step checklist for ‘it installs and imports correctly’ beyond local dev.

3. Docstrings & documentation with Sphinx
- Model: ChatGPT5.2
- Prompt:
  1) Pick a docstring standard (Google vs NumPy) for this repo and justify it. Then provide a docstring template for: (a) module-level docs, (b) a function that handles storm tensors (include shapes/dtypes/units), and (c) a model class.
  2) Write docstring guidance specific to our domain: every function that touches storm data must document tensor layout (e.g., (T,C,H,W)), normalization range, and any threshold constants. Provide a ‘must include’ checklist.
  3) We use Sphinx + autodoc. Provide a minimal docs workflow: how to generate API docs (sphinx-apidoc) and how to keep them from breaking in CI. Include common failure modes (import errors, missing deps, heavy imports) and how to avoid them.
  4) Propose a documentation strategy that balances notebooks vs API docs: what should be written in README/TEAM_CONVENTIONS vs Sphinx API pages vs the final task notebooks. Include an outline of each document’s responsibilities.

4. Engineering workflow: testing, linting, CI, and collaboration
- Model: ChatGPT5.2
- Prompt:
  1) Given CI runs flake8, compileall, import smoke tests, pytest, and pip check, propose a local developer workflow to match CI (commands to run before pushing) and a troubleshooting guide when CI fails.
  2) Propose a testing strategy for this repo: what should be unit-tested in predicting_unpredictable/tests/ vs what should be left to notebook validation. Include examples of good tests for: preprocessing, splitting, metrics, and submission I/O.
  3) Recommend a branching/PR workflow compatible with the existing CI triggers (main/dev/feature branches). Include commit hygiene, review checklist, and how to handle notebooks (e.g., output clearing, naming, where final notebooks go).
  4) Design a reproducibility checklist for experiments: random seeds, deterministic settings, config logging, checkpoint naming, and how to ensure the final notebook can be re-run. Keep it practical for limited compute.
