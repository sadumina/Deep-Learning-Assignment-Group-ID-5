# EfficientNetB0 — Diabetic Retinopathy Classification

Part of the **SE4050 Deep Learning** group project (SLIIT, Group G05).
**Author:** Insath MMM (IT23183872)
**Notebook:** `EfficientNetB0-Apitos2019.ipynb`

---

## Overview

This notebook classifies retinal fundus images into five Diabetic Retinopathy (DR) severity levels (No DR, Mild, Moderate, Severe, Proliferative DR) using **transfer learning with EfficientNetB0**.

EfficientNetB0 uses compound scaling of network depth, width and input resolution, which gives strong accuracy for a small number of parameters.

## Setup

| Item | Value |
|------|-------|
| Dataset | APTOS 2019 Blindness Detection (3,662 images) |
| Input size | 224 × 224 RGB |
| Split | 80 / 20 stratified (2,929 train / 733 validation) |
| Backbone | EfficientNetB0, ImageNet weights |
| Head | Global average pooling + 5-class softmax output |
| Optimizer | Adam, learning rate 1e-4 |
| Batch size / epochs | 32 / up to 30 |
| Imbalance handling | Class weights computed from training labels |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

## Results (validation set, n = 733)

| Metric | Value |
|--------|:-----:|
| Accuracy | 0.79 |
| Macro F1 | 0.64 |
| Weighted F1 | 0.79 |
| Model parameters | ≈ 4.06 M |

Per-class F1: No DR **0.97** · Moderate **0.72** · Mild **0.55** · Severe **0.50** · Proliferative DR **0.47**

The model separates *No DR* very well. The rarer classes (Mild, Severe, Proliferative DR) are harder because of the class imbalance in the dataset.

## Explainability and Inference

- **Grad-CAM** heatmaps are generated from the last convolutional layer (`top_conv`, 7 × 7 feature map) for all five classes, so we can see which retinal regions drive each prediction.
- The notebook also includes an inference function that returns the predicted severity, class probabilities and Base64-encoded Grad-CAM images, ready for use in an API.
- The best model is saved as `best_efficientnetb0.keras`.

## How to Run

1. Download APTOS 2019 from [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
2. Update the `BASE_PATH` / `PROJECT_ROOT` variables at the top of the notebook.
3. Install dependencies: `pip install tensorflow scikit-learn pandas numpy matplotlib seaborn`
4. Run the notebook top to bottom (a GPU is recommended).

## Reference

Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* https://arxiv.org/abs/1905.11946
