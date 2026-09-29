# ResNet50 — Diabetic Retinopathy Classification

Part of the **SE4050 Deep Learning** group project (SLIIT, Group G05).
**Author:** Purijjala R W M A H (IT23431676)
**Notebook:** `resnet50_train.ipynb`

---

## Overview

This notebook classifies retinal fundus images into five Diabetic Retinopathy (DR) severity levels (No DR, Mild, Moderate, Severe, Proliferative DR) using **transfer learning with ResNet50**.

ResNet50 uses residual (skip) connections, which allow deep networks to train stably.

## Setup

| Item | Value |
|------|-------|
| Dataset | APTOS 2019 Blindness Detection |
| Input size | 224 × 224 RGB |
| Split | 70 / 15 / 15 stratified (train / validation / test, 550 test images) |
| Backbone | ResNet50, ImageNet weights, original classifier removed |
| Head | Pooling → Dropout → 5-class Softmax |
| Augmentation | Horizontal flip, rotation, zoom, contrast |
| Imbalance handling | Balanced class weights from training labels |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

The train, validation and test splits are saved as CSV files so the experiment can be reproduced.

## Training

1. **Stage 1:** frozen backbone, Adam with learning rate 1e-3.
2. **Stage 2:** unfreeze the final non-BatchNorm layers and fine-tune at learning rate 1e-5.
3. The checkpoint with the lower validation loss is kept as the final model. Stage 1 won (best validation accuracy 0.767 vs 0.758 for Stage 2).

A second variant (V2: Batch Normalization, a 256-unit ReLU layer, L2 regularization and dropout) was also trained and compared. The original model remained the final choice.

## Results (test set, n = 550)

| Metric | Value |
|--------|:-----:|
| Accuracy | 0.764 |
| Macro F1 | 0.582 |
| Weighted F1 | 0.761 |
| Quadratic Weighted Kappa | 0.837 |
| Macro ROC-AUC | 0.914 |
| Weighted ROC-AUC | 0.945 |

Per-class ROC-AUC: No DR **0.990** · Mild **0.886** · Moderate **0.912** · Severe **0.886** · Proliferative DR **0.897**

**Efficiency:** 23.6 M parameters · 90.75 MB model file · about 15 minutes training · about 49 ms per image at inference.

## Analysis Included

- Classification report and confusion matrix
- One-vs-rest ROC curves for all five classes
- Most confident wrong predictions and a ranked list of the most common error types
- **Grad-CAM** heatmaps for a correct prediction and for the most confident misclassification

## How to Run

1. Download APTOS 2019 from [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) and place `train.csv` and `train_images/` in the project folder the notebook expects.
2. Install dependencies: `pip install tensorflow scikit-learn pandas numpy matplotlib opencv-python`
3. Run the notebook top to bottom. Results are saved to a `results/resnet50/` folder.

## Reference

He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.* https://arxiv.org/abs/1512.03385
