# DenseNet121 — Diabetic Retinopathy Classification

Part of the **SE4050 Deep Learning** group project (SLIIT, Group G05).
**Author:** Gayathree M.G.K (IT23334106)
**Notebook:** `densenet-Apitos2019.ipynb`

---

## Overview

This notebook classifies retinal fundus images into five Diabetic Retinopathy (DR) severity levels (No DR, Mild, Moderate, Severe, Proliferative DR) using **transfer learning with DenseNet121**.

DenseNet connects every layer to all later layers, which improves feature reuse, helps gradient flow, and keeps the parameter count low.

## Setup

| Item | Value |
|------|-------|
| Dataset | APTOS 2019 (3,662 fundus images) |
| Input size | 224 × 224 RGB |
| Split | 80 / 20 stratified (2,930 train / 732 validation) |
| Backbone | DenseNet121, ImageNet weights |
| Head | Global average pooling → Dropout (0.3) → 5-class Dense output |
| Augmentation | Random flips, small rotations, zoom (training only) |
| Imbalance handling | Class weights, softened with a square root so rare classes are not over-predicted |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

Labels are remapped to clinical order (0 = No DR → 4 = Proliferative DR) so that Quadratic Weighted Kappa is calculated correctly.

## Training

1. **Stage 1:** train only the new classification head while the DenseNet121 backbone stays frozen.
2. **Stage 2:** unfreeze the top layers of the backbone and fine-tune with a learning rate 100× smaller, so pretrained weights are adjusted gently rather than overwritten.

## Results (validation set, n = 732)

| Metric | Value |
|--------|:-----:|
| Accuracy | 0.742 |
| Macro F1 | 0.571 |
| Weighted F1 | 0.741 |
| Quadratic Weighted Kappa | 0.808 |
| Total parameters | 7.04 M |

The notebook also includes training curves for both stages, a per-class report and a confusion matrix.

## Explainability

**Grad-CAM** heatmaps use the last DenseNet121 feature maps (7 × 7 × 1024) to highlight the retinal regions that most influenced each prediction.

## How to Run

1. Open the notebook in Google Colab (an *Open in Colab* badge is at the top) and select a GPU runtime.
2. Put the dataset zip in Google Drive at the path set in the configuration cell.
3. Run the notebook top to bottom.

## Reference

Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks.* https://arxiv.org/abs/1608.06993
