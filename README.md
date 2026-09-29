# Diabetic Retinopathy Screening — Automated Detection of Retinal Damage from Eye Scans

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Custom%20CNN-ee4c2c)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%20Transfer%20Learning-orange)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-APTOS%202019-20beff)](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

> **SE4050 — Deep Learning | BSc (Hons) in Information Technology**
> Sri Lanka Institute of Information Technology (SLIIT) | Group G05 | 2026

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Team & Contributions](#team--contributions)
- [References](#references)
- [License](#license)

---

## Overview

Diabetic Retinopathy (DR) is a progressive complication of diabetes that damages the blood vessels of the retina and is one of the leading causes of preventable blindness worldwide. Regular retinal screening enables early treatment, but manual grading by ophthalmologists is slow and depends on specialist availability, which is limited in many regions.

This project implements and compares **four distinct deep learning models** for automated, five-class classification of DR severity from retinal fundus images, evaluating how each balances accuracy, generalization on rare classes, and computational cost.

It is the practical submission for the **SE4050 – Deep Learning** module (SLIIT, 2026) under the **Supervised Deep Learning** category.

---

## Problem Statement

Given a retinal fundus image, classify the severity of Diabetic Retinopathy into one of five clinical stages:

| Class | Label | Description |
|:-----:|-------|-------------|
| 0 | No DR | No visible signs of retinopathy |
| 1 | Mild | Microaneurysms present |
| 2 | Moderate | More extensive vascular damage |
| 3 | Severe | Widespread blood vessel blockage |
| 4 | Proliferative DR | Abnormal new vessel growth; highest risk of blindness |

This is a multi-class image classification problem. Because the classes are ordered, **Quadratic Weighted Kappa (QWK)** is reported alongside accuracy and F1.

---

## Dataset

- **Source:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) (Kaggle / Asia Pacific Tele-Ophthalmology Society)
- **Size:** 3,662 labelled fundus images in the training set
- **Labels:** severity grade 0–4 from `train.csv`
- **Class distribution:** highly imbalanced

| Class | No DR | Mild | Moderate | Severe | Proliferative DR |
|-------|:-----:|:----:|:--------:|:------:|:----------------:|
| Images | 1,805 | 370 | 999 | 193 | 295 |

> The dataset is **not included** in this repository. It requires a Kaggle account and acceptance of the competition rules. See [Getting Started](#getting-started).

---

## Models

| # | Model | Framework | Type | Notebook |
|---|-------|-----------|------|----------|
| 1 | **Custom CNN** (5 conv blocks, pooling + batch norm) | PyTorch | Trained from scratch | `CNN-Apitos2019.ipynb` |
| 2 | **EfficientNetB0** | TensorFlow / Keras | Transfer learning (ImageNet) | `EfficientNetB0-Apitos2019.ipynb` |
| 3 | **ResNet50** | TensorFlow / Keras | Transfer learning (ImageNet), two-stage fine-tuning | `resnet50_train.ipynb` |
| 4 | **DenseNet121** | TensorFlow / Keras | Transfer learning (ImageNet), two-stage fine-tuning | `densenet-Apitos2019.ipynb` |

The custom CNN acts as the from-scratch baseline that shows how much transfer learning adds. The three pretrained backbones use a new classification head (global average pooling, dropout, 5-way softmax).

---

## Methodology

1. **Data loading:** APTOS 2019 downloaded via the Kaggle API (CNN notebook) or from a local copy (other notebooks).
2. **Preprocessing:** black-border cropping, resizing to 224×224, and contrast enhancement (CLAHE / Gaussian filtering) in the CNN pipeline; ImageNet-style normalization for the pretrained backbones.
3. **Splitting:** stratified splits with a fixed seed. The CNN and ResNet50 notebooks use **70 / 15 / 15** (train / val / test). The EfficientNetB0 and DenseNet121 notebooks use **80 / 20** (train / val).
4. **Class imbalance:** class weights computed from the training labels (DenseNet121 uses square-root-softened weights to avoid over-predicting rare classes).
5. **Augmentation:** random flips, small rotations, zoom, and contrast changes on the training set only.
6. **Training:**
   - CNN: up to 50 epochs with early stopping, LR scheduler, and a with/without augmentation and with/without class-weight ablation.
   - ResNet50 / DenseNet121: **Stage 1** trains the new head on a frozen backbone; **Stage 2** unfreezes the top layers with a much smaller learning rate. The checkpoint with the lower validation loss is kept.
   - EfficientNetB0: Adam (lr 1e-4), class weights, early stopping, and learning-rate reduction on plateau.
7. **Evaluation:** accuracy, precision / recall / F1 (macro and weighted), QWK, ROC-AUC, and confusion matrices.
8. **Explainability:** Grad-CAM heatmaps for ResNet50, EfficientNetB0 and DenseNet121 to show which retinal regions drive each prediction.

---

## Results

| Model | Evaluated on | Accuracy | Macro F1 | QWK | Macro ROC-AUC | Total Params |
|-------|--------------|:--------:|:--------:|:---:|:-------------:|:------------:|
| Custom CNN | Test (n = 550) | 0.740 | — | — | 0.884 (val) | — |
| EfficientNetB0 | Val (n = 733) | 0.790 | 0.640 | — | — | 6.77 M |
| ResNet50 | Test (n = 550) | 0.764 | 0.582 | 0.837 | 0.914 | 23.6 M |
| DenseNet121 | Val (n = 732) | 0.742 | 0.571 | 0.808 | — | 7.04 M |

Additional figures:

- **Custom CNN:** test loss 0.945, test F1 0.698; best validation F1 0.709.
- **ResNet50:** weighted F1 0.761; training time about 15 minutes; about 49 ms per image at inference; model file 90.75 MB.
- **EfficientNetB0 (per-class F1 on val):** No DR 0.97, Moderate 0.72, Mild 0.55, Severe 0.50, Proliferative DR 0.47.

**Note on comparability:** the notebooks were run on different splits (see [Methodology](#methodology)), so the numbers above are indicative rather than a strictly controlled comparison. Across all four models, the majority *No DR* class is classified very well, while *Mild*, *Severe* and *Proliferative DR* are the weak points, which reflects the class imbalance in the data.

---

## Repository Structure

```
.
├── CNN-Apitos2019.ipynb               # Custom CNN (PyTorch) + ablations
├── EfficientNetB0-Apitos2019.ipynb    # EfficientNetB0 + Grad-CAM + inference helper
├── resnet50_train.ipynb               # ResNet50 two-stage training, evaluation, ROC, Grad-CAM
├── densenet-Apitos2019.ipynb          # DenseNet121 two-stage training, QWK, Grad-CAM
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A Kaggle account (with the APTOS 2019 competition rules accepted) and an API token (`kaggle.json`)
- A GPU is strongly recommended. The CNN and DenseNet121 notebooks are set up for **Google Colab**.

### Running on Google Colab

1. Open a notebook from this branch in Colab (the CNN and DenseNet121 notebooks include an *Open in Colab* badge).
2. Set the runtime to a GPU.
3. Upload your `kaggle.json` when prompted (CNN notebook) so the dataset can be downloaded automatically.
4. Run the cells top to bottom.

### Running locally

```bash
git clone https://github.com/sadumina/Deep-Learning-Assignment-Group-ID-SE4050-G05.git
cd Deep-Learning-Assignment-Group-ID-SE4050-G05
git checkout Finalize-Models

# TensorFlow notebooks (EfficientNetB0, ResNet50, DenseNet121)
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn opencv-python jupyter

# PyTorch notebook (Custom CNN)
pip install torch torchvision kaggle
```

Download the data from the [competition page](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), then update the dataset path variables at the top of each notebook (`PROJECT_ROOT` / `BASE_PATH`) to point to your copy.

### Reproducibility

- Random seeds are fixed in each notebook, and splits are stratified.
- Best checkpoints are saved during training (`ModelCheckpoint` in the Keras notebooks, best-validation weights in the PyTorch notebook).
- Trained weights and datasets are not stored in this repository because of their size.

---

## Team & Contributions

| Name | Student ID | Contribution |
|------|-----------|--------------|
| Bagya R M S *(Group Leader)* | IT23394124 | Custom CNN |
| Insath MMM | IT23183872 | EfficientNetB0 |
| Purijjala R W M A H | IT23431676 | ResNet50 |
| Gayathree M.G.K | IT23334106 | DenseNet121 |

---

## References

1. APTOS 2019 Blindness Detection. Kaggle / Asia Pacific Tele-Ophthalmology Society. https://www.kaggle.com/competitions/aptos2019-blindness-detection
2. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks*. https://arxiv.org/abs/1608.06993
3. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. https://arxiv.org/abs/1905.11946
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. https://arxiv.org/abs/1512.03385
5. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. https://arxiv.org/abs/1610.02391

---

## License

This project was developed as academic coursework for SE4050 – Deep Learning at SLIIT. The APTOS 2019 dataset remains subject to its original Kaggle / APTOS terms and is not redistributed here.

---

<p align="center">
  Built for SE4050 — Deep Learning, SLIIT (2026)
</p>
