# Diabetic Retinopathy Screening — Automated Detection of Retinal Damage from Eye Scans

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

> **SE4050 — Deep Learning | BSc (Hons) in Information Technology**
> Sri Lanka Institute of Information Technology (SLIIT) | 2026

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Getting Started](#getting-started)
- [Reproducibility](#reproducibility)
- [Team & Contributions](#team--contributions)
- [References](#references)
- [License](#license)

---

## Overview

Diabetic Retinopathy (DR) is a progressive complication of diabetes that damages the blood vessels of the retina and remains one of the leading causes of preventable blindness worldwide. Early detection through regular retinal screening is critical, but manual grading by ophthalmologists is time-intensive and depends on specialist availability that is limited in many regions.

This project implements and critically compares **four distinct deep learning architectures** for automated, multi-class classification of Diabetic Retinopathy severity from retinal fundus images, with the goal of evaluating which approach best balances predictive accuracy, generalization, and computational efficiency for real-world screening use cases.

This repository was developed as the practical submission for the **SE4050 – Deep Learning** module assignment (SLIIT, 2026), under the **Supervised Deep Learning** project category.

---

## Problem Statement

Given a retinal fundus image, classify the severity of Diabetic Retinopathy into one of five clinically defined stages:

| Class | Label | Description |
|:-----:|-------|-------------|
| 0 | No DR | No visible signs of retinopathy |
| 1 | Mild NPDR | Microaneurysms present |
| 2 | Moderate NPDR | More extensive vascular damage |
| 3 | Severe NPDR | Widespread blood vessel blockage |
| 4 | Proliferative DR | Abnormal new vessel growth; highest risk of blindness |

This is framed as a **multi-class image classification problem**, with additional attention to the ordinal nature of the severity scale during evaluation.

---

## Dataset

- **Source:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) — Kaggle, hosted by the Asia Pacific Tele-Ophthalmology Society (APTOS)
- **Size:** ~3,660 labeled retinal fundus images (training set)
- **Format:** RGB `.png` images of varying resolution, accompanied by a CSV of per-image severity labels (0–4)
- **License / Access:** Publicly available under Kaggle competition terms; see the dataset page for full licensing details

> Dataset files are **not included** in this repository due to size and licensing. See [Getting Started](#getting-started) for download instructions.

---

## Models

Four architecturally distinct deep learning models are implemented and evaluated under identical experimental conditions:

| # | Model | Type | Key Idea |
|---|-------|------|----------|
| 1 | **Custom CNN** | Trained from scratch | Baseline convolutional network with no pretrained weights; establishes how much transfer learning improves over a naive approach |
| 2 | **DenseNet121** | Transfer learning (ImageNet) | Dense connectivity between layers improves feature reuse and gradient flow |
| 3 | **EfficientNetB0** | Transfer learning (ImageNet) | Compound scaling of depth/width/resolution for strong accuracy-to-parameter efficiency |
| 4 | **ResNet50** | Transfer learning (ImageNet) | Residual connections enable stable training of deeper networks |

Each model is documented in its own notebook with architecture justification, hyperparameter configuration, and training curves.

---

## Methodology

1. **Exploratory Data Analysis** - class distribution, image dimensions, sample visualization per class
2. **Preprocessing** - black-border cropping, resizing, contrast enhancement (CLAHE / Gaussian blend), normalization
3. **Data Splitting** - stratified train / validation / test split; test set held out and used **only** for final evaluation
4. **Class Imbalance Handling** - class weighting to address the natural skew toward "No DR" cases
5. **Model Training** - all four models trained under matched conditions (same splits, batch size philosophy, and augmentation strategy) for fair comparison
6. **Explainability** - Grad-CAM visualizations to interpret which retinal regions drive each model's predictions
7. **Comparative Evaluation** - performance, efficiency, and generalization compared across all four models

---

## Evaluation Metrics

- **Quadratic Weighted Kappa** — primary metric, accounts for the ordinal severity scale
- **Accuracy, Precision, Recall, F1-score** (per-class and macro-averaged)
- **ROC-AUC** (one-vs-rest)
- **Confusion Matrix**
- **Training time & parameter count** — for computational efficiency comparison

---

## Results

| Model | Val. Kappa | Test Kappa | Accuracy | Params | Training Time |
|-------|:----------:|:----------:|:--------:|:------:|:--------------:|
| Custom CNN | — | — | — | — | — |
| DenseNet121 | — | — | — | — | — |
| EfficientNetB0 | — | — | — | — | — |
| ResNet50 | — | — | — | — | — |

*Full results, confusion matrices, and Grad-CAM visualizations are available in `reports/figures/` and discussed in detail in `reports/Report.pdf`.*

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

```bash
git clone https://github.com/<your-org>/diabetic-retinopathy-screening.git
cd diabetic-retinopathy-screening
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate dr-screening
```

### Dataset Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) (requires a Kaggle account and competition acceptance)
2. Place the extracted files under `data/raw/`
3. Run the preprocessing notebook:

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### Training a Model

```bash
python src/train.py --model densenet121 --config configs/config.yaml
```

### Evaluating a Model

```bash
python src/evaluate.py --model densenet121 --weights saved_models/densenet121_best.h5
```

---

## Reproducibility

- Random seed fixed at `42` across NumPy, TensorFlow, and data splitting
- Exact hyperparameters and training configuration for each model are stored in `configs/config.yaml`
- Model checkpoints saved on best validation kappa score (see `saved_models/`)
- Dependency versions pinned in `requirements.txt` / `environment.yml`

---

## Team & Contributions

| Name | Student ID | Role / Focus Area |
|------|-----------|--------------------|
| Bagya R M S *(Leader)* | IT23394124 | Custom CNN + Preprocessing pipeline |
| Insath MMM | IT23183872 | DenseNet121 |
| Gayathree M.G.K | IT23334106 | EfficientNetB0 |
| Purijjala R W M A H | IT23431676 | ResNet50 + Evaluation/Grad-CAM |

See `Members.txt` for full contact details and [commit history](../../commits/main) for individual contributions.

---

## References

1. APTOS 2019 Blindness Detection Dataset. Kaggle / Asia Pacific Tele-Ophthalmology Society. https://www.kaggle.com/competitions/aptos2019-blindness-detection
2. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks*. https://arxiv.org/abs/1608.06993
3. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. https://arxiv.org/abs/1905.11946
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. https://arxiv.org/abs/1512.03385
5. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. https://arxiv.org/abs/1610.02391

---

## License

This project is submitted as academic coursework for SE4050 – Deep Learning at SLIIT. Code is released under the [MIT License](LICENSE) unless otherwise noted. The dataset is subject to its original Kaggle/APTOS licensing terms and is not redistributed in this repository.

---

<p align="center">
  Built for SE4050 — Deep Learning, SLIIT (2026)
</p>
