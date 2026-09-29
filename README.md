# Research Papers — Diabetic Retinopathy Screening

> **SE4050 — Deep Learning | SLIIT | Group G05 | 2026**

This branch collects the research papers our group read while preparing the Diabetic Retinopathy (DR) screening project. The papers were used as background reading to understand deep learning architectures and design choices before building and comparing our four models (Custom CNN, EfficientNetB0, ResNet50, DenseNet121).

The final code and results are on the `Finalize-Models` branch.

---

## Papers Reviewed

| # | Paper | Authors | Venue / Year | File |
|---|-------|---------|--------------|------|
| 1 | **Densely Connected Convolutional Networks** (DenseNet) | Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger | arXiv:1608.06993 (CVPR 2017) | `Densnet Paper.pdf` |
| 2 | **A Systematic Review of Modern Object Detection: Classical CNN Architectures and Transformer-Based Approaches** | Amer Sh. Mohammed | IEEE ElCon-CN 2026 | `A Systematic Review of Modern Object Detection.pdf` |
| 3 | **Multimodal Feature Extraction and Fusion Deep Neural Networks for Short-Term Load Forecasting** | Zhengmin Kong, Chenggang Zhang, He Lv, Feng Xiong, Zhuolin Fu | IEEE Access, vol. 8, 2020 | `Multimodal_Feature_Extraction_and_Fusion_Deep_Neur.pdf` |

---

## Short Summaries

**1. DenseNet**
Introduces the Dense Convolutional Network, where every layer is connected to all later layers in a feed-forward way. This reduces the vanishing-gradient problem, encourages feature reuse, and needs fewer parameters. This is the basis of the DenseNet121 model used in our project.

**2. Modern Object Detection review**
A survey of deep-learning object detection, covering two-stage detectors (R-CNN family), one-stage detectors (YOLO, SSD), and attention/transformer-based models (ViT, DETR). It compares the trade-offs between accuracy, speed, and computational cost, and gives us background on CNN architectures and how they have evolved.

**3. Multimodal feature extraction and fusion for load forecasting**
Proposes a CNN-LSTM scheme with empirical mode decomposition to extract and fuse spatial and temporal features for short-term electricity load forecasting. Although the domain is different, it shows how features from multiple sources can be combined in a deep network.

---

## Repository Contents

```
.
├── Densnet Paper.pdf
├── A Systematic Review of Modern Object Detection.pdf
├── Multimodal_Feature_Extraction_and_Fusion_Deep_Neur.pdf
└── README.md
```

---

## Note

These papers belong to their respective authors and publishers and are included here for academic reference only, as part of coursework for SE4050 – Deep Learning at SLIIT.
