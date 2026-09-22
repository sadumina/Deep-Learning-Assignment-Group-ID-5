# Custom CNN - Diabetic Retinopathy Screening (Baseline Model)

## Overview

This branch implements a **Custom Convolutional Neural Network (CNN)** trained from scratch as the baseline model for our SE4050 Deep Learning group project: **Diabetic Retinopathy Screening — Automated Detection of Retinal Damage from Eye Scans**.

This model does **not** use any pretrained/transfer-learning weights. It exists specifically to answer: *how much does transfer learning actually help on a small, specialized medical imaging dataset?* — the other three models in this project (DenseNet121, EfficientNetB0, ResNet50) use ImageNet pretrained weights, so this CNN is the point of comparison.

## What is a CNN?

A Convolutional Neural Network (CNN) is a deep learning architecture designed primarily for processing grid-like data such as images. Instead of treating an image as a flat vector of pixels, CNNs use **convolution layers** that slide small filters (kernels) across the image to detect local patterns — edges, textures, shapes — and build up to more complex, high-level features (like blood vessel damage or hemorrhages, in our case) as the network gets deeper.

Key building blocks:
- **Convolutional layers (Conv2D):** learn spatial filters that detect features like edges, curves, and textures directly from raw pixels.
- **Pooling layers (MaxPooling):** downsample feature maps, reducing spatial dimensions while retaining the strongest activations — this keeps the model computationally efficient and adds some translation invariance.
- **Dense (fully connected) layers:** placed at the end of the network to combine the extracted features and produce the final classification.
- **Activation functions (ReLU, Softmax):** introduce non-linearity so the network can learn complex patterns, with Softmax used at the output layer for multi-class probability distribution.

CNNs are the backbone architecture behind most modern computer vision tasks, including:
- **Object detection** — locating and classifying multiple objects in an image
- **Face verification** — comparing facial features to confirm identity
- **Image classification / medical image processing** — exactly what we're doing here: classifying retinal fundus images into DR severity stages

## Architecture (this implementation)

A handful of `Conv2D + MaxPooling` blocks feeding into dense layers — no pretrained backbone.

```
Input (Retinal Fundus Image)
    → Conv2D + ReLU → MaxPooling2D
    → Conv2D + ReLU → MaxPooling2D
    → Conv2D + ReLU → MaxPooling2D
    → Flatten
    → Dense + ReLU → Dropout
    → Dense (Softmax, 5 classes)
```

*(Update this diagram to match your final layer counts/filter sizes once finalized.)*

## Dataset

- **Source:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Kaggle)
- **Classes (5):** No DR, Mild, Moderate, Severe, Proliferative DR
- **Size:** ~3,662 labeled retinal fundus images
- Split into train/validation/test using stratified sampling to preserve class balance (dataset is imbalanced — "No DR" dominates).

## Why this model matters in the project

| Model | Pretrained? | Role |
|---|---|---|
| **Custom CNN (this branch)** | No | Baseline — from-scratch performance |
| DenseNet121 | Yes (ImageNet) | Dense connectivity, strong feature reuse |
| EfficientNetB0 | Yes (ImageNet) | Compound scaling, efficiency comparison |
| ResNet50 | Yes (ImageNet) | Residual connections, depth comparison |

This CNN's results are what we compare the transfer-learning models against to demonstrate the accuracy/generalization gains transfer learning provides.

## Files in this branch

```
├── custom_cnn.ipynb / .py     # Model definition, training loop
├── data_preprocessing.py      # Image loading, resizing, augmentation, stratified split
├── evaluation.py               # Accuracy, confusion matrix, classification report
└── README.md                   # This file
```

## Training Setup

- **Input size:** _(e.g. 224x224 — update once fixed)_
- **Optimizer:** _(e.g. Adam)_
- **Loss function:** Categorical Crossentropy
- **Epochs:** _(update)_
- **Batch size:** _(update)_
- **Augmentation:** _(rotation, flips, zoom — update)_

## Results

_(Add accuracy, precision/recall/F1 per class, confusion matrix, and training curves here once training is complete.)_

## References

- CS231n Convolutional Neural Networks notes (conceptual foundation for conv/pooling layers)
- [Keras Conv2D documentation](https://keras.io/api/layers/convolution_layers/convolution2d/)
- APTOS 2019 Blindness Detection dataset — Kaggle

## Author

Bagya R M S (IT23394124) — Group Leader
SE4050 Deep Learning Group Project — SLIIT
