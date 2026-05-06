---
title: AI vs Real Image Detector
emoji: 🔎
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# AI vs Real Image Detector

This Space hosts a course project demo for detecting whether an uploaded image is AI-generated or real.

The app loads four fine-tuned transfer-learning models and uses a validation-weighted ensemble selected on a held-out validation set.

Held-out test performance on the local project dataset:

- Accuracy: 96.7%
- Precision for AI-generated class: 0.946
- Recall for AI-generated class: 0.921
- F1 for AI-generated class: 0.933
- ROC-AUC: 0.993

The report also includes a Tiny GenImage 5k experiment comparing classical machine learning baselines with ResNet18:

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Random Forest + handcrafted features | 98.2% | 0.982 | 1.000 |
| Logistic Regression + handcrafted features | 78.9% | 0.819 | 0.759 |
| ResNet18 transfer learning, 10 epochs | 78.5% | 0.797 | 0.865 |

The main learning is that strong baselines and feature engineering matter. A more complex neural network does not automatically outperform a well-matched classical model.

The prediction is statistical decision support, not proof of image origin.
