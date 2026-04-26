---
title: AI vs Real Image Detector
emoji: 🖼️
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

Held-out test performance on the project dataset:

- Accuracy: 96.7%
- Precision for AI-generated class: 0.946
- Recall for AI-generated class: 0.921
- F1 for AI-generated class: 0.933
- ROC-AUC: 0.993

The prediction is statistical decision support, not proof of image origin.
