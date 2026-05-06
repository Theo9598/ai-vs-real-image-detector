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

The active detector loads the Tiny GenImage 5k ResNet18 transfer-learning model used in the final report. It was fine-tuned for 10 epochs and uses the validation-selected threshold from that experiment.

Active Space model performance on the Tiny GenImage 5k held-out test set:

- Accuracy: 78.5%
- Precision for fake / AI-generated class: 0.755
- Recall for fake / AI-generated class: 0.844
- F1 for fake / AI-generated class: 0.797
- ROC-AUC: 0.865

The report also includes the local project dataset ensemble as a comparison result:

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
