# AI-Generated vs. Real Image Detection

IEOR 142A Spring 2026 project: an interactive detector that estimates whether an uploaded image is AI-generated or real.

## Demo

Hugging Face Space: <https://huggingface.co/spaces/Theo9598/ai-vs-real-image-detector>

The deployed Space uses the Tiny GenImage Random Forest model for the final prediction and uses ResNet18 only for the attention visualization.

## Dataset

The primary dataset for the deployed model and final report experiment is Tiny GenImage on Hugging Face:

<https://huggingface.co/datasets/TheKernel01/Tiny-GenImage>

This project uses a 5,000-image subset:

- 3,600 fit images
- 400 validation images
- 1,000 held-out test images

The local Google Drive dataset below is used only for the smaller local comparison experiment:

<https://drive.google.com/drive/folders/1KjEAbGMLvyMWPQi33EP8J9C16UpsV22O?usp=sharing>

Expected local structure for the Google Drive comparison dataset:

```text
data/
|-- Ai_generated_dataset/
`-- real_dataset/
```

## Results

Active Space model performance on the Tiny GenImage 5k held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 98.2% |
| Precision, fake / AI class | 0.967 |
| Recall, fake / AI class | 0.998 |
| F1, fake / AI class | 0.982 |
| ROC-AUC | 1.000 |

Local held-out test performance for the validation-weighted ensemble:

| Metric | Result |
|---|---:|
| Accuracy | 96.7% |
| Precision, AI class | 0.946 |
| Recall, AI class | 0.921 |
| F1, AI class | 0.933 |
| ROC-AUC | 0.993 |

Confusion matrix with class order `[real, AI-generated]`: `[[110, 2], [3, 35]]`.

## Tiny GenImage 5k Extension

To test the project on a larger external dataset, we ran a 5,000-image Tiny GenImage subset experiment:

- fit: 3,600 images
- validation: 400 images
- held-out test: 1,000 images

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Random Forest + handcrafted features | 98.2% | 0.982 | 1.000 |
| Logistic Regression + handcrafted features | 78.9% | 0.819 | 0.759 |
| ResNet18 transfer learning, 10 epochs | 78.5% | 0.797 | 0.865 |

Main learning: feature engineering and strong baselines matter. A deeper transfer-learning model does not automatically outperform a well-matched classical model.

## Files

- `train_ai_detector.py`: local training and evaluation script.
- `compare_course_vs_transfer.py`: local classical-model vs transfer-learning comparison.
- `tiny_genimage_5k_compare.py`: Tiny GenImage 5k comparison script.
- `hf_space_ai_detector/app.py`: Hugging Face Space inference app.
- `results/`: evaluation artifacts, plots, predictions, and metrics.
- `reports/142A_final_project_ai_detector_appendix.pdf`: final project report with code appendix.

Raw image data and cached parquet shards are not committed to keep the repository lightweight.
