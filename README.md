# AI-Generated vs. Real Image Detection

IEOR 142A Spring 2026 project: an interactive detector that estimates whether an uploaded image is AI-generated or real.

## Demo

Hugging Face Space: <https://huggingface.co/spaces/Theo9598/ai-vs-real-image-detector>

## Results

Held-out test performance for the validation-weighted ensemble:

| Metric | Result |
|---|---:|
| Accuracy | 96.7% |
| Precision, AI class | 0.946 |
| Recall, AI class | 0.921 |
| F1, AI class | 0.933 |
| ROC-AUC | 0.993 |

Confusion matrix with class order `[real, AI-generated]`: `[[110, 2], [3, 35]]`.

## Files

- `train_ai_detector.py`: local training and evaluation script.
- `hf_space_ai_detector/app.py`: Hugging Face Space inference app.
- `results/`: evaluation artifacts, plots, predictions, and metrics.
- `IEOR142A_Project_Report_Formatted.docx`: formatted project report.

The raw image dataset is not committed. Dataset access is documented in the report.
