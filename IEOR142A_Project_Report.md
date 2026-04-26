# AI-Generated vs. Real Image Detection with Transfer Learning

Team: [Names]  
Course: IEOR 142A, Spring 2026  
Presentation recording: [Insert accessible recording link]  
Interactive demo: https://huggingface.co/spaces/Theo9598/ai-vs-real-image-detector

## Abstract

This project builds an interactive machine learning system for estimating whether an uploaded image is AI-generated or real. The final validation-weighted ensemble achieves 96.7% accuracy, 0.946 AI-class precision, 0.921 AI-class recall, 0.933 AI-class F1, and 0.993 ROC-AUC on the held-out test set.

## Appendix

The code appendix is available in the project GitHub repository: https://github.com/Theo9598/ai-vs-real-image-detector. The repository contains the training script train_ai_detector.py, the Hugging Face Space inference app hf_space_ai_detector/app.py, the formatted report, and the results/ folder with dataset_splits.csv, validation_model_comparison.csv, final_results.json, confusion_matrix.png, roc_curve.png, test_predictions.csv, and per_category_metrics.csv. The trained model weights used by the deployed Space are tracked with Git LFS. The raw image dataset is not committed to GitHub because of size and access constraints; the dataset link is provided in the report. The deployed interactive demo is available at https://huggingface.co/spaces/Theo9598/ai-vs-real-image-detector.
