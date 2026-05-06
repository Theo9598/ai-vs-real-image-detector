import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)
from tqdm.auto import tqdm


SEED = 142
DATA_ROOT = Path(r"G:\CodexProjects\New project 3\data")
RESULTS_DIR = Path(r"G:\CodexProjects\New project 3\results")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_TO_ID = {"real": 0, "ai": 1}
ID_TO_LABEL = {0: "real", 1: "ai"}
AI_CLASS_ID = 1


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def collect_records(root: Path, label_name: str) -> list[dict]:
    rows = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append(
                {
                    "path": str(path),
                    "label": LABEL_TO_ID[label_name],
                    "label_name": label_name,
                    "category": path.parent.name,
                }
            )
    return rows


def load_or_create_splits() -> pd.DataFrame:
    split_path = RESULTS_DIR / "dataset_splits.csv"
    if split_path.exists():
        df = pd.read_csv(split_path)
        needed = {"path", "label", "label_name", "category", "split"}
        if needed.issubset(df.columns):
            return df

    records = collect_records(DATA_ROOT / "Ai_generated_dataset", "ai")
    records += collect_records(DATA_ROOT / "real_dataset", "real")
    df = pd.DataFrame(records).drop_duplicates("path").reset_index(drop=True)
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=SEED, stratify=df["label"]
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15 / 0.85,
        random_state=SEED,
        stratify=df.loc[train_val_idx, "label"],
    )
    df["split"] = "unused"
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "validation"
    df.loc[test_idx, "split"] = "test"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_path, index=False)
    return df


def safe_stats(values: np.ndarray) -> list[float]:
    values = values.astype(np.float32).ravel()
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
    ]


def handcrafted_features(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    small_rgb = image.resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(small_rgb).astype(np.float32) / 255.0
    gray = np.asarray(small_rgb.convert("L")).astype(np.float32) / 255.0

    features = []
    features.extend([width / max(height, 1), np.log1p(width * height)])

    # Color statistics.
    features.extend(arr.mean(axis=(0, 1)).tolist())
    features.extend(arr.std(axis=(0, 1)).tolist())
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=8, range=(0.0, 1.0), density=True)
        features.extend(hist.astype(float).tolist())

    gray_hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
    features.extend(gray_hist.astype(float).tolist())

    # Edge and texture statistics from simple gradients.
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    grad_mag = np.sqrt(gx * gx + gy * gy)
    features.extend(safe_stats(grad_mag))

    # Noise residual: original grayscale minus a small box blur.
    blurred = np.asarray(small_rgb.convert("L").filter(ImageFilter.BoxBlur(1))).astype(np.float32) / 255.0
    residual = gray - blurred
    features.extend(safe_stats(np.abs(residual)))

    # Frequency energy bands.
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    yy, xx = np.indices(spectrum.shape)
    center_y, center_x = (np.array(spectrum.shape) - 1) / 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    total_energy = spectrum.sum() + 1e-8
    low = spectrum[radius < 12].sum() / total_energy
    mid = spectrum[(radius >= 12) & (radius < 36)].sum() / total_energy
    high = spectrum[radius >= 36].sum() / total_energy
    features.extend([float(low), float(mid), float(high), float(high / (low + 1e-8))])

    # JPEG/blockiness proxy.
    vertical_edges = np.abs(np.diff(gray, axis=1))
    horizontal_edges = np.abs(np.diff(gray, axis=0))
    block_v = vertical_edges[:, 7::8].mean() if vertical_edges[:, 7::8].size else 0.0
    block_h = horizontal_edges[7::8, :].mean() if horizontal_edges[7::8, :].size else 0.0
    features.extend([float(block_v), float(block_h), float((block_v + block_h) / (grad_mag.mean() + 1e-8))])

    return np.asarray(features, dtype=np.float32)


def metric_dict(y_true, prob_ai, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob_ai = np.asarray(prob_ai).astype(float)
    y_pred = (prob_ai >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_ai": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_ai)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def best_threshold(y_true, prob_ai) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    return float(max(thresholds, key=lambda t: f1_score(y_true, prob_ai >= t, zero_division=0)))


def train_course_models(df: pd.DataFrame):
    split_frames = {name: df[df["split"] == name].reset_index(drop=True) for name in ["train", "validation", "test"]}
    feature_cache = {}
    for split, frame in split_frames.items():
        x = np.vstack(
            [handcrafted_features(path) for path in tqdm(frame["path"], desc=f"Handcrafted features: {split}")]
        )
        y = frame["label"].to_numpy(dtype=int)
        feature_cache[split] = (x, y)

    x_train, y_train = feature_cache["train"]
    x_val, y_val = feature_cache["validation"]
    x_test, y_test = feature_cache["test"]

    models_to_fit = {
        "Logistic Regression (handcrafted features)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        ),
        "Random Forest (handcrafted features)": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    rows = []
    for name, model in models_to_fit.items():
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, AI_CLASS_ID]
        threshold = best_threshold(y_val, val_prob)
        val_metrics = metric_dict(y_val, val_prob, threshold)
        val_metrics.update({"model": name, "model_family": "Course ML", "split": "validation"})
        rows.append(val_metrics)

        test_prob = model.predict_proba(x_test)[:, AI_CLASS_ID]
        test_metrics = metric_dict(y_test, test_prob, threshold)
        test_metrics.update({"model": name, "model_family": "Course ML", "split": "test"})
        rows.append(test_metrics)

    return pd.DataFrame(rows)


class ImagePathDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(row["label"])


def make_transfer_model(name: str):
    if name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        return model
    if name == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        return model
    raise ValueError(name)


def evaluate_transfer_model(model, loader, device):
    model.eval()
    y_true, prob_ai = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Transfer inference"):
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)[:, AI_CLASS_ID].detach().cpu().numpy()
            prob_ai.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.asarray(y_true), np.asarray(prob_ai)


def evaluate_transfer_models(df: pd.DataFrame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_loader = DataLoader(
        ImagePathDataset(df[df["split"] == "validation"], eval_transform),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ImagePathDataset(df[df["split"] == "test"], eval_transform),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model_names = ["resnet18", "efficientnet_b0", "mobilenet_v3_small", "vit_b_16"]
    rows = []
    val_probs_by_model = {}
    test_probs_by_model = {}
    y_val_ref = y_test_ref = None

    for name in model_names:
        weight_path = RESULTS_DIR / f"{name}_best.pt"
        if not weight_path.exists():
            print(f"Skipping {name}: missing {weight_path}")
            continue
        model = make_transfer_model(name).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        y_val, val_prob = evaluate_transfer_model(model, val_loader, device)
        threshold = best_threshold(y_val, val_prob)
        val_metrics = metric_dict(y_val, val_prob, threshold)
        val_metrics.update({"model": name, "model_family": "Transfer Learning", "split": "validation"})
        rows.append(val_metrics)

        y_test, test_prob = evaluate_transfer_model(model, test_loader, device)
        test_metrics = metric_dict(y_test, test_prob, threshold)
        test_metrics.update({"model": name, "model_family": "Transfer Learning", "split": "test"})
        rows.append(test_metrics)

        val_probs_by_model[name] = val_prob
        test_probs_by_model[name] = test_prob
        y_val_ref = y_val
        y_test_ref = y_test

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if val_probs_by_model:
        order = list(val_probs_by_model)
        val_f1s = np.asarray(
            [metric_dict(y_val_ref, val_probs_by_model[name], best_threshold(y_val_ref, val_probs_by_model[name]))["f1_ai"] for name in order]
        )
        weights = (val_f1s + 1e-6) / (val_f1s.sum() + 1e-6 * len(val_f1s))
        val_ensemble = np.average(np.vstack([val_probs_by_model[name] for name in order]), axis=0, weights=weights)
        ensemble_threshold = best_threshold(y_val_ref, val_ensemble)
        val_metrics = metric_dict(y_val_ref, val_ensemble, ensemble_threshold)
        val_metrics.update({"model": "Validation-weighted ensemble", "model_family": "Final Ensemble", "split": "validation"})
        rows.append(val_metrics)

        test_ensemble = np.average(np.vstack([test_probs_by_model[name] for name in order]), axis=0, weights=weights)
        test_metrics = metric_dict(y_test_ref, test_ensemble, ensemble_threshold)
        test_metrics.update({"model": "Validation-weighted ensemble", "model_family": "Final Ensemble", "split": "test"})
        rows.append(test_metrics)

        cm = confusion_matrix(y_test_ref, test_ensemble >= ensemble_threshold, labels=[0, 1])
        (RESULTS_DIR / "course_vs_transfer_ensemble_confusion_matrix.json").write_text(
            json.dumps({"labels": ["real", "ai"], "matrix": cm.tolist(), "weights": dict(zip(order, weights.tolist()))}, indent=2),
            encoding="utf-8",
        )

    return pd.DataFrame(rows)


def main():
    seed_everything()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_create_splits()
    print("Dataset counts:", df["label_name"].value_counts().to_dict())
    print("Split counts:", df["split"].value_counts().to_dict())

    course_results = train_course_models(df)
    transfer_results = evaluate_transfer_models(df)
    all_results = pd.concat([course_results, transfer_results], ignore_index=True)
    all_results = all_results[
        ["model_family", "model", "split", "threshold", "accuracy", "precision_ai", "recall_ai", "f1_ai", "roc_auc"]
    ].sort_values(["split", "model_family", "f1_ai"], ascending=[True, True, False])

    all_results.to_csv(RESULTS_DIR / "course_vs_transfer_comparison.csv", index=False)
    all_results[all_results["split"] == "validation"].to_csv(
        RESULTS_DIR / "course_vs_transfer_validation.csv", index=False
    )
    all_results[all_results["split"] == "test"].to_csv(RESULTS_DIR / "course_vs_transfer_test.csv", index=False)

    test_results = all_results[all_results["split"] == "test"].copy()
    plt.figure(figsize=(9, 4.8))
    colors = {
        "Course ML": "#6f6f6f",
        "Transfer Learning": "#2f2f2f",
        "Final Ensemble": "#000000",
    }
    bar_colors = [colors.get(family, "#444444") for family in test_results["model_family"]]
    plt.barh(test_results["model"], test_results["f1_ai"], color=bar_colors)
    plt.xlabel("Held-out test F1 for AI class")
    plt.xlim(0, 1.05)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "course_vs_transfer_test_f1.png", dpi=200)
    plt.close()

    summary = {
        "dataset_counts": df["label_name"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "results_file": str(RESULTS_DIR / "course_vs_transfer_comparison.csv"),
        "test_results": test_results.to_dict(orient="records"),
    }
    (RESULTS_DIR / "course_vs_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nHeld-out test comparison:")
    print(test_results.to_string(index=False))
    print("\nWrote comparison files to", RESULTS_DIR)


if __name__ == "__main__":
    main()
