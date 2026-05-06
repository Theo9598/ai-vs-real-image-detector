import io
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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
from torchvision.models import ResNet18_Weights
from tqdm.auto import tqdm


SEED = 142
DATASET = "TheKernel01/Tiny-GenImage"
ROOT = Path(r"G:\CodexProjects\New project 3")
OUT_DIR = ROOT / "results" / "tiny_genimage_5k"
CACHE_DIR = ROOT / "data_tiny_genimage_5k" / "parquet"
LABEL_TO_NAME = {0: "real", 1: "fake"}
AI_CLASS_ID = 1


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def get_parquet_files():
    url = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}"
    data = requests.get(url, timeout=120).json()["parquet_files"]
    files = {}
    for row in data:
        files.setdefault(row["split"], []).append(row)
    for split in files:
        files[split] = sorted(files[split], key=lambda x: x["filename"])
    return files


def download_file(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=240) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def image_from_cell(cell):
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")
    if hasattr(cell, "as_py"):
        return image_from_cell(cell.as_py())
    raise ValueError(f"Unsupported image cell type: {type(cell)}")


def read_rows(split: str, n_rows: int, shard_count: int):
    files = get_parquet_files()[split][:shard_count]
    frames = []
    for file_info in files:
        local_path = CACHE_DIR / split / file_info["filename"]
        download_file(file_info["url"], local_path)
        table = pq.read_table(local_path, columns=["image", "label", "generator"])
        frames.append(table.to_pandas())
        if sum(len(frame) for frame in frames) >= n_rows:
            break
    df = pd.concat(frames, ignore_index=True).iloc[:n_rows].copy()
    df["source_split"] = split
    df["label_name"] = df["label"].map(LABEL_TO_NAME)
    return df


def safe_stats(values):
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
    ]


def handcrafted_features(image_cell):
    image = image_from_cell(image_cell)
    width, height = image.size
    small_rgb = image.resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(small_rgb).astype(np.float32) / 255.0
    gray = np.asarray(small_rgb.convert("L")).astype(np.float32) / 255.0

    features = [width / max(height, 1), np.log1p(width * height)]
    features.extend(arr.mean(axis=(0, 1)).tolist())
    features.extend(arr.std(axis=(0, 1)).tolist())
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=8, range=(0.0, 1.0), density=True)
        features.extend(hist.astype(float).tolist())
    gray_hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
    features.extend(gray_hist.astype(float).tolist())

    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    grad_mag = np.sqrt(gx * gx + gy * gy)
    features.extend(safe_stats(grad_mag))

    blurred = np.asarray(small_rgb.convert("L").filter(ImageFilter.BoxBlur(1))).astype(np.float32) / 255.0
    features.extend(safe_stats(np.abs(gray - blurred)))

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    yy, xx = np.indices(spectrum.shape)
    center_y, center_x = (np.array(spectrum.shape) - 1) / 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    total_energy = spectrum.sum() + 1e-8
    low = spectrum[radius < 12].sum() / total_energy
    mid = spectrum[(radius >= 12) & (radius < 36)].sum() / total_energy
    high = spectrum[radius >= 36].sum() / total_energy
    features.extend([float(low), float(mid), float(high), float(high / (low + 1e-8))])

    vertical_edges = np.abs(np.diff(gray, axis=1))
    horizontal_edges = np.abs(np.diff(gray, axis=0))
    block_v = vertical_edges[:, 7::8].mean() if vertical_edges[:, 7::8].size else 0.0
    block_h = horizontal_edges[7::8, :].mean() if horizontal_edges[7::8, :].size else 0.0
    features.extend([float(block_v), float(block_h), float((block_v + block_h) / (grad_mag.mean() + 1e-8))])
    return np.asarray(features, dtype=np.float32)


def metric_dict(y_true, prob_ai, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    prob_ai = np.asarray(prob_ai).astype(float)
    pred = (prob_ai >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_ai": float(precision_score(y_true, pred, zero_division=0)),
        "recall_ai": float(recall_score(y_true, pred, zero_division=0)),
        "f1_ai": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_ai)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def best_threshold(y_true, prob_ai):
    thresholds = np.linspace(0.05, 0.95, 181)
    return float(max(thresholds, key=lambda t: f1_score(y_true, prob_ai >= t, zero_division=0)))


class TinyImageDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = image_from_cell(row["image"])
        return self.transform(image), int(row["label"])


def evaluate_resnet(model, loader, device):
    model.eval()
    y_true, prob_ai = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="ResNet18 eval"):
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)[:, AI_CLASS_ID].detach().cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            prob_ai.extend(probs.tolist())
    return np.asarray(y_true), np.asarray(prob_ai)


def train_resnet(train_df, val_df, test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_loader = DataLoader(TinyImageDataset(train_df, train_transform), batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TinyImageDataset(val_df, eval_transform), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TinyImageDataset(test_df, eval_transform), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    counts = train_df["label"].value_counts().reindex([0, 1]).fillna(0).astype(float).values
    class_weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = None
    best_val_f1 = -1
    history = []
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in tqdm(train_loader, desc=f"ResNet18 epoch {epoch}/10"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        y_val, p_val = evaluate_resnet(model, val_loader, device)
        threshold = best_threshold(y_val, p_val)
        metrics = metric_dict(y_val, p_val, threshold)
        metrics.update({"epoch": epoch, "train_loss": total_loss / max(seen, 1)})
        history.append(metrics)
        print(f"epoch {epoch}: loss={metrics['train_loss']:.4f}, val_f1={metrics['f1_ai']:.4f}, val_auc={metrics['roc_auc']:.4f}, thr={threshold:.3f}")
        if metrics["f1_ai"] > best_val_f1:
            best_val_f1 = metrics["f1_ai"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    y_val, p_val = evaluate_resnet(model, val_loader, device)
    threshold = best_threshold(y_val, p_val)
    y_test, p_test = evaluate_resnet(model, test_loader, device)
    torch.save(model.state_dict(), OUT_DIR / "tiny_genimage_5k_resnet18_best.pt")
    return pd.DataFrame(history), metric_dict(y_test, p_test, threshold), threshold


def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pool = read_rows("train", n_rows=4000, shard_count=2)
    test_df = read_rows("validation", n_rows=1000, shard_count=1)
    fit_idx, val_idx = train_test_split(
        np.arange(len(train_pool)),
        test_size=400,
        random_state=SEED,
        stratify=train_pool["label"],
    )
    fit_df = train_pool.iloc[fit_idx].reset_index(drop=True)
    val_df = train_pool.iloc[val_idx].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_summary = {
        "fit": fit_df["label_name"].value_counts().to_dict(),
        "validation": val_df["label_name"].value_counts().to_dict(),
        "test": test_df["label_name"].value_counts().to_dict(),
    }
    print("Split summary:", split_summary)

    frames = []
    for name, frame in [("fit", fit_df), ("validation", val_df), ("test", test_df)]:
        tmp = frame[["label", "label_name", "generator"]].copy()
        tmp["split"] = name
        frames.append(tmp)
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "tiny_genimage_5k_splits.csv", index=False)

    feature_sets = {}
    for split, frame in [("fit", fit_df), ("validation", val_df), ("test", test_df)]:
        x = np.vstack([handcrafted_features(cell) for cell in tqdm(frame["image"], desc=f"Handcrafted features: {split}")])
        y = frame["label"].to_numpy(dtype=int)
        feature_sets[split] = (x, y)

    x_fit, y_fit = feature_sets["fit"]
    x_val, y_val = feature_sets["validation"]
    x_test, y_test = feature_sets["test"]

    model_specs = {
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
    for model_name, model in model_specs.items():
        model.fit(x_fit, y_fit)
        val_prob = model.predict_proba(x_val)[:, AI_CLASS_ID]
        threshold = best_threshold(y_val, val_prob)
        test_prob = model.predict_proba(x_test)[:, AI_CLASS_ID]
        metrics = metric_dict(y_test, test_prob, threshold)
        metrics.update({"model": model_name, "model_family": "Classical ML", "split": "test"})
        rows.append(metrics)
        print(model_name, metrics)

    history, resnet_metrics, resnet_threshold = train_resnet(fit_df, val_df, test_df)
    history.to_csv(OUT_DIR / "tiny_genimage_5k_resnet18_history.csv", index=False)
    resnet_metrics.update({"model": "ResNet18 transfer learning", "model_family": "Transfer Learning", "split": "test"})
    rows.append(resnet_metrics)
    print("ResNet18 transfer learning", resnet_metrics)

    results = pd.DataFrame(rows)[
        ["model_family", "model", "split", "threshold", "accuracy", "precision_ai", "recall_ai", "f1_ai", "roc_auc"]
    ].sort_values("f1_ai", ascending=False)
    results.to_csv(OUT_DIR / "tiny_genimage_5k_model_comparison.csv", index=False)

    colors = {"Classical ML": "#1f77b4", "Transfer Learning": "#ff7f0e"}
    plt.figure(figsize=(7.6, 3.4))
    plt.barh(results["model"], results["f1_ai"], color=[colors.get(x, "#2ca02c") for x in results["model_family"]])
    plt.xlabel("Tiny GenImage 1k test F1 for fake class")
    plt.xlim(0, 1.05)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.25)
    for idx, (_, row) in enumerate(results.iterrows()):
        plt.text(min(row["f1_ai"] + 0.01, 1.02), idx, f"{row['f1_ai']:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tiny_genimage_5k_test_f1.png", dpi=220)
    plt.close()

    summary = {
        "dataset": DATASET,
        "design": "4000 train-pool rows from train split, internally split into 3600 fit and 400 validation; 1000 held-out rows from validation split used as test.",
        "split_summary": split_summary,
        "results": results.to_dict(orient="records"),
    }
    (OUT_DIR / "tiny_genimage_5k_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFinal Tiny GenImage 5k comparison:")
    print(results.to_string(index=False))
    print("\nWrote results to", OUT_DIR)


if __name__ == "__main__":
    main()
