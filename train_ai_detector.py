import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
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
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
DATA_ROOT = Path(r"G:\CodexProjects\New project 3\data")
OUT_DIR = Path(r"G:\CodexProjects\New project 3\results")

LABEL_TO_ID = {"real": 0, "ai": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
AI_CLASS_ID = LABEL_TO_ID["ai"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
            category = path.parent.name if path.parent != root else "uncategorized"
            rows.append(
                {
                    "path": str(path),
                    "label": LABEL_TO_ID[label_name],
                    "label_name": label_name,
                    "category": category,
                }
            )
    return rows


class ImagePathDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "path": row["path"],
            "category": row["category"],
        }


def metric_dict(y_true, prob_ai, threshold=0.5):
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


def find_best_threshold(y_true, prob_ai):
    thresholds = np.linspace(0.05, 0.95, 181)
    scored = [(thr, f1_score(y_true, prob_ai >= thr, zero_division=0)) for thr in thresholds]
    best_thr, best_f1 = max(scored, key=lambda x: x[1])
    return float(best_thr), float(best_f1)


def make_model(name: str):
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
    raise ValueError(f"Unknown model: {name}")


def evaluate_torch_model(model, loader, device):
    model.eval()
    y_true, prob_ai, paths, categories = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, AI_CLASS_ID].detach().cpu().numpy()
            y_true.extend(batch["label"].cpu().numpy().tolist())
            prob_ai.extend(probs.tolist())
            paths.extend(list(batch["path"]))
            categories.extend(list(batch["category"]))
    return {"y_true": np.asarray(y_true), "prob_ai": np.asarray(prob_ai), "path": paths, "category": categories}


def train_torch_model(model_name, train_loader, val_loader, class_weights, device):
    model = make_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = None
    best_val_f1 = -1
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{EPOCHS}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        val_out = evaluate_torch_model(model, val_loader, device)
        val_threshold, _ = find_best_threshold(val_out["y_true"], val_out["prob_ai"])
        val_metrics = metric_dict(val_out["y_true"], val_out["prob_ai"], val_threshold)
        val_metrics.update({"model": model_name, "epoch": epoch, "train_loss": running_loss / max(n_seen, 1)})
        history.append(val_metrics)
        print(
            f"{model_name} epoch {epoch}: loss={val_metrics['train_loss']:.4f}, "
            f"val_f1={val_metrics['f1_ai']:.4f}, val_auc={val_metrics['roc_auc']:.4f}, "
            f"thr={val_threshold:.3f}",
            flush=True,
        )
        if val_metrics["f1_ai"] > best_val_f1:
            best_val_f1 = val_metrics["f1_ai"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT_DIR / f"{model_name}_best.pt")
    return model, pd.DataFrame(history)


def extract_features(backbone, loader, device):
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting baseline features"):
            images = batch["image"].to(device, non_blocking=True)
            feats.append(backbone(images).detach().cpu().numpy())
            labels.append(batch["label"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def predict_prob_ai_for_loader(loader, final_model_name, use_ensemble, trained_models, model_order, ensemble_weights, baseline, feature_backbone, device):
    if use_ensemble:
        all_probs = []
        y_true = paths = categories = None
        for model_name in model_order:
            out = evaluate_torch_model(trained_models[model_name], loader, device)
            all_probs.append(out["prob_ai"])
            if y_true is None:
                y_true, paths, categories = out["y_true"], out["path"], out["category"]
        final_prob = np.average(np.vstack(all_probs), axis=0, weights=ensemble_weights)
        return {"y_true": y_true, "prob_ai": final_prob, "path": paths, "category": categories}

    if final_model_name == "Frozen ResNet18 + Logistic Regression":
        x_parts, y_parts, paths, categories = [], [], [], []
        feature_backbone.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting final baseline features"):
                images = batch["image"].to(device, non_blocking=True)
                x_parts.append(feature_backbone(images).detach().cpu().numpy())
                y_parts.append(batch["label"].numpy())
                paths.extend(list(batch["path"]))
                categories.extend(list(batch["category"]))
        x = np.concatenate(x_parts)
        y = np.concatenate(y_parts)
        prob = baseline.predict_proba(x)[:, AI_CLASS_ID]
        return {"y_true": y, "prob_ai": prob, "path": paths, "category": categories}

    return evaluate_torch_model(trained_models[final_model_name], loader, device)


def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    records = collect_records(DATA_ROOT / "Ai_generated_dataset", "ai") + collect_records(DATA_ROOT / "real_dataset", "real")
    df = pd.DataFrame(records).drop_duplicates("path").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No images found under {DATA_ROOT}")
    df.to_csv(OUT_DIR / "dataset_records.csv", index=False)
    summary = df.groupby(["label_name", "category"]).size().reset_index(name="n_images")
    summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    print("Dataset:", df["label_name"].value_counts().to_dict())

    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=SEED, stratify=df["label"])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15 / 0.85, random_state=SEED, stratify=df.loc[train_val_idx, "label"]
    )
    df["split"] = "unused"
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "validation"
    df.loc[test_idx, "split"] = "test"
    df.to_csv(OUT_DIR / "dataset_splits.csv", index=False)
    print("Splits:", df["split"].value_counts().to_dict())

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ImagePathDataset(df[df.split == "train"], train_transform)
    val_ds = ImagePathDataset(df[df.split == "validation"], eval_transform)
    test_ds = ImagePathDataset(df[df.split == "test"], eval_transform)
    # num_workers=0 is more reliable for Windows script execution.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    counts = df[df.split == "train"]["label"].value_counts().reindex([0, 1]).fillna(0).astype(float).values
    class_weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    print("Class weights:", class_weights.detach().cpu().numpy())

    feature_train_ds = ImagePathDataset(df[df.split == "train"], eval_transform)
    feature_val_ds = ImagePathDataset(df[df.split == "validation"], eval_transform)
    feature_train_loader = DataLoader(feature_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    feature_val_loader = DataLoader(feature_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    feature_backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_backbone.fc = nn.Identity()
    feature_backbone = feature_backbone.to(device).eval()
    x_train, y_train = extract_features(feature_backbone, feature_train_loader, device)
    x_val, y_val = extract_features(feature_backbone, feature_val_loader, device)
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    baseline.fit(x_train, y_train)
    baseline_val_prob = baseline.predict_proba(x_val)[:, AI_CLASS_ID]
    baseline_thr, _ = find_best_threshold(y_val, baseline_val_prob)
    baseline_metrics = metric_dict(y_val, baseline_val_prob, baseline_thr)
    baseline_metrics.update({"model": "Frozen ResNet18 + Logistic Regression", "split": "validation"})
    print("Baseline validation:", baseline_metrics)

    model_names = ["resnet18", "efficientnet_b0", "mobilenet_v3_small", "vit_b_16"]
    trained_models = {}
    histories = {}
    for model_name in model_names:
        model, history = train_torch_model(model_name, train_loader, val_loader, class_weights, device)
        trained_models[model_name] = model
        histories[model_name] = history

    history_df = pd.concat(histories.values(), ignore_index=True)
    history_df.to_csv(OUT_DIR / "training_history.csv", index=False)

    validation_outputs = {}
    validation_rows = [baseline_metrics]
    for model_name, model in trained_models.items():
        out = evaluate_torch_model(model, val_loader, device)
        threshold, _ = find_best_threshold(out["y_true"], out["prob_ai"])
        metrics = metric_dict(out["y_true"], out["prob_ai"], threshold)
        metrics.update({"model": model_name, "split": "validation"})
        validation_outputs[model_name] = {**out, "threshold": threshold, "metrics": metrics}
        validation_rows.append(metrics)

    model_order = list(trained_models.keys())
    val_scores = np.array([validation_outputs[m]["metrics"]["f1_ai"] for m in model_order], dtype=float)
    ensemble_weights = (val_scores + 1e-6) / (val_scores.sum() + 1e-6 * len(val_scores))
    ensemble_val_prob = np.zeros_like(validation_outputs[model_order[0]]["prob_ai"], dtype=float)
    y_val_final = validation_outputs[model_order[0]]["y_true"]
    for weight, model_name in zip(ensemble_weights, model_order):
        ensemble_val_prob += weight * validation_outputs[model_name]["prob_ai"]
    ensemble_threshold, _ = find_best_threshold(y_val_final, ensemble_val_prob)
    ensemble_val_metrics = metric_dict(y_val_final, ensemble_val_prob, ensemble_threshold)
    ensemble_val_metrics.update({"model": "validation_weighted_ensemble", "split": "validation"})

    candidate_table = pd.DataFrame(validation_rows + [ensemble_val_metrics]).sort_values("f1_ai", ascending=False).reset_index(drop=True)
    candidate_table.to_csv(OUT_DIR / "validation_model_comparison.csv", index=False)
    final_model_name = candidate_table.loc[0, "model"]
    final_threshold = float(candidate_table.loc[0, "threshold"])
    use_ensemble = final_model_name == "validation_weighted_ensemble"
    print("Final predictor:", final_model_name, "threshold:", final_threshold)
    print("Ensemble weights:", dict(zip(model_order, ensemble_weights.tolist())))

    test_out = predict_prob_ai_for_loader(
        test_loader,
        final_model_name,
        use_ensemble,
        trained_models,
        model_order,
        ensemble_weights,
        baseline,
        feature_backbone,
        device,
    )
    test_metrics = metric_dict(test_out["y_true"], test_out["prob_ai"], final_threshold)
    test_metrics.update({"model": final_model_name, "split": "test"})
    print("Test metrics:", test_metrics)

    y_test = test_out["y_true"]
    prob_test = test_out["prob_ai"]
    pred_test = (prob_test >= final_threshold).astype(int)
    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "ai"], yticklabels=["real", "ai"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Final Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, prob_test)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Test ROC Curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=200)
    plt.close()

    error_df = pd.DataFrame(
        {
            "path": test_out["path"],
            "category": test_out["category"],
            "actual": [ID_TO_LABEL[int(x)] for x in y_test],
            "prob_ai": prob_test,
            "predicted": [ID_TO_LABEL[int(x)] for x in pred_test],
        }
    )
    error_df["correct"] = error_df["actual"] == error_df["predicted"]
    error_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    per_category = (
        error_df.groupby("category")
        .agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_prob_ai=("prob_ai", "mean"))
        .reset_index()
        .sort_values("accuracy")
    )
    per_category.to_csv(OUT_DIR / "per_category_metrics.csv", index=False)

    payload = {
        "seed": SEED,
        "data_root": str(DATA_ROOT),
        "dataset_counts": df["label_name"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "final_model_name": final_model_name,
        "final_threshold": final_threshold,
        "test_metrics": test_metrics,
        "confusion_matrix_labels": ["real", "ai"],
        "confusion_matrix": cm.tolist(),
        "ensemble_weights": dict(zip(model_order, [float(x) for x in ensemble_weights])),
    }
    (OUT_DIR / "final_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote results to", OUT_DIR)


if __name__ == "__main__":
    main()
