from pathlib import Path
import json

import gradio as gr
import joblib
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
IMAGE_SIZE = 224
AI_CLASS_ID = 1
ID_TO_LABEL = {0: "Real", 1: "AI-generated"}
RF_MODEL_LABEL = "Random Forest with handcrafted features"
RESNET_MODEL_NAME = "tiny_genimage_resnet18"
RESNET_MODEL_LABEL = "Tiny GenImage ResNet18, 10 epochs"

with (MODEL_DIR / "tiny_genimage_random_forest_metadata.json").open("r", encoding="utf-8") as f:
    RF_METADATA = json.load(f)

RF_THRESHOLD = float(RF_METADATA["threshold"])
RF_METRICS = RF_METADATA["metrics"]
RESNET_METRICS = {
    "test_accuracy": 0.785,
    "test_precision_fake": 0.7549,
    "test_recall_fake": 0.844,
    "test_f1_fake": 0.797,
    "test_roc_auc": 0.865,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_COMPARISON_ROWS = [
    ["Logistic Regression + handcrafted features", "Classical ML", "98.7%", "0.974", "1.000"],
    ["Random Forest + handcrafted features", "Classical ML", "98.7%", "0.974", "1.000"],
    ["Validation-weighted ensemble", "Local comparison", "96.7%", "0.933", "0.993"],
    ["ResNet18", "Transfer learning", "96.0%", "0.919", "0.995"],
]

TINY_GENIMAGE_ROWS = [
    ["Random Forest + handcrafted features", "Classical ML", "98.2%", "0.982", "1.000"],
    ["Logistic Regression + handcrafted features", "Classical ML", "78.9%", "0.819", "0.759"],
    ["ResNet18, 10 epochs", "Transfer learning", "78.5%", "0.797", "0.865"],
]

PROJECT_LINKS = """
- GitHub appendix: https://github.com/Theo9598/ai-vs-real-image-detector
- Final report: see `reports/submission_report_final_ai_detector_tinygenimage_space_aligned.pdf` in GitHub
- Tiny GenImage dataset: https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
"""

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def make_model(name: str) -> nn.Module:
    if name == RESNET_MODEL_NAME:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    raise ValueError(f"Unknown model: {name}")


def load_models() -> dict[str, nn.Module]:
    resnet = make_model(RESNET_MODEL_NAME)
    state = torch.load(MODEL_DIR / f"{RESNET_MODEL_NAME}_best.pt", map_location=DEVICE)
    resnet.load_state_dict(state)
    resnet.to(DEVICE)
    resnet.eval()
    return {
        "random_forest": joblib.load(MODEL_DIR / "tiny_genimage_random_forest.joblib"),
        RESNET_MODEL_NAME: resnet,
    }


MODELS = load_models()


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


def handcrafted_features(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
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


def resnet_probability(image: Image.Image) -> float:
    image = image.convert("RGB")
    x = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = MODELS[RESNET_MODEL_NAME](x)
        prob_ai = torch.softmax(output, dim=1)[0, AI_CLASS_ID].item()
    return float(prob_ai)


def predict_probabilities(image: Image.Image) -> tuple[float, dict[str, float]]:
    features = handcrafted_features(image).reshape(1, -1)
    rf_prob = float(MODELS["random_forest"].predict_proba(features)[0, AI_CLASS_ID])
    resnet_prob = resnet_probability(image)
    return rf_prob, {
        RF_MODEL_LABEL: rf_prob,
        f"{RESNET_MODEL_LABEL} attention model": resnet_prob,
    }


def resnet_attention(image: Image.Image) -> Image.Image:
    """Simple Grad-CAM-style attention overlay for the AI class using ResNet18."""
    model = MODELS[RESNET_MODEL_NAME]
    layer = model.layer4[-1]
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_full_backward_hook(backward_hook)

    try:
        rgb = image.convert("RGB")
        x = EVAL_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
        model.zero_grad(set_to_none=True)
        output = model(x)
        score = output[:, AI_CLASS_ID].sum()
        score.backward()

        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()

        cam_img = Image.fromarray(np.uint8(cam * 255)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        heat = np.asarray(cam_img).astype(np.float32) / 255.0
        base = np.asarray(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

        overlay = base.copy()
        overlay[..., 0] = np.clip(0.55 * base[..., 0] + 0.45 * heat, 0, 1)
        overlay[..., 1] = np.clip(0.65 * base[..., 1], 0, 1)
        overlay[..., 2] = np.clip(0.65 * base[..., 2], 0, 1)
        return Image.fromarray(np.uint8(overlay * 255))
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def analyze(image: Image.Image):
    if image is None:
        return "Upload an image to run the detector.", {}, None

    prob_ai, per_model = predict_probabilities(image)
    pred_id = int(prob_ai >= RF_THRESHOLD)
    verdict = ID_TO_LABEL[pred_id]
    confidence = prob_ai if pred_id == AI_CLASS_ID else 1.0 - prob_ai
    attention = resnet_attention(image)

    text = f"""
### Prediction: {verdict}

**AI probability:** {prob_ai:.4f}  
**Decision threshold:** {RF_THRESHOLD:.4f}  
**Displayed confidence:** {confidence:.4f}  
**Final predictor:** {RF_MODEL_LABEL}

This is the highest-scoring Tiny GenImage 5k model from the report. Its held-out test results were accuracy {RF_METRICS["accuracy"]:.3f}, F1 {RF_METRICS["f1_ai"]:.3f}, and ROC-AUC {RF_METRICS["roc_auc"]:.3f}.

This is a statistical detector, not proof of image origin. The heatmap is generated by the ResNet18 comparison model as an attention visualization, not by the Random Forest itself.
"""
    rounded = {name: round(value, 4) for name, value in per_model.items()}
    return text, rounded, attention


CSS = """
.main-title {max-width: 980px; margin: 0 auto 10px auto;}
.note-box {
    border-left: 4px solid #4b5563;
    background: #f8fafc;
    padding: 12px 14px;
    border-radius: 4px;
}
"""


with gr.Blocks(title="AI vs Real Image Detector", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        """
<div class="main-title">

# AI vs Real Image Detector

IEOR 142A course project demo. Upload an image to estimate whether it is AI-generated.
The final detector uses the Tiny GenImage 5k Random Forest report model.
The output is statistical decision support, not proof of image origin.

</div>
"""
    )

    with gr.Tabs():
        with gr.Tab("Detector"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Image")
                    analyze_button = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=1):
                    result_text = gr.Markdown(label="Final Result")
                    model_json = gr.JSON(label="Model AI Probabilities")
                    attention_image = gr.Image(label="ResNet18 Attention Regions")

            analyze_button.click(
                fn=analyze,
                inputs=image_input,
                outputs=[result_text, model_json, attention_image],
            )

        with gr.Tab("Results"):
            gr.Markdown(
                """
## Model comparison

The deployed detector uses the Tiny GenImage 5k Random Forest model from the report because it has the best held-out test F1.
The ResNet18 model is still used for the attention visualization in the Detector tab.
"""
            )
            gr.Markdown("### Local project dataset, held-out test")
            gr.Dataframe(
                value=LOCAL_COMPARISON_ROWS,
                headers=["Model", "Type", "Accuracy", "F1, AI/fake", "ROC-AUC"],
                interactive=False,
                wrap=True,
            )
            gr.Markdown("### Tiny GenImage 5k subset, held-out test")
            gr.Dataframe(
                value=TINY_GENIMAGE_ROWS,
                headers=["Model", "Type", "Accuracy", "F1, fake", "ROC-AUC"],
                interactive=False,
                wrap=True,
            )

        with gr.Tab("Interpretation"):
            gr.Markdown(
                """
## What the project learned

The main lesson is not simply that one model had the highest accuracy. Classical machine learning
baselines were very strong when images were represented with handcrafted features such as color,
edge, noise, frequency, and blockiness statistics.

On the Tiny GenImage 5k subset, Random Forest outperformed a 10-epoch ResNet18 fine-tuning run.
This suggests that low-level image artifacts are highly informative in this dataset, and that model
complexity alone does not guarantee better performance.

<div class="note-box">
The model-attention image shows regions that influenced the ResNet18 AI-class score. It should not
be interpreted as proof that a region is fake or AI-generated.
</div>
"""
            )

        with gr.Tab("Links"):
            gr.Markdown(PROJECT_LINKS)


if __name__ == "__main__":
    demo.launch()
