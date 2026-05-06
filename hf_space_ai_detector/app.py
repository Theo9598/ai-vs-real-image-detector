from pathlib import Path
import json

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
IMAGE_SIZE = 224
AI_CLASS_ID = 1
ID_TO_LABEL = {0: "Real", 1: "AI-generated"}

with (MODEL_DIR / "final_results.json").open("r", encoding="utf-8") as f:
    METADATA = json.load(f)

THRESHOLD = float(METADATA["final_threshold"])
ENSEMBLE_WEIGHTS = METADATA["ensemble_weights"]
MODEL_ORDER = ["resnet18", "efficientnet_b0", "mobilenet_v3_small", "vit_b_16"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_COMPARISON_ROWS = [
    ["Logistic Regression + handcrafted features", "Course ML", "98.7%", "0.974", "1.000"],
    ["Random Forest + handcrafted features", "Course ML", "98.7%", "0.974", "1.000"],
    ["Validation-weighted ensemble", "Final deployed model", "96.7%", "0.933", "0.993"],
    ["ResNet18", "Transfer learning", "96.0%", "0.919", "0.995"],
]

TINY_GENIMAGE_ROWS = [
    ["Random Forest + handcrafted features", "Course ML", "98.2%", "0.982", "1.000"],
    ["Logistic Regression + handcrafted features", "Course ML", "78.9%", "0.819", "0.759"],
    ["ResNet18, 10 epochs", "Transfer learning", "78.5%", "0.797", "0.865"],
]

PROJECT_LINKS = """
- GitHub appendix: https://github.com/Theo9598/ai-vs-real-image-detector
- Final report: see `reports/submission_report_final_ai_detector_tinygenimage.pdf` in GitHub
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
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        return model
    if name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        return model
    raise ValueError(f"Unknown model: {name}")


def load_models() -> dict[str, nn.Module]:
    loaded = {}
    for name in MODEL_ORDER:
        model = make_model(name)
        state = torch.load(MODEL_DIR / f"{name}_best.pt", map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        loaded[name] = model
    return loaded


MODELS = load_models()


def predict_probabilities(image: Image.Image) -> tuple[float, dict[str, float]]:
    image = image.convert("RGB")
    x = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    per_model = {}
    with torch.no_grad():
        for name in MODEL_ORDER:
            output = MODELS[name](x)
            prob_ai = torch.softmax(output, dim=1)[0, AI_CLASS_ID].item()
            per_model[name] = float(prob_ai)

    ensemble_prob = float(sum(ENSEMBLE_WEIGHTS[name] * per_model[name] for name in MODEL_ORDER))
    per_model["validation_weighted_ensemble"] = ensemble_prob
    return ensemble_prob, per_model


def resnet_attention(image: Image.Image) -> Image.Image:
    """Simple Grad-CAM-style attention overlay for the AI class using ResNet18."""
    model = MODELS["resnet18"]
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
    pred_id = int(prob_ai >= THRESHOLD)
    verdict = ID_TO_LABEL[pred_id]
    confidence = prob_ai if pred_id == AI_CLASS_ID else 1.0 - prob_ai
    attention = resnet_attention(image)

    text = f"""
### Prediction: {verdict}

**AI probability:** {prob_ai:.4f}  
**Decision threshold:** {THRESHOLD:.4f}  
**Displayed confidence:** {confidence:.4f}  
**Final predictor:** validation-weighted ensemble

This is a statistical detector, not proof of image origin. The heatmap shows model attention regions.
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
                    model_json = gr.JSON(label="Per-model AI Probability")
                    attention_image = gr.Image(label="Model Attention Regions")

            analyze_button.click(
                fn=analyze,
                inputs=image_input,
                outputs=[result_text, model_json, attention_image],
            )

        with gr.Tab("Results"):
            gr.Markdown(
                """
## Model comparison

The deployed detector uses the validation-weighted ensemble trained on the local project dataset.
For the report, we also compared course-based models against ResNet18 on a Tiny GenImage 5k subset.
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
