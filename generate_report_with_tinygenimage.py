from pathlib import Path
import html

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(r"G:\CodexProjects\New project 3")
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"
TINY_RESULTS = RESULTS / "tiny_genimage_5k"
OUT = REPORTS / "142A_final_project_ai_detector_code_appendix.pdf"


def register_fonts():
    font_dir = Path(r"C:\Windows\Fonts")
    try:
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(font_dir / "times.ttf")))
        pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(font_dir / "timesbd.ttf")))
        return "TimesNewRoman", "TimesNewRoman-Bold", "Courier"
    except Exception:
        return "Times-Roman", "Times-Bold", "Courier"


BASE_FONT, BOLD_FONT, CODE_FONT = register_fonts()

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="RefTitle",
        parent=styles["Title"],
        fontName=BOLD_FONT,
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="RefHeading",
        parent=styles["Heading1"],
        fontName=BOLD_FONT,
        fontSize=13,
        leading=16,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=9,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="RefSubheading",
        parent=styles["Heading2"],
        fontName=BOLD_FONT,
        fontSize=11.5,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=5,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="RefBody",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=10.0,
        leading=13.4,
        firstLineIndent=0.28 * inch,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceAfter=3.3,
    )
)
styles.add(
    ParagraphStyle(
        name="RefNoIndent",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=10.0,
        leading=13.4,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="RefSmall",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=7.6,
        leading=9.0,
        alignment=TA_LEFT,
        textColor=colors.black,
    )
)
styles.add(
    ParagraphStyle(
        name="RefSmallBold",
        parent=styles["BodyText"],
        fontName=BOLD_FONT,
        fontSize=7.6,
        leading=9.0,
        alignment=TA_CENTER,
        textColor=colors.black,
    )
)
styles.add(
    ParagraphStyle(
        name="RefCaption",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=8.1,
        leading=10.0,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceBefore=2,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="RefCode",
        parent=styles["Code"],
        fontName=CODE_FONT,
        fontSize=5.2,
        leading=6.1,
        leftIndent=0,
        rightIndent=0,
        textColor=colors.black,
        splitLongWords=True,
    )
)


story = []


def h(text):
    story.append(Paragraph(text, styles["RefHeading"]))


def sh(text):
    story.append(Paragraph(text, styles["RefSubheading"]))


def p(text):
    story.append(Paragraph(text, styles["RefBody"]))


def pn(text):
    story.append(Paragraph(text, styles["RefNoIndent"]))


def table(data, widths, numeric_from=None):
    wrapped = []
    for row_idx, row in enumerate(data):
        wrapped.append(
            [
                Paragraph(str(cell), styles["RefSmallBold"] if row_idx == 0 else styles["RefSmall"])
                for cell in row
            ]
        )
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    ts = [
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for row_idx in range(1, len(data)):
        ts.append(("LINEBELOW", (0, row_idx), (-1, row_idx), 0.25, colors.HexColor("#BFBFBF")))
    if numeric_from is not None:
        ts.append(("ALIGN", (numeric_from, 1), (-1, -1), "CENTER"))
    t.setStyle(TableStyle(ts))
    return t


def add_figure(path, width, height, caption):
    if path.exists():
        story.append(KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, styles["RefCaption"])]))


def add_code_file(relative_path):
    path = ROOT / relative_path
    if not path.exists():
        return
    story.append(PageBreak())
    h(f"Appendix Code: {relative_path}")
    code = path.read_text(encoding="utf-8", errors="replace")
    # Escape XML-sensitive characters because ReportLab parses Preformatted text.
    story.append(Preformatted(html.escape(code, quote=False), styles["RefCode"], maxLineLength=118))


REPORTS.mkdir(exist_ok=True)

story.append(Paragraph("AI-Generated vs. Real Image Detection", styles["RefTitle"]))
for meta in [
    "Course: IEOR 142A Spring 2026 Project",
    "Team: Theo Zhang",
    "Interactive demo: https://huggingface.co/spaces/Theo9598/ai-vs-real-image-detector",
    "GitHub appendix: https://github.com/Theo9598/ai-vs-real-image-detector",
    "Primary dataset: Tiny GenImage 5k subset from https://huggingface.co/datasets/TheKernel01/Tiny-GenImage",
    "Local comparison dataset: course project Google Drive dataset",
]:
    pn(meta)

h("Abstract")
p(
    "This project studies AI-generated image detection as a supervised binary classification problem. The deployed model is a Random Forest trained on handcrafted image features from a 5,000-image Tiny GenImage subset. I compare it with Logistic Regression and a 10-epoch ResNet18 transfer-learning model, using validation-only threshold selection and a held-out 1,000-image test set. Random Forest achieves 98.2% test accuracy and 0.982 fake-class F1, outperforming Logistic Regression and ResNet18 on this subset. The main learning is that simple course models can be highly competitive when the feature representation fits the data, but the detector still has important generalization limits."
)

h("1 Problem and Data")
p(
    "The task is binary classification: label 0 means real and label 1 means AI-generated. The final demo uses Tiny GenImage because it is larger and more relevant to the final experiment than the small local folder dataset. I use 3,600 Tiny GenImage examples for fitting, 400 for validation, and 1,000 held-out examples for final testing. The Google Drive dataset with 995 images is kept as a smaller local comparison experiment, not as the active deployed model."
)
story.append(Paragraph("Table 1. Dataset protocols.", styles["RefCaption"]))
story.append(
    table(
        [
            ["Dataset", "Fit / Train", "Validation", "Test", "Role in project"],
            ["Tiny GenImage 5k", "3,600", "400", "1,000", "Primary experiment and deployed Random Forest model."],
            ["Local Google Drive dataset", "695", "150", "150", "Smaller comparison experiment and sanity check."],
        ],
        [1.25 * inch, 0.75 * inch, 0.75 * inch, 0.65 * inch, 3.0 * inch],
        numeric_from=1,
    )
)

h("2 Models")
sh("Random Forest with handcrafted features.")
p(
    "Random Forest is the final deployed model. Each uploaded image is first converted into 63 handcrafted numerical features. The forest contains many decision trees; each tree votes using different feature splits, and the final probability is the average vote. This is useful here because the relationship between image artifacts and the fake label is nonlinear."
)
story.append(Paragraph("Table 2. Random Forest feature groups.", styles["RefCaption"]))
story.append(
    table(
        [
            ["Feature group", "What it measures", "Why it may help"],
            ["Aspect ratio and image size", "Width/height ratio and log image area.", "Some generated or collected images have repeated size/crop patterns."],
            ["RGB color means and standard deviations", "Average and spread of red, green, and blue channels.", "Synthetic images may have different color balance or smoother color distribution."],
            ["RGB histograms", "Eight-bin histogram for each color channel.", "Captures whether colors are concentrated, flat, or unusually distributed."],
            ["Grayscale histogram", "Sixteen-bin brightness distribution.", "Summarizes contrast and lighting patterns."],
            ["Edge-strength statistics", "Mean, standard deviation, 10th percentile, and 90th percentile of gradient magnitude.", "Generated images can differ in sharpness and fine edge consistency."],
            ["Noise residual statistics", "Difference between grayscale image and a blurred version.", "Real camera photos often contain sensor/compression noise that synthetic images may smooth out."],
            ["Frequency energy ratios", "Low, mid, high Fourier energy and high/low ratio.", "AI images can show different texture frequency patterns."],
            ["Blockiness indicators", "Average edge changes along 8-pixel grid boundaries.", "Detects JPEG-like or generator/compression artifacts."],
        ],
        [1.25 * inch, 2.05 * inch, 3.05 * inch],
    )
)
sh("Logistic Regression and ResNet18.")
p(
    "Logistic Regression uses the same handcrafted features but learns one linear decision boundary, so it is easier to interpret but less flexible. ResNet18 is a transfer-learning baseline: it starts from ImageNet-pretrained weights, replaces the final layer with a two-class classifier, and is fine-tuned for 10 epochs. In the demo, ResNet18 is not the final predictor; it is kept for the Grad-CAM-style attention image."
)

h("3 Validation and Results")
p(
    "All thresholds are selected using validation data only. The held-out test set is evaluated once after model and threshold choices are fixed. This prevents test-set leakage. On Tiny GenImage, the Random Forest threshold selected on validation is 0.545."
)
story.append(Paragraph("Table 3. Tiny GenImage 5k held-out test comparison.", styles["RefCaption"]))
story.append(
    table(
        [
            ["Model", "Type", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            ["Random Forest with handcrafted features", "Classical ML", "98.2%", "0.967", "0.998", "0.982", "1.000"],
            ["Logistic Regression with handcrafted features", "Classical ML", "78.9%", "0.717", "0.956", "0.819", "0.759"],
            ["ResNet18 transfer learning, 10 epochs", "Transfer learning", "78.5%", "0.755", "0.844", "0.797", "0.865"],
        ],
        [2.05 * inch, 1.0 * inch, 0.62 * inch, 0.62 * inch, 0.58 * inch, 0.48 * inch, 0.62 * inch],
        numeric_from=2,
    )
)
add_figure(
    TINY_RESULTS / "tiny_genimage_5k_test_f1.png",
    5.5 * inch,
    2.45 * inch,
    "Figure 1. Tiny GenImage 5k F1 comparison. Random Forest performs best on this subset.",
)
p(
    "The smaller local dataset gives a similar lesson: classical handcrafted-feature models are strong, while a validation-weighted transfer-learning ensemble reaches 96.7% accuracy and 0.933 AI-class F1. I treat this as supporting evidence, not as the main deployed result."
)

h("4 Interpretation and Limitations")
p(
    "The Random Forest result does not mean the detector has solved AI-image detection. It means that this dataset contains low-level signals that the handcrafted features capture well. The model may rely on dataset-specific artifacts such as compression style, resolution patterns, generator texture, or repeated preprocessing. These signals can be useful for a course project, but they may not transfer to every real-world image source."
)
p(
    "A key limitation is generalization to newer AI generators and image-to-image systems. If a new model produces images with more realistic camera noise, different compression, or fewer frequency artifacts, the Random Forest may perform worse. It may also struggle with edited real photos, screenshots, low-resolution uploads, or image-to-image outputs where a real photo is partially modified. In a quick qualitative check, image-to-image style examples such as the user's 'image2' case were less reliable, which is consistent with the limitation that the model learned Tiny GenImage patterns rather than universal proof of image origin."
)
p(
    "For this reason, the Hugging Face Space presents the output as statistical decision support, not proof. False positives may unfairly label real images as fake, while false negatives may miss synthetic content. A real deployment would need larger multi-generator training data, uncertainty thresholds, periodic retraining, and human review."
)

h("5 Deployment and Conclusion")
p(
    "The deployed Hugging Face Space uses the Tiny GenImage Random Forest model for final prediction and the ResNet18 model only for the attention visualization. This design matches the report result while still giving an interpretable visual bonus. Overall, the project demonstrates supervised learning, stratified splitting, feature engineering, validation-based threshold selection, model comparison, error analysis, and deployment. Readers who want the complete project files, result artifacts, and model files can visit the GitHub repository linked on the first page."
)

story.append(PageBreak())
for code_path in [
    Path("tiny_genimage_5k_compare.py"),
    Path("hf_space_ai_detector") / "app.py",
    Path("compare_course_vs_transfer.py"),
]:
    add_code_file(code_path)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BASE_FONT, 9)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(letter[0] / 2, 0.42 * inch, str(doc.page))
    canvas.setFont(BASE_FONT, 8.5)
    canvas.drawCentredString(letter[0] / 2, 0.22 * inch, "AI-Generated vs. Real Image Detection")
    canvas.restoreState()


pdf = SimpleDocTemplate(
    str(OUT),
    pagesize=letter,
    leftMargin=0.85 * inch,
    rightMargin=0.85 * inch,
    topMargin=0.78 * inch,
    bottomMargin=0.7 * inch,
)
pdf.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
