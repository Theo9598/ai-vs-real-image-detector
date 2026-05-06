from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle


ROOT = Path(r"G:\CodexProjects\New project 3")
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"
TINY_RESULTS = RESULTS / "tiny_genimage_5k"
OUT = REPORTS / "submission_report_final_ai_detector_tinygenimage.pdf"


def register_fonts():
    font_dir = Path(r"C:\Windows\Fonts")
    try:
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(font_dir / "times.ttf")))
        pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(font_dir / "timesbd.ttf")))
        return "TimesNewRoman", "TimesNewRoman-Bold"
    except Exception:
        return "Times-Roman", "Times-Bold"


BASE_FONT, BOLD_FONT = register_fonts()

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
        spaceBefore=10,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="RefSubheading",
        parent=styles["Heading2"],
        fontName=BOLD_FONT,
        fontSize=11.3,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="RefBody",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=10.05,
        leading=13.9,
        firstLineIndent=0.3 * inch,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="RefNoIndent",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=10.05,
        leading=13.9,
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
        fontSize=7.8,
        leading=9.2,
        alignment=TA_LEFT,
        textColor=colors.black,
    )
)
styles.add(
    ParagraphStyle(
        name="RefSmallBold",
        parent=styles["BodyText"],
        fontName=BOLD_FONT,
        fontSize=7.8,
        leading=9.2,
        alignment=TA_CENTER,
        textColor=colors.black,
    )
)
styles.add(
    ParagraphStyle(
        name="RefCaption",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=8.3,
        leading=10.3,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceBefore=3,
        spaceAfter=5,
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
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
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


REPORTS.mkdir(exist_ok=True)

story.append(Paragraph("AI-Generated vs. Real Image Detection", styles["RefTitle"]))
for meta in [
    "Course: IEOR 142A Spring 2026 Project",
    "Team: Theo Zhang",
    "Interactive demo: https://huggingface.co/spaces/Theo9598/ai-vs-real-image-detector",
    "GitHub appendix: https://github.com/Theo9598/ai-vs-real-image-detector",
    "Dataset: local dataset and Tiny GenImage 5k subset from https://huggingface.co/datasets/TheKernel01/Tiny-GenImage",
    "Presentation recording: [Insert accessible recording link before Gradescope submission]",
]:
    pn(meta)

h("Abstract")
p(
    "We study AI-generated image detection as a supervised binary classification problem. Each input is an image and each label is either real or AI-generated. To connect the project with IEOR 142A, we compare classical machine learning baselines with a representative transfer-learning model. The classical baselines are Logistic Regression and Random Forest trained on handcrafted color, edge, noise, frequency, and blockiness features. The transfer-learning model is ResNet18 fine-tuned from pretrained image weights. On the original 995-image local dataset, Random Forest and Logistic Regression both perform strongly, while a validation-weighted transfer-learning ensemble achieves 96.7% held-out test accuracy. To test whether the conclusion depends on the small local dataset, we run a second experiment on a 5,000-image Tiny GenImage subset: 3,600 fit images, 400 validation images, and 1,000 held-out test images. On this larger subset, Random Forest reaches 98.2% test accuracy and 0.982 F1, while Logistic Regression reaches 78.9% accuracy and ResNet18 fine-tuned for 10 epochs reaches 78.5% accuracy. The main learning is that stronger-looking neural networks do not automatically outperform classical baselines; feature engineering, validation design, and careful interpretation are central to credible machine learning."
)

h("1 Motivation and Learning Problem")
p(
    "AI-generated images are now common in social media, advertising, education, and online news. A visually convincing synthetic image can be harmless in a creative setting, but it can also create trust and misinformation problems when viewers assume the image is a real photograph. This project studies a practical screening use-case: a user uploads an image and receives a probability estimate for whether the image appears AI-generated."
)
p(
    "We formulate the task using the supervised learning framework from class. Each example is a pair (Xi, Yi), where Xi is an RGB image and Yi is a binary label: 0 for real and 1 for AI-generated. A model outputs a score or probability for the AI-generated class. A decision threshold then converts that probability into a final predicted label, creating a tradeoff between false positives and false negatives."
)

h("2 Data")
p(
    "The first dataset is the local project dataset with 995 total images: 250 AI-generated images and 745 real images across animals, city, food, nature, and people. We use a stratified 70/15/15 train/validation/test split, resulting in 695 training images, 150 validation images, and 150 held-out test images."
)
p(
    "The second dataset is Tiny GenImage from Hugging Face, a smaller version of GenImage. For a computationally manageable external experiment, we use 5,000 images. We take 4,000 rows from the Tiny GenImage train split, then internally split them into 3,600 fit images and 400 validation images. We take 1,000 rows from the Tiny GenImage validation split as a held-out test set. The Tiny GenImage experiment is balanced: the fit set contains 1,800 real and 1,800 fake images, the validation set contains 200 real and 200 fake images, and the test set contains 500 real and 500 fake images."
)
story.append(Paragraph("Table 1. Dataset protocols.", styles["RefCaption"]))
story.append(
    table(
        [
            ["Dataset", "Fit / Train", "Validation", "Test", "Purpose"],
            ["Local dataset", "695", "150", "150", "Main project training and deployed detector."],
            ["Tiny GenImage 5k", "3,600", "400", "1,000", "Larger external comparison of course models vs. ResNet18."],
        ],
        [1.15 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch, 3.1 * inch],
        numeric_from=1,
    )
)

h("3 Methodology")
sh("Classical machine learning baselines.")
p(
    "For Logistic Regression and Random Forest, raw images are first converted into handcrafted numerical features. These features include RGB color statistics, grayscale histograms, edge-strength statistics, noise residual statistics, frequency-domain energy ratios from the Fourier spectrum, and blockiness indicators. Logistic Regression learns a linear decision boundary on these features. Random Forest trains an ensemble of decision trees and can capture nonlinear interactions among the same features."
)
sh("Transfer-learning model.")
p(
    "For the neural-network comparison, we use ResNet18 as a representative transfer-learning model. ResNet18 starts from ImageNet-pretrained weights, then replaces the final classification layer with a two-class output layer. On Tiny GenImage, we fine-tune ResNet18 for 10 epochs using minibatches, cross-entropy loss, AdamW, and validation-based threshold selection. This keeps the comparison focused: classical feature engineering versus a standard pretrained CNN."
)
sh("Validation and leakage prevention.")
p(
    "All thresholds are selected using validation data only. The held-out test sets are not used to choose thresholds, model weights, or the best model. This is an important part of the project because a high accuracy number is not meaningful if the test set was used during model selection."
)

h("4 Results")
sh("Original local dataset.")
p(
    "On the local dataset, the classical baselines perform surprisingly well. Logistic Regression and Random Forest both achieve 98.7% test accuracy and 0.974 AI-class F1. The validation-weighted transfer-learning ensemble achieves 96.7% accuracy, 0.933 F1, and 0.993 ROC-AUC. This suggests that the local data contains low-level features that are already highly informative."
)
story.append(Paragraph("Table 2. Local held-out test comparison.", styles["RefCaption"]))
story.append(
    table(
        [
            ["Model", "Type", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            ["Logistic Regression with handcrafted features", "Classical ML", "98.7%", "0.950", "1.000", "0.974", "1.000"],
            ["Random Forest with handcrafted features", "Classical ML", "98.7%", "0.950", "1.000", "0.974", "1.000"],
            ["Validation-weighted ensemble", "Final ensemble", "96.7%", "0.946", "0.921", "0.933", "0.993"],
            ["ResNet18", "Transfer learning", "96.0%", "0.944", "0.895", "0.919", "0.995"],
        ],
        [1.95 * inch, 1.0 * inch, 0.62 * inch, 0.62 * inch, 0.58 * inch, 0.48 * inch, 0.62 * inch],
        numeric_from=2,
    )
)

sh("Tiny GenImage 5k experiment.")
p(
    "The Tiny GenImage experiment tests whether the same lesson appears on a larger and more diverse dataset. Random Forest on handcrafted features performs best, with 98.2% accuracy, 0.982 F1, and almost perfect ROC-AUC. Logistic Regression performs moderately, with 78.9% accuracy and 0.819 F1. ResNet18 improves after 10 epochs compared with the earlier quick run, reaching 78.5% accuracy and 0.797 F1, but it still does not match Random Forest on this subset."
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
    5.6 * inch,
    2.5 * inch,
    "Figure 1. Tiny GenImage 5k F1 comparison. Random Forest performs best on this subset.",
)

h("5 Interpretation and Learning")
p(
    "The most important result is not that Random Forest has the highest number. The important learning is that a classical model can be very strong when the engineered features match the structure of the data. In this project, low-level image statistics such as texture, frequency energy, edge strength, noise residuals, and blockiness appear to separate many real and fake images. Random Forest benefits because it can combine these features through nonlinear decision-tree rules."
)
p(
    "Logistic Regression uses the same features but is less flexible because it learns a linear boundary. Its lower Tiny GenImage performance suggests that the features are useful but not linearly separable in a simple way. ResNet18 is a stronger representation-learning model in principle, but with a 5k subset and 10 epochs it does not automatically outperform a well-matched classical baseline. This supports the course lesson that model selection should be empirical and baseline-driven, not based only on model popularity."
)

h("6 Error Analysis, Ethics, and Deployment")
p(
    "In AI-image detection, false positives and false negatives have different costs. A false positive may wrongly label a real image as synthetic, while a false negative may let a fake image pass as real. The deployed Hugging Face Space therefore presents the output as statistical decision support, not proof of image origin. The Grad-CAM-style visualization is also described as model attention rather than evidence of actual AI artifacts."
)
p(
    "The Hugging Face demo is available at https://huggingface.co/spaces/Theo9598/ai-vs-real-image-detector. The GitHub appendix is available at https://github.com/Theo9598/ai-vs-real-image-detector. A real deployment would require larger multi-source validation, uncertainty thresholds, human review, and careful policy design."
)

h("7 Conclusion")
p(
    "This project applies the supervised learning workflow from IEOR 142A to a current image-classification problem. The final message is that careful evaluation matters more than simply adding advanced models. The local experiment and the Tiny GenImage 5k experiment both show that classical baselines are essential. Random Forest with handcrafted image features is highly competitive, while ResNet18 transfer learning provides a useful but not automatically superior comparison. The project demonstrates data preparation, feature engineering, model comparison, validation-based threshold selection, classifier metrics, deployment, and honest limitation analysis."
)

h("Appendix")
p(
    "Code and artifacts are available in the GitHub repository. Key files include train_ai_detector.py, compare_course_vs_transfer.py, tiny_genimage_5k_compare.py, hf_space_ai_detector/app.py, and the results folder. Tiny GenImage outputs are stored under results/tiny_genimage_5k, including tiny_genimage_5k_model_comparison.csv, tiny_genimage_5k_resnet18_history.csv, tiny_genimage_5k_test_f1.png, and tiny_genimage_5k_summary.json."
)


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
