"""
agents/report_agent.py
FedASIO-YOLO26: ReportAgent
Generates clinical PDF reports using ReportLab + Qwen2.5-7B-Instruct LLM.
Falls back to rule-based text when HF API is unavailable.
"""
import os
import logging
import datetime
import requests
from typing import Dict, Optional
from agents.state import FedASIOState

logger = logging.getLogger(__name__)


def _qwen_generate(patient_id: str, metrics: Dict, volume_cc: float, hf_token: Optional[str]) -> str:
    """Call Qwen2.5-7B-Instruct via HuggingFace Inference API."""
    if not hf_token:
        return None

    dice = metrics.get("dice", 0.0)
    hd95 = metrics.get("hd95", 200.0)
    iou = metrics.get("iou", 0.0)

    prompt = f"""You are an expert neuroradiologist specializing in pediatric brain tumors.
Generate a structured clinical report for a pediatric patient based on AI segmentation results.

Patient ID: {patient_id}
AI Model: YOLO26-ASIO Federated Segmentation
Tumor Volume: {volume_cc:.2f} cc
Dice Score: {dice:.4f}
IoU Score: {iou:.4f}
Hausdorff Distance 95%: {hd95:.1f} mm

Write a 3-paragraph clinical report including:
1. Imaging findings and tumor characterization
2. AI model confidence and segmentation quality assessment
3. Clinical recommendations

Be concise, professional, and clinically accurate."""

    try:
        url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "return_full_text": False,
            },
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
    except Exception as e:
        logger.warning(f"[ReportAgent] Qwen API error: {e}")
    return None


def _rule_based_report(patient_id: str, metrics: Dict, volume_cc: float) -> str:
    """Generate structured clinical report using rule-based logic."""
    dice = metrics.get("dice", 0.0)
    hd95 = metrics.get("hd95", 200.0)
    iou = metrics.get("iou", 0.0)
    per_class = metrics.get("per_class_metrics", {})

    # Reliability classification
    if dice >= 0.90:
        reliability = "excellent (Dice ≥ 0.90)"
        confidence = "high confidence"
    elif dice >= 0.75:
        reliability = "good (0.75 ≤ Dice < 0.90)"
        confidence = "moderate confidence"
    else:
        reliability = "limited (Dice < 0.75)"
        confidence = "low confidence — recommend manual review"

    snfh_dice = per_class.get("SNFH", {}).get("dice", 0.0)
    tc_dice = per_class.get("tumor_core", {}).get("dice", 0.0)
    et_dice = per_class.get("enhancing_tumor", {}).get("dice", 0.0)

    report = f"""
PEDIATRIC BRAIN TUMOR SEGMENTATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient ID: {patient_id}
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
AI Model: FedASIO-YOLO26 (Federated Learning + ASIO Optimizer)

IMAGING FINDINGS:
AI-assisted segmentation of pediatric brain MRI (BraTS-PEDs protocol) identified
tumor regions with an estimated total volume of {volume_cc:.2f} cc. The analysis
encompassed four MRI sequences (T1-contrast, T1-native, T2-FLAIR, T2-weighted).
Sub-regional analysis: SNFH (Dice: {snfh_dice:.3f}), Tumor Core (Dice: {tc_dice:.3f}),
Enhancing Tumor (Dice: {et_dice:.3f}).

SEGMENTATION QUALITY ASSESSMENT:
The YOLO26-ASIO federated model achieved {reliability} segmentation performance.
Overall Dice Score: {dice:.4f} | IoU: {iou:.4f} | HD95: {hd95:.1f} mm.
Assessment: {confidence}. The federated learning framework ensures patient data
privacy while maintaining competitive segmentation accuracy across participating institutions.

CLINICAL RECOMMENDATION:
{"Results are suitable for clinical decision support. Radiologist verification recommended before treatment planning." if dice >= 0.75 else "Segmentation quality is below the 0.75 Dice threshold. MANDATORY radiologist review required. Do not use AI output directly for clinical decisions."}
Tumor volume of {volume_cc:.2f} cc should be correlated with clinical symptoms,
patient age, and prior imaging studies for comprehensive assessment.

⚠ DISCLAIMER: This report is AI-generated and must be reviewed by a qualified
  pediatric neuroradiologist before any clinical use.
""".strip()

    return report


def _generate_pdf(patient_id: str, report_text: str, metrics: Dict,
                  volume_cc: float, xai_paths: list, output_dir: str) -> str:
    """Generate a clinical PDF report using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"FedASIO_Report_{patient_id}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=16, textColor=colors.HexColor("#1a1a2e"),
                                  spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"],
                                    fontSize=12, textColor=colors.HexColor("#16213e"),
                                    spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=9, leading=14, spaceAfter=6)
    warn_style = ParagraphStyle("Warn", parent=styles["Normal"],
                                 fontSize=8, textColor=colors.red, leading=12)

    story = []

    # Header
    story.append(Paragraph("FedASIO-YOLO26 Clinical Intelligence", title_style))
    story.append(Paragraph(f"Pediatric Brain Tumor Segmentation Report", heading_style))
    story.append(Spacer(1, 0.3*cm))

    # Patient info table
    info_data = [
        ["Patient ID", patient_id, "Report Date", datetime.datetime.now().strftime("%Y-%m-%d")],
        ["AI Model", "YOLO26-ASIO", "FL Framework", "Flower + FedASIO"],
        ["Dataset", "BraTS-PEDs-v1", "Privacy", "DP (ε=1.0, δ=1e-5)"],
    ]
    info_table = Table(info_data, colWidths=[3*cm, 5*cm, 3*cm, 5*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1a1a2e")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#e8f0fe"), colors.white]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # Metrics table
    story.append(Paragraph("Segmentation Metrics", heading_style))
    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Dice Score", f"{metrics.get('dice', 0):.4f}", "IoU", f"{metrics.get('iou', 0):.4f}"],
        ["Precision", f"{metrics.get('precision', 0):.4f}", "Recall", f"{metrics.get('recall', 0):.4f}"],
        ["F1-Score", f"{metrics.get('f1', 0):.4f}", "HD95 (mm)", f"{metrics.get('hd95', 0):.1f}"],
        ["Tumor Volume", f"{volume_cc:.2f} cc", "mAP@50", f"{metrics.get('mAP50', 0):.4f}"],
    ]
    m_table = Table(metrics_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    m_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 0.4*cm))

    # Report text
    story.append(Paragraph("Clinical Report", heading_style))
    for line in report_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), body_style))

    # XAI figure
    if xai_paths:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Explainability Visualization", heading_style))
        story.append(Paragraph(
            "Left to right: Original MRI | Ground Truth | YOLO26 Prediction | GradCAM Confidence",
            ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
        ))
        try:
            img = Image(xai_paths[0], width=16*cm, height=4*cm)
            story.append(img)
        except Exception:
            pass

    # Disclaimer
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by an AI system and must be reviewed by a "
        "qualified pediatric neuroradiologist before any clinical use. FedASIO-YOLO26 is a "
        "research prototype and is NOT approved for clinical deployment.",
        warn_style
    ))

    doc.build(story)
    return pdf_path


def report_agent(state: FedASIOState) -> FedASIOState:
    """
    ReportAgent: Generates clinical PDF report with Qwen LLM text.
    """
    if state.get("error"):
        return state

    patient_id = state.get("patient_id", "unknown")
    metrics = state.get("metrics", {})
    volume_cc = state.get("tumor_volume_cc", 0.0)
    xai_paths = state.get("xai_figure_paths", [])

    logger.info(f"[ReportAgent] Generating report for {patient_id}")

    try:
        # Try Qwen API first
        hf_token = os.environ.get("HF_TOKEN")
        report_text = _qwen_generate(patient_id, metrics, volume_cc, hf_token)

        if not report_text:
            logger.info("  Qwen API unavailable, using rule-based report")
            report_text = _rule_based_report(patient_id, metrics, volume_cc)

        # Generate PDF
        output_dir = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/pdf_reports"
        pdf_path = _generate_pdf(patient_id, report_text, metrics, volume_cc, xai_paths, output_dir)
        logger.info(f"  PDF saved: {pdf_path}")

        return {
            **state,
            "report_text": report_text,
            "pdf_path": pdf_path,
            "report_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": "YOLO26-ASIO",
                "fl_round": state.get("fl_round", 0),
            },
            "error": None,
        }

    except Exception as e:
        logger.error(f"[ReportAgent] Error: {e}", exc_info=True)
        return {**state, "error": str(e), "error_agent": "ReportAgent"}
