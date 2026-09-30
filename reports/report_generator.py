import sys
from pathlib import Path
from datetime import datetime

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# REPORTLAB
# =========================================================

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib import colors
from reportlab.lib.units import inch


# =========================================================
# REPORT DIRECTORY
# =========================================================

REPORT_FOLDER = PROJECT_ROOT / "reports"

REPORT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SAFE TEXT FUNCTION
# =========================================================

def safe_text(value):

    if value is None:
        return ""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


# =========================================================
# GENERATE PDF
# =========================================================

def generate_pdf_report(
    text,
    detection_result,
    genai_explanation
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    pdf_path = REPORT_FOLDER / (
        f"voight_kampff_report_{timestamp}.pdf"
    )


    # -----------------------------------------------------
    # DOCUMENT
    # -----------------------------------------------------

    document = SimpleDocTemplate(

        str(pdf_path),

        pagesize=A4,

        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    # -----------------------------------------------------
    # STYLES
    # -----------------------------------------------------

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "VKTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=22,

        leading=26,

        spaceAfter=10
    )


    subtitle_style = ParagraphStyle(

        "VKSubtitle",

        parent=styles["Heading2"],

        alignment=TA_CENTER,

        fontSize=13,

        spaceAfter=20
    )


    heading_style = ParagraphStyle(

        "VKHeading",

        parent=styles["Heading2"],

        fontSize=14,

        leading=18,

        spaceBefore=12,

        spaceAfter=8
    )


    normal_style = ParagraphStyle(

        "VKNormal",

        parent=styles["BodyText"],

        fontSize=10,

        leading=15,

        spaceAfter=8
    )


    small_style = ParagraphStyle(

        "VKSmall",

        parent=styles["BodyText"],

        fontSize=8,

        leading=12
    )


    # -----------------------------------------------------
    # CONTENT
    # -----------------------------------------------------

    content = []


    # =====================================================
    # TITLE
    # =====================================================

    content.append(

        Paragraph(
            "VOIGHT-KAMPFF",
            title_style
        )

    )


    content.append(

        Paragraph(
            "Generative AI Detection Report",
            subtitle_style
        )

    )


    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    content.append(

        Paragraph(

            f"<b>Analysis Time:</b> {current_time}",

            normal_style

        )

    )


    # =====================================================
    # DETECTION RESULT
    # =====================================================

    content.append(

        Paragraph(
            "1. Detection Result",
            heading_style
        )

    )


    prediction = detection_result.get(
        "prediction",
        "Unknown"
    )


    ai_probability = detection_result.get(
        "ai_probability",
        0
    )


    human_probability = detection_result.get(
        "human_probability",
        0
    )


    perplexity = detection_result.get(
        "perplexity",
        0
    )


    burstiness = detection_result.get(
        "burstiness",
        {}
    )


    if isinstance(burstiness, dict):

        burstiness_score = burstiness.get(
            "burstiness_score",
            0
        )

        burstiness_level = burstiness.get(
            "burstiness_level",
            "Unknown"
        )

    else:

        burstiness_score = burstiness

        burstiness_level = "Unknown"


    # -----------------------------------------------------
    # RESULT TABLE
    # -----------------------------------------------------

    result_data = [

        ["Metric", "Value"],

        ["Prediction", str(prediction)],

        [
            "AI Probability",
            f"{ai_probability}%"
        ],

        [
            "Human Probability",
            f"{human_probability}%"
        ],

        [
            "Perplexity",
            str(perplexity)
        ],

        [
            "Burstiness Score",
            str(burstiness_score)
        ],

        [
            "Burstiness Level",
            str(burstiness_level)
        ]

    ]


    result_table = Table(

        result_data,

        colWidths=[
            2.5 * inch,
            3.5 * inch
        ]

    )


    result_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "FONTNAME",
                (0, 1),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])

    )


    content.append(result_table)


    # =====================================================
    # INPUT TEXT
    # =====================================================

    content.append(

        Paragraph(
            "2. Input Text",
            heading_style
        )

    )


    content.append(

        Paragraph(
            safe_text(text),
            normal_style
        )

    )


    # =====================================================
    # LINGUISTIC FEATURES
    # =====================================================

    content.append(

        Paragraph(
            "3. Linguistic Analysis",
            heading_style
        )

    )


    linguistic_features = detection_result.get(
        "linguistic_features",
        {}
    )


    linguistic_data = [

        ["Feature", "Value"]

    ]


    if isinstance(
        linguistic_features,
        dict
    ):

        for key, value in linguistic_features.items():

            linguistic_data.append([

                key.replace(
                    "_",
                    " "
                ).title(),

                str(value)

            ])


    linguistic_table = Table(

        linguistic_data,

        colWidths=[
            3.5 * inch,
            2.5 * inch
        ]

    )


    linguistic_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )

        ])

    )


    content.append(linguistic_table)


    # =====================================================
    # BURSTINESS
    # =====================================================

    content.append(

        Paragraph(
            "4. Burstiness Analysis",
            heading_style
        )

    )


    burstiness_text = f"""
    <b>Burstiness Score:</b> {burstiness_score}<br/>
    <b>Burstiness Level:</b> {burstiness_level}<br/>
    <b>Mean Sentence Length:</b>
    {burstiness.get("mean_sentence_length", 0)}<br/>
    <b>Sentence Length Standard Deviation:</b>
    {burstiness.get("sentence_length_std", 0)}
    """


    content.append(

        Paragraph(
            burstiness_text,
            normal_style
        )

    )


    # =====================================================
    # PERPLEXITY
    # =====================================================

    content.append(

        Paragraph(
            "5. Perplexity Analysis",
            heading_style
        )

    )


    perplexity_text = f"""
    <b>Perplexity Score:</b> {perplexity}<br/><br/>

    Perplexity measures how predictable a sequence of
    words is to a language model. It is used as a
    supporting signal in the Voight-Kampff system and
    should not be treated as independent proof of
    AI-generated content.
    """


    content.append(

        Paragraph(
            perplexity_text,
            normal_style
        )

    )


    # =====================================================
    # GENAI EXPLANATION
    # =====================================================

    content.append(PageBreak())


    content.append(

        Paragraph(
            "6. GenAI Explanation",
            heading_style
        )

    )


    # -----------------------------------------------------
    # Handle Gemini list response safely
    # -----------------------------------------------------

    if isinstance(
        genai_explanation,
        list
    ):

        explanation_parts = []


        for item in genai_explanation:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    explanation_parts.append(

                        item.get(
                            "text",
                            ""
                        )

                    )

            elif isinstance(item, str):

                explanation_parts.append(item)


        genai_explanation = "\n".join(
            explanation_parts
        )


    content.append(

        Paragraph(

            safe_text(
                genai_explanation
            ),

            normal_style

        )

    )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    content.append(Spacer(1, 20))


    content.append(

        Paragraph(

            """
            <b>Disclaimer:</b>
            Voight-Kampff provides a probabilistic assessment
            of whether text may be AI-generated. Detection
            results are not absolute proof. Human-written text
            may sometimes appear AI-like, and AI-generated
            text may be modified or paraphrased.
            """,

            small_style

        )

    )


    # =====================================================
    # BUILD PDF
    # =====================================================

    document.build(content)


    return str(pdf_path)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VOIGHT-KAMPFF PDF REPORT")
    print("=" * 60)


    # -----------------------------------------------------
    # Import detector
    # -----------------------------------------------------

    from utils.helpers import analyze_text


    # -----------------------------------------------------
    # Import Gemini explanation
    # -----------------------------------------------------

    from genai.explanation import (
        generate_explanation
    )


    sample_text = """
    Artificial intelligence is transforming modern industries.
    Machine learning systems can analyze large amounts of data
    and help organizations make better decisions.
    """


    # -----------------------------------------------------
    # Detector
    # -----------------------------------------------------

    print("\nRunning detector...")

    result = analyze_text(
        sample_text
    )


    print("\nPrediction:")
    print(result["prediction"])


    print(
        "AI Probability:",
        result["ai_probability"],
        "%"
    )


    print(
        "Human Probability:",
        result["human_probability"],
        "%"
    )


    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    print("\nGenerating explanation...")

    explanation = generate_explanation(

        sample_text,

        result

    )


    print("Explanation generated.")


    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    print("\nGenerating PDF...")

    pdf_file = generate_pdf_report(

        sample_text,

        result,

        explanation

    )


    print("\n" + "=" * 60)

    print("PDF GENERATED SUCCESSFULLY")

    print("=" * 60)

    print("\nPDF file:")

    print(pdf_file)

    print("\nDone.")