import os
import sys
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# =========================================================
# NUMBERED CANVAS WITH HEADER & FOOTER
# =========================================================

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Suppress header and footer on cover page (Page 1)
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#475569"))
            self.drawString(54, 750, "VOIGHT-KAMPFF AI DETECTION SYSTEM")
            self.setFont("Helvetica", 8)
            self.drawRightString(612 - 54, 750, "TECHNICAL DOCUMENTATION")
            
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)
            
            # Footer
            self.line(54, 48, 612 - 54, 48)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(54, 34, "Confidential & Proprietary — Voight-Kampff NLP Engine")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 54, 34, page_text)
            
        self.restoreState()

# =========================================================
# PDF BUILDER FUNCTION
# =========================================================

def build_pdf(filename="Voight_Kampff_Project_Documentation.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=64,
        bottomMargin=64
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#0F172A")    # Slate 900
    ACCENT = colors.HexColor("#2563EB")     # Blue 600
    SECONDARY = colors.HexColor("#475569")  # Slate 600
    BG_LIGHT = colors.HexColor("#F8FAFC")   # Slate 50
    CARD_BG = colors.HexColor("#F1F5F9")    # Slate 100
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Custom Typography Styles
    style_cover_title = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=12
    )

    style_cover_subtitle = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        leading=20,
        textColor=ACCENT,
        alignment=TA_LEFT,
        spaceAfter=24
    )

    style_h1 = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=ACCENT,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=8,
        alignment=TA_LEFT
    )

    style_body_bold = ParagraphStyle(
        'BodyBoldCustom',
        parent=style_body,
        fontName='Helvetica-Bold'
    )

    style_bullet = ParagraphStyle(
        'BulletCustom',
        parent=style_body,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    style_code = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        backColor=CARD_BG,
        borderColor=BORDER_COLOR,
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8,
        borderRadius=4
    )

    style_callout = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E3A8A"),
        backColor=colors.HexColor("#EFF6FF"),
        borderColor=colors.HexColor("#93C5FD"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=10,
        borderRadius=4
    )

    story = []

    # =========================================================
    # COVER PAGE / HEADER BLOCK
    # =========================================================
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("VOIGHT-KAMPFF SYSTEM", ParagraphStyle('SubHeaderTag', fontName='Helvetica-Bold', fontSize=10, textColor=ACCENT, spaceAfter=8)))
    story.append(Paragraph("Generative AI Text Detection Engine", style_cover_title))
    story.append(Paragraph("Complete Technical Documentation & Architectural Reference Manual", style_cover_subtitle))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceBefore=0, spaceAfter=20))

    meta_data = [
        [Paragraph("<b>Document Version:</b> 2.0 (High-Performance)", style_body), Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", style_body)],
        [Paragraph("<b>Primary Classifier:</b> Fine-Tuned RoBERTa Transformer", style_body), Paragraph("<b>Perplexity Engine:</b> DistilGPT2 Causal LM", style_body)],
        [Paragraph("<b>Synthesis Engine:</b> LangChain + Gemini 3.6 Flash", style_body), Paragraph("<b>Database & UI:</b> SQLite + Streamlit Lab UI", style_body)]
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 20))

    # Executive Summary Callout
    story.append(Paragraph(
        "<b>EXECUTIVE SUMMARY:</b> The Voight-Kampff AI Detection System is an end-to-end NLP framework designed to accurately identify AI-generated text versus human-authored content. Combining deep transformer classification (RoBERTa) with linguistic metrics (lexical diversity, burstiness) and causal language model perplexity (DistilGPT2), the platform delivers robust detection scores alongside GenAI-synthesized explanations.",
        style_callout
    ))
    story.append(Spacer(1, 15))

    # =========================================================
    # SECTION 1: SYSTEM OVERVIEW & OBJECTIVES
    # =========================================================
    story.append(Paragraph("1. System Overview & Objectives", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=0, spaceAfter=10))
    
    story.append(Paragraph(
        "As Large Language Models (LLMs) proliferate, distinguishing synthetic machine text from organic human prose has become critical for academic integrity, media authenticity, and cybersecurity. The Voight-Kampff project provides a reliable, multi-signal forensic analysis tool inspired by dual-processing NLP methodology.",
        style_body
    ))
    story.append(Paragraph("Key Objectives:", style_h2))
    story.append(Paragraph("• <b>High-Accuracy Classification:</b> Fine-tuned RoBERTa transformer model serving as the primary detection authority.", style_bullet))
    story.append(Paragraph("• <b>Multi-Signal Verification:</b> Supporting statistical signals including Lexical Diversity, Sentence Burstiness, and DistilGPT2 Perplexity.", style_bullet))
    story.append(Paragraph("• <b>High-Speed Inference:</b> Caching mechanisms (`@st.cache_resource`, `lru_cache`) delivering analysis results in sub-second time (0.16s).", style_bullet))
    story.append(Paragraph("• <b>Explainable AI (XAI):</b> Automated plain-English explanations via Gemini 3.6 Flash via LangChain.", style_bullet))
    story.append(Paragraph("• <b>Persistence & Auditability:</b> SQLite storage and automated PDF report compilation.", style_bullet))
    story.append(Spacer(1, 15))

    # =========================================================
    # SECTION 2: SYSTEM ARCHITECTURE & COMPONENTS
    # =========================================================
    story.append(Paragraph("2. System Architecture & Components", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=0, spaceAfter=10))
    
    story.append(Paragraph(
        "The architecture is structured in a modular fashion, separating data preprocessing, model inference, metric evaluation, database logging, and front-end visualization.",
        style_body
    ))

    arch_table_data = [
        [Paragraph("<b>Component Module</b>", style_body_bold), Paragraph("<b>Target Path</b>", style_body_bold), Paragraph("<b>Core Responsibility</b>", style_body_bold)],
        [Paragraph("Master Entrypoint", style_body), Paragraph("<code>train.py</code>", style_code), Paragraph("Orchestrates dataset preprocessing and transformer training.", style_body)],
        [Paragraph("Data Preprocessor", style_body), Paragraph("<code>data/prepare_dataset.py</code>", style_code), Paragraph("Auto-detects columns, handles numeric/string labels, balances data.", style_body)],
        [Paragraph("Transformer Classifier", style_body), Paragraph("<code>transformer/predict.py</code>", style_code), Paragraph("RoBERTa sequence classifier with singleton lazy model caching.", style_body)],
        [Paragraph("Perplexity Engine", style_body), Paragraph("<code>nlp/perplexity.py</code>", style_code), Paragraph("DistilGPT2 cross-entropy loss computation for predictability.", style_body)],
        [Paragraph("Burstiness & NLP", style_body), Paragraph("<code>nlp/burstiness.py</code><br/><code>nlp/linguistic_features.py</code>", style_code), Paragraph("Sentence length variance, lexical diversity (TTR), word counts.", style_body)],
        [Paragraph("GenAI Explanation", style_body), Paragraph("<code>genai/explanation.py</code>", style_code), Paragraph("LangChain prompt pipeline with Gemini for human-readable reports.", style_body)],
        [Paragraph("Database Engine", style_body), Paragraph("<code>database/database.py</code>", style_code), Paragraph("SQLite database creation and historical record tracking.", style_body)],
        [Paragraph("PDF Report Generator", style_body), Paragraph("<code>reports/report_generator.py</code>", style_code), Paragraph("ReportLab PDF generation for individual analysis audits.", style_body)],
        [Paragraph("Web Application", style_body), Paragraph("<code>app.py</code>", style_code), Paragraph("Streamlit interactive lab dashboard with tabbed UI and dark theme.", style_body)]
    ]

    t_arch = Table(arch_table_data, colWidths=[120, 140, 244])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD_BG),
        ('TEXTCOLOR', (0,0), (-1,0), PRIMARY),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 15))

    # =========================================================
    # SECTION 3: DATA PREPARATION & MODEL TRAINING PIPELINE
    # =========================================================
    story.append(Paragraph("3. Data Preparation & Model Training Pipeline", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph("Dataset Auto-Detection & Cleaning", style_h2))
    story.append(Paragraph(
        "The dataset preprocessor (<code>data/prepare_dataset.py</code>) handles large datasets (e.g. 1.11 GB / 487,235 raw rows) without memory overhead. It features intelligent auto-detection for text and label columns across diverse dataset formats:",
        style_body
    ))
    story.append(Paragraph("• <b>Text Column Matchers:</b> <code>['text', 'text_content', 'content', 'document', 'body', 'essay']</code>", style_bullet))
    story.append(Paragraph("• <b>Label Column Matchers:</b> <code>['generated', 'label', 'target', 'class', 'is_ai']</code>", style_bullet))
    story.append(Paragraph("• <b>Label Normalization:</b> Automatically converts numeric floats (0.0 / 1.0) and strings ('human' / 'ai') into binary integers (0 = Human, 1 = AI).", style_bullet))
    story.append(Paragraph("• <b>Stratified Subsampling:</b> Extracts an equal distribution (5,000 Human, 5,000 AI) into 80% Train (8,000), 10% Validation (1,000), and 10% Test (1,000) splits.", style_bullet))

    story.append(Paragraph("RoBERTa Fine-Tuning Setup", style_h2))
    story.append(Paragraph(
        "The primary detector is fine-tuned using PyTorch and HuggingFace Transformers (<code>transformer/train.py</code>):",
        style_body
    ))

    hyper_table_data = [
        [Paragraph("<b>Hyperparameter</b>", style_body_bold), Paragraph("<b>Value / Setting</b>", style_body_bold), Paragraph("<b>Rationale</b>", style_body_bold)],
        [Paragraph("Base Architecture", style_body), Paragraph("<code>roberta-base</code>", style_code), Paragraph("Robust pre-trained bidirectional transformer encoder.", style_body)],
        [Paragraph("Max Sequence Length", style_body), Paragraph("128 tokens", style_code), Paragraph("Optimal trade-off between semantic context & CPU speed.", style_body)],
        [Paragraph("Optimizer", style_body), Paragraph("AdamW (lr=2e-5)", style_code), Paragraph("Standard decoupled weight decay optimizer for NLP fine-tuning.", style_body)],
        [Paragraph("Batch Size", style_body), Paragraph("8 (CPU) / 16 (GPU)", style_code), Paragraph("Balanced memory consumption and gradient stability.", style_body)],
        [Paragraph("Epochs", style_body), Paragraph("2 Epochs", style_code), Paragraph("Prevents overfitting while achieving high validation accuracy.", style_body)],
        [Paragraph("Scheduler", style_body), Paragraph("Linear Warmup (10%)", style_code), Paragraph("Smooth learning rate warm-up and decay schedule.", style_body)]
    ]

    t_hyper = Table(hyper_table_data, colWidths=[130, 130, 244])
    t_hyper.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD_BG),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_hyper)
    story.append(Spacer(1, 15))

    # =========================================================
    # SECTION 4: FORENSIC METRICS & DETECTION SIGNALS
    # =========================================================
    story.append(Paragraph("4. Forensic Metrics & Detection Signals", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph("Primary Signal: RoBERTa Classification", style_h2))
    story.append(Paragraph(
        "The fine-tuned model outputs classification logits for binary classes. Applying a Softmax function yields probability distributions for Human versus AI origin:",
        style_body
    ))
    story.append(Paragraph("<code>P(AI) = softmax(Logits)[1],   P(Human) = softmax(Logits)[0]</code>", style_code))

    story.append(Paragraph("Secondary Signal: Perplexity (DistilGPT2)", style_h2))
    story.append(Paragraph(
        "Perplexity measures how surprised a causal language model is by the text. Synthetic texts generated by LLMs typically exhibit lower perplexity (predictable token sequences), whereas human text demonstrates higher perplexity.",
        style_body
    ))
    story.append(Paragraph("<code>Perplexity = exp( CrossEntropyLoss(Text, DistilGPT2) )</code>", style_code))

    story.append(Paragraph("Secondary Signal: Burstiness Index", style_h2))
    story.append(Paragraph(
        "Burstiness quantifies the variation in sentence length across a document. Human writers naturally mix concise sentences with long, descriptive structures (high standard deviation). AI generators tend to maintain uniform sentence lengths.",
        style_body
    ))
    story.append(Paragraph("<code>Burstiness Score = StdDev(Sentence Lengths) / Mean(Sentence Lengths)</code>", style_code))

    story.append(Paragraph("Linguistic Feature Suite", style_h2))
    story.append(Paragraph("Extracts key stylistic markers:", style_body))
    story.append(Paragraph("• <b>Lexical Diversity (Type-Token Ratio):</b> Ratio of unique words to total words.", style_bullet))
    story.append(Paragraph("• <b>Average Word Length:</b> Character count divided by word count.", style_bullet))
    story.append(Paragraph("• <b>Punctuation Ratio:</b> Punctuation density relative to total character length.", style_bullet))
    story.append(Spacer(1, 15))

    # =========================================================
    # SECTION 5: PERFORMANCE & INFERENCE SPEED BENCHMARKS
    # =========================================================
    story.append(Paragraph("5. Performance & Speed Optimizations", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph(
        "Prior to optimization, initializing HuggingFace transformer models on every analysis request created a 70-second latency bottleneck. By implementing singleton lazy model loading and Streamlit resource caching (<code>@st.cache_resource</code> / <code>@st.cache_data</code>), the runtime latency was dramatically reduced.",
        style_body
    ))

    bench_data = [
        [Paragraph("<b>Metric / Stage</b>", style_body_bold), Paragraph("<b>Uncached Execution</b>", style_body_bold), Paragraph("<b>Optimized Cached Execution</b>", style_body_bold), Paragraph("<b>Speedup Factor</b>", style_body_bold)],
        [Paragraph("Model Cold Load", style_body), Paragraph("~70.0 seconds", style_body), Paragraph("5.23 seconds (Startup Only)", style_body), Paragraph("13.3x Faster", style_body)],
        [Paragraph("Repeated Text Analysis", style_body), Paragraph("~70.0 seconds", style_body), Paragraph("<b>0.166 seconds</b>", style_body), Paragraph("<b>> 420x Faster</b>", style_body)],
        [Paragraph("Memory Overhead", style_body), Paragraph("Repeated Allocations", style_body), Paragraph("Single Resident Process", style_body), Paragraph("Stable Footprint", style_body)]
    ]
    t_bench = Table(bench_data, colWidths=[130, 120, 144, 110])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD_BG),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 15))

    # =========================================================
    # SECTION 6: USER INTERFACE & OPERATION MANUAL
    # =========================================================
    story.append(Paragraph("6. User Interface & Operation Manual", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph("Command Line Operations", style_h2))
    story.append(Paragraph("1. <b>Prepare Dataset & Fine-Tune Model:</b>", style_body))
    story.append(Paragraph("<code>python train.py</code>", style_code))
    story.append(Paragraph("2. <b>Launch Web Dashboard:</b>", style_body))
    story.append(Paragraph("<code>streamlit run app.py</code>", style_code))

    story.append(Paragraph("Streamlit Dashboard Features", style_h2))
    story.append(Paragraph("• <b>Preset Sample Loader:</b> One-click loading of AI-generated vs Human-written sample texts in the sidebar.", style_bullet))
    story.append(Paragraph("• <b>Tab 1 - Detection Overview:</b> Color-coded banner (AI Red vs Human Green), dual progress bars, quick signal summary.", style_bullet))
    story.append(Paragraph("• <b>Tab 2 - Linguistic Features:</b> Metric cards for Lexical Diversity, Word Count, Sentence Length, and raw JSON export.", style_bullet))
    story.append(Paragraph("• <b>Tab 3 - Perplexity & Burstiness:</b> Detailed breakdown of DistilGPT2 predictability and sentence length variance.", style_bullet))
    story.append(Paragraph("• <b>Tab 4 - GenAI Synthesis Report:</b> Gemini 3.6 Flash structured reasoning and confidence assessment.", style_bullet))
    story.append(Paragraph("• <b>Tab 5 - Audit PDF & SQLite History:</b> Generate and download downloadable PDF report audits and explore historical scans.", style_bullet))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=10, spaceAfter=10))
    story.append(Paragraph("End of Documentation — Voight-Kampff AI Detection Engine v2.0", ParagraphStyle('FooterTag', fontName='Helvetica-Bold', fontSize=9, textColor=SECONDARY, alignment=TA_CENTER)))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Documentation PDF generated successfully at: {os.path.abspath(filename)}")

if __name__ == "__main__":
    build_pdf()
