import sys
import os
from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# STREAMLIT & LIBRARIES
# =========================================================

import streamlit as st
from utils.helpers import analyze_text
from genai.explanation import generate_explanation
from database.database import (
    create_table,
    save_detection,
    get_detection_history
)
from reports.report_generator import generate_pdf_report

# Initialize SQLite database
create_table()

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Voight-Kampff | GenAI Text Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM PROFESSIONAL CSS STYLING
# =========================================================

st.markdown("""
<style>
    /* Dark Modern Theme Styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Result Cards */
    .result-card-ai {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid #ef4444;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
    }
    
    .result-card-human {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(21, 128, 61, 0.05) 100%);
        border: 1px solid #22c55e;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
    }
    
    .badge-ai {
        color: #ef4444;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    
    .badge-human {
        color: #22c55e;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    /* Metric Container Cards */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CACHED ANALYSIS WRAPPER (SPEED OPTIMIZATION)
# =========================================================

@st.cache_data(show_spinner=False)
def run_fast_analysis(input_text):
    """
    Cached call to text analyzer to prevent slow re-computations.
    """
    return analyze_text(input_text)

# =========================================================
# SIDEBAR CONTROLS & PRESETS
# =========================================================

with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/000000/brain.png", width=70)
    st.title("🛡️ Voight-Kampff")
    st.caption("AI vs Human Text Analysis Engine v2.0")
    st.divider()
    
    st.subheader("⚙️ System Status")
    st.markdown("• **Model:** Fine-Tuned RoBERTa")
    st.markdown("• **NLP Metrics:** Burstiness & Perplexity")
    st.markdown("• **LLM Engine:** Gemini 3.6 Flash")
    st.divider()

    st.subheader("💡 Try Sample Texts")
    preset = st.radio(
        "Load sample content:",
        ["Custom Input", "AI-Generated Sample", "Human-Written Sample"],
        index=0
    )
    
    st.divider()
    st.info("ℹ️ **Tip:** Enter at least 3-4 sentences for high-precision detection.")

# Sample Preset Texts
SAMPLE_AI = """Artificial intelligence is rapidly expanding across global industries. Machine learning architectures facilitate rapid decision-making by analyzing vast streams of structured data. Furthermore, automated neural network frameworks eliminate traditional human operational bottlenecks, allowing enterprises to optimize workflow efficiency seamlessly."""

SAMPLE_HUMAN = """I spent hours trying to get my old coffee machine working this morning. It kept leaking water all over the kitchen counter, and no matter how many times I unplugged it, the red indicator light just kept flashing. Eventually I gave up and bought a cold brew on my way to work."""

# =========================================================
# HEADER BANNER
# =========================================================

st.markdown("""
<div class="header-box">
    <div class="header-title">🛡️ Voight-Kampff AI Detector</div>
    <div class="header-subtitle">Advanced Multi-Signal Text Forensic Analysis Engine</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TEXT INPUT SECTION
# =========================================================

st.subheader("📝 Enter Text for Analysis")

# Determine default text based on preset selection
default_text = ""
if preset == "AI-Generated Sample":
    default_text = SAMPLE_AI
elif preset == "Human-Written Sample":
    default_text = SAMPLE_HUMAN

user_text = st.text_area(
    "Paste the document or essay text below:",
    value=default_text,
    height=220,
    placeholder="Type or paste text here (minimum 20 words recommended)..."
)

# Text Stats Counter
word_count = len(user_text.split()) if user_text else 0
char_count = len(user_text) if user_text else 0

col_stat1, col_stat2, col_stat3 = st.columns([2, 2, 6])
with col_stat1:
    st.caption(f"📏 Words: **{word_count}**")
with col_stat2:
    st.caption(f"🔤 Characters: **{char_count}**")

st.divider()

# =========================================================
# ANALYZE ACTION
# =========================================================

analyze_btn = st.button("🚀 Analyze Text", type="primary", use_container_width=True)

if analyze_btn:
    if not user_text.strip():
        st.warning("⚠️ Please paste or type some text first.")
        st.stop()
        
    with st.spinner("⚡ Running Voight-Kampff Multi-Signal Analysis..."):
        # 1. Fast detection analysis
        result = run_fast_analysis(user_text)
        
        if "error" in result:
            st.error(result["error"])
            st.stop()
            
        st.session_state["text"] = user_text
        st.session_state["result"] = result
        
    with st.spinner("✨ Synthesizing GenAI Detailed Report..."):
        # 2. GenAI explanation
        explanation = generate_explanation(user_text, result)
        st.session_state["explanation"] = explanation
        
    # 3. Save to database history
    burstiness_val = result.get("burstiness", {}).get("burstiness_score", 0.0)
    save_detection(
        input_text=user_text,
        prediction=result["prediction"],
        ai_probability=result["ai_probability"],
        human_probability=result["human_probability"],
        perplexity=result["perplexity"],
        burstiness=burstiness_val,
        genai_explanation=explanation
    )

# =========================================================
# RESULTS DISPLAY AREA
# =========================================================

if "result" in st.session_state:
    result = st.session_state["result"]
    prediction = result["prediction"]
    ai_prob = result["ai_probability"]
    human_prob = result["human_probability"]
    
    st.header("🎯 Detection Dashboard")
    
    # Overview Tabs
    tab_overview, tab_nlp, tab_metrics, tab_genai, tab_history = st.tabs([
        "🎯 Overview",
        "📊 Linguistic Features",
        "🧠 Perplexity & Burstiness",
        "✨ GenAI Report",
        "📄 PDF & History"
    ])
    
    # -----------------------------------------------------
    # TAB 1: OVERVIEW
    # -----------------------------------------------------
    with tab_overview:
        col_res1, col_res2 = st.columns([4, 6])
        
        with col_res1:
            if prediction == "AI Generated":
                st.markdown(f"""
                <div class="result-card-ai">
                    <div style="color: #f87171; font-weight: 700; font-size: 0.9rem;">PRIMARY PREDICTION</div>
                    <div class="badge-ai">🤖 AI GENERATED</div>
                    <div style="color: #cbd5e1; margin-top: 8px;">Confidence: <b>{ai_prob}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-human">
                    <div style="color: #4ade80; font-weight: 700; font-size: 0.9rem;">PRIMARY PREDICTION</div>
                    <div class="badge-human">👤 HUMAN WRITTEN</div>
                    <div style="color: #cbd5e1; margin-top: 8px;">Confidence: <b>{human_prob}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                
        with col_res2:
            st.subheader("Probability Breakdown")
            st.write(f"**AI Generated Probability:** {ai_prob}%")
            st.progress(ai_prob / 100.0)
            st.write(f"**Human Written Probability:** {human_prob}%")
            st.progress(human_prob / 100.0)

        st.divider()
        st.subheader("⚡ Quick Signals Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("RoBERTa Classification", prediction)
        m2.metric("Perplexity Score", f"{result['perplexity']}")
        m3.metric("Burstiness Index", f"{result['burstiness'].get('burstiness_score', 0)}")

    # -----------------------------------------------------
    # TAB 2: LINGUISTIC FEATURES
    # -----------------------------------------------------
    with tab_nlp:
        st.subheader("📊 Linguistic & Syntactic Analysis")
        ling = result.get("linguistic_features", {})
        
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        lcol1.metric("Word Count", ling.get("word_count", 0))
        lcol2.metric("Sentence Count", ling.get("sentence_count", 0))
        lcol3.metric("Lexical Diversity", ling.get("lexical_diversity", 0))
        lcol4.metric("Avg Word Length", f"{ling.get('average_word_length', 0)} chars")
        
        st.divider()
        st.subheader("Detailed Feature Dictionary")
        st.json(ling)

    # -----------------------------------------------------
    # TAB 3: PERPLEXITY & BURSTINESS
    # -----------------------------------------------------
    with tab_metrics:
        st.subheader("🧠 Language Model Metrics")
        
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("### Perplexity Score")
            st.metric("Perplexity", result["perplexity"])
            st.caption(
                "Perplexity evaluates text predictability via distilgpt2. "
                "Lower scores (< 40) often indicate machine predictability, while higher scores indicate human variance."
            )
            
        with pcol2:
            st.markdown("### Burstiness Analysis")
            burst = result.get("burstiness", {})
            st.metric("Burstiness Score", burst.get("burstiness_score", 0))
            st.write(f"**Variation Level:** {burst.get('burstiness_level', 'N/A')}")
            st.caption(
                "Burstiness measures the variation in sentence lengths. "
                "Humans naturally mix short and long sentences, resulting in higher burstiness."
            )

    # -----------------------------------------------------
    # TAB 4: GENAI REPORT
    # -----------------------------------------------------
    with tab_genai:
        st.subheader("✨ Gemini AI Synthesis & Explanation")
        if "explanation" in st.session_state:
            st.markdown(st.session_state["explanation"])
        else:
            st.info("GenAI explanation will appear here after analysis.")

    # -----------------------------------------------------
    # TAB 5: PDF & HISTORY
    # -----------------------------------------------------
    with tab_history:
        st.subheader("📄 Export & Historical Log")
        
        # PDF Generator Button
        if st.button("📥 Generate PDF Audit Report", type="secondary"):
            with st.spinner("Generating PDF document..."):
                pdf_path = generate_pdf_report(
                    st.session_state["text"],
                    result,
                    st.session_state.get("explanation", "")
                )
            st.success("PDF generated successfully!")
            
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_file,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf"
                )
                
        st.divider()
        st.subheader("🗄️ Past Detections")
        history = get_detection_history()
        
        if history:
            for row in history[:10]:
                with st.expander(f"Detection #{row[0]} | Prediction: {row[2]} (AI: {row[3]}%) | Date: {row[7]}"):
                    st.write(f"**Text Snippet:** {row[1][:300]}...")
                    st.write(f"**AI Probability:** {row[3]}% | **Human Probability:** {row[4]}%")
                    st.write(f"**Perplexity:** {row[5]} | **Burstiness:** {row[6]}")
                    if row[8]:
                        st.markdown("**Explanation:**")
                        st.markdown(row[8])
        else:
            st.info("No saved detection history yet.")

# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("Voight-Kampff Engine v2.0 | Powered by Fine-Tuned RoBERTa + DistilGPT2 + LangChain Gemini")