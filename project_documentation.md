# Voight-Kampff AI Detection System — Project Documentation

**Version:** 2.0 (High-Performance Multi-Signal Edition)  
**PDF Location:** [`Voight_Kampff_Project_Documentation.pdf`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/reports/Voight_Kampff_Project_Documentation.pdf)

---

## 1. Executive Summary & Objectives

The **Voight-Kampff AI Detection System** is an end-to-end Natural Language Processing (NLP) framework designed to identify synthetic AI-generated text versus organic human-authored content. Combining deep transformer classification (RoBERTa) with linguistic metrics (lexical diversity, burstiness) and causal language model perplexity (DistilGPT2), the platform delivers robust detection scores alongside GenAI-synthesized explanations.

### Key Objectives:
- **High-Accuracy Classification:** Fine-tuned RoBERTa transformer model serving as the primary detection authority.
- **Multi-Signal Verification:** Supporting statistical signals including Lexical Diversity, Sentence Burstiness, and DistilGPT2 Perplexity.
- **High-Speed Inference:** Caching mechanisms (`@st.cache_resource`, `lru_cache`) delivering analysis results in sub-second time (**0.16s**).
- **Explainable AI (XAI):** Automated plain-English explanations via Gemini 3.6 Flash via LangChain.
- **Persistence & Auditability:** SQLite storage and automated PDF report compilation.

---

## 2. System Architecture & Components

```
+-----------------------------------------------------------------------+
|                         STREAMLIT USER DASHBOARD                      |
|                                 (app.py)                              |
+-----------------------------------------------------------------------+
                                   |
       +---------------------------+---------------------------+
       |                           |                           |
       v                           v                           v
+------------------+     +-------------------+     +--------------------+
|  RoBERTa Model   |     | Perplexity Engine |     |  Linguistic Features|
|  (predict.py)    |     |  (perplexity.py)  |     |  & Burstiness      |
+------------------+     +-------------------+     +--------------------+
       |                           |                           |
       +---------------------------+---------------------------+
                                   |
                                   v
                      +-------------------------+
                      | LangChain + Gemini LLM  |
                      |    (explanation.py)     |
                      +-------------------------+
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v                                   v
    +------------------------+          +------------------------+
    | SQLite Database Store  |          | ReportLab PDF Audit    |
    |      (database.py)     |          | (report_generator.py)  |
    +------------------------+          +------------------------+
```

| Component Module | File Path | Core Responsibility |
| :--- | :--- | :--- |
| **Master Entrypoint** | [`train.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/train.py) | Orchestrates dataset preprocessing and transformer training. |
| **Data Preprocessor** | [`data/prepare_dataset.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/data/prepare_dataset.py) | Auto-detects columns, handles numeric/string labels, balances data. |
| **Transformer Classifier** | [`transformer/predict.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/transformer/predict.py) | RoBERTa sequence classifier with singleton lazy model caching. |
| **Perplexity Engine** | [`nlp/perplexity.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/nlp/perplexity.py) | DistilGPT2 cross-entropy loss computation for predictability. |
| **Burstiness & NLP** | [`nlp/burstiness.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/nlp/burstiness.py)<br/>[`nlp/linguistic_features.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/nlp/linguistic_features.py) | Sentence length variance, lexical diversity (TTR), word counts. |
| **GenAI Explanation** | [`genai/explanation.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/genai/explanation.py) | LangChain prompt pipeline with Gemini for human-readable reports. |
| **Database Engine** | [`database/database.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/database/database.py) | SQLite database creation and historical record tracking. |
| **PDF Report Generator** | [`reports/report_generator.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/reports/report_generator.py) | ReportLab PDF generation for individual analysis audits. |
| **Web Application** | [`app.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/app.py) | Streamlit interactive lab dashboard with tabbed UI and dark theme. |

---

## 3. Data Preparation & Model Training Pipeline

### Dataset Auto-Detection & Cleaning
The dataset preprocessor ([`data/prepare_dataset.py`](file:///c:/Users/saidh/OneDrive/Desktop/voight_kampff/data/prepare_dataset.py)) handles large datasets (e.g. 1.11 GB / 487,235 raw rows) without memory overhead. It features intelligent auto-detection for text and label columns across diverse dataset formats:
- **Text Column Matchers:** `['text', 'text_content', 'content', 'document', 'body', 'essay']`
- **Label Column Matchers:** `['generated', 'label', 'target', 'class', 'is_ai']`
- **Label Normalization:** Automatically converts numeric floats (`0.0` / `1.0`) and strings (`'human'` / `'ai'`) into binary integers (`0` = Human, `1` = AI).
- **Stratified Subsampling:** Extracts an equal distribution (5,000 Human, 5,000 AI) into 80% Train (8,000), 10% Validation (1,000), and 10% Test (1,000) splits.

### RoBERTa Fine-Tuning Hyperparameters

| Hyperparameter | Value / Setting | Rationale |
| :--- | :--- | :--- |
| **Base Architecture** | `roberta-base` | Robust pre-trained bidirectional transformer encoder. |
| **Max Sequence Length** | 128 tokens | Optimal trade-off between semantic context & CPU speed. |
| **Optimizer** | AdamW (`lr=2e-5`) | Standard decoupled weight decay optimizer for NLP fine-tuning. |
| **Batch Size** | 8 (CPU) / 16 (GPU) | Balanced memory consumption and gradient stability. |
| **Epochs** | 2 Epochs | Prevents overfitting while achieving high validation accuracy. |
| **Scheduler** | Linear Warmup (10%) | Smooth learning rate warm-up and decay schedule. |

---

## 4. Forensic Metrics & Detection Signals

1. **Primary Signal: RoBERTa Classification**
   The fine-tuned model outputs classification logits for binary classes:
   $$\text{P(AI)} = \text{Softmax}(\text{Logits})[1], \quad \text{P(Human)} = \text{Softmax}(\text{Logits})[0]$$

2. **Secondary Signal: Perplexity (DistilGPT2)**
   Perplexity measures how predictable text is to a causal model:
   $$\text{Perplexity} = \exp\left( \text{CrossEntropyLoss}(\text{Text}, \text{DistilGPT2}) \right)$$
   *Lower perplexity (< 40)* indicates machine predictability, while *higher perplexity (> 70)* indicates human stylistic variation.

3. **Secondary Signal: Burstiness Index**
   Burstiness measures sentence length variation:
   $$\text{Burstiness} = \frac{\text{StdDev}(\text{Sentence Lengths})}{\text{Mean}(\text{Sentence Lengths})}$$

---

## 5. Performance & Speed Optimizations

| Metric / Stage | Uncached Execution | Optimized Cached Execution | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **Model Cold Load** | ~70.0 seconds | 5.23 seconds (Startup Only) | 13.3x Faster |
| **Repeated Text Analysis** | ~70.0 seconds | **0.166 seconds** | **> 420x Faster** |

---

## 6. How to Run the Project

1. **Train / Fine-Tune Model on New Dataset:**
   ```bash
   python train.py
   ```

2. **Launch Streamlit Web App:**
   ```bash
   streamlit run app.py
   ```

3. **Re-compile PDF Project Documentation:**
   ```bash
   python generate_project_docs_pdf.py
   ```
