import os
import gc
import torch
from functools import lru_cache
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "models/detector_model"
FALLBACK_MODEL = "roberta-base"
MAX_LENGTH = 128

# Limit CPU threads to reduce memory footprint on Render free tier
if not torch.cuda.is_available():
    torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def load_detector_model():
    """
    Lazy singleton loader for RoBERTa detector model and tokenizer.
    Cached in memory to prevent slow disk reloads.
    Applies CPU dynamic quantization to stay well within RAM limits on Render.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        load_target = MODEL_PATH
        print(f"Loading trained Voight-Kampff model from: {load_target}")
    else:
        load_target = FALLBACK_MODEL
        print(f"Trained model not found at {MODEL_PATH}. Loading fallback model: {load_target}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(load_target)
        model = AutoModelForSequenceClassification.from_pretrained(load_target, num_labels=2)
        model.to(device)
        model.eval()

        if device.type == "cpu":
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"Quantization notice: {e}")

        return tokenizer, model
    except Exception as e:
        print(f"Failed to load detector model {load_target}: {e}")
        return None, None

def _heuristic_prediction(text):
    """
    Fallback classifier based on text entropy & sentence characteristics if heavy ML model cannot be loaded.
    """
    words = text.split()
    if not words:
        return {"label": "Human Written", "human_probability": 80.0, "ai_probability": 20.0}
    
    unique_ratio = len(set(words)) / len(words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    
    # Highly repetitive, uniform formal structure -> higher AI probability indicator
    if unique_ratio < 0.65 and avg_word_len > 5.2:
        ai_prob = min(88.5, round(60.0 + (5.2 - avg_word_len)*5 + (0.65 - unique_ratio)*50, 2))
        human_prob = round(100.0 - ai_prob, 2)
        label = "AI Generated" if ai_prob >= human_prob else "Human Written"
    else:
        human_prob = min(92.0, round(55.0 + (unique_ratio * 35.0), 2))
        ai_prob = round(100.0 - human_prob, 2)
        label = "Human Written" if human_prob >= ai_prob else "AI Generated"
        
    return {
        "label": label,
        "human_probability": human_prob,
        "ai_probability": ai_prob
    }

def predict_text(text):
    """
    Run RoBERTa prediction on input text.
    Returns label, human_probability (%), and ai_probability (%).
    """
    if not text or not text.strip():
        return {
            "label": "Unknown",
            "human_probability": 0.0,
            "ai_probability": 0.0
        }

    try:
        tokenizer, model = load_detector_model()
        if tokenizer is None or model is None:
            return _heuristic_prediction(text)

        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        human_probability = probabilities[0][0].item()
        ai_probability = probabilities[0][1].item()

        if ai_probability >= human_probability:
            label = "AI Generated"
        else:
            label = "Human Written"

        return {
            "label": label,
            "human_probability": round(human_probability * 100, 2),
            "ai_probability": round(ai_probability * 100, 2)
        }
    except Exception as e:
        print(f"Prediction execution error: {e}")
        return _heuristic_prediction(text)
    finally:
        gc.collect()

if __name__ == "__main__":
    sample_text = "Artificial intelligence is transforming modern industries with machine learning algorithms."
    res = predict_text(sample_text)
    print("Prediction Result:", res)
