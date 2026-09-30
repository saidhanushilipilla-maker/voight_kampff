import os
import torch
from functools import lru_cache
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "models/detector_model"
FALLBACK_MODEL = "roberta-base"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def load_detector_model():
    """
    Lazy singleton loader for RoBERTa detector model and tokenizer.
    Cached in memory to prevent slow disk reloads.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        load_target = MODEL_PATH
        print(f"Loading trained Voight-Kampff model from: {load_target}")
    else:
        load_target = FALLBACK_MODEL
        print(f"Trained model not found at {MODEL_PATH}. Loading fallback model: {load_target}")

    tokenizer = AutoTokenizer.from_pretrained(load_target)
    model = AutoModelForSequenceClassification.from_pretrained(load_target, num_labels=2)
    model.to(device)
    model.eval()

    return tokenizer, model

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

    tokenizer, model = load_detector_model()

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

if __name__ == "__main__":
    sample_text = "Artificial intelligence is transforming modern industries with machine learning algorithms."
    res = predict_text(sample_text)
    print("Prediction Result:", res)