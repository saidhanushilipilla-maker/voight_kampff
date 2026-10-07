import math
import gc
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

if not torch.cuda.is_available():
    torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def load_perplexity_model():
    """
    Lazy singleton loader for perplexity evaluation model (distilgpt2).
    Applies CPU dynamic quantization to stay well within RAM on Render.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()

        if device.type == "cpu":
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"Perplexity model quantization notice: {e}")

        return tokenizer, model
    except Exception as e:
        print(f"Failed to load perplexity model {MODEL_NAME}: {e}")
        return None, None

def _heuristic_perplexity(text):
    """
    Fallback heuristic perplexity calculation based on word variety and sentence structure
    when transformer model cannot be loaded in memory-constrained environment.
    """
    words = text.split()
    if not words:
        return 45.0
    unique_ratio = len(set(words)) / len(words)
    avg_len = sum(len(w) for w in words) / len(words)
    base_perp = 25.0 + (unique_ratio * 40.0) + (avg_len * 2.5)
    return round(min(max(base_perp, 15.0), 95.0), 2)

def calculate_perplexity(text, max_length=128):
    """
    Calculate text perplexity score using distilgpt2 or fallback heuristic.
    Lower perplexity implies higher predictability (typical of AI text).
    Higher perplexity implies higher variation (typical of human text).
    """
    if not text or not text.strip():
        return 0.0

    try:
        tokenizer, model = load_perplexity_model()
        if tokenizer is None or model is None:
            return _heuristic_perplexity(text)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )

        input_ids = inputs["input_ids"].to(device)

        if input_ids.shape[1] < 2:
            return 35.0

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        return round(perplexity, 2)
    except Exception as e:
        print(f"Perplexity calculation notice: {e}")
        return _heuristic_perplexity(text)
    finally:
        gc.collect()

if __name__ == "__main__":
    sample_text = "Artificial intelligence is transforming many industries rapidly across the globe."
    value = calculate_perplexity(sample_text)
    print("Perplexity Score:", value)
