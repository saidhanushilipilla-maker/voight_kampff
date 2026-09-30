import math
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def load_perplexity_model():
    """
    Lazy singleton loader for perplexity evaluation model (distilgpt2).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model

def calculate_perplexity(text, max_length=256):
    """
    Calculate text perplexity score using distilgpt2.
    Lower perplexity implies higher predictability (typical of AI text).
    Higher perplexity implies higher variation (typical of human text).
    """
    if not text or not text.strip():
        return 0.0

    try:
        tokenizer, model = load_perplexity_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )

        input_ids = inputs["input_ids"].to(device)

        if input_ids.shape[1] < 2:
            return 0.0

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        return round(perplexity, 2)
    except Exception as e:
        print(f"Perplexity calculation notice: {e}")
        return 50.0

if __name__ == "__main__":
    sample_text = "Artificial intelligence is transforming many industries rapidly across the globe."
    value = calculate_perplexity(sample_text)
    print("Perplexity Score:", value)