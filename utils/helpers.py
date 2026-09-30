import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from nlp.linguistic_features import extract_linguistic_features
from nlp.burstiness import calculate_burstiness
from nlp.perplexity import calculate_perplexity
from transformer.predict import predict_text
from nlp.linguistic_features import extract_linguistic_features
from nlp.burstiness import calculate_burstiness
from nlp.perplexity import calculate_perplexity
from transformer.predict import predict_text


def analyze_text(text):
    """
    Run the complete Voight-Kampff analysis.

    RoBERTa is the primary detector.
    NLP metrics provide supporting evidence.
    """

    if not text or not text.strip():
        return {
            "error": "Text cannot be empty."
        }

    # --------------------------------------------------------
    # 1. Linguistic analysis
    # --------------------------------------------------------

    linguistic_features = extract_linguistic_features(text)

    # --------------------------------------------------------
    # 2. Burstiness analysis
    # --------------------------------------------------------

    burstiness = calculate_burstiness(text)

    # --------------------------------------------------------
    # 3. Perplexity analysis
    # --------------------------------------------------------

    perplexity = calculate_perplexity(text)

    # --------------------------------------------------------
    # 4. RoBERTa prediction
    # --------------------------------------------------------

    transformer_result = predict_text(text)

    # --------------------------------------------------------
    # 5. Final result
    # --------------------------------------------------------

    result = {

        "prediction": transformer_result["label"],

        "human_probability":
            transformer_result["human_probability"],

        "ai_probability":
            transformer_result["ai_probability"],

        "linguistic_features":
            linguistic_features,

        "burstiness":
            burstiness,

        "perplexity":
            perplexity
    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is transforming modern industries.
    Machine learning systems can analyze large amounts of data.
    These technologies help organizations make better decisions.
    """

    result = analyze_text(sample_text)

    print("\n" + "=" * 60)
    print("VOIGHT-KAMPFF ANALYSIS")
    print("=" * 60)

    print("\nPrediction:")
    print(result["prediction"])

    print(
        "\nHuman Probability:",
        result["human_probability"],
        "%"
    )

    print(
        "AI Probability:",
        result["ai_probability"],
        "%"
    )

    print("\nLinguistic Features:")

    for key, value in result["linguistic_features"].items():
        print(f"{key}: {value}")

    print("\nBurstiness:")

    for key, value in result["burstiness"].items():
        print(f"{key}: {value}")

    print("\nPerplexity:")
    print(result["perplexity"])