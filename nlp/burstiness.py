import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.preprocessing import split_sentences, word_tokens


def calculate_burstiness(text):

    sentences = split_sentences(text)

    sentence_lengths = [
        len(word_tokens(sentence))
        for sentence in sentences
    ]

    if not sentence_lengths:
        return {
            "mean_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "burstiness_score": 0.0,
            "burstiness_level": "Unknown"
        }

    mean_length = statistics.mean(sentence_lengths)

    if len(sentence_lengths) > 1:
        std_length = statistics.stdev(sentence_lengths)
    else:
        std_length = 0.0

    # Coefficient of variation
    if mean_length > 0:
        burstiness_score = std_length / mean_length
    else:
        burstiness_score = 0.0

    # These are only descriptive heuristic ranges.
    if burstiness_score < 0.25:
        level = "Low"
    elif burstiness_score < 0.50:
        level = "Medium"
    else:
        level = "High"

    return {
        "mean_sentence_length": round(mean_length, 4),
        "sentence_length_std": round(std_length, 4),
        "burstiness_score": round(burstiness_score, 4),
        "burstiness_level": level
    }


if __name__ == "__main__":

    sample_text = """
    AI can write text.
    However, humans can also write structured content.
    Sometimes writing becomes very unpredictable and varied.
    This variation can be useful for linguistic analysis.
    """

    result = calculate_burstiness(sample_text)

    print("\nBurstiness Analysis")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")