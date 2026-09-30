# word_count
# sentence_count
# character_count
# average_sentence_length
# sentence_length_std
# unique_words
# lexical_diversity
# punctuation_count
# question_count
# exclamation_count

# python -m nlp.linguistic_features we should use this to run linguistic features instead of running it directly 

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.preprocessing import clean_text, split_sentences, word_tokens


def extract_linguistic_features(text):

    text = clean_text(text)

    words = word_tokens(text)
    sentences = split_sentences(text)

    word_count = len(words)
    sentence_count = len(sentences)

    unique_words = len(set(words))

    # Lexical diversity
    if word_count > 0:
        lexical_diversity = unique_words / word_count
    else:
        lexical_diversity = 0.0

    # Sentence lengths
    sentence_lengths = [
        len(word_tokens(sentence))
        for sentence in sentences
    ]

    if sentence_lengths:
        avg_sentence_length = statistics.mean(sentence_lengths)
    else:
        avg_sentence_length = 0.0

    if len(sentence_lengths) > 1:
        sentence_length_std = statistics.stdev(sentence_lengths)
    else:
        sentence_length_std = 0.0

    # Punctuation
    punctuation_count = len(
        re.findall(r"[^\w\s]", text)
    )

    if len(text) > 0:
        punctuation_ratio = punctuation_count / len(text)
    else:
        punctuation_ratio = 0.0

    # Character statistics
    character_count = len(text)

    # Average word length
    if word_count > 0:
        average_word_length = (
            sum(len(word) for word in words)
            / word_count
        )
    else:
        average_word_length = 0.0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_words": unique_words,
        "lexical_diversity": round(lexical_diversity, 4),
        "avg_sentence_length": round(avg_sentence_length, 4),
        "sentence_length_std": round(sentence_length_std, 4),
        "punctuation_count": punctuation_count,
        "punctuation_ratio": round(punctuation_ratio, 4),
        "character_count": character_count,
        "average_word_length": round(average_word_length, 4)
    }


if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is becoming increasingly important.
    It can help people analyze large amounts of information.
    However, humans still need to verify generated content.
    """

    features = extract_linguistic_features(sample_text)

    print("\nLinguistic Features")
    print("-" * 40)

    for key, value in features.items():
        print(f"{key}: {value}")