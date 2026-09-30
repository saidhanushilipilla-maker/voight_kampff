# Raw text
#    ↓
# Remove unwanted characters
#    ↓
# Normalize whitespace
#    ↓
# Sentence segmentation
#    ↓
# Tokenization
#    ↓
# Optional lemmatization
#    ↓
# Clean text

import re


def clean_text(text):
    """
    Basic text cleaning.

    We preserve punctuation and words because they can contain
    useful signals for AI-text detection.
    """

    if text is None:
        return ""

    text = str(text)

    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def split_sentences(text):
    """
    Split text into sentences.
    """

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def word_tokens(text):
    """
    Extract words from text.
    """

    text = clean_text(text)

    return re.findall(r"\b[\w'-]+\b", text.lower())


if __name__ == "__main__":

    sample = """
    Artificial intelligence is changing the way people work.
    Large language models can generate human-like text.
    """

    print("Original:")
    print(sample)

    print("\nCleaned:")
    print(clean_text(sample))

    print("\nSentences:")
    print(split_sentences(sample))

    print("\nWords:")
    print(word_tokens(sample))