import os

# Hide harmless Transformers loading warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# 1. MODEL CONFIGURATION

MODEL_NAME = "roberta-base"

NUM_LABELS = 2

MAX_LENGTH = 256


# 2. LABEL CONFIGURATION

# 0 = Human-written
# 1 = AI-generated

ID2LABEL = {
    0: "HUMAN",
    1: "AI_GENERATED"
}

LABEL2ID = {
    "HUMAN": 0,
    "AI_GENERATED": 1
}



# 3. LOAD TOKENIZER


def load_tokenizer():

    print("Loading RoBERTa tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")

    return tokenizer

# 4. LOAD MODEL
def load_model():

    print("Loading RoBERTa model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,

        # Binary classification
        num_labels=NUM_LABELS,

        # Human / AI labels
        id2label=ID2LABEL,
        label2id=LABEL2ID,

        # Classification problem
        problem_type="single_label_classification"
    )

    print("Model loaded successfully.")

    return model

# 5. PRINT MODEL INFORMATION
def print_model_info(model):

    print("\n" + "=" * 60)
    print("VOIGHT-KAMPFF TRANSFORMER")
    print("=" * 60)

    print("\nModel:")
    print(MODEL_NAME)

    print("\nNumber of labels:")
    print(NUM_LABELS)

    print("\nMaximum sequence length:")
    print(MAX_LENGTH)

    print("\nLabels:")
    print("0 = Human")
    print("1 = AI Generated")

    print("\nLabel mapping:")
    print(model.config.id2label)

    print("\nModel type:")
    print(type(model))

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nTotal parameters:")
    print(f"{total_parameters:,}")

# 6. TEST TOKENIZER
def test_tokenizer(tokenizer):

    sample_text = (
        "Artificial intelligence can generate text "
        "that looks similar to human writing."
    )

    print("\n" + "=" * 60)
    print("TESTING TOKENIZER")
    print("=" * 60)

    encoded = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    print("\nSample text:")
    print(sample_text)

    print("\nInput IDs shape:")
    print(encoded["input_ids"].shape)

    print("\nAttention mask shape:")
    print(encoded["attention_mask"].shape)

    return encoded

# 7. TEST MODEL
def test_model(model, encoded):

    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)

    # Evaluation mode
    model.eval()

    # No gradient calculation during testing
    with torch.no_grad():

        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )

    print("\nLogits shape:")
    print(outputs.logits.shape)

    print("\nLogits:")
    print(outputs.logits)

    # Get predicted class
    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    print("\nPrediction ID:")
    print(prediction)

    print("\nPrediction:")
    print(ID2LABEL[prediction])

    return prediction

# 8. MAIN
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("STARTING VOIGHT-KAMPFF MODEL")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load model
    model = load_model()

    # Print model information
    print_model_info(model)

    # Test tokenizer
    encoded = test_tokenizer(tokenizer)

    # Test model
    test_model(
        model,
        encoded
    )

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

