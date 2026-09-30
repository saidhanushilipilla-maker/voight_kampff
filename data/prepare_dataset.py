import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. FILE PATHS & CONFIGURATION


INPUT_FILE = "data/dataset.csv"

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"
TEST_FILE = "data/test.csv"

# Samples to take per class (0 = Human, 1 = AI) for training
# Set to 5000 for balanced, fast, effective RoBERTa fine-tuning
SAMPLE_SIZE_PER_CLASS = 5000


# 2. LOAD DATASET WITH AUTO COLUMN DETECTION

print("=" * 60)
print("VOIGHT-KAMPFF DATASET PREPARATION")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input dataset file not found at: {INPUT_FILE}")

print(f"\nLoading dataset from: {INPUT_FILE}")
print("Reading CSV (this may take a few seconds for large files)...")

# Read dataset
df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("Available columns:", df.columns.tolist())

# 3. AUTO DETECT TEXT AND LABEL COLUMNS

possible_text_cols = ["text", "text_content", "content", "document", "body", "essay", "sentence"]
possible_label_cols = ["generated", "label", "target", "class", "is_ai", "ai_generated", "prediction"]

text_col = None
for col in possible_text_cols:
    if col in df.columns:
        text_col = col
        break

label_col = None
for col in possible_label_cols:
    if col in df.columns:
        label_col = col
        break

if not text_col:
    raise ValueError(f"Could not automatically detect text column. Found columns: {df.columns.tolist()}")

if not label_col:
    raise ValueError(f"Could not automatically detect label column. Found columns: {df.columns.tolist()}")

print(f"\nDetected Text Column : '{text_col}'")
print(f"Detected Label Column: '{label_col}'")

# Rename columns to standard 'text' and 'label'
df = df.rename(columns={text_col: "text", label_col: "label"})

# Keep only necessary columns
df = df[["text", "label"]]

# ------------------------------------------------------------
# 4. CLEAN DATA (MISSING VALUES & DUPLICATES)
# ------------------------------------------------------------

print("\nCleaning data...")
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()

# Remove empty strings
df = df[df["text"] != ""]

# Remove duplicate text rows
df = df.drop_duplicates(subset=["text"])
print(f"Shape after cleaning & deduplication: {df.shape[0]} rows")

# 5. NORMALIZE LABELS TO INT (0 = HUMAN, 1 = AI)
# 

print("\nProcessing labels...")

# Handle numeric labels (e.g. 0.0, 1.0 or 0, 1)
if pd.api.types.is_numeric_dtype(df["label"]):
    df["label"] = df["label"].astype(int)
else:
    # String mapping
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    label_mapping = {
        "0": 0, "0.0": 0, "human": 0, "real": 0, "original": 0,
        "1": 1, "1.0": 1, "ai": 1, "ai_generated": 1, "generated": 1, "machine": 1, "fake": 1
    }
    df["label"] = df["label"].map(label_mapping)

# Check for unmapped labels
if df["label"].isna().any():
    unmapped_count = df["label"].isna().sum()
    print(f"Warning: Dropping {unmapped_count} rows with unrecognized label formats.")
    df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

print("\nOverall Label Distribution in Raw Data:")
label_counts = df["label"].value_counts().to_dict()
print(f"  Class 0 (Human): {label_counts.get(0, 0)}")
print(f"  Class 1 (AI)   : {label_counts.get(1, 0)}")

# 6. BALANCE AND SUBSAMPLE DATASET

human_df = df[df["label"] == 0]
ai_df = df[df["label"] == 1]

if len(human_df) == 0 or len(ai_df) == 0:
    raise ValueError("Dataset must contain both Human (0) and AI (1) examples for binary classification.")

actual_sample_size = min(len(human_df), len(ai_df), SAMPLE_SIZE_PER_CLASS)

print("\nBalancing dataset...")
print(f"Selecting {actual_sample_size} samples per class ({actual_sample_size * 2} total)...")

human_sampled = human_df.sample(n=actual_sample_size, random_state=42)
ai_sampled = ai_df.sample(n=actual_sample_size, random_state=42)

balanced_df = pd.concat([human_sampled, ai_sampled], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 7. TRAIN / VALIDATION / TEST SPLIT (80% / 10% / 10%)

train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.20,
    random_state=42,
    stratify=balanced_df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# 
# 8. SAVE SPLITS TO CSV

os.makedirs("data", exist_ok=True)

train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VALIDATION_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)
print(f"Train set saved to      : {TRAIN_FILE} ({len(train_df)} samples)")
print(f"Validation set saved to : {VALIDATION_FILE} ({len(val_df)} samples)")
print(f"Test set saved to       : {TEST_FILE} ({len(test_df)} samples)")
print("=" * 60)
