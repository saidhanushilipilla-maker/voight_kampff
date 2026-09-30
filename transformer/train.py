import os
import time
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "roberta-base"

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"

MODEL_OUTPUT_DIR = "models/detector_model"

MAX_LENGTH = 128
BATCH_SIZE = 8 if not torch.cuda.is_available() else 16
EPOCHS = 2
LEARNING_RATE = 2e-5

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("VOIGHT-KAMPFF MODEL TRAINING")
print("=" * 60)
print("Device        :", device)
if torch.cuda.is_available():
    print("GPU           :", torch.cuda.get_device_name(0))
print("Base Model    :", MODEL_NAME)
print("Batch Size    :", BATCH_SIZE)
print("Max Length    :", MAX_LENGTH)
print("Epochs        :", EPOCHS)
print("Output Path   :", MODEL_OUTPUT_DIR)
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset splits...")
train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(validation_df)}")

# ============================================================
# DATASET CLASS
# ============================================================

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# LOAD TOKENIZER & CREATE LOADERS
# ============================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = TextDataset(train_df, tokenizer, max_len=MAX_LENGTH)
validation_dataset = TextDataset(validation_df, tokenizer, max_len=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# LOAD MODEL & OPTIMIZER
# ============================================================

print("Loading pre-trained RoBERTa model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ============================================================
# TRAINING LOOP
# ============================================================

start_time = time.time()
print("\nStarting training loop...")

for epoch in range(EPOCHS):
    print("\n" + "=" * 60)
    print(f"EPOCH {epoch + 1} / {EPOCHS}")
    print("=" * 60)

    model.train()
    total_train_loss = 0
    epoch_start_time = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
            avg_step_loss = total_train_loss / (step + 1)
            elapsed = time.time() - epoch_start_time
            print(f"Step [{step + 1}/{len(train_loader)}] - Loss: {loss.item():.4f} (Avg: {avg_step_loss:.4f}) [{elapsed:.1f}s]", flush=True)

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"\nAverage Training Loss for Epoch {epoch + 1}: {avg_train_loss:.4f}")

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL ON VALIDATION SET")
print("=" * 60)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in validation_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, zero_division=0)
recall = recall_score(all_labels, all_predictions, zero_division=0)
f1 = f1_score(all_labels, all_predictions, zero_division=0)

print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving fine-tuned model...")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

total_elapsed = time.time() - start_time
print(f"\nModel saved successfully to: {MODEL_OUTPUT_DIR}")
print(f"Total Training Time: {total_elapsed / 60:.2f} minutes")
print("=" * 60)