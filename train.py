import os
import sys
import subprocess
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("VOIGHT-KAMPFF PIPELINE: DATA PREPARATION & TRAINING")
    print("=" * 60)

    train_file = PROJECT_ROOT / "data" / "train.csv"
    val_file = PROJECT_ROOT / "data" / "validation.csv"
    input_file = PROJECT_ROOT / "data" / "dataset.csv"

    # Step 1: Prepare dataset if needed
    if not train_file.exists() or not val_file.exists():
        print("\n[Step 1/2] Training data split files missing. Running prepare_dataset.py...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "data" / "prepare_dataset.py")], check=True)
    else:
        print(f"\n[Step 1/2] Using existing prepared dataset splits:")
        print(f"  - Train      : {train_file}")
        print(f"  - Validation : {val_file}")

    # Step 2: Run training
    print("\n[Step 2/2] Launching RoBERTa Transformer Fine-Tuning...")
    train_script = PROJECT_ROOT / "transformer" / "train.py"
    subprocess.run([sys.executable, str(train_script)], check=True)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
