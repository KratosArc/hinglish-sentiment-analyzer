"""
preprocessing.py
Hinglish Sentiment Analyzer — Data loading and light cleaning.

Source data: SemEval-2020 Task 9 (SentiMix) Hinglish dataset.
Format: TSV with columns [uid, text, label]
Labels: 0 = Negative, 1 = Neutral, 2 = Positive

Preprocessing philosophy: LIGHT-TOUCH.
We are fine-tuning a transformer (XLM-RoBERTa), not building a TF-IDF model.
Transformers learn useful signal from punctuation, casing, and emojis —
so we avoid aggressive cleaning (no lowercasing, no emoji/punctuation removal).
We only fix things that are genuinely noise, not signal.
"""

import pandas as pd
import re
from pathlib import Path

# ---- Paths ----
RAW_DIR = Path("data/raw/temp_repo/data/hinglish")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---- Label mapping (confirmed by cross-referencing counts with the SemEval paper) ----
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def load_raw(filename: str) -> pd.DataFrame:
    """Load a raw TSV file with correct UTF-8 encoding."""
    path = RAW_DIR / filename
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    return df


def light_clean(text: str) -> str:
    """
    Minimal cleaning — only fixes genuine noise, preserves signal.
    - Collapses repeated whitespace (multiple spaces/tabs -> one space)
    - Normalizes smart quotes (' " etc.) to standard ASCII quotes
    - Strips leading/trailing whitespace
    Does NOT: lowercase, remove punctuation, remove emojis, remove URLs/hashtags.
    """
    if not isinstance(text, str):
        return ""

    # Normalize smart quotes / odd unicode punctuation seen in the raw data
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace(""", '"').replace(""", '"')

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process(filename: str, output_name: str) -> pd.DataFrame:
    """Load, clean, map labels, and save a processed split."""
    df = load_raw(filename)

    # Basic sanity checks before proceeding
    assert set(df.columns) == {"uid", "text", "label"}, f"Unexpected columns: {df.columns}"
    assert df["label"].isin(LABEL_MAP.keys()).all(), "Found a label outside 0/1/2"

    df["text"] = df["text"].apply(light_clean)
    df["sentiment"] = df["label"].map(LABEL_MAP)

    # Drop any rows that ended up empty after cleaning
    before = len(df)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} empty rows from {filename}")

    out_path = PROCESSED_DIR / output_name
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {out_path}")

    return df


if __name__ == "__main__":
    print("Processing train.txt ...")
    train_df = process("train.txt", "train_processed.csv")

    print("\nProcessing test.txt ...")
    test_df = process("test.txt", "test_processed.csv")

    print("\n--- Train label distribution ---")
    print(train_df["sentiment"].value_counts())

    print("\n--- Test label distribution ---")
    print(test_df["sentiment"].value_counts())

    print("\n--- Sample rows ---")
    print(train_df[["text", "sentiment"]].head(5).to_string(index=False))