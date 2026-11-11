"""Re-extract all 16 features from the raw URLs in the dataset.

Run this after any change to ``features.py`` so the training data and the live
app stay in sync. The original CSV is preserved as
``dataset/cleaned_dataset_legacy.csv`` on the first run.
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from features import FEATURE_COLUMNS, extract_features  # noqa: E402

DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
LEGACY_FILE = BASE_DIR / "dataset" / "cleaned_dataset_legacy.csv"


def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Input rows: {len(df):,}")

    if not LEGACY_FILE.exists():
        df.to_csv(LEGACY_FILE, index=False)
        print(f"Legacy dataset saved: {LEGACY_FILE}")

    rows = []
    skipped = 0

    for url, label in zip(df["URL"], df["ClassLabel"]):
        features = extract_features(url)
        if features is None:
            skipped += 1
            continue
        row = {"URL": url}
        row.update(features)
        row["ClassLabel"] = int(label)
        rows.append(row)

    out = pd.DataFrame(rows, columns=["URL"] + FEATURE_COLUMNS + ["ClassLabel"])
    out = out.drop_duplicates(subset=["URL"], keep="first")
    out.to_csv(DATA_FILE, index=False)

    print(f"Rebuilt rows: {len(out):,} (skipped {skipped})")
    print(f"Saved: {DATA_FILE}")
    print("\nClass distribution:")
    print(out["ClassLabel"].value_counts().sort_index().to_string())
    print("\nSample:")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
