
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "dataset" / "legitphish.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original Shape:", df.shape)

# Remove rows with missing labels
df = df.dropna(subset=["ClassLabel"])

# Remove duplicate URLs
df = df.drop_duplicates(subset=["URL"], keep="first")

# Ensure binary labels
df["ClassLabel"] = df["ClassLabel"].astype(int)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning Completed!")
print("Cleaned Shape:", df.shape)

print("\nClass Distribution:")
print(df["ClassLabel"].value_counts().sort_index())

print("\nSaved to:", OUTPUT_FILE)
