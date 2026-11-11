
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"

df = pd.read_csv(DATA_FILE)

target = "ClassLabel"
features = [col for col in df.columns
            if col not in ["URL", target]]

print("=" * 50)
print("PHISHGUARD AI - DATASET ANALYSIS")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df[target].value_counts())

print("\nClass Percentage:")
print((df[target].value_counts(normalize=True) * 100).round(2))

print("\nFeature Data Types:")
print(df[features].dtypes)

print("\nFeature Statistics:")
print(df[features].describe().round(3).T)

print("\nUnique Values Per Feature:")
print(df[features].nunique().sort_values())

print("\nConstant Features:")
constant = [
    col for col in features
    if df[col].nunique() <= 1
]
print(constant if constant else "No constant features found.")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCorrelation with Target:")
print(
    df[features + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(ascending=False)
    .round(3)
)

print("\nEDA Completed!")
