import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tldextract

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "domain_model_a.joblib"
OUTPUT_DIR = BASE_DIR / "models"

# Load dataset and trained model
df = pd.read_csv(DATA_FILE)
saved = joblib.load(MODEL_FILE)

model = saved["model"]
features = saved["features"]

X = df[features]
y = df["ClassLabel"]

# Extract registered domains
extractor = tldextract.TLDExtract(suffix_list_urls=())

groups = df["URL"].apply(
    lambda url: extractor(url).top_domain_under_public_suffix or url
)

# Recreate the same domain-based split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Calculate permutation importance
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance_Mean": result.importances_mean,
    "Importance_Std": result.importances_std
}).sort_values(
    by="Importance_Mean",
    ascending=False
)

print("\nPERMUTATION FEATURE IMPORTANCE")
print(importance.to_string(index=False))

# Save results
importance.to_csv(
    OUTPUT_DIR / "permutation_importance.csv",
    index=False
)

# Plot
plt.figure(figsize=(10, 7))

plt.barh(
    importance["Feature"][::-1],
    importance["Importance_Mean"][::-1],
    xerr=importance["Importance_Std"][::-1]
)

plt.xlabel("Decrease in Macro F1 Score")
plt.ylabel("Features")
plt.title("Permutation Feature Importance")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "permutation_importance.png",
    dpi=300
)

plt.show()

print("\nPermutation importance analysis completed!")