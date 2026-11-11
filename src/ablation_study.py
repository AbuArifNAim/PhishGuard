
import pandas as pd
import tldextract

from urllib.parse import urlsplit
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
OUTPUT_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_FILE)

print("Original Dataset Shape:", df.shape)


# ==========================================
# 3. DOMAIN EXTRACTION
# ==========================================

extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_domain(url):
    try:
        parsed = urlsplit(str(url))
        hostname = parsed.hostname

        if not hostname:
            return "unknown"

        result = extractor(hostname)

        return (
            result.top_domain_under_public_suffix
            or hostname.lower()
        )

    except Exception:
        return "unknown"


df["domain_group"] = df["URL"].apply(get_domain)

# Remove unidentified domains
df = df[df["domain_group"] != "unknown"].copy()

print("Dataset After Domain Filtering:", df.shape)


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

all_features = [
    col for col in df.columns
    if col not in ["URL", "ClassLabel", "domain_group"]
]

X = df[all_features]
y = df["ClassLabel"]
groups = df["domain_group"]


# ==========================================
# 5. DOMAIN-BASED SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# ==========================================
# 6. SPLIT VERIFICATION
# ==========================================

train_domains = set(groups.iloc[train_idx])
test_domains = set(groups.iloc[test_idx])

print("\nSPLIT VERIFICATION")

print("Training Rows:", len(train_idx))
print("Testing Rows:", len(test_idx))

print("Training Domains:", len(train_domains))
print("Testing Domains:", len(test_domains))

print("Domain Overlap:", len(train_domains & test_domains))

print("\nTraining Class Distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting Class Distribution:")
print(y_test.value_counts().sort_index())


# ==========================================
# 7. ABLATION EXPERIMENTS
# ==========================================

experiments = {

    "Model A - All Features": all_features,

    "Model B - Path Only": [
        "path_length"
    ],

    "Model C - Without Path": [
        f for f in all_features
        if f != "path_length"
    ],

    "Model D - Without HTTPS": [
        f for f in all_features
        if f != "https_flag"
    ]

}


# ==========================================
# 8. MODEL TRAINING AND EVALUATION
# ==========================================

results = []

for name, selected_features in experiments.items():

    print("\nTraining:", name)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train[selected_features],
        y_train
    )

    predictions = model.predict(
        X_test[selected_features]
    )

    results.append({

        "Model": name,

        "Features": len(selected_features),

        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "Macro_F1": f1_score(
            y_test,
            predictions,
            average="macro"
        ),

        "Phishing_Precision": precision_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "Phishing_Recall": recall_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0
        )

    })


# ==========================================
# 9. DISPLAY AND SAVE RESULTS
# ==========================================

result_df = pd.DataFrame(results)

print("\nABLATION STUDY RESULTS")

print(result_df.to_string(index=False))

result_df.to_csv(
    OUTPUT_DIR / "ablation_results.csv",
    index=False
)

print("\nResults saved to:", OUTPUT_DIR / "ablation_results.csv")

print("\nAblation study completed!")
