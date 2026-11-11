
import pandas as pd
import joblib
import tldextract

from pathlib import Path
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(DATA_FILE)

# Extract registrable domain without downloading suffix data
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

# Remove rows where domain could not be identified
df = df[df["domain_group"] != "unknown"].copy()

X = df.drop(columns=["URL", "ClassLabel", "domain_group"])
y = df["ClassLabel"]
groups = df["domain_group"]

# Split by domain
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

train_domains = set(groups.iloc[train_idx])
test_domains = set(groups.iloc[test_idx])

print("Training Rows:", len(train_idx))
print("Testing Rows:", len(test_idx))
print("Training Domains:", len(train_domains))
print("Testing Domains:", len(test_domains))
print("Domain Overlap:", len(train_domains & test_domains))

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

experiments = {
    "domain_model_a": X.columns.tolist(),
    "domain_model_b": [
        col for col in X.columns
        if col != "https_flag"
    ]
}

for model_name, feature_names in experiments.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train[feature_names],
        y_train
    )

    predictions = model.predict(X_test[feature_names])
    probabilities = model.predict_proba(
        X_test[feature_names]
    )[:, 1]

    print("\nAccuracy:",
          round(accuracy_score(y_test, predictions), 6))

    print("Precision:",
          round(precision_score(y_test, predictions), 6))

    print("Recall:",
          round(recall_score(y_test, predictions), 6))

    print("F1 Score:",
          round(f1_score(y_test, predictions), 6))

    print("ROC-AUC:",
          round(roc_auc_score(y_test, probabilities), 6))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))

    joblib.dump(
        {
            "model": model,
            "features": feature_names,
            "positive_class": 1,
            "negative_class": 0
        },
        MODEL_DIR / f"{model_name}.joblib"
    )

import pandas as pd
import joblib
import tldextract

from pathlib import Path
from urllib.parse import urlsplit

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_FILE)

print("Original Dataset Shape:", df.shape)


# ==========================================
# 3. DOMAIN EXTRACTION
# ==========================================

# Use local suffix data; no online download
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

X = df.drop(
    columns=["URL", "ClassLabel", "domain_group"]
)

y = df["ClassLabel"]

groups = df["domain_group"]


# ==========================================
# 5. DOMAIN-BASED TRAIN-TEST SPLIT
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

print("\n" + "=" * 50)
print("DOMAIN SPLIT VERIFICATION")
print("=" * 50)

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
# 7. MODEL EXPERIMENTS
# ==========================================

experiments = {

    "domain_model_a": X.columns.tolist(),

    "domain_model_b": [
        col for col in X.columns
        if col != "https_flag"
    ]

}


# ==========================================
# 8. MODEL TRAINING AND EVALUATION
# ==========================================

for model_name, feature_names in experiments.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Train model
    model.fit(
        X_train[feature_names],
        y_train
    )

    # Predictions
    predictions = model.predict(
        X_test[feature_names]
    )

    probabilities = model.predict_proba(
        X_test[feature_names]
    )[:, 1]


    # ======================================
    # 9. PERFORMANCE METRICS
    # ======================================

    print("\nAccuracy:",
          round(accuracy_score(y_test, predictions), 6))

    print("Precision:",
          round(precision_score(y_test, predictions), 6))

    print("Recall:",
          round(recall_score(y_test, predictions), 6))

    print("F1 Score:",
          round(f1_score(y_test, predictions), 6))

    print("Macro F1:",
          round(f1_score(
              y_test,
              predictions,
              average="macro"
          ), 6))

    print("ROC-AUC:",
          round(roc_auc_score(y_test, probabilities), 6))


    # ======================================
    # 10. CONFUSION MATRIX
    # ======================================

    print("\nConfusion Matrix:")

    print(confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ))


    # ======================================
    # 11. CLASSIFICATION REPORT
    # ======================================

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))


    # ======================================
    # 12. SAVE TRAINED MODEL
    # ======================================

    joblib.dump(
        {
            "model": model,
            "features": feature_names,
            "positive_class": 1,
            "negative_class": 0
        },
        MODEL_DIR / f"{model_name}.joblib"
    )

    print("\nModel Saved:", model_name)


# ==========================================
# 13. COMPLETION
# ==========================================

print("\n" + "=" * 55)
print("Domain-based evaluation completed!")
print("=" * 55)

print("\nDomain-based evaluation completed!")
