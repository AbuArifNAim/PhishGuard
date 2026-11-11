
import pandas as pd
import numpy as np
import joblib
import time
import tldextract

from pathlib import Path
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
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
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("=" * 60)
print("PHISHGUARD AI - TRAINING ALL MODELS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
print("\nDataset Shape:", df.shape)


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
df = df[df["domain_group"] != "unknown"].copy()

print("Dataset After Domain Filtering:", df.shape)


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

feature_columns = [
    "url_length",
    "path_length",
    "dot_count",
    "subdomain_count",
    "tld_length",
    "has_ip_address",
    "has_hyphen_in_domain",
    "https_flag",
    "percentage_numeric_chars",
    "number_of_digits",
    "token_count",
    "url_entropy",
    "domain_name_length",
    "suspicious_file_extension",
    "tld_popularity",
    "query_param_count"
]

X = df[feature_columns]
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

train_domains = set(groups.iloc[train_idx])
test_domains = set(groups.iloc[test_idx])

print("\n" + "=" * 60)
print("DOMAIN-BASED SPLIT VERIFICATION")
print("=" * 60)

print(f"Training Rows:   {len(train_idx)}")
print(f"Testing Rows:    {len(test_idx)}")
print(f"Training Domains: {len(train_domains)}")
print(f"Testing Domains:  {len(test_domains)}")
print(f"Domain Overlap:   {len(train_domains & test_domains)}")

print("\nTraining Class Distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting Class Distribution:")
print(y_test.value_counts().sort_index())


# ==========================================
# 6. DEFINE ALL MODELS
# ==========================================

models = {

    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "decision_tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "svm": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "xgboost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

}


# ==========================================
# 7. TRAIN AND EVALUATE ALL MODELS
# ==========================================

all_results = []

print("\n" + "=" * 60)
print("TRAINING ALL MODELS")
print("=" * 60)

for model_name, model in models.items():

    print(f"\n{'─' * 60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'─' * 60}")

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict
    start_pred = time.time()
    predictions = model.predict(X_test)
    pred_time = time.time() - start_pred

    # Probability (for ROC-AUC)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    # Store results
    result = {
        "Model": model_name,
        "Accuracy": round(acc, 6),
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1_Score": round(f1, 6),
        "ROC_AUC": round(roc_auc, 6),
        "Train_Time_sec": round(train_time, 3),
        "Predict_Time_sec": round(pred_time, 4),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    }

    all_results.append(result)

    # Print metrics
    print(f"\nAccuracy:  {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1 Score:  {f1:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")
    print(f"Train Time: {train_time:.3f}s")
    print(f"Predict Time: {pred_time:.4f}s")

    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

    print(f"\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))

    # Save model
    model_data = {
        "model": model,
        "features": feature_columns,
        "positive_class": 1,
        "negative_class": 0,
        "model_name": model_name
    }

    output_file = MODEL_DIR / f"all_{model_name}.joblib"
    joblib.dump(model_data, output_file)
    print(f"Model Saved: {output_file}")


# ==========================================
# 8. COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)

comparison_cols = ["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
print(results_df[comparison_cols].to_string(index=False))

# Find best model
best_model = results_df.loc[results_df["Accuracy"].idxmax()]
print(f"\n🏆 Best Model: {best_model['Model'].upper()}")
print(f"   Accuracy: {best_model['Accuracy']:.6f}")
print(f"   F1 Score: {best_model['F1_Score']:.6f}")
print(f"   ROC-AUC:  {best_model['ROC_AUC']:.6f}")


# ==========================================
# 9. SAVE RESULTS TO CSV
# ==========================================

# Full results
results_file = RESULT_DIR / "model_comparison.csv"
results_df.to_csv(results_file, index=False)
print(f"\nResults Saved: {results_file}")

# Detailed confusion matrix
cm_data = []
for _, row in results_df.iterrows():
    cm_data.append({
        "Model": row["Model"],
        "True_Negative": row["TN"],
        "False_Positive": row["FP"],
        "False_Negative": row["FN"],
        "True_Positive": row["TP"]
    })

cm_df = pd.DataFrame(cm_data)
cm_file = RESULT_DIR / "confusion_matrices.csv"
cm_df.to_csv(cm_file, index=False)
print(f"Confusion Matrices Saved: {cm_file}")


# ==========================================
# 10. SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print(f"\nTotal Models Trained: {len(models)}")
print(f"Models Saved to: {MODEL_DIR}")
print(f"Results Saved to: {RESULT_DIR}")

print("\nTrained Models:")
for name in models.keys():
    print(f"  - all_{name}.joblib")

print("\nAll models trained successfully!")
