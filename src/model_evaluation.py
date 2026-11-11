
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tldextract

from pathlib import Path
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("=" * 60)
print("PHISHGUARD AI - MODEL EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE)


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
        return result.top_domain_under_public_suffix or hostname.lower()
    except Exception:
        return "unknown"


df["domain_group"] = df["URL"].apply(get_domain)
df = df[df["domain_group"] != "unknown"].copy()

print("Dataset Shape:", df.shape)


# ==========================================
# 4. PREPARE DATA
# ==========================================

feature_columns = [
    "url_length", "path_length", "dot_count", "subdomain_count",
    "tld_length", "has_ip_address", "has_hyphen_in_domain", "https_flag",
    "percentage_numeric_chars", "number_of_digits", "token_count",
    "url_entropy", "domain_name_length", "suspicious_file_extension",
    "tld_popularity", "query_param_count"
]

X = df[feature_columns]
y = df["ClassLabel"]
groups = df["domain_group"]

# Same split as training
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print(f"Test Samples: {len(X_test)}")


# ==========================================
# 5. LOAD ALL MODELS
# ==========================================

model_files = {
    "Logistic Regression": "all_logistic_regression.joblib",
    "Decision Tree": "all_decision_tree.joblib",
    "Random Forest": "all_random_forest.joblib",
    "SVM": "all_svm.joblib",
    "XGBoost": "all_xgboost.joblib"
}

models = {}
for name, filename in model_files.items():
    path = MODEL_DIR / filename
    if path.exists():
        data = joblib.load(path)
        models[name] = data["model"]
        print(f"Loaded: {name}")
    else:
        print(f"NOT FOUND: {filename}")


# ==========================================
# 6. ROC CURVE COMPARISON
# ==========================================

print("\n" + "=" * 60)
print("GENERATING ROC CURVES")
print("=" * 60)

plt.figure(figsize=(10, 8))

colors = {
    "Logistic Regression": "#2196F3",
    "Decision Tree": "#FF9800",
    "Random Forest": "#4CAF50",
    "SVM": "#9C27B0",
    "XGBoost": "#F44336"
}

roc_results = {}

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_results[name] = roc_auc

    plt.plot(
        fpr, tpr,
        color=colors[name],
        lw=2,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

# Random baseline
plt.plot(
    [0, 1], [0, 1],
    color="gray",
    lw=1,
    linestyle="--",
    label="Random (AUC = 0.5000)"
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve Comparison - All Models", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

roc_file = RESULT_DIR / "roc_comparison.png"
plt.savefig(roc_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"ROC Curve Saved: {roc_file}")


# ==========================================
# 7. CONFUSION MATRIX PLOTS
# ==========================================

print("\n" + "=" * 60)
print("GENERATING CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    if idx >= len(axes):
        break

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    # Calculate percentages
    cm_pct = cm.astype("float") / cm.sum() * 100

    # Create annotations
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
            )

    sns.heatmap(
        cm,
        annot=np.array(annotations).reshape(2, 2),
        fmt="",
        cmap="Blues",
        xticklabels=["Phishing", "Legitimate"],
        yticklabels=["Phishing", "Legitimate"],
        ax=axes[idx],
        annot_kws={"size": 12}
    )

    axes[idx].set_title(
        f"{name}\nAccuracy: {accuracy_score(y_test, predictions):.4f}",
        fontsize=12,
        fontweight="bold"
    )
    axes[idx].set_ylabel("Actual")
    axes[idx].set_xlabel("Predicted")

# Hide empty subplot
if len(models) < len(axes):
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

plt.suptitle(
    "Confusion Matrices - All Models",
    fontsize=16,
    fontweight="bold",
    y=1.02
)
plt.tight_layout()

cm_file = RESULT_DIR / "confusion_matrices.png"
plt.savefig(cm_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Confusion Matrices Saved: {cm_file}")


# ==========================================
# 8. INDIVIDUAL CONFUSION MATRICES
# ==========================================

print("\nGenerating individual confusion matrices...")

for name, model in models.items():
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    cm_pct = cm.astype("float") / cm.sum() * 100

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)")

    sns.heatmap(
        cm,
        annot=np.array(annotations).reshape(2, 2),
        fmt="",
        cmap="Blues",
        xticklabels=["Phishing", "Legitimate"],
        yticklabels=["Phishing", "Legitimate"],
        annot_kws={"size": 14}
    )

    plt.title(
        f"Confusion Matrix - {name}",
        fontsize=14,
        fontweight="bold"
    )
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    individual_file = RESULT_DIR / f"confusion_matrix_{safe_name}.png"
    plt.savefig(individual_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {individual_file}")


# ==========================================
# 9. METRICS BAR CHART
# ==========================================

print("\n" + "=" * 60)
print("GENERATING METRICS BAR CHART")
print("=" * 60)

metrics_data = []

for name, model in models.items():
    predictions = model.predict(X_test)
    probabilities = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics_data.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

metrics_df = pd.DataFrame(metrics_data)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(metrics_df))
width = 0.15
multiplier = 0

metric_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]):
    offset = width * multiplier
    bars = ax.bar(
        x + offset,
        metrics_df[metric],
        width,
        label=metric,
        color=metric_colors[i],
        alpha=0.85
    )
    multiplier += 1

ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison - All Metrics", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics_df["Model"], fontsize=10)
ax.set_ylim(0.95, 1.005)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

bar_file = RESULT_DIR / "metrics_comparison.png"
plt.savefig(bar_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Metrics Bar Chart Saved: {bar_file}")


# ==========================================
# 10. SAVE METRICS TO CSV
# ==========================================

metrics_file = RESULT_DIR / "detailed_metrics.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"\nMetrics CSV Saved: {metrics_file}")


# ==========================================
# 11. PRINT SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)

print("\n📊 Model Comparison:")
print(metrics_df.to_string(index=False))

print("\n🏆 Best Model by ROC-AUC:")
best_idx = metrics_df["ROC-AUC"].idxmax()
best = metrics_df.iloc[best_idx]
print(f"   {best['Model']} (AUC = {best['ROC-AUC']:.4f})")

print("\n📁 Generated Files:")
print(f"  1. {roc_file.name}")
print(f"  2. {cm_file.name}")
print(f"  3. confusion_matrix_*.png (5 files)")
print(f"  4. {bar_file.name}")
print(f"  5. {metrics_file.name}")

print("\nAll evaluation plots saved successfully!")
