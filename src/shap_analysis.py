
import pandas as pd
import joblib
import shap
import tldextract

from pathlib import Path
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "domain_model_a.joblib"

# ==========================================
# 2. LOAD DATASET AND MODEL
# ==========================================

df = pd.read_csv(DATA_FILE)

model_data = joblib.load(MODEL_FILE)

# Extract model and features from dictionary
model = model_data["model"]
model_features = model_data["features"]

print("Model Loaded Successfully")
print("Model Type:", type(model))
print("Model Features:", len(model_features))
print("Positive Class:", model_data["positive_class"])
print("Negative Class:", model_data["negative_class"])
print("SHAP Version:", shap.__version__)

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

# Remove unknown domains
df = df[df["domain_group"] != "unknown"].copy()

print("\nDataset After Domain Filtering:", df.shape)

# ==========================================
# 4. PREPARE FEATURES AND LABELS
# ==========================================

X = df[model_features]
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

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print("\nTest Dataset Size:", len(X_test))

# ==========================================
# 6. SELECT TEST SAMPLES
# ==========================================

X_sample = X_test.sample(
    n=min(500, len(X_test)),
    random_state=42
)

print("SHAP Test Samples:", len(X_sample))
print("Feature Count:", len(model_features))

# ==========================================
# 7. INITIALIZE SHAP EXPLAINER
# ==========================================

print("\nInitializing SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

# ==========================================
# 8. CALCULATE SHAP VALUES
# ==========================================

print("Calculating SHAP Values...")

shap_values = explainer.shap_values(X_sample)

# ==========================================
# 9. CHECK SHAP OUTPUT
# ==========================================

print("\nSHAP Analysis Completed!")

print("SHAP Output Type:", type(shap_values))

if isinstance(shap_values, list):

    print("Number of Classes:", len(shap_values))
    print("Class 0 SHAP Shape:", shap_values[0].shape)
    print("Class 1 SHAP Shape:", shap_values[1].shape)

else:

    print("SHAP Shape:", shap_values.shape)

    if len(shap_values.shape) == 3:
        print("Class 0 SHAP Shape:", shap_values[:, :, 0].shape)
        print("Class 1 SHAP Shape:", shap_values[:, :, 1].shape)

print("\nSHAP Analysis Script Finished Successfully!")

import matplotlib.pyplot as plt

# ==========================================
# 10. EXTRACT PHISHING CLASS SHAP VALUES
# ==========================================

# Class 0 = Phishing
phishing_shap_values = shap_values[:, :, 0]

print("\nPhishing SHAP Shape:", phishing_shap_values.shape)

# Create results directory
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 11. SHAP SUMMARY PLOT
# ==========================================

plt.figure()

shap.summary_plot(
    phishing_shap_values,
    X_sample,
    show=False
)

plt.title("SHAP Summary Plot - Phishing Detection")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP Summary Plot Saved!")

# ==========================================
# 12. SHAP FEATURE IMPORTANCE BAR PLOT
# ==========================================

plt.figure()

shap.summary_plot(
    phishing_shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance - Phishing Detection")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP Feature Importance Plot Saved!")

# ==========================================
# 13. LOCAL EXPLANATION - WATERFALL PLOT
# ==========================================

# Find an actual phishing sample from test data
phishing_indices = y_test[y_test == 0].index

if len(phishing_indices) > 0:

    sample_index = phishing_indices[0]

    local_X = X_test.loc[[sample_index]]

    local_shap = explainer.shap_values(local_X)

    if isinstance(local_shap, list):
        local_phishing_shap = local_shap[0][0]
    else:
        local_phishing_shap = local_shap[0, :, 0]

    expected_value = explainer.expected_value

    if hasattr(expected_value, "__len__"):
        base_value = expected_value[0]
    else:
        base_value = expected_value

    explanation = shap.Explanation(
        values=local_phishing_shap,
        base_values=base_value,
        data=local_X.iloc[0].values,
        feature_names=model_features
    )

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.title("Local SHAP Explanation - Phishing URL")
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP Waterfall Plot Saved!")

else:
    print("No phishing sample found!")

print("\nAll SHAP Visualizations Completed!")
print("Results Directory:", RESULT_DIR)
