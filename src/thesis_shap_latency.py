"""Regenerate the SHAP artefacts and the latency figure from the reported model.

``src/shap_analysis.py`` explains ``domain_model_a.joblib``, which is not the
Random Forest reported in the results tables. This script keeps the thesis
internally consistent by explaining ``all_random_forest.joblib`` instead and by
writing the global importances to ``results/shap_global_importance.csv``.
"""

import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import matplotlib
import pandas as pd
import shap
import tldextract
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "all_random_forest.joblib"
RESULT_DIR = BASE_DIR / "results"

extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_domain(url):
    try:
        hostname = urlsplit(str(url)).hostname
        if not hostname:
            return "unknown"
        return extractor(hostname).top_domain_under_public_suffix or hostname.lower()
    except Exception:
        return "unknown"


def main():
    df = pd.read_csv(DATA_FILE)
    df["domain_group"] = df["URL"].apply(get_domain)
    df = df[df["domain_group"] != "unknown"].copy()

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    features = bundle["features"]

    X = df[features]
    y = df["ClassLabel"]
    groups = df["domain_group"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_test = X.iloc[test_idx]

    sample = X_test.sample(n=min(600, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        phish = values[0]
    else:
        phish = values[:, :, 0]

    importance = (
        pd.Series(abs(phish).mean(axis=0), index=features)
        .sort_values(ascending=False)
        .round(6)
    )
    importance.to_csv(RESULT_DIR / "shap_global_importance.csv",
                      header=["mean_abs_shap"])

    # summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(phish, sample, show=False)
    plt.title("SHAP Summary Plot - Phishing Detection")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # global bar plot
    plt.figure()
    shap.summary_plot(phish, sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance - Phishing Detection")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "shap_feature_importance.png", dpi=200,
                bbox_inches="tight")
    plt.close()

    # local waterfall for the highest-scoring phishing sample
    proba = model.predict_proba(sample)[:, 0]  # P(phishing)
    idx = int(proba.argmax())
    local = explainer.shap_values(sample.iloc[[idx]])
    local_values = local[0][0] if isinstance(local, list) else local[0, :, 0]
    base = explainer.expected_value
    base = base[0] if hasattr(base, "__len__") else base
    explanation = shap.Explanation(
        values=local_values,
        base_values=base,
        data=sample.iloc[idx].values,
        feature_names=features,
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.title("Local SHAP Explanation - Phishing URL")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "shap_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close()

    # latency chart from the measured benchmark
    with open(RESULT_DIR / "latency_benchmark.json", encoding="utf-8") as fh:
        bench = json.load(fh)

    labels = ["FExtract", "LR", "DT", "RF", "SVM", "XGB", "SHAP"]
    medians = [
        bench["feature_extraction_ms"]["median"],
        bench["models"]["logistic_regression"]["median_ms"],
        bench["models"]["decision_tree"]["median_ms"],
        bench["models"]["random_forest"]["median_ms"],
        bench["models"]["svm"]["median_ms"],
        bench["models"]["xgboost"]["median_ms"],
        bench["shap_ms"]["median"],
    ]
    colors = ["#9e9e9e", "#4c72b0", "#4c72b0", "#c44e52", "#4c72b0",
              "#55a868", "#8172b2"]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars = ax.bar(labels, medians, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("Median latency per URL (ms, log scale)")
    ax.set_title("Measured single-URL latency by component")
    for bar, value in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value * 1.25, f"{value:g}", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(0.01, 400)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "latency_comparison.png", dpi=200)
    plt.close(fig)

    print(importance.to_string())
    print("\nLatency medians:", dict(zip(labels, medians)))


if __name__ == "__main__":
    main()
