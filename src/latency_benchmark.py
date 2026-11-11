"""Single-URL latency benchmark for the thesis computational-feasibility section.

Measures, on the same machine and in the same interpreter:

* feature extraction time for one raw URL,
* one-instance ``predict_proba`` time for every trained model,
* one-instance SHAP explanation time for the deployed Random Forest.

Results are written to ``results/latency_benchmark.json``.
Run with::

    python src/latency_benchmark.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from features import FEATURE_COLUMNS, extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_URLS = 300
N_SHAP = 200
REPEATS = 3


def timed(fn, repeats=REPEATS):
    """Return median, mean and p95 wall-clock time per call in milliseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)

    samples_ns = []
    # Per-call measurements (used for the per-URL figures).
    return {
        "median_ms": round(float(np.median(samples)), 4),
        "mean_ms": round(float(np.mean(samples)), 4),
        "min_ms": round(float(np.min(samples)), 4),
        "max_ms": round(float(np.max(samples)), 4),
    }


def main():
    df = pd.read_csv(DATA_FILE)
    urls = df["URL"].sample(n=min(N_URLS, len(df)), random_state=42).tolist()

    print(f"Benchmarking feature extraction on {len(urls)} URLs ...")
    extract_times = []
    feature_rows = []
    for url in urls:
        start = time.perf_counter_ns()
        row = extract_features(url)
        extract_times.append((time.perf_counter_ns() - start) / 1e6)
        feature_rows.append([row[c] for c in FEATURE_COLUMNS])

    X = np.asarray(feature_rows, dtype=float)

    results = {
        "environment": {
            "python": __import__("sys").version.split()[0],
            "platform": __import__("platform").platform(),
            "processor": __import__("platform").processor(),
            "n_urls": len(urls),
        },
        "feature_extraction_ms": {
            "median": round(float(np.median(extract_times)), 4),
            "mean": round(float(np.mean(extract_times)), 4),
            "p95": round(float(np.percentile(extract_times, 95)), 4),
            "max": round(float(np.max(extract_times)), 4),
        },
        "models": {},
        "shap_ms": None,
    }

    model_files = {
        "logistic_regression": "all_logistic_regression.joblib",
        "decision_tree": "all_decision_tree.joblib",
        "random_forest": "all_random_forest.joblib",
        "svm": "all_svm.joblib",
        "xgboost": "all_xgboost.joblib",
    }

    for name, filename in model_files.items():
        model = joblib.load(MODEL_DIR / filename)["model"]
        x = X[:1]
        # warm-up
        model.predict_proba(x)

        times = []
        for _ in range(N_SHAP * 5):
            start = time.perf_counter_ns()
            model.predict_proba(x)
            times.append((time.perf_counter_ns() - start) / 1e6)

        results["models"][name] = {
            "median_ms": round(float(np.median(times)), 4),
            "mean_ms": round(float(np.mean(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "throughput_per_sec": round(1000.0 / float(np.median(times)), 1),
        }
        print(f"{name}: median {results['models'][name]['median_ms']} ms")

    # SHAP explanation latency for the deployed model.
    rf = joblib.load(MODEL_DIR / "all_random_forest.joblib")["model"]
    explainer = shap.TreeExplainer(rf)
    explainer.shap_values(X[:1])

    shap_times = []
    for i in range(N_SHAP):
        x = X[i : i + 1]
        start = time.perf_counter_ns()
        explainer.shap_values(x)
        shap_times.append((time.perf_counter_ns() - start) / 1e6)

    results["shap_ms"] = {
        "median": round(float(np.median(shap_times)), 4),
        "mean": round(float(np.mean(shap_times)), 4),
        "p95": round(float(np.percentile(shap_times, 95)), 4),
    }
    print("SHAP median:", results["shap_ms"]["median"], "ms")

    out = RESULT_DIR / "latency_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
