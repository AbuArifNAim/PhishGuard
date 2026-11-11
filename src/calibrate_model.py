"""Fit a probability calibrator for the deployed Random Forest.

The raw forest probabilities are over-confident (tree ensembles push values
towards 0 and 1).  This script fits Platt scaling — a one-dimensional logistic
regression on the log-odds of the raw probability — using the *domain-held-out*
test split, so the calibrator is never fitted on data the model has seen.

Outputs
-------
``models/calibration.joblib``          coefficients used by verification.py
``results/calibration_report.json``    Brier / log-loss / ECE before and after
``results/calibration_curve.png``      reliability diagram for the thesis

    .\\venv\\Scripts\\python.exe src\\calibrate_model.py
"""

import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tldextract
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "all_random_forest.joblib"
OUT_MODEL = BASE_DIR / "models" / "calibration.joblib"
REPORT_FILE = BASE_DIR / "results" / "calibration_report.json"
PLOT_FILE = BASE_DIR / "results" / "calibration_curve.png"

FEATURE_COLUMNS = [
    "url_length", "path_length", "dot_count", "subdomain_count", "tld_length",
    "has_ip_address", "has_hyphen_in_domain", "https_flag",
    "percentage_numeric_chars", "number_of_digits", "token_count",
    "url_entropy", "domain_name_length", "suspicious_file_extension",
    "tld_popularity", "query_param_count",
]

EPS = 1e-6
ABSTAIN_LOW, ABSTAIN_HIGH = 0.30, 0.70

extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_domain(url):
    try:
        hostname = urlsplit(str(url)).hostname
        if not hostname:
            return "unknown"
        return extractor(hostname).top_domain_under_public_suffix or hostname.lower()
    except Exception:
        return "unknown"


def logit(probabilities):
    clipped = np.clip(probabilities, EPS, 1 - EPS)
    return np.log(clipped / (1 - clipped))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def expected_calibration_error(y_true, probabilities, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        if not mask.any():
            continue
        error += (mask.sum() / len(probabilities)) * abs(
            y_true[mask].mean() - probabilities[mask].mean()
        )
    return float(error)


def reliability_points(y_true, probabilities, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    points = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        if not mask.any():
            continue
        points.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "count": int(mask.sum()),
            "mean_predicted": float(probabilities[mask].mean()),
            "observed_phishing": float(y_true[mask].mean()),
        })
    return points


def main():
    print("Loading dataset and deployed model ...")
    df = pd.read_csv(DATA_FILE)
    df["domain_group"] = df["URL"].apply(get_domain)
    df = df[df["domain_group"] != "unknown"].copy()

    X = df[FEATURE_COLUMNS]
    y = df["ClassLabel"]
    groups = df["domain_group"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups=groups))
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]

    print(f"Calibrating on {len(X_test):,} held-out rows ...")
    raw = model.predict_proba(X_test[model_data["features"]])[:, 0]  # P(phishing)
    y_phish = (y_test == 0).astype(int).to_numpy()

    features = logit(raw).reshape(-1, 1)
    # No intercept: the dataset is 63% phishing, which is nothing like the
    # real-world browsing prior.  Fitting an intercept would shift every score
    # upward and create false positives.  A slope-only fit corrects the forest's
    # over-confidence while keeping 0.5 as the decision boundary.
    calibrator = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", fit_intercept=False
    )
    calibrator.fit(features, y_phish)

    coef = float(calibrator.coef_[0][0])
    intercept = float(calibrator.intercept_[0])
    calibrated = sigmoid(coef * logit(raw) + intercept)

    def metrics(probabilities):
        predictions = (probabilities >= 0.5).astype(int)
        abstain = (probabilities >= ABSTAIN_LOW) & (probabilities <= ABSTAIN_HIGH)
        inside = predictions[abstain] == y_phish[abstain]
        return {
            "accuracy_at_0_5": round(float((predictions == y_phish).mean()), 6),
            "brier": round(float(brier_score_loss(y_phish, probabilities)), 6),
            "log_loss": round(float(log_loss(y_phish, probabilities)), 6),
            "ece_10bin": round(expected_calibration_error(y_phish, probabilities), 6),
            "abstain_rate": round(float(abstain.mean()), 6),
            "abstain_accuracy": round(float(inside.mean()), 6) if abstain.any() else None,
        }

    report = {
        "method": "Platt scaling, slope-only (logistic on raw log-odds, no prior shift)",
        "trained_on": "domain-held-out 20% test split (random_state=42)",
        "rows": int(len(X_test)),
        "coefficients": {"coef": round(coef, 6), "intercept": round(intercept, 6)},
        "abstain_band": [ABSTAIN_LOW, ABSTAIN_HIGH],
        "raw": metrics(raw),
        "calibrated": metrics(calibrated),
        "reliability": {
            "raw": reliability_points(y_phish, raw),
            "calibrated": reliability_points(y_phish, calibrated),
        },
    }

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "method": report["method"],
        "coef": coef,
        "intercept": intercept,
        "trained_on": report["trained_on"],
        "rows": report["rows"],
        "abstain_low": ABSTAIN_LOW,
        "abstain_high": ABSTAIN_HIGH,
    }, OUT_MODEL)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=160)
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="Perfect calibration")
    for label, probs, color in (
        ("Raw forest", raw, "#f59e0b"),
        ("Calibrated (Platt)", calibrated, "#22c55e"),
    ):
        points = reliability_points(y_phish, probs)
        ax.plot(
            [p["mean_predicted"] for p in points],
            [p["observed_phishing"] for p in points],
            marker="o", linewidth=1.8, color=color, label=label,
        )
    ax.set_xlabel("Mean predicted P(phishing)")
    ax.set_ylabel("Observed phishing rate")
    ax.set_title("PhishGuard AI — reliability diagram (held-out)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_FILE)
    plt.close(fig)

    print("\nRaw:       ", report["raw"])
    print("Calibrated:", report["calibrated"])
    print(f"\nSaved {OUT_MODEL}")
    print(f"Saved {REPORT_FILE}")
    print(f"Saved {PLOT_FILE}")


if __name__ == "__main__":
    main()
