"""Regression checks for real-world URLs after a retrain.

Runs the deployed Random Forest directly (no network / Flask needed) and checks
that known-good sites are not flagged and known-bad URLs still are. Exits with
a non-zero status if any case fails, so it can be used in CI or before a demo.

    .\\venv\\Scripts\\python.exe src\\regression_check.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from features import extract_features  # noqa: E402

THRESHOLD = 70.0

CASES = [
    # (url, expected verdict)
    ("https://ahmedimtiaz.com", "Legitimate"),
    ("https://atmoshuddhi.vercel.app/", "Legitimate"),
    ("https://www.google.com/", "Legitimate"),
    ("https://example.com", "Legitimate"),
    ("https://github.com/login", "Legitimate"),
    ("https://stripe.com", "Legitimate"),
    ("http://paypal-secure-login-verify.xyz/account/confirm.php?id=88213&user=test", "Phishing"),
    ("http://testsite.xyz/", "Phishing"),
]


def main():
    data = joblib.load(BASE_DIR / "models" / "all_random_forest.joblib")
    model, feature_names = data["model"], data["features"]

    failures = 0
    print(f"{'URL':<70} {'P(phish)':>9} {'P(legit)':>9} {'verdict':<11} result")
    print("-" * 110)

    for url, expected in CASES:
        features = extract_features(url)
        if features is None:
            print(f"{url[:68]:<70} {'-':>9} {'-':>9} {'-':<11} SKIP (no features)")
            failures += 1
            continue

        row = np.array([[features[k] for k in feature_names]])
        proba = model.predict_proba(row)[0]
        confidence = max(proba) * 100
        verdict = "Phishing" if proba[0] > proba[1] else "Legitimate"
        if confidence < THRESHOLD:
            verdict = "Unknown"

        ok = verdict == expected
        failures += 0 if ok else 1
        print(
            f"{url[:68]:<70} {proba[0]*100:8.2f}% {proba[1]*100:8.2f}% "
            f"{verdict:<11} {'PASS' if ok else 'FAIL (expected ' + expected + ')'}"
        )

    print("-" * 110)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
