"""Reproduce the domain-disjoint test split and inspect misclassified URLs.

The split in ``src/train_all_models.py`` is deterministic (GroupShuffleSplit with
``random_state=42``), so the exact test partition can be reconstructed and the
saved models can be asked which URLs they got wrong. The output is used for the
error analysis in Chapter 4.
"""

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
import tldextract
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

FEATURES = [
    "url_length", "path_length", "dot_count", "subdomain_count", "tld_length",
    "has_ip_address", "has_hyphen_in_domain", "https_flag",
    "percentage_numeric_chars", "number_of_digits", "token_count",
    "url_entropy", "domain_name_length", "suspicious_file_extension",
    "tld_popularity", "query_param_count",
]

extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_domain(url):
    try:
        hostname = urlsplit(str(url)).hostname
        if not hostname:
            return "unknown"
        result = extractor(hostname)
        return result.top_domain_under_public_suffix or hostname.lower()
    except Exception:
        return "unknown"


def main():
    df = pd.read_csv(DATA_FILE)
    df["domain_group"] = df["URL"].apply(get_domain)
    df = df[df["domain_group"] != "unknown"].copy()

    X = df[FEATURES]
    y = df["ClassLabel"]
    groups = df["domain_group"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    urls_test = df["URL"].iloc[test_idx].reset_index(drop=True)
    domains_test = groups.iloc[test_idx].reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    summary = {"n_test": int(len(y_test))}

    for name in ("random_forest", "xgboost"):
        bundle = joblib.load(MODEL_DIR / f"all_{name}.joblib")
        model = bundle["model"]
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        summary[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "confusion": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            },
        }

        # label 0 = phishing, 1 = legitimate
        fn_mask = (y_test == 0) & (pred == 1)   # missed phishing
        fp_mask = (y_test == 1) & (pred == 0)   # legitimate flagged

        errors = pd.DataFrame({
            "url": urls_test[fn_mask | fp_mask],
            "domain": domains_test[fn_mask | fp_mask],
            "true_label": y_test[fn_mask | fp_mask],
            "error_type": ["FN" if t == 0 else "FP"
                           for t in y_test[fn_mask | fp_mask]],
            "path_length": X_test.loc[fn_mask | fp_mask, "path_length"],
            "url_length": X_test.loc[fn_mask | fp_mask, "url_length"],
            "https_flag": X_test.loc[fn_mask | fp_mask, "https_flag"],
            "has_ip": X_test.loc[fn_mask | fp_mask, "has_ip_address"],
            "dots": X_test.loc[fn_mask | fp_mask, "dot_count"],
        })
        errors.to_csv(RESULT_DIR / f"errors_{name}.csv", index=False)

        summary[name]["error_profile"] = {
            "n_fn": int(fn_mask.sum()),
            "n_fp": int(fp_mask.sum()),
            "fn_https_share": float(X_test.loc[fn_mask, "https_flag"].mean())
            if fn_mask.any() else None,
            "fp_https_share": float(X_test.loc[fp_mask, "https_flag"].mean())
            if fp_mask.any() else None,
            "fn_median_path": float(X_test.loc[fn_mask, "path_length"].median())
            if fn_mask.any() else None,
            "fp_median_path": float(X_test.loc[fp_mask, "path_length"].median())
            if fp_mask.any() else None,
            "fn_median_url_len": float(X_test.loc[fn_mask, "url_length"].median())
            if fn_mask.any() else None,
            "fp_median_url_len": float(X_test.loc[fp_mask, "url_length"].median())
            if fp_mask.any() else None,
            "fn_has_ip_share": float(X_test.loc[fn_mask, "has_ip_address"].mean())
            if fn_mask.any() else None,
            "fp_has_ip_share": float(X_test.loc[fp_mask, "has_ip_address"].mean())
            if fp_mask.any() else None,
            "fp_top_domains": Counter(domains_test[fp_mask]).most_common(8),
            "fn_top_domains": Counter(domains_test[fn_mask]).most_common(8),
        }

    with open(RESULT_DIR / "error_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
