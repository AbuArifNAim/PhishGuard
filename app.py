
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import shap
import numpy as np
import re
import socket
import ipaddress
import requests
import json
import csv
import uuid
import threading
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlparse

from features import extract_features, get_tld
from verification import (
    ABSTAIN_HIGH,
    ABSTAIN_LOW,
    assess,
    calibrate,
    load_calibration,
)

# ==========================================
# VERIFICATION LAYER SETTINGS
# ==========================================

PROBE_TIMEOUT = 5             # Seconds for DNS / HTTP checks
PROBE_HEADERS = {"User-Agent": "PhishGuard-AI/1.0 (verification-layer)"}

# ==========================================
# 1. FLASK APP CONFIGURATION
# ==========================================

app = Flask(__name__)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/all_random_forest.joblib"
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
feature_names = model_data["features"]

print("Model Loaded Successfully")
print("Features:", feature_names)

# SHAP explainer for per-prediction explanations (XAI)
xai_explainer = shap.TreeExplainer(model)

# Platt-scaling coefficients fitted on the domain-held-out split (optional).
CALIBRATION = load_calibration()

# Friendly labels used across the UI (scanner, XAI, analytics)
FEATURE_LABELS = {
    "url_length": "URL Length",
    "path_length": "Path Length",
    "dot_count": "Dot Count",
    "subdomain_count": "Subdomains",
    "tld_length": "TLD Length",
    "has_ip_address": "IP Address in URL",
    "has_hyphen_in_domain": "Hyphen in Domain",
    "https_flag": "HTTPS",
    "percentage_numeric_chars": "Numeric Chars %",
    "number_of_digits": "Number of Digits",
    "token_count": "Token Count",
    "url_entropy": "URL Entropy",
    "domain_name_length": "Domain Name Length",
    "suspicious_file_extension": "Suspicious Extension",
    "tld_popularity": "Popular TLD",
    "query_param_count": "Query Parameters",
}

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "scan_history.json"
HISTORY_LIMIT = 300

_history_lock = threading.Lock()


# ==========================================
# 2b. SCAN HISTORY STORE (powers Dashboard)
# ==========================================

def load_history():
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        print(f"History load error: {e}")
    return []


def save_history(records):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records[:HISTORY_LIMIT], f, ensure_ascii=False)
    except Exception as e:
        print(f"History save error: {e}")


def record_scan(result):
    """Persist a completed scan so Dashboard / XAI / Analytics can use it."""
    now = datetime.now()
    record = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%I:%M %p"),
        "url": result.get("url"),
        "status": result.get("status"),
        "status_reason": result.get("status_reason"),
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "phishing_probability": result.get("phishing_probability"),
        "legitimate_probability": result.get("legitimate_probability"),
        "raw_phishing_probability": result.get("raw_phishing_probability"),
        "raw_legitimate_probability": result.get("raw_legitimate_probability"),
        "risk_score": result.get("risk_score"),
        "indicators": result.get("indicators", []),
        "features": result.get("features"),
        "explanation": result.get("explanation"),
        "layers": result.get("layers", []),
        "calibration": result.get("calibration"),
    }
    with _history_lock:
        records = load_history()
        records.insert(0, record)
        save_history(records)
    return record


# ==========================================
# 2c. XAI — SHAP EXPLANATION FOR ONE URL
# ==========================================

def explain_prediction(features):
    """
    SHAP values for a single instance.
    Class 0 = Phishing, Class 1 = Legitimate.
    base_value[0] + sum(shap_class0) == model.predict_proba(...)[0]
    """
    try:
        x = np.array([[features[f] for f in feature_names]])
        sv = np.asarray(xai_explainer.shap_values(x))
        base = np.asarray(xai_explainer.expected_value)

        if sv.ndim == 3:              # (1, n_features, n_classes)
            shap_phish = sv[0, :, 0]
            shap_legit = sv[0, :, 1]
        elif isinstance(sv, list):    # legacy list format
            shap_phish = np.asarray(sv[0]).ravel()
            shap_legit = np.asarray(sv[1]).ravel()
        else:
            return None

        expected_phishing = float(base[0]) if np.ndim(base) else float(base)

        contributions = []
        for i, fname in enumerate(feature_names):
            contributions.append({
                "feature": fname,
                "label": FEATURE_LABELS.get(fname, fname),
                "value": features.get(fname),
                "shap_phishing": round(float(shap_phish[i]), 4),
                "shap_legitimate": round(float(shap_legit[i]), 4),
            })

        # Sort by absolute influence on the phishing decision
        contributions.sort(key=lambda c: abs(c["shap_phishing"]), reverse=True)

        return {
            "base_value": round(expected_phishing, 4),
            "base_percent": round(expected_phishing * 100, 2),
            "contributions": contributions,
            "method": "SHAP (TreeExplainer on Random Forest)",
        }
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        return None


# ==========================================
# 2d. RULE-BASED RISK INDICATORS
# ==========================================

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def build_indicators(features, status, tld=None):
    """Human-readable risk indicators derived from the extracted features."""
    ind = []

    def add(sev, title, detail):
        ind.append({"severity": sev, "title": title, "detail": detail})

    if features.get("has_ip_address"):
        add("high", "IP Address Instead of Domain",
            "The URL uses a raw IP address as the host. Legitimate services almost "
            "always use a domain name, so this is a strong phishing signal.")

    if features.get("subdomain_count", 0) >= 3:
        add("high", "Excessive Subdomains",
            f"{features['subdomain_count']} subdomains were detected. Attackers often "
            "chain subdomains to hide the real destination (e.g. login.verify.bank.com.evil.io).")
    elif features.get("subdomain_count", 0) >= 1:
        add("low", "Multiple Subdomains",
            f"{features['subdomain_count']} subdomain level(s) present — worth a closer look.")

    if features.get("suspicious_file_extension"):
        add("high", "Suspicious File Extension",
            "The URL ends with an executable or script extension (.exe, .zip, .bat, .js …), "
            "a common delivery mechanism for malware.")

    if not features.get("https_flag"):
        add("medium", "No HTTPS Encryption",
            "The site is served over plain HTTP. Any data entered (credentials, card numbers) "
            "can be intercepted in transit.")

    if features.get("has_hyphen_in_domain"):
        add("medium", "Hyphen in Domain Name",
            "Hyphens are frequently used to make a malicious domain look like a trusted "
            "brand (e.g. pay-pal-login.com).")

    if features.get("url_length", 0) >= 100:
        add("high", "Very Long URL",
            f"{features['url_length']} characters long. Phishers pad URLs to push the "
            "malicious domain out of view in the address bar.")
    elif features.get("url_length", 0) >= 75:
        add("medium", "Long URL",
            f"{features['url_length']} characters long — longer than typical legitimate URLs.")

    if features.get("percentage_numeric_chars", 0) >= 30:
        add("medium", "High Numeric Content",
            f"{features['percentage_numeric_chars']:.1f}% of the URL is digits. Randomly "
            "generated phishing URLs tend to be digit-heavy.")

    if features.get("tld_popularity") == 0:
        tld_txt = f".{tld}" if tld else "this TLD"
        add("medium", "Uncommon Top-Level Domain",
            f"Domain ends with {tld_txt}, which is outside the popular set "
            "(com, org, net, edu, gov, io, co, app, dev, ai, me). Cheap or uncommon "
            "TLDs are preferred by attackers.")

    if features.get("url_entropy", 0) >= 4.5:
        add("medium", "High Character Entropy",
            f"Entropy {features['url_entropy']} indicates a random-looking string — "
            "typical of automatically generated phishing links.")

    if features.get("dot_count", 0) >= 5:
        add("medium", "Many Dots in URL",
            f"{features['dot_count']} dots detected, which can be used to disguise the "
            "real host (e.g. paypal.com.evil-site.io/login).")

    if features.get("token_count", 0) >= 10:
        add("medium", "Deep Path Structure",
            f"{features['token_count']} path tokens — deep, generated paths are common "
            "on credential-harvesting pages.")

    if features.get("query_param_count", 0) >= 4:
        add("low", "Many Query Parameters",
            f"{features['query_param_count']} parameters. Excessive parameters are often "
            "used to track or redirect victims.")

    if features.get("domain_name_length", 0) >= 25:
        add("low", "Long Domain Name",
            f"{features['domain_name_length']} characters. Long domains are often "
            "brand-impersonation attempts.")

    if status in ("Unknown", "Unreachable"):
        add("low", "Automated Verdict Limited",
            "The model could not issue a full verdict for this URL, so the indicators "
            "above should be reviewed manually.")

    ind.sort(key=lambda i: SEVERITY_ORDER.get(i["severity"], 9))
    return ind[:7]



# ==========================================
# 3. FEATURE EXTRACTION FUNCTION
# ==========================================
#
# Feature extraction lives in features.py so that the training pipeline and the
# live app use the exact same logic (see that module for the normalisation
# rules). extract_features / get_tld are imported at the top of this file.


# ==========================================
# 4. VERIFICATION LAYER
# ==========================================

HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)"
    r"[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
)


def validate_url(url):
    """
    Step 1: URL format & scheme validation.
    Returns (is_valid, reason).
    """
    try:
        parts = urlsplit(url)
    except ValueError as e:
        return False, f"URL could not be parsed ({e})"

    if parts.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported scheme '{parts.scheme}' (only http/https allowed)"

    hostname = parts.hostname
    if not hostname:
        return False, "URL has no hostname"

    # IP literal (e.g. http://192.168.1.1) is syntactically valid
    try:
        ipaddress.ip_address(hostname)
        return True, "Valid URL (IP-literal host)"
    except ValueError:
        pass

    # IDN domains (e.g. http://বাংলা.com) -> ASCII (punycode) form
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeEncodeError):
        return False, "Hostname contains invalid characters"

    if " " in ascii_host or not HOSTNAME_RE.match(ascii_host):
        return False, "Invalid hostname format"

    if "." not in ascii_host:
        return False, "Hostname must be a fully qualified domain (must contain a dot)"

    return True, "Valid URL format and scheme (http/https)"


def is_private_address(address):
    """True for loopback / private / link-local / reserved targets."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def probe_http(url, timeout=PROBE_TIMEOUT):
    """
    HTTP reachability probe.
    Returns (reachable: bool, detail: str).
    Any HTTP response (even 4xx/5xx) means the website IS reachable.
    Redirects are NOT followed (avoids SSRF via redirect chains).
    """
    # 1st attempt: HEAD
    try:
        resp = requests.head(
            url, timeout=timeout, allow_redirects=False, headers=PROBE_HEADERS
        )
        return True, f"HTTP {resp.status_code} (server responded)"
    except requests.exceptions.SSLError:
        # TLS handshake failed, but the server DID respond -> host is up
        return True, "Server responded with a TLS/SSL certificate error"
    except requests.exceptions.TooManyRedirects:
        return True, "Server responded (redirect limit exceeded)"
    except requests.exceptions.Timeout:
        return False, "Connection timed out"
    except requests.exceptions.ConnectionError:
        # Some servers reject HEAD -> retry once with GET
        try:
            resp = requests.get(
                url, timeout=timeout, allow_redirects=False,
                headers=PROBE_HEADERS, stream=True
            )
            resp.close()
            return True, f"HTTP {resp.status_code} (server responded, GET)"
        except requests.exceptions.SSLError:
            return True, "Server responded with a TLS/SSL certificate error"
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed ({type(e).__name__})"
    except requests.exceptions.RequestException as e:
        return False, f"HTTP probe failed ({type(e).__name__})"

    return False, "Unknown probe result"


def verify_domain(url, hostname, timeout=PROBE_TIMEOUT):
    """
    Step 2: Domain verification = DNS resolution + HTTP reachability probe.
    Returns dict: dns_resolved, reachable, private_network, detail.
    """
    result = {
        "dns_resolved": False,
        "reachable": False,
        "private_network": False,
        "detail": "",
    }

    # --- DNS resolution ---
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
        addresses = {info[4][0] for info in addrinfo}
        result["dns_resolved"] = True
    except socket.gaierror as e:
        result["detail"] = f"DNS resolution failed ({e.strerror or e})"
        return result
    except (socket.herror, OSError) as e:
        result["detail"] = f"DNS lookup error ({e})"
        return result

    # --- Private / local target: skip HTTP probe (safety) ---
    if any(is_private_address(a) for a in addresses):
        result["private_network"] = True
        result["reachable"] = True
        result["detail"] = "Domain resolves to a private/local address; HTTP probe skipped"
        return result

    # --- HTTP reachability probe ---
    reachable, detail = probe_http(url, timeout)
    result["reachable"] = reachable
    result["detail"] = detail
    return result


# ==========================================
# 5. ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("dashboard.html", active="dashboard")


@app.route("/scanner")
def scanner_page():
    return render_template("scanner.html", active="scanner")


@app.route("/xai")
def xai_page():
    return render_template("xai.html", active="xai")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html", active="analytics")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        url = data.get("url", "").strip()

        if not url:
            return jsonify({
                "error": "URL is required"
            }), 400

        # Add scheme if missing (a URL that already carries a scheme
        # like "ftp://" must NOT get "http://" prepended)
        if "://" not in url:
            url = "http://" + url

        # Verification state shared across all steps
        verification = {
            "url_valid": False,
            "url_detail": "",
            "dns_resolved": None,   # None = not checked yet
            "reachable": None,
            "domain_detail": "Not checked"
        }

        def unknown_response(reason, features=None):
            payload = {
                "url": url,
                "status": "Unknown",
                "status_reason": reason,
                "prediction": None,
                "confidence": None,
                "model_confidence": None,
                "phishing_probability": None,
                "legitimate_probability": None,
                "raw_phishing_probability": None,
                "raw_legitimate_probability": None,
                "risk_score": None,
                "indicators": build_indicators(features, "Unknown") if features else [],
                "explanation": None,
                "verification": verification,
                "features": features,
                "layers": [],
                "calibration": None,
            }
            try:
                record = record_scan(payload)
                payload["scan_id"] = record["id"]
                payload["timestamp"] = record["timestamp"]
            except Exception as rec_err:
                print(f"History record error: {rec_err}")
            return jsonify(payload)

        # ---- Step 1: URL Validation (format & scheme) ----
        valid, reason = validate_url(url)
        verification["url_detail"] = reason

        if not valid:
            return unknown_response(f"URL validation failed: {reason}")

        verification["url_valid"] = True

        hostname = urlsplit(url).hostname

        # ---- Step 2: Domain Verification (DNS + HTTP probe) ----
        domain_check = verify_domain(url, hostname)
        verification["dns_resolved"] = domain_check["dns_resolved"]
        verification["reachable"] = domain_check["reachable"]
        verification["domain_detail"] = domain_check["detail"]

        # ---- Step 3: Machine Learning Prediction (Random Forest) ----
        # The 16 features are purely URL-based, so the model runs even
        # when the site is unreachable (its verdict is shown separately).
        features = extract_features(url)
        if features is None:
            return unknown_response(
                "Feature extraction failed — insufficient information"
            )

        try:
            input_array = np.array([[features[f] for f in feature_names]])
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0]
        except Exception as model_error:
            return unknown_response(
                f"Model inference failed — insufficient information ({model_error})",
                features=features
            )

        model_prediction = "Phishing" if prediction == 0 else "Legitimate"
        raw_phishing = float(probability[0])
        raw_confidence = round(max(probability) * 100, 2)
        calibrated_phishing = calibrate(raw_phishing)

        # ---- Step 4: Post-model verification layers + final assessment ----
        # Reputation, WHOIS, TLS, page content, redirects and the top-domain
        # allowlist are combined with the calibrated model probability.  This is
        # what turns "the model thinks" into a verdict with evidence behind it.
        assessment = assess(
            url=url,
            hostname=hostname,
            raw_probability=raw_phishing,
            calibrated_probability=calibrated_phishing,
            reachable=bool(domain_check["reachable"]),
        )

        status = assessment["status"]
        confidence = assessment["confidence"]
        status_reason = assessment["reason"]

        tld = get_tld(hostname or "")

        indicators = build_indicators(features, status, tld)
        indicators.extend(assessment.get("extra_indicators") or [])
        indicators.sort(key=lambda i: SEVERITY_ORDER.get(i["severity"], 9))
        indicators = indicators[:9]

        result = {
            "url": url,
            "status": status,
            "status_reason": status_reason,
            "prediction": model_prediction,
            "confidence": confidence,
            "model_confidence": raw_confidence,
            # Main probabilities are calibrated; raw model output is kept too.
            "phishing_probability": round(calibrated_phishing * 100, 2),
            "legitimate_probability": round((1 - calibrated_phishing) * 100, 2),
            "raw_phishing_probability": round(raw_phishing * 100, 2),
            "raw_legitimate_probability": round((1 - raw_phishing) * 100, 2),
            "calibration": assessment["calibration"],
            "layers": assessment["layers"],
            # 0-100 risk score: combined phishing risk after all layers
            "risk_score": assessment["risk_score"],
            "indicators": indicators,
            "explanation": explain_prediction(features),
            "verification": verification,
            "features": features,
        }

        try:
            record = record_scan(result)
            result["scan_id"] = record["id"]
            result["timestamp"] = record["timestamp"]
        except Exception as rec_err:
            print(f"History record error: {rec_err}")

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# 5b. DASHBOARD / ANALYTICS DATA API
# ==========================================

DATASET_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"

_analytics_cache = None


def _read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"CSV read error ({path}): {e}")
        return []


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.route("/api/stats")
def api_stats():
    records = load_history()
    today = datetime.now()

    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    labels = [d.strftime("%a") for d in days]
    keys = [d.strftime("%Y-%m-%d") for d in days]

    series = {k: {"phishing": 0, "legitimate": 0, "other": 0} for k in keys}
    counts = {"Phishing": 0, "Legitimate": 0, "Unreachable": 0, "Unknown": 0}
    confidences = []

    for r in records:
        status = r.get("status")
        if status in counts:
            counts[status] += 1
        if r.get("confidence") is not None:
            confidences.append(r["confidence"])
        key = r.get("date")
        if key in series:
            if status == "Phishing":
                series[key]["phishing"] += 1
            elif status == "Legitimate":
                series[key]["legitimate"] += 1
            else:
                series[key]["other"] += 1

    total = len(records)
    classified = counts["Phishing"] + counts["Legitimate"]
    security_score = (
        round(counts["Legitimate"] / classified * 100) if classified else None
    )

    return jsonify({
        "total_scans": total,
        "phishing": counts["Phishing"],
        "legitimate": counts["Legitimate"],
        "unreachable": counts["Unreachable"],
        "unknown": counts["Unknown"],
        "avg_confidence": (
            round(sum(confidences) / len(confidences), 1) if confidences else None
        ),
        "security_score": security_score,
        "scans_last_7_days": sum(sum(v.values()) for v in series.values()),
        "activity": {
            "labels": labels,
            "phishing": [series[k]["phishing"] for k in keys],
            "legitimate": [series[k]["legitimate"] for k in keys],
            "other": [series[k]["other"] for k in keys],
        },
        "recent": records[:8],
        "has_data": total > 0,
    })


@app.route("/api/history")
def api_history():
    limit = request.args.get("limit", 20, type=int)
    limit = max(1, min(limit, HISTORY_LIMIT))
    records = load_history()
    return jsonify({"records": records[:limit], "total": len(records)})


@app.route("/api/analytics")
def api_analytics():
    global _analytics_cache
    if _analytics_cache is not None:
        return jsonify(_analytics_cache)

    models = []
    for row in _read_csv(RESULTS_DIR / "model_comparison.csv"):
        models.append({
            "model": row.get("Model"),
            "accuracy": _num(row.get("Accuracy")),
            "precision": _num(row.get("Precision")),
            "recall": _num(row.get("Recall")),
            "f1": _num(row.get("F1_Score")),
            "roc_auc": _num(row.get("ROC_AUC")),
            "train_time": _num(row.get("Train_Time_sec")),
            "predict_time": _num(row.get("Predict_Time_sec")),
        })

    confusion = []
    for row in _read_csv(RESULTS_DIR / "confusion_matrices.csv"):
        confusion.append({
            "model": row.get("Model"),
            "tn": int(_num(row.get("True_Negative"))),
            "fp": int(_num(row.get("False_Positive"))),
            "fn": int(_num(row.get("False_Negative"))),
            "tp": int(_num(row.get("True_Positive"))),
        })

    ablation = []
    for row in _read_csv(MODELS_DIR / "ablation_results.csv"):
        ablation.append({
            "name": row.get("Model"),
            "features": int(_num(row.get("Features"))),
            "accuracy": _num(row.get("Accuracy")),
            "macro_f1": _num(row.get("Macro_F1")),
            "precision": _num(row.get("Phishing_Precision")),
            "recall": _num(row.get("Phishing_Recall")),
        })

    feature_importance = []
    for row in _read_csv(MODELS_DIR / "feature_importance.csv"):
        name = row.get("Feature")
        feature_importance.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "importance": _num(row.get("Importance")),
        })
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    dataset_rows = 0
    try:
        with open(DATASET_FILE, encoding="utf-8") as f:
            dataset_rows = max(0, sum(1 for _ in f) - 1)
    except Exception as e:
        print(f"Dataset read error: {e}")

    calibration = None
    calibration_file = RESULTS_DIR / "calibration_report.json"
    try:
        if calibration_file.exists():
            with open(calibration_file, encoding="utf-8") as f:
                calibration = json.load(f)
    except Exception as e:
        print(f"Calibration report read error: {e}")

    _analytics_cache = {
        "models": models,
        "confusion": confusion,
        "ablation": ablation,
        "feature_importance": feature_importance,
        "dataset_rows": dataset_rows,
        "calibration": calibration,
        "images": sorted(p.name for p in RESULTS_DIR.glob("*.png")),
    }
    return jsonify(_analytics_cache)


@app.route("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


# ==========================================
# 6. RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
