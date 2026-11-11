"""End-to-end regression checks for the verification-layer pipeline.

Unlike ``regression_check.py`` (which only tests the Random Forest), this script
runs the same decision logic as the live app: model → calibration → reputation,
WHOIS, TLS, page content, redirects and allowlist.  It needs network access.

    .\\venv\\Scripts\\python.exe src\\verification_check.py

Cases that legitimately go offline are accepted as ``Unreachable``.
"""

import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from features import extract_features  # noqa: E402
from verification import assess, calibrate  # noqa: E402

HEADERS = {"User-Agent": "PhishGuard-AI/2.0 (regression-check)"}

# (url, acceptable statuses)
CASES = [
    ("https://ahmedimtiaz.com", {"Legitimate"}),
    ("https://atmoshuddhi.vercel.app/", {"Legitimate"}),
    ("https://example.com", {"Legitimate"}),
    ("https://www.google.com/", {"Legitimate"}),
    ("https://github.com/login", {"Legitimate"}),
    ("https://stripe.com", {"Legitimate"}),
    ("https://www.prothomalo.com/", {"Legitimate"}),
    ("http://testsite.xyz/", {"Phishing", "Unreachable"}),
    ("http://paypal-secure-login-verify.xyz/account/confirm.php?id=1",
     {"Phishing", "Unreachable"}),
]


def is_reachable(url):
    host = urlsplit(url).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
    except OSError:
        return False

    for method in (requests.head, requests.get):
        try:
            response = method(url, timeout=6, allow_redirects=False,
                              headers=HEADERS, stream=True)
            response.close()
            return True
        except requests.exceptions.SSLError:
            return True
        except Exception:
            continue
    return False


def main():
    data = joblib.load(BASE_DIR / "models" / "all_random_forest.joblib")
    model, feature_names = data["model"], data["features"]

    failures = 0
    print(f"{'URL':<58} {'raw':>7} {'cal':>7} {'status':<12} {'risk':>5} result")
    print("-" * 108)

    for url, expected in CASES:
        features = extract_features(url)
        if features is None:
            print(f"{url[:56]:<58} {'-':>7} {'-':>7} {'-':<12} {'-':>5} FAIL (no features)")
            failures += 1
            continue

        row = [[features[name] for name in feature_names]]
        raw = float(model.predict_proba(row)[0][0])
        calibrated = calibrate(raw)
        reachable = is_reachable(url)

        result = assess(url, urlsplit(url).hostname, raw, calibrated, reachable)
        ok = result["status"] in expected
        failures += 0 if ok else 1

        print(
            f"{url[:56]:<58} {raw*100:6.1f}% {calibrated*100:6.1f}% "
            f"{result['status']:<12} {result['risk_score']:>5} "
            f"{'PASS' if ok else 'FAIL (expected ' + '/'.join(sorted(expected)) + ')'}"
        )
        if not ok:
            print(f"    reason: {result['reason']}")
            for layer in result["layers"]:
                print(f"    [{layer['state']}] {layer['name']}: {layer['detail'][:100]}")

    print("-" * 108)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
