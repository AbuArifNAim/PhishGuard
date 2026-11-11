# Verification layers

The Random Forest in `features.py` / `models/all_random_forest.joblib` only sees
16 lexical URL features. `verification.py` adds independent evidence and turns
the model output into an explainable verdict.

```
URL
 ├─ 1. URL validation          scheme, hostname, IDN
 ├─ 2. DNS + HTTP probe        reachable? private target?
 ├─ 3. Random Forest           P(phishing) on 16 lexical features
 ├─ 4. Calibration             slope-only Platt scaling (no prior shift)
 ├─ 5. Reputation              Safe Browsing · VirusTotal · OpenPhish
 ├─ 6. WHOIS / RDAP            domain creation date and age
 ├─ 7. TLS                     certificate validity, issuer, expiry
 ├─ 8. Page content / DOM      password forms, off-domain actions, brands
 ├─ 9. Redirect chain          cross-domain hops, URL shorteners
 └─ 10. Tranco allowlist       false-positive guard for top domains
```

Every layer returns a `state` (`pass`, `fail`, `warn`, `info`, `skip`), a
human-readable `detail` and its numeric `influence` on the final risk score.
The scanner UI renders them in the **Verification Layers** card.

## How the verdict is produced

| Situation | Verdict |
|---|---|
| Any reputation feed reports a hit | **Phishing** (decisive) |
| Site unreachable, no reputation hit | **Unreachable** |
| Combined risk ≥ 70% | **Phishing** |
| Combined risk ≤ 30% | **Legitimate** |
| Combined risk 30–70% | **Unknown** (abstain) |

The combined risk starts from the calibrated model probability and is adjusted:

| Evidence | Adjustment |
|---|---|
| Domain registered < 30 days ago | +0.12 |
| Domain registered < 180 days ago | +0.04 |
| Domain registered ≥ 2 years ago | −0.05 |
| Invalid / expired TLS certificate | +0.15 |
| Valid TLS certificate | −0.03 |
| Password form posts to an off-domain host | +0.25 |
| Password form + brand impersonation | +0.20 |
| Password form present | +0.05 |
| Meta refresh / obfuscated JS | +0.03 |
| URL shortener in redirect chain | +0.05 |
| Cross-domain redirects | +0.05 |
| Domain is in Tranco top 100k (no hard phishing evidence) | capped at 0.25 |

A top-domain allowlist hit can only *pull a score down* to 25%; it never
overrides a reputation hit or a credential-harvesting form. This is what fixes
false positives such as `ahmedimtiaz.com`, while `paypal-secure-login-verify.xyz`
is still flagged.

## Configuration

All layers degrade gracefully when offline or unconfigured. Set these
environment variables before starting the app:

```powershell
$env:GSB_API_KEY = "..."          # Google Safe Browsing v4 (optional)
$env:VT_API_KEY  = "..."          # VirusTotal v3 (optional)
$env:PG_ALLOWLIST_MAX_RANK = "100000"
$env:PG_LAYER_CACHE_TTL = "21600" # seconds
.\venv\Scripts\python.exe app.py
```

Without API keys the reputation layer still uses the free OpenPhish feed.
Google Safe Browsing and VirusTotal then show `not_configured` and are ignored.

## Data files

| File | Purpose |
|---|---|
| `data/tranco_top1m.txt` | Tranco top-1M list (`rank,domain`), refresh with `src/update_tranco.py` |
| `data/brands.json` | brand → canonical domain map for impersonation checks |
| `models/calibration.joblib` | Platt coefficients used by `verification.calibrate` |
| `results/calibration_report.json` | Brier / log-loss / ECE before and after calibration |
| `results/calibration_curve.png` | reliability diagram shown on the Analytics page |

## Scripts

```powershell
# refresh the allowlist (downloads the latest Tranco list)
.\venv\Scripts\python.exe src\update_tranco.py

# fit calibration after retraining
.\venv\Scripts\python.exe src\calibrate_model.py

# model-only regression checks
.\venv\Scripts\python.exe src\regression_check.py

# full pipeline (network required)
.\venv\Scripts\python.exe src\verification_check.py

# rebuild → retrain → calibrate → regenerate every artefact
.\venv\Scripts\python.exe src\run_pipeline.py
```

## Design notes

* **Calibration has no intercept.** The dataset is 63% phishing, which is not
  the real-world browsing prior. Fitting an intercept shifted every score
  upward and created false positives; a slope-only fit corrects the forest's
  over-confidence while keeping 0.5 as the decision boundary.
* **Redirects are followed manually.** Each hop is resolved and rejected if it
  points at a private or loopback address, so the page fetcher cannot be used
  as an SSRF proxy.
* **WHOIS has an RDAP fallback.** Some TLDs (for example `.app`) have no
  reachable WHOIS server; `rdap.org` supplies the registration date instead.
* **Private suffixes are handled.** `evil.vercel.app` is not trusted just
  because `vercel.app` is in Tranco, and WHOIS lookups collapse hosting
  subdomains to the registry domain.
* **Results are cached in memory for 6 hours** (configurable), so repeat scans
  of the same host are fast.

## Limitations

* A lexical model cannot understand intent. A well-crafted phishing page on a
  clean, old domain with a valid certificate can still reach `Unknown`.
* Reputation feeds are lagging indicators; a brand-new campaign will not be
  listed yet.
* Page-content checks are heuristic (forms, brand strings, meta refresh,
  `eval`/`atob`). They are evidence, not proof.
* Google Safe Browsing / VirusTotal require API keys; without them the
  reputation layer only covers OpenPhish.
