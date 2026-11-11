
import pandas as pd
import numpy as np
import joblib
import tldextract
import math

from urllib.parse import urlsplit

# ==========================================
# LOAD MODEL AND DATA
# ==========================================

BASE_DIR = r"E:\21-22\7th Sem\Thesis\PhishGuard-AI"

model_data = joblib.load(BASE_DIR + r"\models\all_random_forest.joblib")
model = model_data["model"]
feature_names = model_data["features"]

df = pd.read_csv(BASE_DIR + r"\dataset\cleaned_dataset.csv")

print("=" * 60)
print("FALSE POSITIVE DIAGNOSIS")
print("=" * 60)

# ==========================================
# 1. CHECK HOW GOOGLE URL LOOKS IN DATASET
# ==========================================

# Find Google-like URLs in dataset
google_urls = df[df["URL"].str.contains("google", case=False, na=False)]
print(f"\nGoogle URLs in dataset: {len(google_urls)}")
print(f"Class distribution:")
print(google_urls["ClassLabel"].value_counts().sort_index())
print(f"  (0=Phishing, 1=Legitimate)")

# Show some examples
print("\nSample Google URLs from dataset:")
for _, row in google_urls.head(10).iterrows():
    label = "Phishing" if row["ClassLabel"] == 0 else "Legit"
    print(f"  [{label}] {row['URL'][:80]}")

# ==========================================
# 2. COMPARE FEATURES: DATASET vs APP EXTRACTION
# ==========================================

extractor = tldextract.TLDExtract(suffix_list_urls=())

def calculate_entropy(text):
    if not text:
        return 0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    length = len(text)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )
    return round(entropy, 4)

def extract_features_app(url):
    """Same as app.py"""
    try:
        parsed = urlsplit(url)
        url_length = len(url)
        path_length = len(parsed.path) if parsed.path else 0
        dot_count = url.count(".")
        hostname = parsed.hostname or ""
        subdomain_count = hostname.count(".")
        tld = extractor(hostname).suffix
        tld_length = len(tld) if tld else 0
        has_ip_address = 1 if hostname.replace(".", "").isdigit() else 0
        has_hyphen_in_domain = 1 if "-" in hostname else 0
        https_flag = 1 if parsed.scheme.lower() == "https" else 0
        numeric_count = sum(1 for c in url if c.isdigit())
        percentage_numeric_chars = round(numeric_count / url_length * 100, 4) if url_length > 0 else 0
        number_of_digits = numeric_count
        tokens = url.split("/")
        token_count = len(tokens)
        url_entropy = calculate_entropy(url)
        domain_name = extractor(hostname).domain
        domain_name_length = len(domain_name) if domain_name else 0
        suspicious_extensions = [".exe", ".zip", ".rar", ".bat", ".cmd", ".scr", ".js", ".php"]
        suspicious_file_extension = 1 if any(url.lower().endswith(ext) for ext in suspicious_extensions) else 0
        popular_tlds = [".com", ".org", ".net", ".edu", ".gov", ".io", ".co"]
        tld_popularity = 1 if any(url.lower().endswith(tld) for tld in popular_tlds) else 0
        query = parsed.query
        query_param_count = query.count("&") + 1 if query else 0

        return {
            "url_length": url_length,
            "path_length": path_length,
            "dot_count": dot_count,
            "subdomain_count": subdomain_count,
            "tld_length": tld_length,
            "has_ip_address": has_ip_address,
            "has_hyphen_in_domain": has_hyphen_in_domain,
            "https_flag": https_flag,
            "percentage_numeric_chars": percentage_numeric_chars,
            "number_of_digits": number_of_digits,
            "token_count": token_count,
            "url_entropy": url_entropy,
            "domain_name_length": domain_name_length,
            "suspicious_file_extension": suspicious_file_extension,
            "tld_popularity": tld_popularity,
            "query_param_count": query_param_count
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# ==========================================
# 3. TEST PROBLEMATIC URL
# ==========================================

test_url = "https://www.google.com/search?q=phishing"
print(f"\n{'=' * 60}")
print(f"TEST URL: {test_url}")
print(f"{'=' * 60}")

# Extract with app logic
app_features = extract_features_app(test_url)
print("\nFeatures (app.py extraction):")
for k, v in app_features.items():
    print(f"  {k}: {v}")

# Predict
input_array = np.array([[app_features[f] for f in feature_names]])
prediction = model.predict(input_array)[0]
probability = model.predict_proba(input_array)[0]

print(f"\nPrediction: {'Phishing' if prediction == 0 else 'Legitimate'}")
print(f"Phishing Prob: {probability[0]:.4f}")
print(f"Legit Prob: {probability[1]:.4f}")

# ==========================================
# 4. COMPARE WITH SIMILAR DATASET ROWS
# ==========================================

print(f"\n{'=' * 60}")
print("COMPARING WITH DATASET (google search URLs)")
print(f"{'=' * 60}")

# Find similar URLs in dataset
similar = df[df["URL"].str.contains("google.com/search", case=False, na=False)]
print(f"\nFound {len(similar)} similar URLs in dataset")

if len(similar) > 0:
    print("\nTheir features:")
    feature_cols = [f for f in feature_names if f in df.columns]
    
    print(f"\n{'Feature':<30} {'Our URL':>15} {'Dataset Avg (Legit)':>20} {'Dataset Avg (Phish)':>20}")
    print("-" * 90)
    
    legit_rows = similar[similar["ClassLabel"] == 1]
    phish_rows = similar[similar["ClassLabel"] == 0]
    
    for feat in feature_cols:
        our_val = app_features[feat]
        legit_avg = legit_rows[feat].mean() if len(legit_rows) > 0 else "N/A"
        phish_avg = phish_rows[feat].mean() if len(phish_rows) > 0 else "N/A"
        
        if isinstance(legit_avg, float):
            legit_avg = f"{legit_avg:.2f}"
        if isinstance(phish_avg, float):
            phish_avg = f"{phish_avg:.2f}"
            
        print(f"{feat:<30} {our_val:>15} {legit_avg:>20} {phish_avg:>20}")

# ==========================================
# 5. FEATURE IMPORTANCE FOR THIS DECISION
# ==========================================

print(f"\n{'=' * 60}")
print("FEATURE CONTRIBUTION ANALYSIS")
print(f"{'=' * 60}")

# Get feature importances
importances = model.feature_importances_
feature_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

print("\nTop features by importance:")
for feat, imp in feature_imp:
    print(f"  {feat:<30} {imp:.4f}")

# ==========================================
# 6. CHECK FEATURE EXTRACTION CONSISTENCY
# ==========================================

print(f"\n{'=' * 60}")
print("FEATURE EXTRACTION CONSISTENCY CHECK")
print(f"{'=' * 60}")

# Take a dataset row and re-extract features from its URL
sample = df.iloc[0]
print(f"\nDataset row URL: {sample['URL']}")
print(f"Dataset label: {sample['ClassLabel']}")

reextracted = extract_features_app(sample["URL"])

print(f"\n{'Feature':<30} {'Dataset Value':>15} {'Re-extracted':>15} {'Match':>10}")
print("-" * 75)

mismatches = 0
for feat in feature_names:
    if feat in df.columns:
        original = sample[feat]
        new_val = reextracted.get(feat, "N/A")
        match = "YES" if original == new_val else "NO"
        if match == "NO":
            mismatches += 1
        print(f"{feat:<30} {original:>15} {new_val:>15} {match:>10}")

print(f"\nTotal Mismatches: {mismatches}")
