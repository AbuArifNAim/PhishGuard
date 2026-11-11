
import sys
sys.path.insert(0, r"E:\21-22\7th Sem\Thesis\PhishGuard-AI")

from app import extract_features
import numpy as np
import joblib

model_data = joblib.load(r"E:\21-22\7th Sem\Thesis\PhishGuard-AI\models\all_random_forest.joblib")
model = model_data["model"]
feature_names = model_data["features"]

test_urls = [
    "https://github.com",
    "https://www.google.com/search?q=phishing",
    "http://paypa1-secure-login.verify-account.xyz/signin",
    "https://www.facebook.com",
    "http://myglobaltech.in/",
    "https://keraekken-loagginnusa.godaddysites.com/",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://stackoverflow.com/questions/12345",
]

print("=" * 70)
print("FIXED FEATURE EXTRACTION TEST")
print("=" * 70)

for url in test_urls:
    features = extract_features(url)
    input_array = np.array([[features[f] for f in feature_names]])
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0]

    result = "Phishing" if prediction == 0 else "Legitimate"
    print(f"\nURL: {url}")
    print(f"  Result: {result} (Phish: {prob[0]*100:.1f}%, Legit: {prob[1]*100:.1f}%)")
    print(f"  query_param_count = {features['query_param_count']}")
    print(f"  tld_popularity     = {features['tld_popularity']}")
    print(f"  subdomain_count    = {features['subdomain_count']}")
    print(f"  token_count        = {features['token_count']}")
