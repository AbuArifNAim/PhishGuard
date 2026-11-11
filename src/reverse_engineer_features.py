
import pandas as pd
import numpy as np
from urllib.parse import urlsplit
import tldextract

# ==========================================
# REVERSE-ENGINEER FEATURE EXTRACTION
# ==========================================

BASE_DIR = r"E:\21-22\7th Sem\Thesis\PhishGuard-AI"
df = pd.read_csv(BASE_DIR + r"\dataset\cleaned_dataset.csv")

extractor = tldextract.TLDExtract(suffix_list_urls=())

print("=" * 70)
print("REVERSE-ENGINEERING FEATURE EXTRACTION LOGIC")
print("=" * 70)

# ==========================================
# 1. ANALYZE subdomain_count
# ==========================================

print("\n1. SUBDOMAIN_COUNT ANALYSIS")
print("-" * 70)

samples = df.head(20)
print(f"{'URL':<60} {'hostname':<35} {'dots':>5} {'dataset':>8}")
print("-" * 110)

for _, row in samples.iterrows():
    url = row["URL"]
    dataset_val = row["subdomain_count"]
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        dots = hostname.count(".")
        print(f"{url[:58]:<60} {hostname[:33]:<35} {dots:>5} {dataset_val:>8}")
    except:
        pass

# Test hypothesis: subdomain_count = dots - 1?
print("\nHypothesis: subdomain_count = count of dots in hostname - 1")
matches = 0
total = 0
for _, row in df.head(1000).iterrows():
    try:
        parsed = urlsplit(row["URL"])
        hostname = parsed.hostname or ""
        dots = hostname.count(".")
        predicted = max(0, dots - 1)
        if predicted == row["subdomain_count"]:
            matches += 1
        total += 1
    except:
        pass

print(f"  Match rate (dots-1): {matches}/{total} = {matches/total*100:.1f}%")

# Alternative: subdomain count using tldextract
matches2 = 0
total2 = 0
for _, row in df.head(1000).iterrows():
    try:
        parsed = urlsplit(row["URL"])
        hostname = parsed.hostname or ""
        ext = extractor(hostname)
        # Count subdomain parts
        subdomain = ext.subdomain
        predicted = len(subdomain.split(".")) if subdomain else 0
        if predicted == row["subdomain_count"]:
            matches2 += 1
        total2 += 1
    except:
        pass

print(f"  Match rate (tldextract subdomain parts): {matches2}/{total2} = {matches2/total2*100:.1f}%")

# ==========================================
# 2. ANALYZE token_count
# ==========================================

print("\n\n2. TOKEN_COUNT ANALYSIS")
print("-" * 70)

print(f"{'URL':<70} {'dataset':>8} {'split(/)':>10} {'split(/+&)':>12}")
print("-" * 110)

for _, row in samples.head(15).iterrows():
    url = row["URL"]
    dataset_val = row["token_count"]
    
    # Hypothesis 1: split by /
    t1 = len(url.split("/"))
    
    # Hypothesis 2: split by / and &
    import re
    t2 = len(re.split(r'[/:&=?]', url))
    
    print(f"{url[:68]:<70} {dataset_val:>8} {t1:>10} {t2:>12}")

# Test different token hypotheses
hypotheses = {
    "split(/)": lambda u: len(u.split("/")),
    "split(/ and ?)": lambda u: len(re.split(r'[/?]', u)),
    "split(/ : & = ?)": lambda u: len(re.split(r'[/:&=?]', u)),
    "split(words)": lambda u: len(u.split()),
    "count path segments": lambda u: len([x for x in u.split("/") if x]),
}

print("\nToken count hypothesis testing:")
for name, func in hypotheses.items():
    matches = 0
    total = 0
    for _, row in df.head(500).iterrows():
        try:
            predicted = func(row["URL"])
            if predicted == row["token_count"]:
                matches += 1
            total += 1
        except:
            pass
    if total > 0:
        print(f"  {name:<30}: {matches}/{total} = {matches/total*100:.1f}%")

# ==========================================
# 3. ANALYZE tld_popularity
# ==========================================

print("\n\n3. TLD_POPULARITY ANALYSIS")
print("-" * 70)

print(f"{'URL':<70} {'tld':>8} {'dataset':>8}")
print("-" * 90)

for _, row in samples.head(15).iterrows():
    url = row["URL"]
    dataset_val = row["tld_popularity"]
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        ext = extractor(hostname)
        tld = ext.suffix
        print(f"{url[:68]:<70} {tld:>8} {dataset_val:>8}")
    except:
        pass

# What TLDs are popular in dataset?
print("\nTLD distribution for tld_popularity=1:")
pop_tlds = df[df["tld_popularity"] == 1]["URL"].apply(
    lambda u: extractor(urlsplit(u).hostname or "").suffix
).value_counts()
print(pop_tlds.head(15))

print("\nTLD distribution for tld_popularity=0:")
nonpop_tlds = df[df["tld_popularity"] == 0]["URL"].apply(
    lambda u: extractor(urlsplit(u).hostname or "").suffix
).value_counts()
print(nonpop_tlds.head(15))

# ==========================================
# 4. ANALYZE query_param_count
# ==========================================

print("\n\n4. QUERY_PARAM_COUNT ANALYSIS")
print("-" * 70)

print(f"{'URL':<80} {'dataset':>8}")
print("-" * 95)

for _, row in samples.head(15).iterrows():
    url = row["URL"]
    dataset_val = row["query_param_count"]
    print(f"{url[:78]:<80} {dataset_val:>8}")

# Test hypotheses
def q1(u):
    try:
        q = urlsplit(u).query
        return q.count("&") + 1 if q else 0
    except:
        return 0

def q2(u):
    try:
        q = urlsplit(u).query
        return len(q.split("&")) if q else 0
    except:
        return 0

def q3(u):
    return u.count("=")

print("\nQuery count hypotheses:")
for name, func in [("split & + 1", q1), ("split &", q2), ("count =", q3)]:
    matches = 0
    total = 0
    for _, row in df.head(500).iterrows():
        try:
            predicted = func(row["URL"])
            if predicted == row["query_param_count"]:
                matches += 1
            total += 1
        except:
            pass
    if total > 0:
        print(f"  {name:<25}: {matches}/{total} = {matches/total*100:.1f}%")

# ==========================================
# 5. ANALYZE url_entropy precision
# ==========================================

print("\n\n5. URL_ENTROPY PRECISION ANALYSIS")
print("-" * 70)

import math

def entropy_variants(text):
    if not text:
        return {}
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    length = len(text)
    
    # Natural log
    e_ln = -sum((count/length) * math.log(count/length) for count in freq.values())
    # Log2
    e_log2 = -sum((count/length) * math.log2(count/length) for count in freq.values())
    # Log10
    e_log10 = -sum((count/length) * math.log10(count/length) for count in freq.values())
    
    return {"ln": e_ln, "log2": e_log2, "log10": e_log10}

# Test which log base matches
for _, row in df.head(10).iterrows():
    url = row["URL"]
    dataset_val = row["url_entropy"]
    variants = entropy_variants(url)
    
    print(f"\nURL: {url[:60]}")
    print(f"  Dataset: {dataset_val}")
    for name, val in variants.items():
        match = "MATCH" if abs(val - dataset_val) < 0.001 else ""
        print(f"  {name}: {val:.6f} {match}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
