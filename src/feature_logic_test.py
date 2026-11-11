
import pandas as pd
import re
from urllib.parse import urlsplit

BASE_DIR = r"E:\21-22\7th Sem\Thesis\PhishGuard-AI"
df = pd.read_csv(BASE_DIR + r"\dataset\legitphish.csv")

print("=" * 70)
print("PRECISE FEATURE REVERSE-ENGINEERING")
print("=" * 70)

# ==========================================
# 1. TOKEN_COUNT: Test split by / and .
# ==========================================

print("\n1. TOKEN_COUNT = len(re.split(r'[/.]', url)) ?")
print("-" * 70)

matches = 0
total = 0
for _, row in df.head(2000).iterrows():
    url = row["URL"]
    predicted = len(re.split(r'[/.]', url))
    if predicted == row["token_count"]:
        matches += 1
    total += 1

print(f"  Match rate: {matches}/{total} = {matches/total*100:.2f}%")

# Show mismatches
print("\n  Sample mismatches:")
shown = 0
for _, row in df.head(2000).iterrows():
    url = row["URL"]
    predicted = len(re.split(r'[/.]', url))
    if predicted != row["token_count"] and shown < 5:
        print(f"    URL: {url[:70]}")
        print(f"      dataset={row['token_count']}, predicted={predicted}")
        shown += 1

# ==========================================
# 2. SUBDOMAIN_COUNT: Test hostname parts - 2
# ==========================================

print("\n\n2. SUBDOMAIN_COUNT = hostname.split('.') - 2 ?")
print("-" * 70)

matches = 0
total = 0
for _, row in df.head(2000).iterrows():
    try:
        hostname = urlsplit(row["URL"]).hostname or ""
        predicted = max(0, len(hostname.split(".")) - 2)
        if predicted == row["subdomain_count"]:
            matches += 1
        total += 1
    except:
        pass

print(f"  Match rate: {matches}/{total} = {matches/total*100:.2f}%")

# Show mismatches
print("\n  Sample mismatches:")
shown = 0
for _, row in df.head(2000).iterrows():
    try:
        hostname = urlsplit(row["URL"]).hostname or ""
        predicted = max(0, len(hostname.split(".")) - 2)
        if predicted != row["subdomain_count"] and shown < 8:
            print(f"    hostname: {hostname}")
            print(f"      dataset={row['subdomain_count']}, predicted={predicted}")
            shown += 1
    except:
        pass

# ==========================================
# 3. QUERY_PARAM_COUNT: What gives 1 for no query?
# ==========================================

print("\n\n3. QUERY_PARAM_COUNT analysis")
print("-" * 70)

# Check: what's the distribution?
print("  Distribution:")
print(df["query_param_count"].value_counts().sort_index().head(10))

# Check: URLs with no query string
no_query = df[~df["URL"].str.contains(r"\?", regex=True, na=False)]
print(f"\n  URLs without '?': {len(no_query)}")
print(f"  Their query_param_count values:")
print(no_query["query_param_count"].value_counts().sort_index())

# Check: URLs with query string
with_query = df[df["URL"].str.contains(r"\?", regex=True, na=False)]
print(f"\n  URLs with '?': {len(with_query)}")
print(f"  Their query_param_count values:")
print(with_query["query_param_count"].value_counts().sort_index())

# Hypothesis: query_param_count = query.count("=") + 1? or 1 if no query?
print("\n  Testing: query_param_count = len(query.split('&')) if query else 1")
matches = 0
total = 0
for _, row in df.head(2000).iterrows():
    try:
        query = urlsplit(row["URL"]).query
        predicted = len(query.split("&")) if query else 1
        if predicted == row["query_param_count"]:
            matches += 1
        total += 1
    except:
        pass
print(f"    Match rate: {matches}/{total} = {matches/total*100:.2f}%")

# Hypothesis: count = in query + 1?
print("\n  Testing: query_param_count = query.count('=') + 1")
matches = 0
total = 0
for _, row in df.head(2000).iterrows():
    try:
        query = urlsplit(row["URL"]).query
        predicted = query.count("=") + 1
        if predicted == row["query_param_count"]:
            matches += 1
        total += 1
    except:
        pass
print(f"    Match rate: {matches}/{total} = {matches/total*100:.2f}%")

# Hypothesis: always 1 minimum
print("\n  Testing: query_param_count = max(1, len(query.split('&')))")
matches = 0
total = 0
for _, row in df.head(2000).iterrows():
    try:
        query = urlsplit(row["URL"]).query
        predicted = max(1, len(query.split("&")))
        if predicted == row["query_param_count"]:
            matches += 1
        total += 1
    except:
        pass
print(f"    Match rate: {matches}/{total} = {matches/total*100:.2f}%")

# ==========================================
# 4. TLD_POPULARITY: exact list
# ==========================================

print("\n\n4. TLD_POPULARITY = TLD in [com, org, net, edu, gov] ?")
print("-" * 70)

import tldextract
extractor = tldextract.TLDExtract(suffix_list_urls=())

matches = 0
total = 0
popular_tlds = {"com", "org", "net", "edu", "gov"}
for _, row in df.head(2000).iterrows():
    try:
        hostname = urlsplit(row["URL"]).hostname or ""
        ext = extractor(hostname)
        tld = ext.suffix
        predicted = 1 if tld in popular_tlds else 0
        if predicted == row["tld_popularity"]:
            matches += 1
        total += 1
    except:
        pass
print(f"  Match rate: {matches}/{total} = {matches/total*100:.2f}%")

# ==========================================
# 5. VERIFY other features match app.py
# ==========================================

print("\n\n5. OTHER FEATURES verification")
print("-" * 70)

import math

def calc_entropy(text):
    if not text:
        return 0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    return -sum((count/length) * math.log2(count/length) for count in freq.values())

# Test url_length
matches = sum(1 for _, row in df.head(1000).iterrows() if len(row["URL"]) == row["url_length"])
print(f"  url_length: {matches}/1000 = {matches/10:.1f}%")

# Test url_entropy
matches = 0
for _, row in df.head(1000).iterrows():
    e = calc_entropy(row["URL"])
    if abs(e - row["url_entropy"]) < 0.01:
        matches += 1
print(f"  url_entropy: {matches}/1000 = {matches/10:.1f}%")

# Test path_length
matches = 0
for _, row in df.head(1000).iterrows():
    path = urlsplit(row["URL"]).path
    plen = len(path) if path else 0
    if plen == row["path_length"]:
        matches += 1
print(f"  path_length: {matches}/1000 = {matches/10:.1f}%")

# Test tld_length
matches = 0
for _, row in df.head(1000).iterrows():
    hostname = urlsplit(row["URL"]).hostname or ""
    ext = extractor(hostname)
    tlen = len(ext.suffix) if ext.suffix else 0
    if tlen == row["tld_length"]:
        matches += 1
print(f"  tld_length: {matches}/1000 = {matches/10:.1f}%")

# Test domain_name_length
matches = 0
for _, row in df.head(1000).iterrows():
    hostname = urlsplit(row["URL"]).hostname or ""
    ext = extractor(hostname)
    dlen = len(ext.domain) if ext.domain else 0
    if dlen == row["domain_name_length"]:
        matches += 1
print(f"  domain_name_length: {matches}/1000 = {matches/10:.1f}%")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
