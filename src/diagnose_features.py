
import pandas as pd
from pathlib import Path
from urllib.parse import urlsplit

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 55)
print("PHISHGUARD AI - FEATURE DIAGNOSTICS")
print("=" * 55)

# 1. HTTPS feature vs label
print("\n1. HTTPS FLAG vs CLASS LABEL")
print(pd.crosstab(
    df["https_flag"],
    df["ClassLabel"],
    margins=True
))

# 2. Verify HTTPS flag against URL scheme
def get_https_flag(url):
    try:
        return int(urlsplit(str(url)).scheme.lower() == "https")
    except Exception:
        return -1

actual_https = df["URL"].apply(get_https_flag)

mismatch = actual_https != df["https_flag"]

print("\n2. HTTPS FLAG CONSISTENCY")
print("Total mismatches:", mismatch.sum())
print("Mismatch percentage:", round(mismatch.mean() * 100, 2), "%")

# 3. Inspect unusual feature values
print("\n3. SUBDOMAIN COUNT = -1")
subdomain_rows = df[df["subdomain_count"] == -1]
print("Rows:", len(subdomain_rows))
print(subdomain_rows["ClassLabel"].value_counts())

print("\n4. TLD LENGTH = 0")
tld_rows = df[df["tld_length"] == 0]
print("Rows:", len(tld_rows))
print(tld_rows["ClassLabel"].value_counts())

# 4. Feature averages by class
print("\n5. FEATURE MEANS BY CLASS")
features = [
    col for col in df.select_dtypes(include="number").columns
    if col != "ClassLabel"
]

print(
    df.groupby("ClassLabel")[features]
    .mean()
    .round(3)
    .T
)

# 5. Check remaining duplicate URLs
print("\n6. DUPLICATE URL CHECK")
print("Duplicate URLs:", df["URL"].duplicated().sum())

print("\nDiagnostics completed.")
