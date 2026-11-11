"""Download the latest Tranco top-1M domain list.

The scanner uses the list as a false-positive guard: a domain in the top ranks
is very unlikely to be a phishing site, so the verification layer refuses to
issue a Phishing verdict for it on lexical evidence alone.

    .\\venv\\Scripts\\python.exe src\\update_tranco.py

The result is written to ``data/tranco_top1m.txt`` (``rank,domain`` per line).
The app only loads the first ``PG_ALLOWLIST_MAX_RANK`` entries (default 100k),
so the file can stay in place without slowing startup.
"""

import gzip
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = BASE_DIR / "data" / "tranco_top1m.txt"
API_URL = "https://tranco-list.eu/api/lists/date/latest"


def main():
    print("Fetching Tranco list metadata ...")
    meta = requests.get(API_URL, timeout=30).json()
    download = meta.get("download")
    if not download:
        print("No download URL returned by the Tranco API.")
        return 1

    print(f"List {meta.get('list_id')} created {meta.get('created_on')}")
    print(f"Downloading {download} ...")
    response = requests.get(download, timeout=180)
    response.raise_for_status()

    payload = response.content
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)

    text = payload.decode("utf-8", "ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Saved {len(lines):,} domains to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
