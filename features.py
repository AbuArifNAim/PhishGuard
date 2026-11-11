"""Single source of truth for URL feature extraction.

Both the training pipeline (``src/rebuild_dataset.py``) and the Flask app import
this module, so a model always sees exactly the features it was trained on.

Two normalisation fixes live here:

1. A bare trailing slash ("https://site.com/") is removed before features are
   computed, so ``path_length`` is 0 for the site root instead of 1.
2. ``tldextract`` is configured with ``include_psl_private_domains=True`` so
   hosting platforms (vercel.app, netlify.app, github.io, pages.dev, web.app,
   ...) are recognised as public suffixes. The platform is then not counted as
   a suspicious subdomain, and the effective TLD (``app``) is used for the
   TLD-length / TLD-popularity features.
3. ``www.`` is stripped from the URL before any feature is computed. It is a
   presentation prefix, and in the training data 99% of legitimate URLs start
   with it while only 0.5% of phishing URLs do — leaving it in taught the model
   spurious "no www = phishing" rules through the subdomain, dot, token and
   length features.
"""

import math
import re
from urllib.parse import urlsplit, urlunsplit

import tldextract


FEATURE_COLUMNS = [
    "url_length",
    "path_length",
    "dot_count",
    "subdomain_count",
    "tld_length",
    "has_ip_address",
    "has_hyphen_in_domain",
    "https_flag",
    "percentage_numeric_chars",
    "number_of_digits",
    "token_count",
    "url_entropy",
    "domain_name_length",
    "suspicious_file_extension",
    "tld_popularity",
    "query_param_count",
]

# Mainstream, low-abuse TLDs. `.app`, `.dev`, `.io`, `.co`, `.ai` and `.me` were
# added because modern legitimate sites and hosting platforms use them heavily.
# High-abuse TLDs (.xyz, .top, .site, .online, .club, .icu, .buzz, ...) stay out
# on purpose, so the feature keeps its signal.
POPULAR_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "io", "co", "app", "dev", "ai", "me", "tv",
}

SUSPICIOUS_EXTENSIONS = (
    ".exe", ".zip", ".rar", ".bat", ".cmd", ".scr", ".js", ".php",
)

_extractor = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)


def normalize_url(url):
    """Trim whitespace, drop a bare trailing slash and a leading ``www.``.

    ``www`` is dropped before any feature is computed (not just for the
    subdomain count). Otherwise ``dot_count``, ``token_count``, ``url_length``
    and ``url_entropy`` keep acting as hidden "has www" detectors, which made
    the model flag short, clean, bare domains as phishing.
    """
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url

    # Strip a leading "www." while keeping any userinfo and port intact.
    userinfo, sep, hostport = parts.netloc.rpartition("@")
    if hostport.lower().startswith("www."):
        hostport = hostport[4:]
    netloc = userinfo + sep + hostport

    path = "" if parts.path == "/" else parts.path
    return urlunsplit((parts.scheme, netloc, path, parts.query, parts.fragment))


def calculate_entropy(text):
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    length = len(text)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )
    return round(entropy, 4)


def get_tld(hostname):
    """Effective public TLD, e.g. "app" for atmoshuddhi.vercel.app."""
    suffix = _extractor(hostname or "").suffix
    return suffix.split(".")[-1] if suffix else ""


def extract_features(url):
    try:
        url = normalize_url(url)
        parsed = urlsplit(url)

        # Basic
        url_length = len(url)
        path_length = len(parsed.path) if parsed.path else 0
        dot_count = url.count(".")

        # Domain
        hostname = parsed.hostname or ""
        ext = _extractor(hostname)

        # Subdomains relative to the public suffix (hosting platforms included).
        # `www` is a presentation prefix, not a real subdomain: 99% of legitimate
        # URLs in the dataset start with it, so counting it made the model treat
        # "no subdomain" as a phishing signal.
        if hostname:
            parts = [p for p in (ext.subdomain or "").split(".") if p and p != "www"]
            subdomain_count = len(parts)
        else:
            subdomain_count = -1

        tld = ext.suffix.split(".")[-1] if ext.suffix else ""
        tld_length = len(tld)

        # IP address check
        has_ip_address = 1 if hostname.replace(".", "").isdigit() else 0

        # Hyphen in domain
        has_hyphen_in_domain = 1 if "-" in hostname else 0

        # HTTPS
        https_flag = 1 if parsed.scheme.lower() == "https" else 0

        # Numeric characters
        numeric_count = sum(1 for c in url if c.isdigit())
        percentage_numeric_chars = (
            round(numeric_count / url_length * 100, 4) if url_length > 0 else 0
        )
        number_of_digits = numeric_count

        # Tokens: split by / and .
        token_count = len(re.split(r"[/.]", url))

        # Entropy
        url_entropy = calculate_entropy(url)

        # Domain name length
        domain_name = ext.domain
        domain_name_length = len(domain_name) if domain_name else 0

        # Suspicious file extensions
        lowered = url.lower()
        suspicious_file_extension = 1 if lowered.endswith(SUSPICIOUS_EXTENSIONS) else 0

        # TLD popularity
        tld_popularity = 1 if tld in POPULAR_TLDS else 0

        # Query params: if no query -> 1, else len(query.split('&'))
        query = parsed.query
        query_param_count = len(query.split("&")) if query else 1

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
            "query_param_count": query_param_count,
        }

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None
