"""Post-model verification layers for PhishGuard AI.

The Random Forest only sees 16 lexical URL features.  This module adds the
independent evidence a security analyst would collect next, and combines it with
the calibrated model probability into one explainable verdict:

1. Reputation      Google Safe Browsing, VirusTotal and the OpenPhish feed.
2. WHOIS           domain creation date / age (raw WHOIS over TCP port 43).
3. TLS             certificate validity, issuer and expiry.
4. Page content    credential forms, off-domain form actions, brand spoofing.
5. Redirect chain  cross-domain hops and URL shorteners.
6. Allowlist       Tranco top-domain membership (downloaded by
                   ``src/update_tranco.py``; a small seed list is always active).

Every layer is optional and degrades gracefully: if a network call fails the
layer reports ``skip``/``info`` instead of failing the scan.  Results are cached
in memory so repeated scans of the same host stay fast.

Environment variables
---------------------
``GSB_API_KEY`` / ``GOOGLE_SAFE_BROWSING_API_KEY``
    Enables Google Safe Browsing lookups.
``VT_API_KEY`` / ``VIRUSTOTAL_API_KEY``
    Enables VirusTotal domain lookups.
``PG_LAYER_CACHE_TTL``       cache lifetime in seconds (default 21600 = 6 h).
``PG_ALLOWLIST_MAX_RANK``    largest Tranco rank treated as trusted (default 100000).
``PG_WHOIS_TIMEOUT``, ``PG_TLS_TIMEOUT``, ``PG_HTTP_TIMEOUT``
    Per-layer network timeouts in seconds.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
import tldextract

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CACHE_TTL = int(os.getenv("PG_LAYER_CACHE_TTL", str(6 * 3600)))
ALLOWLIST_MAX_RANK = int(os.getenv("PG_ALLOWLIST_MAX_RANK", "100000"))
WHOIS_TIMEOUT = float(os.getenv("PG_WHOIS_TIMEOUT", "5"))
TLS_TIMEOUT = float(os.getenv("PG_TLS_TIMEOUT", "5"))
HTTP_TIMEOUT = float(os.getenv("PG_HTTP_TIMEOUT", "6"))
LAYER_TIMEOUT = float(os.getenv("PG_LAYER_TIMEOUT", "14"))

ABSTAIN_LOW = 0.30
ABSTAIN_HIGH = 0.70

USER_AGENT = "Mozilla/5.0 (compatible; PhishGuard-AI/2.0; +verification-layer)"
PROBE_HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"}

OPENPHISH_FEED = "https://openphish.com/feed.txt"
GSB_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
VT_DOMAIN_ENDPOINT = "https://www.virustotal.com/api/v3/domains/{host}"

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc", "s.id",
    "lnkd.in", "t.ly", "v.gd", "qr.ae", "adf.ly", "shorte.st",
}

# Always trusted even when the Tranco file has not been downloaded.
SEED_TRUSTED = {
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "twitter.com",
    "x.com", "instagram.com", "linkedin.com", "microsoft.com", "apple.com",
    "amazon.com", "netflix.com", "github.com", "gitlab.com", "stackoverflow.com",
    "cloudflare.com", "mozilla.org", "paypal.com", "stripe.com", "vercel.com",
    "netlify.app", "openai.com", "bbc.co.uk", "nytimes.com", "prothomalo.com",
    "example.com", "example.org", "example.net",
}

_extractor = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)

# WHOIS only knows ICANN-registered domains, so private suffixes (vercel.app,
# github.io, ...) must be collapsed to their registry domain for lookups.
_icann_extractor = tldextract.TLDExtract(suffix_list_urls=())


# ==========================================================
# Small helpers
# ==========================================================

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key):
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if not item:
            return None
        stamp, value = item
        if time.time() - stamp > CACHE_TTL:
            _CACHE.pop(key, None)
            return None
        return value


def _cache_set(key, value):
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


def _cached(key, producer):
    hit = _cache_get(key)
    if hit is not None:
        return hit
    value = producer()
    _cache_set(key, value)
    return value


def registrable_domain(hostname):
    """Return the registrable domain, honouring private suffixes.

    ``sites.google.com`` -> ``google.com``;
    ``evil.vercel.app`` -> ``evil.vercel.app`` (private PSL entry).
    """
    host = (hostname or "").strip().strip(".").lower()
    if not host:
        return ""
    ext = _extractor(host)
    return ext.top_domain_under_public_suffix or host


def icann_domain(hostname):
    """Registry domain used for WHOIS (private suffixes collapsed).

    ``atmoshuddhi.vercel.app`` -> ``vercel.app`` so the lookup asks the
    registry that actually holds the registration.
    """
    host = (hostname or "").strip().strip(".").lower()
    if not host:
        return ""
    ext = _icann_extractor(host)
    return ext.top_domain_under_public_suffix or host


def is_private_ip(value):
    try:
        ip = ipaddress.ip_address(value)
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


def _host_is_public(hostname):
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, OSError):
        return False
    addresses = {info[4][0] for info in infos}
    return bool(addresses) and not any(is_private_ip(a) for a in addresses)


def _normalise_url_for_match(url):
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{host}{path}?{parts.query}" if parts.query else f"{parts.scheme.lower()}://{host}{path}"


# ==========================================================
# 1. Top-domain allowlist (Tranco)
# ==========================================================

_ALLOWLIST = None
_ALLOWLIST_LOCK = threading.Lock()


def _load_allowlist():
    """Load ``data/tranco_top1m.txt`` (``rank,domain`` per line)."""
    global _ALLOWLIST
    if _ALLOWLIST is not None:
        return _ALLOWLIST

    with _ALLOWLIST_LOCK:
        if _ALLOWLIST is not None:
            return _ALLOWLIST

        table = {}
        path = DATA_DIR / "tranco_top1m.txt"
        source = "seed list"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split(",", 1)
                        if len(parts) != 2:
                            continue
                        try:
                            rank = int(parts[0])
                        except ValueError:
                            continue
                        if rank > ALLOWLIST_MAX_RANK:
                            break
                        table[parts[1].strip().lower()] = rank
                source = f"Tranco top {ALLOWLIST_MAX_RANK:,}"
                print(f"Allowlist loaded: {len(table):,} domains ({source})")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Allowlist load error: {exc}")

        for domain in SEED_TRUSTED:
            table.setdefault(domain, 0)

        _ALLOWLIST = {"domains": table, "source": source}
        return _ALLOWLIST


def check_allowlist(hostname):
    """Check the host (and its registrable domain) against the allowlist."""
    data = _load_allowlist()
    domains = data["domains"]
    host = (hostname or "").lower().strip(".")
    root = registrable_domain(host)

    rank = domains.get(host)
    matched = host if rank is not None else None
    if rank is None and root:
        rank = domains.get(root)
        matched = root if rank is not None else None

    if rank is None:
        return {
            "key": "allowlist",
            "name": "Top-domain allowlist",
            "state": "info",
            "detail": f"{root or host} is not in the {data['source']} (neutral signal).",
            "listed": False,
            "rank": None,
            "domain": root or host,
        }

    label = f"rank #{rank:,}" if rank else "seed trusted domain"
    return {
        "key": "allowlist",
        "name": "Top-domain allowlist",
        "state": "pass",
        "detail": f"{matched} is a well-known domain ({label}, {data['source']}).",
        "listed": True,
        "rank": rank,
        "domain": matched,
    }


# ==========================================================
# 2. Reputation (Google Safe Browsing / VirusTotal / OpenPhish)
# ==========================================================

_OPENPHISH = {"stamp": 0.0, "hosts": set(), "urls": set()}
_OPENPHISH_LOCK = threading.Lock()


def _load_openphish():
    now = time.time()
    with _OPENPHISH_LOCK:
        if now - _OPENPHISH["stamp"] < 3600:
            return _OPENPHISH

        hosts, urls = set(), set()
        try:
            resp = requests.get(OPENPHISH_FEED, timeout=HTTP_TIMEOUT, headers=PROBE_HEADERS)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                urls.add(_normalise_url_for_match(line))
                host = (urlsplit(line).hostname or "").lower()
                if host:
                    hosts.add(host)
            print(f"OpenPhish feed loaded: {len(urls):,} URLs")
        except Exception as exc:
            print(f"OpenPhish feed unavailable: {exc}")

        _OPENPHISH.update({"stamp": now, "hosts": hosts, "urls": urls})
        return _OPENPHISH


def _gsb_check(url):
    key = os.getenv("GSB_API_KEY") or os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")
    if not key:
        return {"name": "Google Safe Browsing", "state": "not_configured",
                "detail": "No API key configured (set GSB_API_KEY)."}

    body = {
        "client": {"clientId": "phishguard-ai", "clientVersion": "2.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE",
                            "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = requests.post(GSB_ENDPOINT, params={"key": key}, json=body, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        matches = resp.json().get("matches") or []
        if matches:
            kinds = sorted({m.get("threatType", "THREAT") for m in matches})
            return {"name": "Google Safe Browsing", "state": "hit",
                    "detail": f"Listed as {', '.join(kinds)}."}
        return {"name": "Google Safe Browsing", "state": "clean",
                "detail": "No threat matches."}
    except Exception as exc:
        return {"name": "Google Safe Browsing", "state": "unavailable",
                "detail": f"Lookup failed ({type(exc).__name__})."}


def _vt_check(hostname):
    key = os.getenv("VT_API_KEY") or os.getenv("VIRUSTOTAL_API_KEY")
    if not key:
        return {"name": "VirusTotal", "state": "not_configured",
                "detail": "No API key configured (set VT_API_KEY)."}

    try:
        resp = requests.get(
            VT_DOMAIN_ENDPOINT.format(host=hostname),
            headers={"x-apikey": key, "User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 404:
            return {"name": "VirusTotal", "state": "unknown",
                    "detail": "Domain is not in the VirusTotal dataset."}
        resp.raise_for_status()
        stats = (resp.json().get("data", {}).get("attributes", {})
                 .get("last_analysis_stats", {}))
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        if malicious or suspicious:
            return {"name": "VirusTotal", "state": "hit",
                    "detail": f"{malicious} malicious / {suspicious} suspicious engine verdicts."}
        return {"name": "VirusTotal", "state": "clean",
                "detail": "No malicious engine verdicts."}
    except Exception as exc:
        return {"name": "VirusTotal", "state": "unavailable",
                "detail": f"Lookup failed ({type(exc).__name__})."}


def _openphish_check(url, hostname):
    feed = _load_openphish()
    if not feed["urls"] and not feed["hosts"]:
        return {"name": "OpenPhish", "state": "unavailable",
                "detail": "Feed could not be downloaded."}

    host = (hostname or "").lower()
    if _normalise_url_for_match(url) in feed["urls"] or host in feed["hosts"]:
        return {"name": "OpenPhish", "state": "hit",
                "detail": "URL or host appears in the OpenPhish phishing feed."}
    return {"name": "OpenPhish", "state": "clean",
            "detail": "Not present in the current OpenPhish feed."}


def reputation_check(url, hostname):
    def produce():
        sources = [_gsb_check(url), _vt_check(hostname), _openphish_check(url, hostname)]
        hits = [s for s in sources if s["state"] == "hit"]
        configured = [s for s in sources if s["state"] in ("hit", "clean")]

        if hits:
            detail = "; ".join(f"{s['name']}: {s['detail']}" for s in hits)
            state = "fail"
        elif configured:
            detail = "; ".join(f"{s['name']}: clean" for s in configured)
            state = "pass"
        else:
            detail = "No reputation source is configured or reachable."
            state = "skip"

        return {
            "key": "reputation",
            "name": "Reputation feeds",
            "state": state,
            "detail": detail,
            "malicious": bool(hits),
            "hits": [s["name"] for s in hits],
            "sources": sources,
        }

    return _cached(f"reputation:{_normalise_url_for_match(url)}", produce)


# ==========================================================
# 3. WHOIS domain age
# ==========================================================

_WHOIS_SERVERS: dict[str, str] = {}
_WHOIS_SERVER_LOCK = threading.Lock()

_CREATION_PATTERNS = [
    r"(?im)^\s*creation date\s*:\s*(.+)$",
    r"(?im)^\s*created(?: on)?\s*:\s*(.+)$",
    r"(?im)^\s*registered on\s*:\s*(.+)$",
    r"(?im)^\s*registration (?:date|time)\s*:\s*(.+)$",
    r"(?im)^\s*domain registration date\s*:\s*(.+)$",
    r"(?im)^\s*registered\s*:\s*(.+)$",
    r"(?im)^\s*activation\s*:\s*(.+)$",
    r"(?im)^\s*issued\s*:\s*(.+)$",
]


def _whois_query_raw(server, query, timeout=WHOIS_TIMEOUT):
    with socket.create_connection((server, 43), timeout=timeout) as sock:
        sock.sendall((query + "\r\n").encode("utf-8", "ignore"))
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks).decode("utf-8", "ignore")


def _whois_server(tld):
    with _WHOIS_SERVER_LOCK:
        if tld in _WHOIS_SERVERS:
            return _WHOIS_SERVERS[tld]

    server = None
    try:
        text = _whois_query_raw("whois.iana.org", tld, timeout=4)
        match = re.search(r"(?im)^(?:whois|refer)[ \t]*:[ \t]*(\S+)", text)
        if match:
            server = match.group(1).strip()
    except Exception:
        pass

    if not server:
        server = f"whois.nic.{tld}"

    with _WHOIS_SERVER_LOCK:
        _WHOIS_SERVERS[tld] = server
    return server


def _parse_whois_date(value):
    try:
        from dateutil import parser as date_parser
    except Exception:  # pragma: no cover
        date_parser = None

    cleaned = re.split(r"\s*[\(\[]", value.strip())[0].strip().strip(".")
    if not cleaned:
        return None

    if date_parser is not None:
        try:
            parsed = date_parser.parse(cleaned, fuzzy=False)
            if parsed.tzinfo:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            if 1985 <= parsed.year <= datetime.utcnow().year + 1:
                return parsed
        except Exception:
            pass

    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(cleaned[:11], fmt)
            if 1985 <= parsed.year <= datetime.utcnow().year + 1:
                return parsed
        except ValueError:
            continue
    return None


def _extract_creation(text):
    for pattern in _CREATION_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue
        parsed = _parse_whois_date(match.group(1))
        if parsed:
            return parsed
    return None


def _rdap_lookup(domain):
    """RDAP fallback for TLDs whose WHOIS server is unavailable (e.g. .app)."""
    result = {"created": None, "registrar": None, "available": None}
    try:
        response = requests.get(
            f"https://rdap.org/domain/{domain}",
            timeout=WHOIS_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "application/rdap+json"},
        )
    except Exception:
        return result

    if response.status_code == 404:
        result["available"] = True
        return result
    if response.status_code != 200:
        return result

    try:
        data = response.json()
    except Exception:
        return result

    for event in data.get("events") or []:
        action = (event.get("eventAction") or "").lower()
        if action in ("registration", "registered", "created"):
            created = _parse_whois_date(event.get("eventDate", ""))
            if created:
                result["created"] = created
                break

    for entity in data.get("entities") or []:
        if "registrar" not in (entity.get("roles") or []):
            continue
        vcard = entity.get("vcardArray") or [None, []]
        try:
            for item in vcard[1]:
                if item[0] == "fn":
                    result["registrar"] = str(item[3])[:120]
                    break
        except Exception:
            pass
        if result["registrar"]:
            break

    return result


def whois_lookup(hostname):
    domain = icann_domain(hostname)

    def produce():
        result = {
            "key": "whois",
            "name": "WHOIS domain age",
            "state": "info",
            "detail": "WHOIS data unavailable.",
            "checked": False,
            "available": None,
            "created": None,
            "age_days": None,
            "age_years": None,
            "registrar": None,
            "expires": None,
        }

        if not domain or "." not in domain:
            return result

        tld = domain.rsplit(".", 1)[-1]
        text = ""
        whois_error = None
        try:
            server = _whois_server(tld)
            if server:
                text = _whois_query_raw(server, domain)
        except Exception as exc:
            whois_error = type(exc).__name__

        if text:
            result["checked"] = True

            if re.search(r"(?i)(no match|not found|no data found|domain not found|no entries found)", text):
                result.update({
                    "state": "fail",
                    "available": True,
                    "detail": f"{domain} is not registered (available for registration).",
                })
                return result

            registrar_match = re.search(r"(?im)^\s*registrar\s*:\s*(.+)$", text)
            expiry_match = re.search(r"(?im)^\s*(?:registry expiry date|expiry date|expiration date)\s*:\s*(.+)$", text)
            if registrar_match:
                result["registrar"] = registrar_match.group(1).strip()[:120]
            if expiry_match:
                expires = _parse_whois_date(expiry_match.group(1))
                if expires:
                    result["expires"] = expires.strftime("%Y-%m-%d")

        created = _extract_creation(text) if text else None

        # RDAP is the fallback for TLDs without a reachable WHOIS server
        # (for example .app) and for registries that omit the creation date.
        if not created:
            rdap = _rdap_lookup(domain)
            result["checked"] = True
            if rdap.get("available"):
                result.update({
                    "state": "fail",
                    "available": True,
                    "detail": f"{domain} is not registered (available for registration).",
                })
                return result
            created = rdap.get("created")
            if not result["registrar"]:
                result["registrar"] = rdap.get("registrar")

        if not created:
            if whois_error:
                result["detail"] = f"WHOIS/RDAP lookup failed ({whois_error})."
            else:
                result["detail"] = f"{domain} is registered, but no creation date could be retrieved."
            return result

        age_days = max(0, (datetime.utcnow() - created).days)
        result.update({
            "created": created.strftime("%Y-%m-%d"),
            "age_days": age_days,
            "age_years": round(age_days / 365.25, 1),
        })

        if age_days < 30:
            result["state"] = "fail"
            result["detail"] = f"{domain} was registered {age_days} day(s) ago — very new domains are a strong phishing signal."
        elif age_days < 180:
            result["state"] = "warn"
            result["detail"] = f"{domain} was registered {age_days} days ago (under 6 months) — worth caution."
        elif age_days >= 730:
            result["state"] = "pass"
            result["detail"] = f"{domain} has been registered for {result['age_years']} years."
        else:
            result["detail"] = f"{domain} was registered {age_days} days ago."
        return result

    return _cached(f"whois:{domain}", produce)


# ==========================================================
# 4. TLS certificate
# ==========================================================

def _cert_name(name):
    try:
        return ", ".join(f"{key}={value}" for part in name for key, value in part)
    except Exception:
        return ""


def tls_check(hostname, port=443):
    def produce():
        result = {
            "key": "tls",
            "name": "TLS certificate",
            "state": "info",
            "detail": "TLS not checked.",
            "checked": False,
            "valid": None,
            "issuer": None,
            "subject": None,
            "not_before": None,
            "not_after": None,
            "days_left": None,
        }

        if not hostname:
            return result

        context = ssl.create_default_context()
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
        try:
            with socket.create_connection((hostname, port), timeout=TLS_TIMEOUT) as raw:
                with context.wrap_socket(raw, server_hostname=hostname) as tls:
                    cert = tls.getpeercert()
        except ssl.SSLCertVerificationError as exc:
            result.update({
                "checked": True,
                "valid": False,
                "state": "fail",
                "detail": f"Certificate validation failed: {getattr(exc, 'verify_message', None) or exc}",
            })
            return result
        except (socket.timeout, TimeoutError):
            result.update({"checked": True, "detail": "TLS handshake timed out."})
            return result
        except (socket.gaierror, ConnectionRefusedError, OSError) as exc:
            result.update({
                "checked": True,
                "detail": f"TLS handshake failed ({type(exc).__name__}).",
            })
            return result

        not_after = cert.get("notAfter")
        not_before = cert.get("notBefore")
        days_left = None
        if not_after:
            try:
                seconds = ssl.cert_time_to_seconds(not_after)
                days_left = int((seconds - time.time()) // 86400)
            except Exception:
                pass

        issuer = _cert_name(cert.get("issuer", []))
        subject = _cert_name(cert.get("subject", []))

        result.update({
            "checked": True,
            "valid": True,
            "issuer": issuer or None,
            "subject": subject or None,
            "not_before": not_before,
            "not_after": not_after,
            "days_left": days_left,
            "state": "pass",
            "detail": (
                f"Certificate is valid and matches the host"
                + (f" (expires in {days_left} days)" if days_left is not None else "")
                + (f"; issuer: {issuer}" if issuer else "")
                + "."
            ),
        })
        return result

    return _cached(f"tls:{hostname}:{port}", produce)


# ==========================================================
# 5. Page fetch, redirect chain and content/DOM analysis
# ==========================================================

MAX_HTML_BYTES = 400_000
MAX_TEXT_CHARS = 120_000
MAX_REDIRECTS = 6

_BRANDS = None


def _brands():
    global _BRANDS
    if _BRANDS is not None:
        return _BRANDS
    path = DATA_DIR / "brands.json"
    table = {}
    try:
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    except Exception as exc:
        print(f"Brand list load error: {exc}")
    _BRANDS = {str(k).lower(): str(v).lower() for k, v in table.items()}
    return _BRANDS


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.forms = []
        self.password_inputs = 0
        self.input_count = 0
        self.iframes = 0
        self.meta_refresh = False
        self.script_srcs = []
        self.inline_js = []
        self.text_parts = []
        self._in_title = False
        self._in_script = False
        self._in_style = False
        self._text_len = 0
        self._js_len = 0

    def handle_starttag(self, tag, attrs):
        attributes = {key.lower(): (value or "") for key, value in attrs}
        tag = tag.lower()

        if tag == "form":
            self.forms.append({
                "action": attributes.get("action", ""),
                "method": (attributes.get("method", "get") or "get").lower(),
            })
        elif tag == "input":
            self.input_count += 1
            if (attributes.get("type", "") or "").lower() == "password":
                self.password_inputs += 1
        elif tag == "iframe":
            self.iframes += 1
        elif tag == "meta":
            if (attributes.get("http-equiv", "") or "").lower() == "refresh":
                self.meta_refresh = True
        elif tag == "script":
            src = attributes.get("src", "")
            if src:
                self.script_srcs.append(src)
            self._in_script = True
        elif tag == "style":
            self._in_style = True
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "script":
            self._in_script = False
        elif tag == "style":
            self._in_style = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_script:
            if self._js_len < 200_000:
                self.inline_js.append(data)
                self._js_len += len(data)
            return
        if self._in_style:
            return
        if self._in_title:
            self.title += data
        if self._text_len < MAX_TEXT_CHARS:
            self.text_parts.append(data)
            self._text_len += len(data)


def _read_limited(response):
    chunks = []
    size = 0
    for chunk in response.iter_content(8192):
        if not chunk:
            continue
        chunks.append(chunk)
        size += len(chunk)
        if size >= MAX_HTML_BYTES:
            break
    encoding = response.encoding or "utf-8"
    return b"".join(chunks).decode(encoding, "ignore")


def page_probe(url):
    """Fetch a page (manual, SSRF-safe redirects) and parse its DOM once."""

    def produce():
        result = {
            "checked": False,
            "ok": False,
            "status": None,
            "chain": [url],
            "final_url": url,
            "error": None,
            "html_length": 0,
            "content": None,
        }

        current = url
        seen = {url}

        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(current)
            if parts.scheme.lower() not in ("http", "https"):
                result["error"] = f"Blocked non-HTTP redirect target ({parts.scheme or 'unknown'})."
                break
            if not parts.hostname or not _host_is_public(parts.hostname):
                result["error"] = "Blocked redirect to a private or unresolvable host."
                break

            try:
                response = requests.get(
                    current,
                    timeout=(3, HTTP_TIMEOUT),
                    allow_redirects=False,
                    headers=PROBE_HEADERS,
                    stream=True,
                )
            except Exception as exc:
                result["error"] = f"Page fetch failed ({type(exc).__name__})."
                break

            result["checked"] = True
            result["status"] = response.status_code
            location = response.headers.get("Location")

            if response.is_redirect and location:
                response.close()
                nxt = urljoin(current, location)
                if nxt in seen:
                    result["error"] = "Redirect loop detected."
                    break
                seen.add(nxt)
                result["chain"].append(nxt)
                current = nxt
                continue

            ctype = (response.headers.get("Content-Type") or "").lower()
            if "html" in ctype or not ctype:
                try:
                    html = _read_limited(response)
                    result["html_length"] = len(html)
                    result["content"] = _analyse_html(html, current)
                except Exception as exc:
                    result["error"] = f"HTML parse failed ({type(exc).__name__})."
            response.close()
            break

        result["final_url"] = current
        result["ok"] = result["status"] is not None
        return result

    return _cached(f"page:{url}", produce)


def _analyse_html(html, page_url):
    parser = _PageParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass

    page_host = (urlsplit(page_url).hostname or "").lower()
    page_root = registrable_domain(page_host)

    text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
    title = re.sub(r"\s+", " ", parser.title).strip()[:200]

    # Off-domain form action: credentials would be posted to another site.
    external_actions = []
    for form in parser.forms:
        action = (form.get("action") or "").strip()
        if not action or action.startswith("#") or action.lower().startswith("javascript:"):
            continue
        target = urljoin(page_url, action)
        target_host = (urlsplit(target).hostname or "").lower()
        if target_host and registrable_domain(target_host) != page_root:
            external_actions.append(target_host)

    # Brand impersonation: a known brand is mentioned on the page but the
    # domain is not owned by that brand.
    mentioned = []
    haystack = f"{title} {text[:60000]}".lower()
    for brand, domain in _brands().items():
        if brand in haystack:
            if page_root == domain or page_root.endswith("." + domain):
                continue
            mentioned.append(brand)
    mentioned = sorted(set(mentioned))[:6]

    js = " ".join(parser.inline_js)
    obfuscated_js = bool(re.search(r"\beval\s*\(|\batob\s*\(|unescape\s*\(|fromCharCode", js))

    return {
        "title": title,
        "form_count": len(parser.forms),
        "input_count": parser.input_count,
        "password_inputs": parser.password_inputs,
        "has_password_form": parser.password_inputs > 0,
        "external_actions": sorted(set(external_actions))[:5],
        "has_external_form_action": bool(external_actions),
        "brand_mentions": mentioned,
        "brand_mismatch": bool(mentioned),
        "meta_refresh": parser.meta_refresh,
        "iframe_count": parser.iframes,
        "script_count": len(parser.script_srcs),
        "obfuscated_js": obfuscated_js,
        "text_excerpt": text[:400],
    }


def redirect_layer(page):
    chain = page.get("chain") or []
    hops = max(0, len(chain) - 1)

    if not page.get("checked"):
        return {
            "key": "redirects",
            "name": "Redirect chain",
            "state": "skip",
            "detail": page.get("error") or "Redirect chain could not be inspected.",
            "hops": hops,
            "chain": chain,
            "cross_domain": False,
            "shortener": False,
        }

    roots = []
    for link in chain:
        host = (urlsplit(link).hostname or "").lower()
        roots.append(registrable_domain(host))

    cross_domain = any(a != b for a, b in zip(roots, roots[1:]))
    used_shortener = any(
        (urlsplit(link).hostname or "").lower().lstrip("www.") in SHORTENERS
        for link in chain[:-1]
    )

    if used_shortener or hops >= 4 or (cross_domain and hops >= 2):
        state = "fail"
    elif cross_domain or hops >= 2:
        state = "warn"
    else:
        state = "pass"

    if hops == 0:
        detail = "No redirects — the URL responded directly."
    else:
        detail = f"{hops} redirect(s)"
        if cross_domain:
            detail += f"; crosses {len(set(roots))} registrable domains"
        if used_shortener:
            detail += "; uses a known URL shortener"
        detail += "."
        if chain:
            detail += f" Final destination: {chain[-1][:120]}"

    return {
        "key": "redirects",
        "name": "Redirect chain",
        "state": state,
        "detail": detail,
        "hops": hops,
        "chain": chain,
        "cross_domain": cross_domain,
        "shortener": used_shortener,
    }


def content_layer(page, hostname):
    content = page.get("content")

    if not content:
        return {
            "key": "content",
            "name": "Page content / DOM",
            "state": "skip",
            "detail": page.get("error") or "No HTML content could be analysed.",
            "has_password_form": False,
            "external_form_action": False,
            "brand_mismatch": False,
            "brand_mentions": [],
            "title": None,
        }

    password = content["has_password_form"]
    external = content["has_external_form_action"]
    brand = content["brand_mismatch"]

    if external and password:
        state = "fail"
        detail = (
            "Credential form posts to an off-domain host "
            f"({', '.join(content['external_actions'])}) — a classic credential-harvesting pattern."
        )
    elif password and brand:
        state = "fail"
        detail = (
            "Password form plus brand impersonation: the page mentions "
            f"{', '.join(content['brand_mentions'])}, but the domain is not owned by that brand."
        )
    elif external:
        state = "warn"
        detail = f"A form submits to an off-domain host ({', '.join(content['external_actions'])})."
    elif password:
        state = "warn"
        detail = "The page contains a password field — verify the domain before entering credentials."
    elif content["meta_refresh"] or content["obfuscated_js"]:
        state = "warn"
        detail = "The page uses a meta refresh or obfuscated JavaScript, a common cloaking technique."
    else:
        state = "pass"
        detail = (
            f"No credential form or brand mismatch found "
            f"({content['form_count']} form(s), {content['input_count']} input(s))."
        )

    if content["brand_mentions"] and not password:
        detail += f" Brand mentions (informational): {', '.join(content['brand_mentions'])}."

    return {
        "key": "content",
        "name": "Page content / DOM",
        "state": state,
        "detail": detail,
        "has_password_form": password,
        "external_form_action": external,
        "brand_mismatch": brand,
        "brand_mentions": content["brand_mentions"],
        "title": content["title"],
    }


# ==========================================================
# 6. Calibration (Platt scaling fitted on held-out data)
# ==========================================================

_CALIBRATION = None
_CALIBRATION_LOCK = threading.Lock()


def load_calibration():
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION

    with _CALIBRATION_LOCK:
        if _CALIBRATION is not None:
            return _CALIBRATION
        data = {"available": False, "method": "none", "coef": 1.0, "intercept": 0.0}
        path = MODELS_DIR / "calibration.joblib"
        if path.exists():
            try:
                import joblib
                loaded = joblib.load(path)
                if isinstance(loaded, dict) and "coef" in loaded:
                    data = {"available": True, **loaded}
                    print(f"Calibration loaded: {data.get('method', 'platt')}")
            except Exception as exc:
                print(f"Calibration load error: {exc}")
        _CALIBRATION = data
        return data


def calibrate(probability):
    """Map a raw phishing probability to a calibrated one."""
    data = load_calibration()
    if not data.get("available"):
        return float(probability)

    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p))
    z = data["coef"] * logit + data["intercept"]
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


# ==========================================================
# 7. Orchestration + verdict combination
# ==========================================================

def _run_layers(url, hostname, reachable):
    tasks = {
        "allowlist": lambda: check_allowlist(hostname),
        "reputation": lambda: reputation_check(url, hostname),
        "whois": lambda: whois_lookup(hostname),
        "tls": lambda: tls_check(hostname),
    }
    if reachable:
        tasks["page"] = lambda: page_probe(url)

    results = {}
    executor = ThreadPoolExecutor(max_workers=len(tasks))
    futures = {executor.submit(fn): key for key, fn in tasks.items()}

    from concurrent.futures import wait

    done, pending = wait(futures, timeout=LAYER_TIMEOUT)
    for future in done:
        key = futures[future]
        try:
            results[key] = future.result()
        except Exception as exc:
            results[key] = {
                "key": key,
                "name": key.title(),
                "state": "skip",
                "detail": f"Layer failed ({type(exc).__name__}).",
            }
    for future in pending:
        key = futures[future]
        results[key] = {
            "key": key,
            "name": key.title(),
            "state": "skip",
            "detail": "Layer timed out — treated as inconclusive.",
        }

    executor.shutdown(wait=False, cancel_futures=True)

    page = results.pop("page", None)
    if page is None:
        page = {
            "checked": False,
            "ok": False,
            "chain": [url],
            "final_url": url,
            "error": "Page was not fetched because the site is unreachable.",
        }
    results["redirects"] = redirect_layer(page)
    results["content"] = content_layer(page, hostname)
    return results


def _fmt_delta(value):
    if not value:
        return "—"
    return f"{value:+.2f}"


def assess(url, hostname, raw_probability, calibrated_probability, reachable):
    """Run every verification layer and combine them into one verdict.

    Parameters
    ----------
    raw_probability
        The Random Forest's P(phishing).
    calibrated_probability
        Platt-scaled P(phishing).
    reachable
        Result of the DNS + HTTP probe.

    Returns a dict with ``status``, ``reason``, ``confidence``,
    ``risk_score``, ``layers``, ``extra_indicators`` and ``calibration``.
    """
    layers = _run_layers(url, hostname, reachable)

    allow = layers["allowlist"]
    reputation = layers["reputation"]
    whois = layers["whois"]
    tls = layers["tls"]
    content = layers["content"]
    redirects = layers["redirects"]

    score = float(calibrated_probability)
    positive_signals = []
    negative_signals = []
    extra_indicators = []

    def add_indicator(severity, title, detail):
        extra_indicators.append({"severity": severity, "title": title, "detail": detail})

    # ---- Reputation: a hit is decisive ----
    hard_phishing = False
    hard_reason = None
    if reputation.get("malicious"):
        hard_phishing = True
        hard_reason = "Reputation layer flagged this URL: " + reputation["detail"]
        add_indicator("high", "Listed by Reputation Feeds", reputation["detail"])
        positive_signals.append("reputation hit")
        score = max(score, 0.95)

    # ---- WHOIS age ----
    age_days = whois.get("age_days")
    if whois.get("available"):
        score += 0.10
        positive_signals.append("unregistered domain")
    elif age_days is not None:
        if age_days < 30:
            score += 0.12
            positive_signals.append(f"domain registered {age_days} days ago")
            add_indicator("high", "Newly Registered Domain",
                          f"The domain was created {age_days} days ago. Phishing domains are usually "
                          "registered days or weeks before an attack.")
        elif age_days < 180:
            score += 0.04
            positive_signals.append(f"domain only {age_days} days old")
        elif age_days >= 730:
            score -= 0.05
            negative_signals.append(f"domain registered {whois['age_years']} years ago")

    # ---- TLS ----
    if tls.get("valid") is True:
        score -= 0.03
        negative_signals.append("valid TLS certificate")
    elif tls.get("valid") is False:
        score += 0.15
        positive_signals.append("invalid TLS certificate")
        add_indicator("high", "Invalid TLS Certificate",
                      "The HTTPS certificate failed validation (expired, self-signed or "
                      "issued for a different host).")

    # ---- Page content ----
    content_fail = content["state"] == "fail"
    if content.get("external_form_action") and content.get("has_password_form"):
        score += 0.25
        positive_signals.append("credential form posts off-domain")
        add_indicator("high", "Credential Form Posts Off-Domain",
                      content["detail"])
    elif content.get("has_password_form") and content.get("brand_mismatch"):
        score += 0.20
        positive_signals.append("brand impersonation with password form")
        add_indicator("high", "Brand Impersonation",
                      content["detail"])
    elif content.get("has_password_form"):
        score += 0.05
        positive_signals.append("password form present")
    elif content.get("brand_mismatch"):
        # Informational only: news sites mention brands all the time.
        pass

    if content["state"] == "warn" and not content.get("has_password_form"):
        score += 0.03
        positive_signals.append("page cloaking signals")

    # ---- Redirects ----
    if redirects.get("shortener"):
        score += 0.05
        positive_signals.append("URL shortener in redirect chain")
    if redirects.get("cross_domain"):
        score += 0.05
        positive_signals.append("cross-domain redirects")
        add_indicator("medium", "Cross-Domain Redirects",
                      redirects["detail"])

    # ---- Allowlist override (false-positive guard) ----
    allowlist_applied = False
    if allow.get("listed") and not hard_phishing and not content_fail:
        if score > 0.25:
            score = 0.25
            allowlist_applied = True
            negative_signals.append("domain is on the top-domain allowlist")

    score = max(0.0, min(1.0, score))

    # ---- Final verdict ----
    if hard_phishing:
        status = "Phishing"
        reason = hard_reason
    elif not reachable:
        status = "Unreachable"
        reason = (
            "Domain/website could not be reached. This is neither a Legitimate nor "
            "a Phishing verdict."
        )
        if score >= 0.70:
            reason += f" The model's calibrated phishing risk is {score:.0%}, so treat the link with caution."
    elif score >= ABSTAIN_HIGH:
        status = "Phishing"
        driver = positive_signals[0] if positive_signals else (
            f"model probability {calibrated_probability:.0%}"
        )
        reason = f"{driver}; combined risk {score:.0%}."
    elif score <= ABSTAIN_LOW:
        status = "Legitimate"
        driver = negative_signals[0] if negative_signals else "no phishing evidence"
        reason = f"{driver}; combined risk {score:.0%}."
    else:
        status = "Unknown"
        reason = (
            f"Combined risk {score:.0%} falls inside the "
            f"{ABSTAIN_LOW:.0%}–{ABSTAIN_HIGH:.0%} abstain band — not enough reliable "
            "evidence for a binary verdict."
        )

    confidence = round(max(score, 1 - score) * 100, 2)

    # Attach the numeric influence of each layer for the UI.
    influences = {
        "allowlist": "capped at 25%" if allowlist_applied else "—",
        "reputation": "decisive" if hard_phishing else ("clean" if reputation["state"] == "pass" else "—"),
        "whois": _fmt_delta(0.12 if age_days is not None and age_days < 30
                            else 0.04 if age_days is not None and age_days < 180
                            else -0.05 if age_days is not None and age_days >= 730
                            else (0.10 if whois.get("available") else 0)),
        "tls": _fmt_delta(-0.03 if tls.get("valid") is True else 0.15 if tls.get("valid") is False else 0),
        "content": _fmt_delta(
            0.25 if content.get("external_form_action") and content.get("has_password_form")
            else 0.20 if content.get("has_password_form") and content.get("brand_mismatch")
            else 0.05 if content.get("has_password_form")
            else 0.03 if content["state"] == "warn" else 0
        ),
        "redirects": _fmt_delta(
            (0.05 if redirects.get("shortener") else 0)
            + (0.05 if redirects.get("cross_domain") else 0)
        ),
    }

    ordered = [allow, reputation, whois, tls, content, redirects]
    for layer in ordered:
        layer["influence"] = influences.get(layer["key"], "—")

    layers["calibration"] = {
        "key": "calibration",
        "name": "Confidence calibration",
        "state": "info",
        "detail": (
            f"Raw model risk {raw_probability:.1%} → calibrated {calibrated_probability:.1%}; "
            f"combined risk after verification layers {score:.1%} "
            f"(abstain band {ABSTAIN_LOW:.0%}–{ABSTAIN_HIGH:.0%})."
        ),
        "influence": "—",
    }

    layer_order = ["reputation", "whois", "tls", "content", "redirects", "allowlist", "calibration"]
    return {
        "status": status,
        "reason": reason,
        "confidence": confidence,
        "risk_score": int(round(score * 100)),
        "score": score,
        "layers": [layers[key] for key in layer_order],
        "extra_indicators": extra_indicators,
        "calibration": {
            "raw_phishing_probability": round(raw_probability * 100, 2),
            "calibrated_phishing_probability": round(score * 100, 2),
            "method": load_calibration().get("method", "none"),
            "abstain_low": round(ABSTAIN_LOW * 100, 1),
            "abstain_high": round(ABSTAIN_HIGH * 100, 1),
        },
    }
