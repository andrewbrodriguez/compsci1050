# Utility functions for privacy policy analysis
import re, time, math, json, string, statistics as stats
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import quote_plus
import requests
import tldextract
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from readability import Document
import textstat
from wordfreq import zipf_frequency

WS_RE = re.compile(r"\s+")

LEGAL_TERMS = set([
    "arbitration","waiver","indemnify","indemnification","liable","liability",
    "jurisdiction","venue","assignment","severability","warranty","limitation","limitation of liability",
    "consent","processing","processor","controller","ccpa","gdpr","transfer","retention","opt-out","opt in",
    "cookies","beacons","do not sell","profiling","biometric","geolocation"
])

def domain_from_url(url: str) -> str:
    parts = tldextract.extract(url)
    return ".".join(p for p in [parts.domain, parts.suffix] if p)

def clean_text(s: str) -> str:
    # Normalize whitespace & remove zero-width / non-printables
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = WS_RE.sub(" ", s).strip()
    return s

def html_to_text(html: str) -> str:
    # Favor readability-lxml to get main content; fallback to BeautifulSoup
    try:
        doc = Document(html)
        content_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(content_html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")
    # Remove scripts/styles/nav/footer
    for tag in soup(["script","style","noscript","header","footer","nav","form"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return clean_text(text)

def fetch_url(url: str, timeout: int = 20) -> str:
    hdrs = {"User-Agent":"Mozilla/5.0 (policy-study bot; +https://example.com)"}
    r = requests.get(url, headers=hdrs, timeout=timeout)
    r.raise_for_status()
    return r.text

def get_wayback_snapshots(url: str, from_year: int=2015, to_year: int=2025, limit_per_year: int=2):
    """
    Query the CDX API to get snapshot timestamps. We pick up to N per year (first & last if available).
    """
    cdx = f"https://web.archive.org/cdx/search/cdx?url={quote_plus(url)}&output=json&filter=statuscode:200&from={from_year}&to={to_year}&fl=timestamp,original,mimetype,statuscode"
    r = requests.get(cdx, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = data[1:] if isinstance(data, list) and len(data) > 1 else []
    # group by year
    by_year = {}
    for ts, orig, mime, status in rows:
        year = int(ts[:4])
        by_year.setdefault(year, []).append(ts)
    snapshots = []
    for y in sorted(by_year.keys()):
        ts_list = sorted(by_year[y])
        picks = []
        if len(ts_list) == 1:
            picks = ts_list
        else:
            picks = [ts_list[0], ts_list[-1]]
        snapshots.extend([(y, ts) for ts in picks[:limit_per_year]])
    return snapshots

def fetch_wayback_content(url: str, timestamp: str) -> str:
    wb_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
    return fetch_url(wb_url)

def basic_counts(text: str) -> Dict[str,int]:
    chars = len(text)
    words = textstat.lexicon_count(text, removepunct=True)
    sents = max(1, textstat.sentence_count(text))
    return {"n_chars": chars, "n_words": words, "n_sents": sents}

def lexical_stats(text: str) -> Dict[str,float]:
    # simple tokenization on words
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
    if not tokens:
        return {"ttr":0.0, "yule_k":0.0, "entropy":0.0, "avg_zipf":0.0, "rare_frac":0.0, "legalese_frac":0.0}
    types = set(tokens)
    ttr = len(types)/len(tokens)
    # Yule's K
    from collections import Counter
    freqs = Counter(tokens)
    m1 = sum(freqs.values())
    m2 = sum(v*v for v in freqs.values())
    yule_k = 1e4 * (m2 - m1) / (m1*m1) if m1 > 0 else 0.0
    # Entropy (Shannon)
    probs = np.array([c/m1 for c in freqs.values()])
    entropy = -float(np.sum(probs * np.log2(probs)))
    # Zipf freq
    zipfs = [zipf_frequency(tok, "en") for tok in tokens]
    avg_zipf = float(np.mean(zipfs)) if zipfs else 0.0
    rare_frac = float(np.mean([1.0 if z < 3.0 else 0.0 for z in zipfs])) if zipfs else 0.0
    legalese = float(np.mean([1.0 if t in LEGAL_TERMS else 0.0 for t in tokens]))
    return {"ttr":ttr, "yule_k":yule_k, "entropy":entropy, "avg_zipf":avg_zipf, "rare_frac":rare_frac, "legalese_frac":legalese}

def readability_metrics(text: str) -> Dict[str,float]:
    # Wrap in try/except since some metrics can error for edge cases
    def safe(fn, default=np.nan):
        try:
            return float(fn(text))
        except Exception:
            return float(default)
    return {
        "flesch_reading_ease":     safe(textstat.flesch_reading_ease),
        "flesch_kincaid_grade":    safe(textstat.flesch_kincaid_grade),
        "gunning_fog":             safe(textstat.gunning_fog),
        "smog_index":              safe(textstat.smog_index),
        "dale_chall":              safe(textstat.dale_chall_readability_score),
        "coleman_liau":            safe(textstat.coleman_liau_index),
        "ari":                     safe(textstat.automated_readability_index),
        "avg_sentence_length":     safe(lambda s: basic_counts(s)["n_words"]/max(1,basic_counts(s)["n_sents"]))
    }

def analyze_text(name: str, text: str) -> Dict[str, Any]:
    counts = basic_counts(text)
    lex = lexical_stats(text)
    read = readability_metrics(text)
    out = {"name": name}
    out.update(counts)
    out.update(read)
    out.update(lex)
    return out

def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())

def load_local_texts(folder: str) -> Dict[str,str]:
    import os
    texts = {}
    for fn in os.listdir(folder):
        if fn.lower().endswith((".txt",".md",".html")):
            path = os.path.join(folder, fn)
            raw = load_file(path)
            # if looks like HTML, convert to text
            if "<html" in raw.lower() or "</p>" in raw.lower():
                raw = html_to_text(raw)
            name = os.path.splitext(fn)[0]
            texts[name] = raw
    return texts

def fetch_policy_text(url: str) -> Tuple[str,str]:
    html = fetch_url(url)
    text = html_to_text(html)
    name = domain_from_url(url)
    return name, text

def polite_sleep(seconds: float = 1.0):
    time.sleep(seconds)
