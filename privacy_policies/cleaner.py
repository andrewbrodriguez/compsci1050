# cleaner.py — cleaned copies of policy .txt files using NLTK stopwords
# Usage: python cleaner.py

import re
from pathlib import Path
import ssl
import certifi
import nltk

# ---------------- NLTK + SSL setup ----------------
NLTK_LOCAL = Path(__file__).parent / "nltk_data"
NLTK_LOCAL.mkdir(exist_ok=True)
if str(NLTK_LOCAL) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_LOCAL))

# IMPORTANT: set a FUNCTION, not an instance
def _ssl_context_with_certifi(*args, **kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return ssl.create_default_context(*args, **kwargs)

ssl._create_default_https_context = _ssl_context_with_certifi

def ensure_stopwords():
    try:
        from nltk.corpus import stopwords
        _ = stopwords.words("english")
        return stopwords
    except LookupError:
        nltk.download("stopwords", download_dir=str(NLTK_LOCAL), raise_on_error=True)
        from nltk.corpus import stopwords
        _ = stopwords.words("english")
        return stopwords

stopwords = ensure_stopwords()
STOPWORDS = set(stopwords.words("english"))
print(f"[nltk] stopwords loaded ({len(STOPWORDS)}) from {NLTK_LOCAL}")

# --------------- optional: wordfreq filter ---------------
try:
    from wordfreq import zipf_frequency
    def is_english_word(word, threshold=1.5):
        return zipf_frequency(word, "en") > threshold
except Exception:
    def is_english_word(word, threshold=None):
        return True

# ---------------- cleaning function ----------------
def clean_policy_text(raw: str) -> str:
    text = raw.lower()
    text = re.sub(r"\d+", " ", text)                 # remove numbers
    tokens = re.findall(r"[a-zA-Z]+", text)          # alphabetic only
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2 and is_english_word(t)]
    return " ".join(tokens)

# ---------------- I/O paths ----------------
RAW_DIR = Path("/Users/andrewrodriguez/Desktop/compsci1050/privacy_policies/policy_texts")
CLEAN_DIR = Path("/Users/andrewrodriguez/Desktop/compsci1050/privacy_policies/policy_texts_cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[paths] RAW   = {RAW_DIR.resolve()}")
print(f"[paths] CLEAN = {CLEAN_DIR.resolve()}")

# ---------------- process files ----------------
count = 0
for raw_fp in RAW_DIR.rglob("*.txt"):
    try:
        raw_text = raw_fp.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_policy_text(raw_text)
        out_fp = CLEAN_DIR / raw_fp.name
        out_fp.write_text(cleaned, encoding="utf-8")
        count += 1
        print(f"✔ {raw_fp.name} -> {out_fp.name} (clean words={len(cleaned.split())})")
    except Exception as e:
        print(f"✘ error on {raw_fp}: {e}")

print(f"\nDone. Cleaned {count} files.")