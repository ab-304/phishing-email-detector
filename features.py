import re
import numpy as np

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
URGENT = ["urgent", "immediately", "verify", "suspend", "expire", "confirm",
          "password", "login", "limited time", "act now", "click here",
          "winner", "prize", "free", "bank", "security alert", "update your"]

FEATURE_NAMES = ["n_urls", "has_ip_url", "n_urgent_words", "n_exclaims",
                 "caps_ratio", "log_length", "has_html"]


def extract(texts):
    """Hand-crafted signals that phishing emails tend to show."""
    rows = []
    for t in texts:
        t = str(t)
        low = t.lower()
        letters = [c for c in t if c.isalpha()]
        caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
        rows.append([
            len(URL_RE.findall(t)),
            int(bool(IP_URL_RE.search(t))),
            sum(low.count(w) for w in URGENT),
            t.count("!"),
            caps,
            np.log1p(len(t)),
            int("<html" in low or "<a href" in low),
        ])
    return np.array(rows, dtype=float)