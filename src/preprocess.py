"""Tweet cleaning pipeline shared by the notebooks and the Streamlit app.

Using ONE implementation everywhere guarantees that text seen at inference
time is cleaned exactly like the text the models were trained on.
"""
import html
import re
from functools import lru_cache

import nltk

# --- Regex patterns ---------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # "sooooo" -> "soo"
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

# Order matters: handle irregular forms before the generic "n't" rule.
CONTRACTIONS = [
    (r"won't", "will not"), (r"can't", "can not"), (r"cannot", "can not"),
    (r"n't", " not"), (r"'re", " are"), (r"'m", " am"),
    (r"'ve", " have"), (r"'ll", " will"), (r"'d", " would"),
]

# Negations / intensifiers carry sentiment, so we do NOT treat them as stop words.
KEEP_WORDS = {"no", "nor", "not", "never", "against", "off", "too", "very"}

_READY = False


def ensure_nltk():
    """Download the NLTK resources we need (idempotent, quiet)."""
    global _READY
    if _READY:
        return
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    _READY = True


@lru_cache(maxsize=1)
def _resources():
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop = set(stopwords.words("english")) - KEEP_WORDS
    return stop, WordNetLemmatizer()


def _lemmatize(lemmatizer, token):
    # verb first ("running" -> "run", "delayed" -> "delay"), then noun ("bags" -> "bag")
    return lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos="v"), pos="n")


def clean_tweet(text, lemmatize=True, remove_stopwords=True, keep_hashtag_words=True):
    """Clean a raw tweet and return a space-separated string of tokens.

    Parameters
    ----------
    lemmatize : convert words to their dictionary form.
    remove_stopwords : drop common words (negations are always kept).
        Use False for sequence models (LSTM) so word order/context is preserved.
    keep_hashtag_words : "#fail" -> "fail" (True) or remove the whole hashtag (False).
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalisation
    text = html.unescape(text).lower().replace("\u2019", "'")

    # 2. Noise removal
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1" if keep_hashtag_words else " ", text)
    for pattern, repl in CONTRACTIONS:
        text = re.sub(pattern, repl, text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = NON_ALPHA_RE.sub(" ", text)          # digits, punctuation, emojis

    tokens = [t for t in text.split() if len(t) > 1]

    # 3. Linguistic processing
    stop, lemmatizer = _resources()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop]
    if lemmatize:
        tokens = [_lemmatize(lemmatizer, t) for t in tokens]
    return " ".join(tokens)
