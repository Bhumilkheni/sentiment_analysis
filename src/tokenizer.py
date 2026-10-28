"""Tiny word-level tokenizer + padding helper.

Does the same job as Keras' Tokenizer/pad_sequences, but has no dependency on
a specific TensorFlow/Keras version (the legacy Keras Tokenizer was removed
from the public API in Keras 3). It is also easy to read and to save as JSON.

Index 0 = padding, index 1 = out-of-vocabulary (OOV), real words start at 2.
"""
import json
from collections import Counter

import numpy as np


class SimpleTokenizer:
    PAD, OOV = 0, 1

    def __init__(self, num_words=20000):
        self.num_words = num_words
        self.word_index = {}

    def fit_on_texts(self, texts):
        counts = Counter(w for t in texts for w in t.split())
        vocab = [w for w, _ in counts.most_common(self.num_words - 2)]
        self.word_index = {w: i + 2 for i, w in enumerate(vocab)}
        return self

    def texts_to_sequences(self, texts):
        return [[self.word_index.get(w, self.OOV) for w in t.split()] for t in texts]

    @property
    def vocab_size(self):
        return len(self.word_index) + 2

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_words": self.num_words, "word_index": self.word_index}, f)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(num_words=data["num_words"])
        tok.word_index = data["word_index"]
        return tok


def pad_sequences(sequences, maxlen, padding="post", truncating="post"):
    """Pad / truncate a list of integer lists into an int32 array (n, maxlen)."""
    out = np.zeros((len(sequences), maxlen), dtype="int32")
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen] if truncating == "post" else seq[-maxlen:]
        if not seq:
            continue
        if padding == "post":
            out[i, : len(seq)] = seq
        else:
            out[i, -len(seq):] = seq
    return out
