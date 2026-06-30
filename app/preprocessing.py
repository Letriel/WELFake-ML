"""Pure-Python reimplementation of the Keras text pipeline used in training.

It must produce byte-for-byte the same integer sequences as
``keras.preprocessing.text.Tokenizer.texts_to_sequences`` +
``keras.preprocessing.sequence.pad_sequences`` so the ONNX model receives the
exact inputs it was trained on -- without depending on TensorFlow.
"""
import json
import re

import numpy as np

from .config import MAXLEN, NUM_WORDS, OOV_TOKEN, PAD_FILTERS, WORD_INDEX_PATH

# Map every Keras "filter" character to a space (str.translate is fast + exact).
_TRANSLATE_MAP = {ord(c): " " for c in PAD_FILTERS}


def clean_text(text: str) -> str:
    """Upgraded text preprocessing that removes raw UTF-8 byte leaks (\\xe2...)
    and isolates punctuation for optimal fake news classification.

    Replicates tf_preprocess behavior using pure Python re.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # 1. Convert to lowercase
    lowercase = text.lower()

    # 2. CRITICAL FIX: Strip raw UTF-8 byte escape patterns (\\xe2\\x80\\x93, \\xe2\\x80\\x9d, etc.)
    no_bytes = re.sub(r'\\xe2\\x80\\x[0-9a-fA-F]{2}', ' ', lowercase)
    no_bytes = re.sub(r'\\x[0-9a-fA-F]{2}', ' ', no_bytes)

    # 3. Standardize URLs and Web Links
    no_urls = re.sub(r'https?://\S+|www\.\S+', ' <url> ', no_bytes)

    # 4. Standardize Email addresses
    no_emails = re.sub(r'\S+@\S+', ' <email> ', no_urls)

    # 5. Isolate punctuation marks with spaces on both sides
    isolated_punct = re.sub(r'([.,!?();:$\-\"\'\[\]])', r' \1 ', no_emails)

    # 6. Collapse multiple spaces down to a single space
    clean_spaces = re.sub(r'\s+', ' ', isolated_punct)

    return clean_spaces.strip()


def text_to_word_sequence(text):
    """Replicates keras text_to_word_sequence(lower=True, split=' ')."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower().translate(_TRANSLATE_MAP)
    return [w for w in text.split(" ") if w]


class Tokenizer:
    """Minimal, serving-only Tokenizer backed by an exported ``word_index``."""

    def __init__(self, word_index, num_words=NUM_WORDS, oov_token=OOV_TOKEN):
        # JSON loads values as ints already; coerce defensively.
        self.word_index = {w: int(i) for w, i in word_index.items()}
        self.num_words = num_words
        self.oov_token = oov_token
        self.oov_index = self.word_index.get(oov_token) if oov_token else None

    @classmethod
    def from_json_file(cls, path=WORD_INDEX_PATH, num_words=NUM_WORDS, oov_token=OOV_TOKEN):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict) and data.get("class_name") == "Tokenizer" and "config" in data:
            word_index_raw = data["config"].get("word_index", "{}")
            word_index = json.loads(word_index_raw) if isinstance(word_index_raw, str) else word_index_raw
        else:
            word_index = data

        return cls(word_index, num_words=num_words, oov_token=oov_token)

    def text_to_sequence(self, text):
        """Mirror of Tokenizer.texts_to_sequences for a single string."""
        seq = []
        num_words = self.num_words
        for w in text_to_word_sequence(text):
            i = self.word_index.get(w)
            if i is not None:
                if num_words and i >= num_words:
                    if self.oov_index is not None:
                        seq.append(self.oov_index)
                else:
                    seq.append(i)
            elif self.oov_index is not None:
                seq.append(self.oov_index)
        return seq


def pad_sequence(seq, maxlen=MAXLEN, value=0):
    """Keras pad_sequences defaults: padding='pre', truncating='pre'."""
    if len(seq) > maxlen:
        seq = seq[-maxlen:]                       # truncating='pre' keeps the tail
    return [value] * (maxlen - len(seq)) + list(seq)   # padding='pre'


def preprocess(text, tokenizer, maxlen=MAXLEN):
    """One string -> float32 array of shape (1, maxlen)."""
    cleaned = clean_text(text)
    padded = pad_sequence(tokenizer.text_to_sequence(cleaned), maxlen=maxlen)
    return np.array([padded], dtype=np.float32)


def preprocess_batch(texts, tokenizer, maxlen=MAXLEN):
    """List of strings -> float32 array of shape (N, maxlen)."""
    rows = [pad_sequence(tokenizer.text_to_sequence(clean_text(t)), maxlen=maxlen) for t in texts]
    return np.array(rows, dtype=np.float32)
