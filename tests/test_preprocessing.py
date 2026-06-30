"""Parity tests for the pure-Python Keras tokenization replica.

The `text_to_word_sequence` tests are self-contained. The end-to-end parity test
(known notebook confidences) requires models/ artifacts and is skipped if absent.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import MODEL_PATH, WORD_INDEX_PATH  # noqa: E402
from app.preprocessing import clean_text, pad_sequence, text_to_word_sequence  # noqa: E402


def test_clean_text():
    # 1. Lowercase and byte strip
    assert clean_text("Hello \\xe2\\x80\\x93 World \\x9d") == "hello world"
    
    # 2. URLs
    assert clean_text("Visit https://google.com or www.example.com for info") == "visit <url> or <url> for info"
    
    # 3. Emails
    assert clean_text("Contact me at test@example.com now") == "contact me at <email> now"
    
    # 4. Punctuation isolation
    assert clean_text("Hello, world!") == "hello , world !"


def test_word_sequence_lowercases_and_strips_punctuation():
    assert text_to_word_sequence("Hello, WORLD! It's fake-news.") == [
        "hello",
        "world",
        "it's",  # apostrophe is NOT in Keras default filters -> kept
        "fake",
        "news",
    ]


def test_word_sequence_collapses_filtered_chars():
    assert text_to_word_sequence("a\tb\nc...d") == ["a", "b", "c", "d"]


def test_pad_pre_truncates_tail_and_left_pads():
    assert pad_sequence([1, 2, 3], maxlen=5) == [0, 0, 1, 2, 3]
    assert pad_sequence([1, 2, 3, 4, 5, 6], maxlen=3) == [4, 5, 6]  # keeps last 3


# --- End-to-end parity against the notebook's documented predictions ---
NOTEBOOK_CASES = [
    ("Breaking: New discovery found in the deep ocean by researchers.", "FAKE", 0.9655),
    (
        "trump is doing some nganu nganu in the public restroom it was caught in a public journal",
        "FAKE",
        0.9771,
    ),
]


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and WORD_INDEX_PATH.exists()),
    reason="models/model.onnx or tokenizer_word_index.json not present",
)
@pytest.mark.parametrize("text,label,conf", NOTEBOOK_CASES)
def test_matches_notebook_predictions(text, label, conf):
    from app.inference import FakeNewsModel
    from app.preprocessing import Tokenizer, preprocess

    tok = Tokenizer.from_json_file(WORD_INDEX_PATH)
    model = FakeNewsModel(MODEL_PATH)
    prob = model.predict_proba(preprocess(text, tok))[0]
    got_label, got_conf, _ = model.classify(prob)
    assert got_label == label
    assert abs(got_conf - conf) < 0.02, f"confidence {got_conf:.4f} vs notebook {conf}"
