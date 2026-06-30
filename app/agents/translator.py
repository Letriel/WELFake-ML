"""Agent 1 -- Translator.

Auto-detects the input language and translates it to English before the model
runs, so non-English news (e.g. Indonesian) can be classified. Mirrors the
notebook's ``translator_agent`` but adds chunking for long articles (Google's
endpoint rejects requests longer than ~5000 chars).
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_CHARS = 4500


def _split_chunks(text, max_chars=_MAX_CHARS):
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    return chunks


def _detect_language(text) -> Optional[str]:
    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return None


def translate_to_english(text: str) -> Tuple[str, bool, Optional[str]]:
    """Return (english_text, was_translated, detected_language)."""
    text = (text or "").strip()
    if not text:
        return "", False, None
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target="en")
        parts = [translator.translate(c) or "" for c in _split_chunks(text)]
        translated = " ".join(p for p in parts if p).strip()
    except Exception as e:  # network error, rate limit, etc. -> fall back gracefully
        logger.warning("Translation failed, using original text: %s", e)
        return text, False, None

    if not translated:
        return text, False, None
    was_translated = translated.lower() != text.lower()
    detected = _detect_language(text) if was_translated else "en"
    return translated, was_translated, detected
