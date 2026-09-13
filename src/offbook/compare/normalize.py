"""Deterministic token normalization. Applied identically to both streams."""

from __future__ import annotations

import re
import unicodedata

_STRIP = re.compile(r"[^a-z0-9']+")


def normalize_word(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = _STRIP.sub("", t.lower())
    return t.strip("'")


def normalize_phoneme(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())
