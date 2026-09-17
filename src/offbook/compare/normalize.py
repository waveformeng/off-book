"""Deterministic token normalization. Applied identically to both streams."""

from __future__ import annotations

import re
import unicodedata

_STRIP = re.compile(r"[^a-z0-9']+")
_FILLERS = frozenset({"uh", "um", "umm", "hmm", "hm", "mm", "mmm", "mhm"})
"""Breath and hesitation tokens the word recognizer emits on sung intakes. Not words;
dropped from both streams alike."""


def normalize_word(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = _STRIP.sub("", t.lower()).strip("'")
    return "" if t in _FILLERS else t


def normalize_phoneme(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())
