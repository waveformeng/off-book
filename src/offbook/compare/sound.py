"""A sound key for English words (classic Metaphone, Philips 1990), so that two spellings
of one sound — "for"/"four", "there"/"their", "right"/"write" — compare equal.

Pure rules, no dictionary. Applied identically to both streams, so a recognizer that
spells a homophone one way on the reference and the other way on the live vocal does
not turn a correctly sung word into a failure.
"""

from __future__ import annotations

import re

_VOWELS = "AEIOU"
_NOT_LETTERS = re.compile(r"[^A-Z]")


def _at(word: str, i: int) -> str:
    return word[i] if 0 <= i < len(word) else ""


def metaphone(text: str) -> str:
    w = _NOT_LETTERS.sub("", text.upper())
    if not w:
        return ""
    # Initial transformations.
    if w[:2] in ("KN", "GN", "PN", "AE", "WR"):
        w = w[1:]
    elif w[0] == "X":
        w = "S" + w[1:]
    elif w[:2] == "WH":
        w = "W" + w[1:]

    out: list[str] = []
    n = len(w)
    for i, c in enumerate(w):
        prev, nxt, nxt2 = _at(w, i - 1), _at(w, i + 1), _at(w, i + 2)
        if c == prev and c != "C":
            continue  # duplicate adjacent letters
        if c in _VOWELS:
            if i == 0:
                out.append(c)
            continue
        if c == "B":
            if not (prev == "M" and i == n - 1):
                out.append("B")
        elif c == "C":
            if nxt == "I" and nxt2 == "A":
                out.append("X")
            elif nxt == "H":
                out.append("K" if prev == "S" else "X")
            elif nxt in "IEY":
                if prev != "S":
                    out.append("S")
            else:
                out.append("K")
        elif c == "D":
            if nxt == "G" and nxt2 in "EYI":
                out.append("J")
            else:
                out.append("T")
        elif c == "G":
            if nxt == "H" and not (i + 2 >= n or nxt2 in _VOWELS):
                continue
            if nxt == "N" and (i + 2 >= n or w[i + 1 :] == "NED"):
                continue
            if nxt in "IEY" and prev != "G":
                out.append("J")
            else:
                out.append("K")
        elif c == "H":
            if prev in "CSPTG" or (prev in _VOWELS and nxt not in _VOWELS):
                continue
            out.append("H")
        elif c == "K":
            if prev != "C":
                out.append("K")
        elif c == "P":
            out.append("F" if nxt == "H" else "P")
        elif c == "Q":
            out.append("K")
        elif c == "S":
            if nxt == "H" or (nxt == "I" and nxt2 in "AO"):
                out.append("X")
            else:
                out.append("S")
        elif c == "T":
            if nxt == "I" and nxt2 in "AO":
                out.append("X")
            elif nxt == "H":
                out.append("0")
            elif not (nxt == "C" and nxt2 == "H"):
                out.append("T")
        elif c == "V":
            out.append("F")
        elif c == "W":
            if nxt in _VOWELS:
                out.append("W")
        elif c == "X":
            out.append("KS")
        elif c == "Y":
            if nxt in _VOWELS:
                out.append("Y")
        elif c == "Z":
            out.append("S")
        else:
            out.append(c)  # F, J, L, M, N, R
    return "".join(out)
