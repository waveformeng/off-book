"""Sliding-window edit-distance alignment of the two token streams.

Tokens are held until they are *decidable*: a reference token once the live stream has
resolved past the latest point a live counterpart could start, and a live token once the
reference stream has resolved past the earliest point its counterpart could start. Each
time either stream advances, the pending window is re-aligned with Levenshtein DP whose
substitution cost carries both the lexical and the timing mismatch; only decidable
tokens are emitted. Being late by up to `max_lag_s` does not cascade: the DP still pairs
the token, and the pair is FAIL_TIMING at worst.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from offbook.asr.base import Token, Unit
from offbook.clock import TransportTime
from offbook.compare.verdict import TokenPair, Verdict
from offbook.config import AlignmentConfig
from offbook.roles import Live, Reference

_INF = float("inf")


def word_distance(a: str, b: str) -> float:
    return 0.0 if a == b else 1.0


def phoneme_distance(a: str, b: str) -> float:
    """Normalized Levenshtein over the IPA characters of a single phoneme token."""
    if a == b:
        return 0.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m] / max(n, m)


class Aligner:
    def __init__(self, cfg: AlignmentConfig, unit: Unit) -> None:
        self.cfg = cfg
        self.unit = unit
        self._lex: Callable[[str, str], float] = (
            word_distance if unit == "word" else phoneme_distance
        )
        self._lex_threshold = 0.0 if unit == "word" else cfg.phoneme_match_threshold
        self._ref: list[Token[Reference]] = []
        self._live: list[Token[Live]] = []
        self._ref_resolved = 0.0
        self._live_resolved = 0.0

    # --- input ---------------------------------------------------------------------

    def add_reference(self, tokens: list[Token[Reference]]) -> None:
        for t in tokens:
            if t.role is not Reference:
                raise TypeError("add_reference given a non-Reference token")
        self._ref.extend(tokens)
        self._ref.sort(key=lambda t: t.start.samples)

    def add_live(self, tokens: list[Token[Live]]) -> None:
        for t in tokens:
            if t.role is not Live:
                raise TypeError("add_live given a non-Live token")
        self._live.extend(tokens)
        self._live.sort(key=lambda t: t.start.samples)

    def advance(self, ref_resolved: TransportTime, live_resolved: TransportTime) -> list[TokenPair]:
        self._ref_resolved = max(self._ref_resolved, ref_resolved.seconds)
        self._live_resolved = max(self._live_resolved, live_resolved.seconds)
        return self._emit()

    def finish(self) -> list[TokenPair]:
        self._ref_resolved = self._live_resolved = _INF
        return self._emit()

    # --- decidability --------------------------------------------------------------

    def _ref_decidable(self, r: Token[Reference]) -> bool:
        return self._live_resolved >= r.start.seconds + self.cfg.tolerance_s + self.cfg.max_lag_s

    def _live_decidable(self, live: Token[Live]) -> bool:
        return self._ref_resolved >= live.start.seconds + self.cfg.tolerance_s

    def _pairable(self, r: Token[Reference], live: Token[Live]) -> bool:
        dt = live.start.seconds - r.start.seconds
        return -self.cfg.tolerance_s <= dt <= self.cfg.tolerance_s + self.cfg.max_lag_s

    # --- alignment -----------------------------------------------------------------

    def _emit(self) -> list[TokenPair]:
        out: list[TokenPair] = []
        while True:
            batch = self._emit_once()
            if not batch:
                return out
            out.extend(batch)

    def _emit_once(self) -> list[TokenPair]:
        if not self._ref and not self._live:
            return []
        refs = self._ref[: self.cfg.window_tokens]
        beyond = self._ref[self.cfg.window_tokens :]
        # A live token that could still pair with a reference token past the window edge
        # is not inserted yet; it waits for the window to reach that token.
        insert_limit = beyond[0].start.seconds - self.cfg.tolerance_s if beyond else _INF
        if refs:
            lo = refs[0].start.seconds - self.cfg.tolerance_s
            hi = refs[-1].start.seconds + self.cfg.tolerance_s + self.cfg.max_lag_s
            lives = [t for t in self._live if lo <= t.start.seconds <= hi]
        else:
            lives = list(self._live)
        pairs = self._dp(refs, lives)

        emitted: list[TokenPair] = []
        used_ref: set[int] = set()
        used_live: set[int] = set()
        paired_live = {id(live) for _, live in pairs if live is not None}
        for r, live in pairs:
            if live is None:
                if self._ref_decidable(r):
                    emitted.append(TokenPair(r, None, Verdict.FAIL_MISSED))
                    used_ref.add(id(r))
            elif self._ref_decidable(r) and self._live_decidable(live):
                emitted.append(TokenPair(r, live, self._verdict(r, live)))
                used_ref.add(id(r))
                used_live.add(id(live))
        for live in lives:
            if (
                id(live) not in paired_live
                and live.start.seconds < insert_limit
                and self._live_decidable(live)
            ):
                emitted.append(TokenPair(None, live, Verdict.FAIL_INSERTED))
                used_live.add(id(live))
        # Live tokens older than the reference window can never pair with anything
        # still pending; decide them now so the window keeps moving.
        if refs:
            for live in self._live:
                if (
                    id(live) not in used_live
                    and live.start.seconds < lo
                    and self._live_decidable(live)
                ):
                    emitted.append(TokenPair(None, live, Verdict.FAIL_INSERTED))
                    used_live.add(id(live))

        self._ref = [t for t in self._ref if id(t) not in used_ref]
        self._live = [t for t in self._live if id(t) not in used_live]
        emitted.sort(key=_pair_order)
        return emitted

    def _verdict(self, r: Token[Reference], live: Token[Live]) -> Verdict:
        if self._lex(r.text, live.text) > self._lex_threshold:
            return Verdict.FAIL_LEXICAL
        if abs(live.start.seconds - r.start.seconds) > self.cfg.tolerance_s:
            return Verdict.FAIL_TIMING
        return Verdict.MATCH

    def _sub_cost(self, r: Token[Reference], live: Token[Live]) -> float:
        if not self._pairable(r, live):
            return _INF
        lex = self._lex(r.text, live.text)
        dt = abs(live.start.seconds - r.start.seconds)
        timing = 0.0 if dt <= self.cfg.tolerance_s else self.cfg.timing_fail_weight
        # A lexical substitution must cost less than delete+insert (2) or it is never chosen;
        # a perfect-but-late pair must cost less than a wrong-but-on-time one. The last term
        # only breaks ties: among equally good pairings prefer the closer one in time, so a
        # repeated token ("the … the", "n … n") pairs with its own occurrence.
        closeness = 0.1 * dt / (self.cfg.tolerance_s + self.cfg.max_lag_s)
        return lex * 1.5 + timing * 0.4 + closeness

    def _dp(
        self, refs: list[Token[Reference]], lives: list[Token[Live]]
    ) -> list[tuple[Token[Reference], Token[Live] | None]]:
        n, m = len(refs), len(lives)
        cost = [[0.0] * (m + 1) for _ in range(n + 1)]
        back = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diag, 1 up (miss), 2 left (insert)
        for i in range(1, n + 1):
            cost[i][0] = float(i)
            back[i][0] = 1
        for j in range(1, m + 1):
            cost[0][j] = float(j)
            back[0][j] = 2
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diag = cost[i - 1][j - 1] + self._sub_cost(refs[i - 1], lives[j - 1])
                up = cost[i - 1][j] + 1.0
                left = cost[i][j - 1] + 1.0
                best = min(diag, up, left)
                cost[i][j] = best
                back[i][j] = (
                    0 if best == diag and not math.isinf(diag) else (1 if best == up else 2)
                )
        pairs: list[tuple[Token[Reference], Token[Live] | None]] = []
        i, j = n, m
        while i > 0 or j > 0:
            move = back[i][j]
            if move == 0:
                pairs.append((refs[i - 1], lives[j - 1]))
                i, j = i - 1, j - 1
            elif move == 1:
                pairs.append((refs[i - 1], None))
                i -= 1
            else:
                j -= 1
        pairs.reverse()
        return pairs


def _pair_order(p: TokenPair) -> float:
    t = p.reference if p.reference is not None else p.live
    assert t is not None
    return t.start.seconds
