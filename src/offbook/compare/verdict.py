"""Per-token verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import final

from offbook.asr.base import Token
from offbook.roles import Live, Reference


class Verdict(StrEnum):
    MATCH = "MATCH"
    FAIL_LEXICAL = "FAIL_LEXICAL"
    """Paired in time, but the words/phonemes differ."""
    FAIL_TIMING = "FAIL_TIMING"
    """Same word/phoneme, but outside the timing tolerance (inside max lag)."""
    FAIL_MISSED = "FAIL_MISSED"
    """Reference token with no live counterpart in its window."""
    FAIL_INSERTED = "FAIL_INSERTED"
    """Live token with no reference counterpart in its window."""

    @property
    def is_match(self) -> bool:
        return self is Verdict.MATCH


@final
@dataclass(frozen=True, slots=True)
class TokenPair:
    reference: Token[Reference] | None
    live: Token[Live] | None
    verdict: Verdict

    def __post_init__(self) -> None:
        if self.reference is None and self.live is None:
            raise ValueError("a TokenPair needs at least one side")
        if self.reference is not None and self.reference.role is not Reference:
            raise TypeError("TokenPair.reference must carry a Reference token")
        if self.live is not None and self.live.role is not Live:
            raise TypeError("TokenPair.live must carry a Live token")
        if (self.reference is None) != (self.verdict is Verdict.FAIL_INSERTED):
            raise ValueError("FAIL_INSERTED iff no reference token")
        if (self.live is None) != (self.verdict is Verdict.FAIL_MISSED):
            raise ValueError("FAIL_MISSED iff no live token")

    @property
    def dt_s(self) -> float | None:
        """live.start − reference.start; positive means the singer is late."""
        if self.reference is None or self.live is None:
            return None
        return self.live.start.seconds - self.reference.start.seconds
