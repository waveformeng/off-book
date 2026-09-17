"""Running aggregate score.

    match_rate = Σ points / Σ verdicts, where MATCH = 1, FAIL_TIMING = 1 − timing_fail_weight,
                 every other FAIL = 0. With graded timing a MATCH earns
                 1 − timing_fail_weight · |dt| / tolerance instead of a flat 1.
    score      = 100 · match_rate · min(1, transport_position / reference_duration)

It climbs from zero through the song and lands at 100 · match_rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import final

from offbook.compare.verdict import TokenPair, Verdict


@final
@dataclass
class RunningScore:
    timing_fail_weight: float
    reference_duration_s: float
    graded_timing: bool = False
    tolerance_s: float = 1.0
    counts: dict[Verdict, int] = field(default_factory=lambda: {v: 0 for v in Verdict})
    points: float = 0.0
    verdicts: int = 0

    def add(self, pair: TokenPair) -> None:
        self.counts[pair.verdict] += 1
        self.verdicts += 1
        if pair.verdict is Verdict.MATCH:
            if self.graded_timing and pair.dt_s is not None:
                fraction = min(abs(pair.dt_s) / self.tolerance_s, 1.0)
                self.points += 1.0 - self.timing_fail_weight * fraction
            else:
                self.points += 1.0
        elif pair.verdict is Verdict.FAIL_TIMING:
            self.points += 1.0 - self.timing_fail_weight

    @property
    def match_rate(self) -> float:
        return self.points / self.verdicts if self.verdicts else 0.0

    def progress(self, transport_position_s: float) -> float:
        if self.reference_duration_s <= 0:
            return 1.0
        return min(1.0, transport_position_s / self.reference_duration_s)

    def score(self, transport_position_s: float) -> float:
        return 100.0 * self.match_rate * self.progress(transport_position_s)
