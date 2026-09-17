"""The transport clock.

The transport clock is the mic input's frame counter: frames captured at the device
sample rate since session start. The reference pacer advances in lockstep with it, so
every `Frames[Reference]` and `Frames[Live]` (and every token derived from them) carries
a `TransportTime` on the same axis. All comparison happens on this axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, slots=True)
class TransportTime:
    """A position on the transport clock, stored as an exact sample index at `rate` Hz."""

    samples: int
    rate: int

    @property
    def seconds(self) -> float:
        return self.samples / self.rate

    @staticmethod
    def from_seconds(seconds: float, rate: int) -> TransportTime:
        return TransportTime(int(round(seconds * rate)), rate)

    def __add__(self, other: TransportTime) -> TransportTime:
        self._same_rate(other)
        return TransportTime(self.samples + other.samples, self.rate)

    def __sub__(self, other: TransportTime) -> TransportTime:
        self._same_rate(other)
        return TransportTime(self.samples - other.samples, self.rate)

    def _same_rate(self, other: TransportTime) -> None:
        if other.rate != self.rate:
            raise ValueError(f"transport rate mismatch: {self.rate} vs {other.rate}")

    def __lt__(self, other: TransportTime) -> bool:
        return self.seconds < other.seconds

    def __le__(self, other: TransportTime) -> bool:
        return self.seconds <= other.seconds

    def __gt__(self, other: TransportTime) -> bool:
        return self.seconds > other.seconds

    def __ge__(self, other: TransportTime) -> bool:
        return self.seconds >= other.seconds


ANALYSIS_RATE = 16_000
"""Sample rate every recognizer consumes. Both streams are resampled to this identically."""
