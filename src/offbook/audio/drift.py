"""Drift between the transport clock (mic ADC) and the backing-track clock (DAC).

At each mic block we know the wall time its first frame hit the ADC. From the most recent
output block we know which backing frame hit the DAC at which wall time, so we can say
where the backing track was, in its own clock, at the instant the mic frame was captured.
Drift is that position minus the transport position, relative to the first block.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DriftSample:
    transport_s: float
    drift_s: float
    """Backing-track position minus transport position, in seconds, relative to the first
    block. Positive means the output clock runs fast relative to the mic clock."""


@dataclass
class DriftMonitor:
    in_rate: int
    out_rate: int
    interval_s: float
    duplex: bool
    samples: list[DriftSample] = field(default_factory=list)
    latest: DriftSample | None = None
    _offset0: float | None = None
    _next_log_s: float = 0.0

    def observe(
        self,
        mic_start: int,
        adc_time: float | None,
        dac_frames_before: int,
        dac_time: float | None,
    ) -> DriftSample | None:
        """Called from the input path on every block. Returns a sample when one is logged."""
        transport_s = mic_start / self.in_rate
        if adc_time is None or dac_time is None:
            backing_s = dac_frames_before / self.out_rate
        else:
            backing_s = dac_frames_before / self.out_rate + (adc_time - dac_time)
        offset = backing_s - transport_s
        if self._offset0 is None:
            self._offset0 = offset
        self.latest = DriftSample(transport_s, offset - self._offset0)
        if transport_s >= self._next_log_s:
            self._next_log_s += self.interval_s
            self.samples.append(self.latest)
            return self.latest
        return None

    @property
    def max_abs_s(self) -> float:
        return max((abs(s.drift_s) for s in self.samples), default=0.0)

    @property
    def final_s(self) -> float:
        return self.samples[-1].drift_s if self.samples else 0.0
