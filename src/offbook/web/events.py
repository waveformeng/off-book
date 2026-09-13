"""A console that turns the session readout into JSON events for the web UI."""

from __future__ import annotations

import queue
import time
from typing import Any

from offbook.asr.base import RecognizerSpec, Token
from offbook.audio.drift import DriftSample
from offbook.compare.verdict import TokenPair, Verdict
from offbook.roles import Live, Reference
from offbook.session.console import SessionConsole

Event = dict[str, Any]


def _token(t: Token[Reference] | Token[Live]) -> Event:
    return {
        "text": t.text,
        "start_s": round(t.start.seconds, 3),
        "end_s": round(t.end.seconds, 3),
        "confidence": round(t.confidence, 3),
    }


class EventConsole(SessionConsole):
    """Emits every readout as an event; also prints to the terminal unless quiet."""

    def __init__(self, sink: queue.Queue[Event | None], quiet: bool = True) -> None:
        super().__init__(quiet=quiet)
        self.sink = sink
        self.t0 = time.monotonic()
        self._last_tick = -1.0

    def emit(self, kind: str, **data: Any) -> None:
        self.sink.put({"kind": kind, "wall_s": round(time.monotonic() - self.t0, 3), **data})

    def header(
        self, spec: RecognizerSpec, mode: str, duplex: bool, rate: int, reference_duration_s: float
    ) -> None:
        super().header(spec, mode, duplex, rate, reference_duration_s)
        self.emit(
            "header",
            mode=mode,
            duplex=duplex,
            transport_rate=rate,
            reference_duration_s=reference_duration_s,
            recognizer={
                "impl": spec.impl,
                "unit": spec.unit,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "model_hash": spec.model_hash,
                "dtype": spec.dtype,
            },
        )

    def resolved(self, ref: list[Token[Reference]], live: list[Token[Live]], t_s: float) -> None:
        super().resolved(ref, live, t_s)
        if ref or live:
            self.emit(
                "resolved",
                transport_s=round(t_s, 3),
                reference=[_token(t) for t in ref],
                live=[_token(t) for t in live],
            )
            self._last_tick = t_s
        elif t_s - self._last_tick >= 0.25:
            self.emit("tick", transport_s=round(t_s, 3))
            self._last_tick = t_s

    def verdict(self, pair: TokenPair, score: float) -> None:
        super().verdict(pair, score)
        self.emit(
            "verdict",
            reference=_token(pair.reference) if pair.reference is not None else None,
            live=_token(pair.live) if pair.live is not None else None,
            verdict=pair.verdict.value,
            dt_s=round(pair.dt_s, 3) if pair.dt_s is not None else None,
            score=round(score, 2),
        )

    def drift(self, s: DriftSample) -> None:
        super().drift(s)
        self.emit("drift", transport_s=round(s.transport_s, 3), drift_ms=round(s.drift_s * 1e3, 4))

    def final(self, score: float, match_rate: float, counts: dict[Verdict, int], path: str) -> None:
        super().final(score, match_rate, counts, path)
        self.emit(
            "final",
            score=round(score, 2),
            match_rate=round(match_rate, 4),
            counts={v.value: n for v, n in counts.items()},
            record_path=path,
        )
