"""A console that turns the session readout into JSON events for the web UI."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from offbook.asr.base import RecognizerSpec, Token
from offbook.audio.drift import DriftSample
from offbook.compare.verdict import TokenPair, Verdict
from offbook.roles import Live, Reference
from offbook.session.console import SessionConsole

Event = dict[str, Any]

# Points the stage receives per mic block. The block is decimated, not averaged: the stage
# draws it as an oscilloscope trace the way the karaoke app draws its analyser buffer, and
# an averaged block would flatten the peaks that make a voice look like a voice.
WAVEFORM_POINTS = 256


class WaveformBuffer:
    """The LIVE VOCAL's recent trace, for the stage view. Deliberately not an event: at
    one block every ~20 ms it would swamp the history that `/api/events` replays to every
    late-joining browser. Frames carry a sequence number so a subscriber polls for what it
    has not yet seen; anything older than the ring is simply gone."""

    def __init__(self, maxlen: int = 256) -> None:
        self._lock = threading.Lock()
        self._frames: deque[tuple[int, Event]] = deque(maxlen=maxlen)
        self._seq = 0

    def push(self, frame: Event) -> None:
        with self._lock:
            self._seq += 1
            self._frames.append((self._seq, frame))

    def since(self, seq: int) -> tuple[int, list[Event]]:
        """Frames newer than `seq`, and the sequence number to ask from next time."""
        with self._lock:
            new = [f for s, f in self._frames if s > seq]
            return self._seq, new


def trace(block: NDArray[np.float32], points: int = WAVEFORM_POINTS) -> list[float]:
    """A mono block decimated to at most `points` samples, evenly spaced."""
    mono = np.asarray(block, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return []
    if mono.size > points:
        mono = mono[np.linspace(0, mono.size - 1, points).astype(int)]
    return [round(float(v), 3) for v in mono]


def _token(t: Token[Reference] | Token[Live]) -> Event:
    return {
        "text": t.text,
        "start_s": round(t.start.seconds, 3),
        "end_s": round(t.end.seconds, 3),
        "confidence": round(t.confidence, 3),
    }


class EventConsole(SessionConsole):
    """Emits every readout as an event; also prints to the terminal unless quiet."""

    def __init__(
        self,
        sink: queue.Queue[Event | None],
        quiet: bool = True,
        waveform: WaveformBuffer | None = None,
    ) -> None:
        super().__init__(quiet=quiet)
        self.sink = sink
        self.waveform = waveform
        self.t0 = time.monotonic()
        self._last_tick = -1.0
        self._last_tentative: tuple[list[Event], list[Event]] | None = None

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

    def tentative(self, ref: list[Token[Reference]], live: list[Token[Live]], t_s: float) -> None:
        # Recognizers only decode once per hop, so this is unchanged for most blocks; the
        # event carries the whole current list (the UI replaces, never appends).
        current = ([_token(t) for t in ref], [_token(t) for t in live])
        if current == self._last_tentative:
            return
        self._last_tentative = current
        self.emit("tentative", transport_s=round(t_s, 3), reference=current[0], live=current[1])

    def live_audio(self, block: NDArray[np.float32], t_s: float) -> None:
        if self.waveform is None:
            return
        self.waveform.push({"transport_s": round(t_s, 3), "pcm": trace(block)})

    def verdict(self, pair: TokenPair, score: float, match_rate: float) -> None:
        super().verdict(pair, score, match_rate)
        self.emit(
            "verdict",
            reference=_token(pair.reference) if pair.reference is not None else None,
            live=_token(pair.live) if pair.live is not None else None,
            verdict=pair.verdict.value,
            dt_s=round(pair.dt_s, 3) if pair.dt_s is not None else None,
            score=round(score, 2),
            match_rate=round(match_rate, 4),
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
