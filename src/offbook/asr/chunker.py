"""Chunked streaming with overlap, shared by every backend.

Every `hop` samples, the last `window` samples are re-decoded. A token is emitted when

  * it ends before the resolve line (window end − `resolve_margin`),
  * it starts past the window's left-edge guard (the first second of a window that cuts
    into the middle of a phrase decodes badly),
  * the previous decode produced the same token, with its start within `agree`
    (recognizer timestamps jitter between overlapping windows; a token that only one
    decode saw is not trusted yet), and
  * nothing overlapping it has been emitted already.

Chunk boundaries are functions of sample counts only, so the same audio produces the
same windows, the same tokens and the same timestamps every run, regardless of callback
timing. The final flush emits everything left without requiring agreement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic

import numpy as np
from numpy.typing import NDArray

from offbook.asr.base import Backend, Frames, RecognizerSpec, Token
from offbook.clock import ANALYSIS_RATE, TransportTime
from offbook.config import ChunkingConfig
from offbook.roles import Role


@dataclass(frozen=True, slots=True)
class _Abs:
    """A backend token placed on the analysis clock, in samples."""

    text: str
    start: int
    end: int
    confidence: float


class Recognizer(Generic[Role]):
    """One stream's streaming recognizer: a chunker around one backend instance."""

    def __init__(self, role: type[Role], backend: Backend, chunking: ChunkingConfig) -> None:
        self.role: type[Role] = role
        self.backend = backend
        self.chunking = chunking
        self._window = int(round(chunking.window_s * ANALYSIS_RATE))
        self._hop = int(round(chunking.hop_s * ANALYSIS_RATE))
        self._margin = int(round(chunking.resolve_margin_s * ANALYSIS_RATE))
        self._agree = int(round(chunking.agree_s * ANALYSIS_RATE))
        self._edge = int(round(chunking.edge_guard_s * ANALYSIS_RATE))

        self._buf: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._buf_start = 0  # absolute sample index of _buf[0]
        self._received = 0  # absolute sample index of end of received audio
        self._next_decode_end = self._hop
        self._resolved_until = 0
        self._previous: list[_Abs] = []  # every token of the previous decode
        self._emitted: list[_Abs] = []  # emitted tokens still inside the window

    @property
    def spec(self) -> RecognizerSpec:
        return self.backend.spec

    @property
    def resolved_until(self) -> TransportTime:
        """Transport time before which no further tokens will be emitted on this stream."""
        return TransportTime(self._resolved_until, ANALYSIS_RATE)

    def feed(self, frames: Frames[Role]) -> list[Token[Role]]:
        if frames.role is not self.role:
            raise TypeError(f"{self.role.__name__} recognizer fed {frames.role.__name__} frames")
        if frames.start.rate != ANALYSIS_RATE:
            raise ValueError(f"frames must be at {ANALYSIS_RATE} Hz")
        if frames.start.samples != self._received:
            raise ValueError(
                f"discontinuous frames: expected sample {self._received}, "
                f"got {frames.start.samples}"
            )
        self._buf = np.concatenate([self._buf, frames.pcm])
        self._received += len(frames.pcm)

        out: list[Token[Role]] = []
        while self._received >= self._next_decode_end:
            decode_end = self._next_decode_end
            self._next_decode_end += self._hop
            out.extend(self._decode(decode_end, decode_end - self._margin, final=False))
        return out

    def flush(self) -> list[Token[Role]]:
        """End of stream: decode whatever remains and resolve everything."""
        if self._received <= self._resolved_until:
            return []
        return self._decode(self._received, self._received, final=True)

    def _decode(self, decode_end: int, resolve_line: int, *, final: bool) -> list[Token[Role]]:
        window_start = max(0, decode_end - self._window)
        self._trim(window_start)
        pcm = self._buf[window_start - self._buf_start : decode_end - self._buf_start]
        raw = self.backend.transcribe(np.ascontiguousarray(pcm))
        now = time.monotonic()

        current = [
            _Abs(
                r.text,
                window_start + int(round(r.start_s * ANALYSIS_RATE)),
                window_start + int(round(r.end_s * ANALYSIS_RATE)),
                r.confidence,
            )
            for r in raw
        ]
        guard = window_start + self._edge if window_start > 0 else 0
        emitted: list[Token[Role]] = []
        for t in current:
            if t.end > resolve_line or t.start < guard:
                continue
            if not final and not any(self._same(p, t) for p in self._previous):
                continue
            if any(self._same(e, t) or self._overlaps(e, t) for e in self._emitted):
                continue
            self._emitted.append(t)
            emitted.append(self._token(t, now))

        self._previous = current
        self._resolved_until = max(self._resolved_until, resolve_line)
        self._emitted = [e for e in self._emitted if e.start >= window_start]
        emitted.sort(key=lambda tok: tok.start.samples)
        return emitted

    def _same(self, a: _Abs, b: _Abs) -> bool:
        return a.text == b.text and abs(a.start - b.start) <= self._agree

    @staticmethod
    def _overlaps(a: _Abs, b: _Abs) -> bool:
        """Two tokens occupying the same span are one token the recognizer re-spelled."""
        overlap = min(a.end, b.end) - max(a.start, b.start)
        return overlap > 0 and overlap >= 0.5 * min(
            max(a.end - a.start, 1), max(b.end - b.start, 1)
        )

    def _token(self, t: _Abs, now: float) -> Token[Role]:
        return Token(
            role=self.role,
            text=t.text,
            start=TransportTime(t.start, ANALYSIS_RATE),
            end=TransportTime(max(t.end, t.start + 1), ANALYSIS_RATE),
            confidence=t.confidence,
            resolved_wall=now,
        )

    def _trim(self, keep_from: int) -> None:
        if keep_from > self._buf_start:
            self._buf = self._buf[keep_from - self._buf_start :]
            self._buf_start = keep_from
