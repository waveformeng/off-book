"""Recognizer interface.

A `Backend` is a pure function from a window of 16 kHz PCM to window-relative tokens.
A `Recognizer[Role]` wraps one backend instance in the streaming chunker and stamps every
token with an absolute `TransportTime`. Both streams in a session must be built from
backends with an identical `RecognizerSpec` — checked at session start (see
`offbook.session.runner`), not at comparison time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, final

import numpy as np
from numpy.typing import NDArray

from offbook.clock import TransportTime
from offbook.config import ChunkingConfig
from offbook.roles import Live, Reference, Role

Unit = Literal["word", "phoneme"]


@final
@dataclass(frozen=True, slots=True)
class Frames(Generic[Role]):
    """A run of analysis-rate PCM belonging to exactly one stream."""

    role: type[Role]
    pcm: NDArray[np.float32]
    start: TransportTime

    def __post_init__(self) -> None:
        if self.pcm.ndim != 1 or self.pcm.dtype != np.float32:
            raise TypeError("Frames.pcm must be 1-D float32")

    @property
    def end(self) -> TransportTime:
        return TransportTime(self.start.samples + len(self.pcm), self.start.rate)


@final
@dataclass(frozen=True, slots=True)
class Token(Generic[Role]):
    """A resolved recognizer output on exactly one stream, on the transport clock."""

    role: type[Role]
    text: str
    start: TransportTime
    end: TransportTime
    confidence: float
    resolved_wall: float
    """time.monotonic() when the chunker resolved this token."""


@final
@dataclass(frozen=True, slots=True)
class RawToken:
    """A backend output; times are seconds relative to the start of the window it decoded."""

    text: str
    start_s: float
    end_s: float
    confidence: float


@final
@dataclass(frozen=True, slots=True)
class RecognizerSpec:
    """Everything that must be identical on both streams for their errors to cancel."""

    impl: str
    unit: Unit
    model_id: str
    revision: str
    model_hash: str
    dtype: str
    chunking: ChunkingConfig

    def describe(self) -> str:
        return (
            f"{self.impl} ({self.unit}) {self.model_id}@{self.revision[:12]} "
            f"hash={self.model_hash[:16]} dtype={self.dtype} chunking={self.chunking}"
        )


class Backend(Protocol):
    """Transcribes one window of 16 kHz mono float32 PCM. Must be deterministic."""

    @property
    def spec(self) -> RecognizerSpec: ...

    def transcribe(self, pcm: NDArray[np.float32]) -> list[RawToken]: ...


class RecognizerMismatchError(RuntimeError):
    def __init__(self, reference: RecognizerSpec, live: RecognizerSpec) -> None:
        super().__init__(
            "both streams must use the same recognizer and model revision:\n"
            f"  reference: {reference.describe()}\n"
            f"  live:      {live.describe()}"
        )


def assert_same_recognizer(reference: RecognizerSpec, live: RecognizerSpec) -> None:
    if reference != live:
        raise RecognizerMismatchError(reference, live)


RoleT = type[Reference] | type[Live]
