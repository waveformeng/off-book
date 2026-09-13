"""Reference-only diagnostic: run one vocal file through a recognizer and show the result.

Used to tune transcription before trusting it for scoring. `--full` also decodes the whole
file in one shot — the recognizer's best effort — and reports the streaming chunker's word
error rate against it, so chunker damage can be told apart from model limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr
from numpy.typing import NDArray
from rich.console import Console

from offbook.asr.base import Backend, Frames, RawToken, Token
from offbook.asr.chunker import Recognizer
from offbook.asr.registry import build_backends
from offbook.audio.decode import ReferencePCM
from offbook.audio.sources import ReferenceVocalFile
from offbook.clock import ANALYSIS_RATE, TransportTime
from offbook.config import SessionConfig
from offbook.roles import Reference


def edit_distance_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Levenshtein distance over tokens divided by the reference length."""
    n, m = len(reference), len(hypothesis)
    row = list(range(m + 1))
    for i in range(1, n + 1):
        prev, row[0] = row[0], i
        for j in range(1, m + 1):
            cur = min(row[j] + 1, row[j - 1] + 1, prev + (reference[i - 1] != hypothesis[j - 1]))
            prev, row[j] = row[j], cur
    return row[m] / max(n, 1)


@dataclass(frozen=True)
class TranscribeResult:
    chunked: list[Token[Reference]]
    full: list[RawToken] | None
    error_rate_vs_full: float | None


def _analysis_pcm(reference: ReferencePCM) -> NDArray[np.float32]:
    # Same path a session takes: transport rate → 16 kHz through soxr HQ.
    frames = reference._slice_for_analysis(0, reference.n_samples)
    if reference.rate == ANALYSIS_RATE:
        return frames
    return np.ascontiguousarray(
        soxr.resample(frames, reference.rate, ANALYSIS_RATE, quality="HQ"), dtype=np.float32
    )


def transcribe_reference(
    cfg: SessionConfig,
    source: ReferenceVocalFile,
    *,
    full: bool,
    seconds: float | None = None,
    block: int = 1024,
    backend: Backend | None = None,
) -> TranscribeResult:
    backend = backend or build_backends(cfg)[0]
    reference = ReferencePCM(source, 48_000)
    pcm = _analysis_pcm(reference)
    if seconds is not None:
        pcm = pcm[: int(seconds * ANALYSIS_RATE)]

    recognizer: Recognizer[Reference] = Recognizer(Reference, backend, cfg.chunking)
    chunked: list[Token[Reference]] = []
    for start in range(0, len(pcm), block):
        frames: Frames[Reference] = Frames(
            Reference, pcm[start : start + block], TransportTime(start, ANALYSIS_RATE)
        )
        chunked.extend(recognizer.feed(frames))
    chunked.extend(recognizer.flush())
    # Emission order is not time order: agreement can hold a token back a hop while a
    # later one goes out. Everything downstream sorts by transport time; so do we.
    chunked.sort(key=lambda t: t.start.samples)

    one_shot = backend.transcribe(pcm) if full else None
    rate = (
        edit_distance_rate([t.text for t in one_shot], [t.text for t in chunked])
        if one_shot is not None
        else None
    )
    return TranscribeResult(chunked, one_shot, rate)


def print_result(result: TranscribeResult, path: Path, console: Console | None = None) -> None:
    console = console or Console(highlight=False)
    console.print(f"[bold]{path}[/bold]")
    console.print(f"[dim]streaming chunker: {len(result.chunked)} tokens[/dim]")
    console.print(
        "  " + " ".join(f"{t.text}[dim]@{t.start.seconds:.1f}[/dim]" for t in result.chunked)
    )
    if result.full is not None:
        console.print(f"[dim]one-shot decode:   {len(result.full)} tokens[/dim]")
        console.print("  " + " ".join(f"{t.text}[dim]@{t.start_s:.1f}[/dim]" for t in result.full))
        assert result.error_rate_vs_full is not None
        console.print(
            f"[bold]chunker error rate vs one-shot: {result.error_rate_vs_full:.3f}[/bold]  "
            "(0 = the streaming path loses nothing over decoding the whole file at once)"
        )
