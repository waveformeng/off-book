"""The streaming chunker: sample-defined windows, one emission per token, absolute times."""

import numpy as np
from numpy.typing import NDArray

from offbook.asr.base import Frames, RawToken, RecognizerSpec
from offbook.asr.chunker import Recognizer
from offbook.clock import ANALYSIS_RATE, TransportTime
from offbook.config import ChunkingConfig
from offbook.roles import Live, Reference

SPEC = RecognizerSpec("fake", "word", "fake/model", "rev", "hash", "float32", ChunkingConfig())


class SpikeBackend:
    """Emits a token for every impulse in the window, 100 ms long, named by amplitude."""

    spec = SPEC

    def __init__(self) -> None:
        self.calls: list[int] = []

    def transcribe(self, pcm: NDArray[np.float32]) -> list[RawToken]:
        self.calls.append(len(pcm))
        idx = np.flatnonzero(pcm > 0.5)
        return [
            RawToken(f"t{int(round(pcm[i] * 10))}", i / ANALYSIS_RATE, i / ANALYSIS_RATE + 0.1, 1.0)
            for i in idx
        ]


def _audio_with_spikes(seconds: float, spikes: dict[float, float]) -> NDArray[np.float32]:
    pcm = np.zeros(int(seconds * ANALYSIS_RATE), dtype=np.float32)
    for t, amp in spikes.items():
        pcm[int(t * ANALYSIS_RATE)] = amp
    return pcm


def _run(pcm: NDArray[np.float32], block: int, cfg: ChunkingConfig) -> list[tuple[str, float]]:
    rec: Recognizer[Live] = Recognizer(Live, SpikeBackend(), cfg)
    out = []
    for start in range(0, len(pcm), block):
        chunk = pcm[start : start + block]
        out += rec.feed(Frames(Live, chunk, TransportTime(start, ANALYSIS_RATE)))
    out += rec.flush()
    return [(t.text, round(t.start.seconds, 3)) for t in out]


def test_tokens_emitted_once_with_absolute_timestamps() -> None:
    cfg = ChunkingConfig(window_s=2.0, hop_s=0.5, resolve_margin_s=0.5)
    spikes = {0.3: 0.6, 1.2: 0.7, 2.9: 0.8, 4.05: 0.9, 5.5: 0.6}
    pcm = _audio_with_spikes(6.0, spikes)
    got = _run(pcm, 1024, cfg)
    assert got == [("t6", 0.3), ("t7", 1.2), ("t8", 2.9), ("t9", 4.05), ("t6", 5.5)]


def test_block_size_does_not_change_output() -> None:
    cfg = ChunkingConfig(window_s=2.0, hop_s=0.5, resolve_margin_s=0.5)
    pcm = _audio_with_spikes(6.0, {0.3: 0.6, 1.2: 0.7, 2.9: 0.8, 4.05: 0.9})
    assert _run(pcm, 1024, cfg) == _run(pcm, 333, cfg) == _run(pcm, 4096, cfg)


def test_straddling_token_is_deferred_not_dropped() -> None:
    # Resolve line at 0.5 s; a token starting at 0.45 s straddles it and must appear once.
    cfg = ChunkingConfig(window_s=2.0, hop_s=1.0, resolve_margin_s=0.5)
    pcm = _audio_with_spikes(3.0, {0.45: 0.6})
    assert _run(pcm, 1024, cfg) == [("t6", 0.45)]


def test_role_mismatch_is_rejected() -> None:
    import pytest

    rec: Recognizer[Reference] = Recognizer(Reference, SpikeBackend(), ChunkingConfig())
    frames = Frames(Live, np.zeros(16, dtype=np.float32), TransportTime(0, ANALYSIS_RATE))
    with pytest.raises(TypeError):
        rec.feed(frames)  # type: ignore[arg-type]


class JitterBackend(SpikeBackend):
    """A spike backend whose timestamps move ±100 ms between decodes and which mis-spells
    a token on every third call — the failure modes real recognizers show at window edges."""

    def transcribe(self, pcm: NDArray[np.float32]) -> list[RawToken]:
        toks = super().transcribe(pcm)
        n = len(self.calls)
        shift = 0.1 if n % 2 else -0.1
        out = []
        for t in toks:
            text = t.text + "x" if n % 3 == 0 else t.text
            out.append(RawToken(text, t.start_s + shift, t.end_s + shift, 1.0))
        return out


def test_jitter_and_respelling_do_not_drop_or_duplicate() -> None:
    cfg = ChunkingConfig(
        window_s=4.0, hop_s=0.5, resolve_margin_s=0.5, agree_s=0.3, edge_guard_s=0.5
    )
    spikes = {0.3: 0.6, 1.2: 0.7, 2.9: 0.8, 4.05: 0.9, 5.5: 0.6, 7.7: 0.7}
    pcm = _audio_with_spikes(9.0, spikes)
    rec: Recognizer[Live] = Recognizer(Live, JitterBackend(), cfg)
    out = []
    for start in range(0, len(pcm), 1024):
        out += rec.feed(
            Frames(Live, pcm[start : start + 1024], TransportTime(start, ANALYSIS_RATE))
        )
    out += rec.flush()
    texts = [t.text.rstrip("x") for t in out]
    assert texts == ["t6", "t7", "t8", "t9", "t6", "t7"], [(t.text, t.start.seconds) for t in out]
    for t, expected in zip(out, sorted(spikes), strict=True):
        assert abs(t.start.seconds - expected) <= 0.11


def test_resolved_until_is_a_promise() -> None:
    """No token is ever emitted with a start before a previously reported resolved_until."""
    cfg = ChunkingConfig(
        window_s=4.0, hop_s=0.5, resolve_margin_s=0.5, agree_s=0.3, edge_guard_s=0.5
    )
    pcm = _audio_with_spikes(9.0, {0.3: 0.6, 1.2: 0.7, 2.9: 0.8, 4.05: 0.9, 5.5: 0.6, 7.7: 0.7})
    rec: Recognizer[Live] = Recognizer(Live, JitterBackend(), cfg)
    promised = 0
    for start in range(0, len(pcm), 1024):
        toks = rec.feed(
            Frames(Live, pcm[start : start + 1024], TransportTime(start, ANALYSIS_RATE))
        )
        assert all(t.start.samples >= promised for t in toks)
        promised = rec.resolved_until.samples
    assert all(t.start.samples >= promised for t in rec.flush())
