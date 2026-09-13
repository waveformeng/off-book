"""File → PCM, decoded in software. `ReferencePCM` has no output path by construction."""

from __future__ import annotations

from typing import final

import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray

from offbook.audio.sources import BackingTrackFile, ReferenceVocalFile
from offbook.roles import Reference


def _read_mono(path: str, rate: int) -> NDArray[np.float32]:
    data, file_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if file_rate != rate:
        mono = soxr.resample(mono, file_rate, rate, quality="HQ").astype(np.float32)
    return np.ascontiguousarray(mono)


@final
class ReferencePCM:
    """The decoded reference vocal, mono at the transport rate.

    Deliberately not array-like: no `__array__`, no `__len__` over samples exposed to
    output code paths. The only reader is `ReferencePacer`, which hands slices to the
    reference recognizer as `Frames[Reference]`.
    """

    role = Reference

    def __init__(self, source: ReferenceVocalFile, rate: int) -> None:
        self.source = source
        self.rate = rate
        self._samples = _read_mono(str(source.path), rate)

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.rate

    def _slice_for_analysis(self, start: int, end: int) -> NDArray[np.float32]:
        out = np.zeros(end - start, dtype=np.float32)
        lo, hi = max(start, 0), min(end, self.n_samples)
        if hi > lo:
            out[lo - start : hi - start] = self._samples[lo:hi]
        return out


@final
class BackingPCM:
    """The backing track, stereo at the output device rate. Output only; never analyzed."""

    def __init__(self, source: BackingTrackFile, rate: int) -> None:
        self.source = source
        self.rate = rate
        data, file_rate = sf.read(str(source.path), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        if data.shape[1] > 2:
            data = data[:, :2]
        if file_rate != rate:
            data = soxr.resample(data, file_rate, rate, quality="HQ")
        self.samples: NDArray[np.float32] = np.ascontiguousarray(data.astype(np.float32))

    @property
    def n_frames(self) -> int:
        return len(self.samples)
