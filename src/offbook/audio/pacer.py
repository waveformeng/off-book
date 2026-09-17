"""The reference pacer: advances the reference cursor in lockstep with mic capture."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from offbook.audio.decode import ReferencePCM


class PacerLockstepError(RuntimeError):
    pass


class ReferencePacer:
    def __init__(self, reference: ReferencePCM) -> None:
        self._reference = reference
        self._cursor = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def done(self) -> bool:
        return self._cursor >= self._reference.n_samples

    def advance(self, n_frames: int, mic_total_after: int) -> NDArray[np.float32]:
        """Return the next `n_frames` of reference audio. `mic_total_after` is the mic
        frame counter after the block that triggered this call; it must equal the cursor
        after advancing, or the two streams have lost lockstep."""
        start = self._cursor
        self._cursor += n_frames
        if self._cursor != mic_total_after:
            raise PacerLockstepError(
                f"reference cursor {self._cursor} != mic frames {mic_total_after}"
            )
        return self._reference._slice_for_analysis(start, self._cursor)
