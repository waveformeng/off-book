"""Live vocal persistence: the mic stream is written to disk for replay."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import final

import numpy as np
import soundfile as sf
from numpy.typing import NDArray


@final
class PerformanceWriter:
    def __init__(self, path: Path, rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = sf.SoundFile(str(path), "w", samplerate=rate, channels=1, subtype="FLOAT")

    def write(self, block: NDArray[np.float32]) -> None:
        self._file.write(block)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> PerformanceWriter:
        return self

    def __exit__(
        self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
