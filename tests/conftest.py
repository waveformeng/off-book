from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

RATE = 16_000


@pytest.fixture
def silence_wav(tmp_path: Path) -> Path:
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(RATE * 2, dtype=np.float32), RATE)
    return path
