from pathlib import Path

import numpy as np
import pytest

from offbook.audio.decode import ReferencePCM
from offbook.audio.pacer import PacerLockstepError, ReferencePacer
from offbook.audio.sources import ReferenceVocalFile


def test_pacer_advances_in_lockstep_and_zero_pads(silence_wav: Path) -> None:
    ref = ReferencePCM(ReferenceVocalFile(silence_wav), 16_000)
    pacer = ReferencePacer(ref)
    a = pacer.advance(1024, 1024)
    b = pacer.advance(1024, 2048)
    assert len(a) == len(b) == 1024 and a.dtype == np.float32
    assert not pacer.done
    tail = pacer.advance(ref.n_samples, ref.n_samples + 2048)  # past the end → zeros
    assert pacer.done and np.all(tail[-2048:] == 0)


def test_pacer_fails_loudly_when_out_of_step(silence_wav: Path) -> None:
    pacer = ReferencePacer(ReferencePCM(ReferenceVocalFile(silence_wav), 16_000))
    with pytest.raises(PacerLockstepError):
        pacer.advance(1024, 1000)
