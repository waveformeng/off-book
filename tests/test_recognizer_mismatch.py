import dataclasses

import pytest

from offbook.asr.base import RecognizerMismatchError, RecognizerSpec, assert_same_recognizer
from offbook.config import ChunkingConfig

A = RecognizerSpec("parakeet", "word", "m", "rev-a", "hash-a", "float32", ChunkingConfig())


def test_identical_specs_pass() -> None:
    assert_same_recognizer(A, dataclasses.replace(A))


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "rev-b"},
        {"model_hash": "hash-b"},
        {"impl": "w2v2-phoneme", "unit": "phoneme"},
        {"dtype": "bfloat16"},
        {"chunking": ChunkingConfig(hop_s=0.5)},
    ],
)
def test_any_difference_is_a_hard_failure(change: dict[str, object]) -> None:
    with pytest.raises(RecognizerMismatchError):
        assert_same_recognizer(A, dataclasses.replace(A, **change))  # type: ignore[arg-type]
