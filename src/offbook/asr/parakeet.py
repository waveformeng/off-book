"""A. Word-level recognizer: Parakeet TDT 0.6b v2 via parakeet-mlx. Greedy TDT decoding."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
from parakeet_mlx import DecodingConfig, Greedy, from_pretrained
from parakeet_mlx.alignment import AlignedToken
from parakeet_mlx.audio import get_logmel
from parakeet_mlx.parakeet import ParakeetTDT

from offbook.asr.base import RawToken, RecognizerSpec
from offbook.asr.hashing import pinned_snapshot, sha256_files, verify_hash
from offbook.compare.normalize import normalize_word
from offbook.config import ChunkingConfig

LOCK_KEY = "parakeet"
_WEIGHT_FILES = ["config.json", "model.safetensors", "tokenizer.model", "vocab.txt"]
_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


class ParakeetBackend:
    def __init__(self, chunking: ChunkingConfig, dtype: str = "float32") -> None:
        snapshot, model_id, revision = pinned_snapshot(LOCK_KEY, _WEIGHT_FILES)
        model_hash = sha256_files([snapshot / f for f in _WEIGHT_FILES if (snapshot / f).exists()])
        verify_hash(LOCK_KEY, model_hash)
        self._spec = RecognizerSpec(
            impl="parakeet",
            unit="word",
            model_id=model_id,
            revision=revision,
            model_hash=model_hash,
            dtype=dtype,
            chunking=chunking,
        )
        self._model = self._load(snapshot, dtype)
        self._decoding = DecodingConfig(decoding=Greedy())

    @staticmethod
    def _load(snapshot: Path, dtype: str) -> ParakeetTDT:
        model = from_pretrained(str(snapshot), dtype=_DTYPES[dtype])
        if not isinstance(model, ParakeetTDT):
            raise TypeError("pinned parakeet model is not a TDT model")
        mx.eval(model.parameters())
        return model

    @classmethod
    def sharing_weights_with(cls, other: ParakeetBackend) -> ParakeetBackend:
        """A second instance with its own decode state but the same weight tensors."""
        inst = cls.__new__(cls)
        inst._spec = other._spec
        inst._model = other._model
        inst._decoding = DecodingConfig(decoding=Greedy())
        return inst

    @property
    def spec(self) -> RecognizerSpec:
        return self._spec

    def transcribe(self, pcm: NDArray[np.float32]) -> list[RawToken]:
        if len(pcm) < self._model.preprocessor_config.n_fft:
            return []
        mel = get_logmel(mx.array(pcm), self._model.preprocessor_config)
        result = self._model.generate(mel, decoding_config=self._decoding)[0]
        return _merge_pieces(result.tokens)


def _merge_pieces(pieces: list[AlignedToken]) -> list[RawToken]:
    """SentencePiece pieces → words. A piece whose text begins with a space starts a word."""
    words: list[RawToken] = []
    cur_text = ""
    cur_start = 0.0
    cur_end = 0.0
    cur_conf: list[float] = []

    def close() -> None:
        text = normalize_word(cur_text)
        if text:
            words.append(RawToken(text, cur_start, cur_end, float(np.mean(cur_conf))))

    for p in pieces:
        text: str = p.text
        start: float = p.start
        end: float = p.end
        conf: float = p.confidence
        if text.startswith(" ") and cur_text:
            close()
            cur_text, cur_conf = "", []
        if not cur_text:
            cur_start = start
        cur_text += text
        cur_end = end
        cur_conf.append(conf)
    if cur_text:
        close()
    return words
