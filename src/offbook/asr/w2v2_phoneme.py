"""B. Phoneme-level recognizer: wav2vec2 CTC over an espeak IPA phoneme inventory (MLX port).

Greedy CTC: argmax per 20 ms frame, collapse repeats, drop blanks. Every emitted phoneme
carries the frame span it was emitted from and the mean posterior over those frames.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from offbook.asr.base import RawToken, RecognizerSpec
from offbook.asr.hashing import pinned_snapshot, sha256_files, verify_hash
from offbook.asr.mlx_wav2vec2.load import load_model, load_vocab
from offbook.asr.mlx_wav2vec2.model import Wav2Vec2ForCTC
from offbook.clock import ANALYSIS_RATE
from offbook.compare.normalize import normalize_phoneme
from offbook.config import ChunkingConfig

LOCK_KEY = "w2v2-phoneme"
_WEIGHT_FILES = ["config.json", "pytorch_model.bin", "vocab.json"]
_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}
_SPECIAL = {"<s>", "<pad>", "</s>", "<unk>", "|"}


class W2v2PhonemeBackend:
    def __init__(self, chunking: ChunkingConfig, dtype: str = "float32") -> None:
        snapshot, model_id, revision = pinned_snapshot(LOCK_KEY, _WEIGHT_FILES)
        model_hash = sha256_files([snapshot / f for f in _WEIGHT_FILES])
        verify_hash(LOCK_KEY, model_hash)
        self._spec = RecognizerSpec(
            impl="w2v2-phoneme",
            unit="phoneme",
            model_id=model_id,
            revision=revision,
            model_hash=model_hash,
            dtype=dtype,
            chunking=chunking,
        )
        self._model: Wav2Vec2ForCTC = load_model(snapshot, revision, _DTYPES[dtype])
        self._vocab = load_vocab(snapshot)
        self._blank = 0
        self._stride = self._model.cfg.total_stride
        self._min_samples = self._model.cfg.receptive_field

    @classmethod
    def sharing_weights_with(cls, other: W2v2PhonemeBackend) -> W2v2PhonemeBackend:
        inst = cls.__new__(cls)
        inst.__dict__.update(other.__dict__)  # stateless between calls; weights shared
        return inst

    @property
    def spec(self) -> RecognizerSpec:
        return self._spec

    def transcribe(self, pcm: NDArray[np.float32]) -> list[RawToken]:
        if len(pcm) < self._min_samples:
            return []
        x = (pcm - pcm.mean()) / np.sqrt(pcm.var() + 1e-7)
        logits = self._model(mx.array(x)[None])
        probs = mx.softmax(logits[0].astype(mx.float32), axis=-1)
        ids = mx.argmax(probs, axis=-1)
        mx.eval(ids, probs)
        id_arr = np.array(ids)
        best = np.array(mx.max(probs, axis=-1))
        return self._collapse(id_arr, best)

    def _collapse(self, ids: NDArray[np.int64], best: NDArray[np.float32]) -> list[RawToken]:
        out: list[RawToken] = []
        frame_s = self._stride / ANALYSIS_RATE
        i, n = 0, len(ids)
        while i < n:
            tok = int(ids[i])
            j = i
            while j < n and int(ids[j]) == tok:
                j += 1
            if tok != self._blank:
                text = normalize_phoneme(self._vocab[tok])
                if text and text not in _SPECIAL:
                    out.append(RawToken(text, i * frame_s, j * frame_s, float(np.mean(best[i:j]))))
            i = j
        return out
