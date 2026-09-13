"""Backend construction. Both streams' backends are built here, from the same config."""

from __future__ import annotations

from offbook.asr.base import Backend
from offbook.config import RecognizerConfig, SessionConfig


def build_backends(cfg: SessionConfig) -> tuple[Backend, Backend]:
    """Return (reference backend, live backend): two instances of the same recognizer."""
    rc: RecognizerConfig = cfg.recognizer
    if rc.impl == "parakeet":
        from offbook.asr.parakeet import ParakeetBackend

        a = ParakeetBackend(cfg.chunking, rc.dtype)
        b = (
            ParakeetBackend.sharing_weights_with(a)
            if rc.share_weights
            else ParakeetBackend(cfg.chunking, rc.dtype)
        )
        return a, b
    if rc.impl == "w2v2-phoneme":
        from offbook.asr.w2v2_phoneme import W2v2PhonemeBackend

        c = W2v2PhonemeBackend(cfg.chunking, rc.dtype)
        d = (
            W2v2PhonemeBackend.sharing_weights_with(c)
            if rc.share_weights
            else W2v2PhonemeBackend(cfg.chunking, rc.dtype)
        )
        return c, d
    raise ValueError(f"unknown recognizer impl {rc.impl!r}")
