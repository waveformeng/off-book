"""Load the pinned HF checkpoint into the MLX model, converting once to safetensors."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from offbook.asr.mlx_wav2vec2.checkpoint import load_torch_checkpoint
from offbook.asr.mlx_wav2vec2.model import Wav2Vec2Config, Wav2Vec2ForCTC, convert_state_dict

CONVERTED_DIR = Path(__file__).resolve().parents[4] / ".models"


def load_config(snapshot: Path) -> Wav2Vec2Config:
    with (snapshot / "config.json").open() as f:
        return Wav2Vec2Config.from_hf(json.load(f))


def load_vocab(snapshot: Path) -> list[str]:
    with (snapshot / "vocab.json").open() as f:
        v: dict[str, int] = json.load(f)
    vocab = [""] * len(v)
    for tok, i in v.items():
        vocab[i] = tok
    return vocab


def converted_weights(snapshot: Path, revision: str) -> Path:
    path = CONVERTED_DIR / f"w2v2-phoneme-{revision}.safetensors"
    if not path.exists():
        CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
        sd = convert_state_dict(load_torch_checkpoint(snapshot / "pytorch_model.bin"))
        mx.save_safetensors(str(path), {k: mx.array(v) for k, v in sd.items()})
    return path


def load_model(snapshot: Path, revision: str, dtype: mx.Dtype) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC(load_config(snapshot))
    weights = mx.load(str(converted_weights(snapshot, revision)))
    if not isinstance(weights, dict):
        raise TypeError("converted weights must be a flat dict")
    model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
    mx.eval(model.parameters())
    return model
