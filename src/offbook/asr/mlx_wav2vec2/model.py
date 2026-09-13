"""wav2vec2 (large, stable-layer-norm variant) with a CTC head, in MLX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Wav2Vec2Config:
    conv_dim: tuple[int, ...]
    conv_kernel: tuple[int, ...]
    conv_stride: tuple[int, ...]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    num_conv_pos_embeddings: int
    num_conv_pos_embedding_groups: int
    layer_norm_eps: float
    vocab_size: int
    do_stable_layer_norm: bool
    feat_extract_norm: str

    @staticmethod
    def from_hf(c: dict[str, Any]) -> Wav2Vec2Config:
        return Wav2Vec2Config(
            conv_dim=tuple(c["conv_dim"]),
            conv_kernel=tuple(c["conv_kernel"]),
            conv_stride=tuple(c["conv_stride"]),
            hidden_size=c["hidden_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            intermediate_size=c["intermediate_size"],
            num_conv_pos_embeddings=c["num_conv_pos_embeddings"],
            num_conv_pos_embedding_groups=c["num_conv_pos_embedding_groups"],
            layer_norm_eps=c["layer_norm_eps"],
            vocab_size=c["vocab_size"],
            do_stable_layer_norm=c["do_stable_layer_norm"],
            feat_extract_norm=c["feat_extract_norm"],
        )

    @property
    def total_stride(self) -> int:
        return int(np.prod(self.conv_stride))

    @property
    def receptive_field(self) -> int:
        rf = 1
        for k, s in zip(reversed(self.conv_kernel), reversed(self.conv_stride), strict=True):
            rf = (rf - 1) * s + k
        return rf


class ConvLayer(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int, eps: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, bias=True)
        self.layer_norm = nn.LayerNorm(cout, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.layer_norm(self.conv(x)))


class FeatureEncoder(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        if cfg.feat_extract_norm != "layer":
            raise NotImplementedError("only feat_extract_norm='layer' is ported")
        dims = (1, *cfg.conv_dim)
        self.conv_layers = [
            ConvLayer(dims[i], dims[i + 1], cfg.conv_kernel[i], cfg.conv_stride[i], 1e-5)
            for i in range(len(cfg.conv_dim))
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.conv_layers:
            x = layer(x)
        return x


class FeatureProjection(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.conv_dim[-1], eps=cfg.layer_norm_eps)
        self.projection = nn.Linear(cfg.conv_dim[-1], cfg.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.projection(self.layer_norm(x))


class PositionalConvEmbedding(nn.Module):
    """Grouped conv with even kernel; the trailing frame is dropped (HF SamePad)."""

    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        k = cfg.num_conv_pos_embeddings
        self.conv = nn.Conv1d(
            cfg.hidden_size,
            cfg.hidden_size,
            k,
            padding=k // 2,
            groups=cfg.num_conv_pos_embedding_groups,
            bias=True,
        )
        self.remove_last = k % 2 == 0

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv(x)
        if self.remove_last:
            y = y[:, :-1, :]
        return nn.gelu(y)


class Attention(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        d, h = cfg.hidden_size, cfg.num_attention_heads
        self.heads = h
        self.scale = (d // h) ** -0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.output_dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_dense(nn.gelu(self.intermediate_dense(x)))


class EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        self.attention = Attention(cfg)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.feed_forward = FeedForward(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.layer_norm(x))
        return x + self.feed_forward(self.final_layer_norm(x))


class Encoder(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        if not cfg.do_stable_layer_norm:
            raise NotImplementedError("only do_stable_layer_norm=True is ported")
        self.pos_conv_embed = PositionalConvEmbedding(cfg)
        self.layers = [EncoderLayerStableLayerNorm(cfg) for _ in range(cfg.num_hidden_layers)]
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.pos_conv_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class Wav2Vec2Model(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        self.feature_extractor = FeatureEncoder(cfg)
        self.feature_projection = FeatureProjection(cfg)
        self.encoder = Encoder(cfg)

    def __call__(self, wav: mx.array) -> mx.array:
        x = self.feature_extractor(wav[:, :, None])
        return self.encoder(self.feature_projection(x))


class Wav2Vec2ForCTC(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.wav2vec2 = Wav2Vec2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def __call__(self, wav: mx.array) -> mx.array:
        """wav: (B, samples), already normalized. Returns logits (B, frames, vocab)."""
        return self.lm_head(self.wav2vec2(wav))


def convert_state_dict(sd: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    """HF PyTorch parameter names/layouts → this module's names/layouts."""
    out: dict[str, NDArray[Any]] = {}
    g = sd.pop("wav2vec2.encoder.pos_conv_embed.conv.weight_g")
    v = sd.pop("wav2vec2.encoder.pos_conv_embed.conv.weight_v")
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=(0, 1), keepdims=True))
    sd["wav2vec2.encoder.pos_conv_embed.conv.weight"] = (g * v / norm).astype(np.float32)
    for k, arr in sd.items():
        if k == "wav2vec2.masked_spec_embed" or k.startswith(("quantizer", "project_")):
            continue
        if k.endswith("conv.weight"):
            arr = np.ascontiguousarray(arr.transpose(0, 2, 1))  # (out, in, k) → (out, k, in)
        out[k] = arr
    return out
