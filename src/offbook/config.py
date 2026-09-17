"""Session configuration. Everything a product decision might want to tune lives here."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel, frozen=True):
    """Chunked streaming with overlap. Boundaries are defined on transport sample counts,
    never wall clock, so identical audio always yields identical chunks.

    Every `hop_s`, the last `window_s` of audio is re-decoded. A token is emitted once it
    ends before `window_end − resolve_margin_s`, starts past the window's left-edge guard,
    and two consecutive decodes agree on it (same text, start within `agree_s`). Long
    windows matter: the recognizers were tuned against one-shot decodes of full vocals,
    and 30 s windows get within a few percent of them; 6 s windows do not."""

    window_s: float = Field(default=30.0, gt=0, description="Audio re-decoded on every hop")
    hop_s: float = Field(default=2.0, gt=0, description="New audio between decodes")
    resolve_margin_s: float = Field(
        default=3.0, ge=0, description="Tokens ending within this of the window end stay tentative"
    )
    agree_s: float = Field(
        default=0.3,
        ge=0,
        description="Two decodes agree on a token if its start moved by at most this",
    )
    edge_guard_s: float = Field(
        default=1.0,
        ge=0,
        description="Tokens starting within this of the window's left edge are ignored",
    )
    confirm_timeout_s: float = Field(
        default=4.0,
        ge=0,
        description="A token still unconfirmed this long past the resolve line is emitted anyway",
    )
    frontier_lag_s: float = Field(
        default=4.0,
        ge=0,
        description="resolved_until trails the resolve line by this, so tokens the recognizer "
        "produces late (phrase onsets after silence) still land ahead of it",
    )


class AlignmentConfig(BaseModel, frozen=True):
    tolerance_s: float = Field(
        default=0.75, gt=0, description="|dt| within which a lexical match is also a timing match"
    )
    max_lag_s: float = Field(
        default=1.5,
        ge=0,
        description="Extra lateness a live token may have and still be considered",
    )
    max_lead_s: float = Field(
        default=1.5, ge=0, description="Extra earliness a live token may have and still pair"
    )
    window_tokens: int = Field(
        default=12, ge=2, description="Reference tokens held for edit-distance"
    )
    timing_fail_weight: float = Field(
        default=0.5, ge=0, le=1, description="Weight of a FAIL_TIMING relative to a lexical failure"
    )
    word_match: Literal["exact", "sound"] = Field(
        default="sound",
        description="Word mode: 'exact' compares spellings; 'sound' also accepts homophones "
        "(Metaphone key), so a recognizer spelling one sound two ways is not a failure",
    )
    graded_timing: bool = Field(
        default=False,
        description="Inside the tolerance a MATCH earns 1 − timing_fail_weight·|dt|/tolerance "
        "instead of a flat 1 (continuous with FAIL_TIMING at the boundary)",
    )
    score_past_reference_end: bool = Field(
        default=False,
        description="Score live tokens sung after the reference vocal has ended (else ignored)",
    )
    phoneme_match_threshold: float = Field(
        default=0.34,
        ge=0,
        le=1,
        description="Phoneme mode: max normalized edit distance still counted as MATCH",
    )


class RecognizerConfig(BaseModel, frozen=True):
    impl: str = Field(
        default="parakeet", description="'parakeet' (word) or 'w2v2-phoneme' (phoneme)"
    )
    dtype: str = Field(default="float32", description="MLX dtype for weights")
    share_weights: bool = Field(
        default=False, description="Load weights once and share between the two instances"
    )


class SessionConfig(BaseModel, frozen=True):
    chunking: ChunkingConfig = ChunkingConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    recognizer: RecognizerConfig = RecognizerConfig()
    seed: int = 0
    drift_log_interval_s: float = 1.0
