"""The session record: the audit artifact. Written as JSON at the end of every session."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from offbook.asr.base import RecognizerSpec, Token
from offbook.compare.score import RunningScore
from offbook.compare.verdict import TokenPair, Verdict
from offbook.config import SessionConfig
from offbook.roles import Live, Reference


class TokenRecord(BaseModel, frozen=True):
    text: str | None = Field(
        description="null unless config.record_transcripts: token text is lyric text"
    )
    start_s: float
    end_s: float
    confidence: float
    resolved_wall_s: float = Field(description="Wall-clock seconds since session start")

    @staticmethod
    def of(t: Token[Reference] | Token[Live], wall0: float, text: bool) -> TokenRecord:
        return TokenRecord(
            text=t.text if text else None,
            start_s=t.start.seconds,
            end_s=t.end.seconds,
            confidence=t.confidence,
            resolved_wall_s=t.resolved_wall - wall0,
        )


class PairRecord(BaseModel, frozen=True):
    reference: TokenRecord | None = Field(description="null ⇒ FAIL_INSERTED")
    live: TokenRecord | None = Field(description="null ⇒ FAIL_MISSED")
    verdict: Verdict
    dt_s: float | None = Field(description="live.start − reference.start; positive = late")
    score_after: float = Field(description="Aggregate score after this verdict")

    @staticmethod
    def of(p: TokenPair, score_after: float, wall0: float, text: bool) -> PairRecord:
        return PairRecord(
            reference=(
                TokenRecord.of(p.reference, wall0, text) if p.reference is not None else None
            ),
            live=TokenRecord.of(p.live, wall0, text) if p.live is not None else None,
            verdict=p.verdict,
            dt_s=p.dt_s,
            score_after=score_after,
        )


class RecognizerRecord(BaseModel, frozen=True):
    impl: str
    unit: str
    model_id: str
    revision: str
    model_hash: str = Field(description="sha256 over the pinned model files")
    dtype: str

    @staticmethod
    def of(s: RecognizerSpec) -> RecognizerRecord:
        return RecognizerRecord(
            impl=s.impl,
            unit=s.unit,
            model_id=s.model_id,
            revision=s.revision,
            model_hash=s.model_hash,
            dtype=s.dtype,
        )


class DriftSampleRecord(BaseModel, frozen=True):
    transport_s: float
    drift_s: float


class DriftRecord(BaseModel, frozen=True):
    duplex: bool = Field(description="Mic and output on one device: drift is zero by design")
    input_rate: int
    output_rate: int
    samples: list[DriftSampleRecord]
    max_abs_s: float
    final_s: float


class CountsRecord(BaseModel, frozen=True):
    match: int
    fail_lexical: int
    fail_timing: int
    fail_missed: int
    fail_inserted: int

    @staticmethod
    def of(score: RunningScore) -> CountsRecord:
        c = score.counts
        return CountsRecord(
            match=c[Verdict.MATCH],
            fail_lexical=c[Verdict.FAIL_LEXICAL],
            fail_timing=c[Verdict.FAIL_TIMING],
            fail_missed=c[Verdict.FAIL_MISSED],
            fail_inserted=c[Verdict.FAIL_INSERTED],
        )


class SessionRecord(BaseModel, frozen=True):
    schema_version: int = 2  # 2: token text is null unless config.record_transcripts
    session_id: str
    started_at: str = Field(description="ISO-8601 UTC")
    mode: str = Field(description="'live' (mic) or 'replay' (recorded performance)")
    reference_path: str
    backing_path: str | None
    performance_wav_path: str
    transport_rate: int
    reference_duration_s: float
    recognizer: RecognizerRecord
    config: SessionConfig
    pairs: list[PairRecord]
    counts: CountsRecord
    match_rate: float
    final_score: float
    drift: DriftRecord
    live_tokens_ignored_after_reference_end: int = Field(
        default=0, description="Live tokens sung after the reference ended, not scored"
    )
    session_duration_s: float
    error: str | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def new_session_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SessionRecord.model_json_schema(), indent=2) + "\n")
