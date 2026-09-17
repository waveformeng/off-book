"""Wires the audio graph, the two recognizers, the aligner and the score into a session."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import final

import mlx.core as mx
import numpy as np
import soxr
from numpy.typing import NDArray

from offbook.asr.base import Backend, Frames, Token, assert_same_recognizer
from offbook.asr.chunker import Recognizer
from offbook.asr.registry import build_backends
from offbook.audio.capture import PerformanceWriter
from offbook.audio.decode import BackingPCM, ReferencePCM
from offbook.audio.graph import AudioGraph, CapturedBlock, input_device_rate, output_device_rate
from offbook.audio.sources import BackingTrackFile, LiveReplayFile, MicInput, ReferenceVocalFile
from offbook.clock import ANALYSIS_RATE, TransportTime
from offbook.compare.align import Aligner
from offbook.compare.score import RunningScore
from offbook.config import SessionConfig
from offbook.roles import Live, Reference, Role
from offbook.session.console import SessionConsole
from offbook.session.record import (
    CountsRecord,
    DriftRecord,
    DriftSampleRecord,
    PairRecord,
    RecognizerRecord,
    SessionRecord,
    new_session_id,
)


class _StreamResampler:
    """Device rate → analysis rate, streaming. One per stream, identical settings."""

    def __init__(self, in_rate: int) -> None:
        self._passthrough = in_rate == ANALYSIS_RATE
        self._rs = (
            None
            if self._passthrough
            else soxr.ResampleStream(in_rate, ANALYSIS_RATE, 1, dtype="float32", quality="HQ")
        )

    def push(self, block: NDArray[np.float32], last: bool = False) -> NDArray[np.float32]:
        if self._rs is None:
            return block
        out = self._rs.resample_chunk(block, last=last)
        return np.ascontiguousarray(out, dtype=np.float32)


class _Stream[R: (Reference, Live)]:
    """Everything one role owns: its resampler, its recognizer, its analysis-clock cursor."""

    def __init__(self, role: type[R], backend: Backend, cfg: SessionConfig, in_rate: int) -> None:
        self.role: type[R] = role
        self.resampler = _StreamResampler(in_rate)
        self.recognizer: Recognizer[R] = Recognizer(role, backend, cfg.chunking)
        self.cursor = 0

    def feed(self, block: NDArray[np.float32], last: bool = False) -> list[Token[R]]:
        pcm = self.resampler.push(block, last)
        frames: Frames[R] = Frames(self.role, pcm, TransportTime(self.cursor, ANALYSIS_RATE))
        self.cursor += len(pcm)
        tokens = self.recognizer.feed(frames)
        if last:
            tokens = tokens + self.recognizer.flush()
        return tokens


@final
class Session:
    def __init__(
        self,
        cfg: SessionConfig,
        reference: ReferenceVocalFile,
        live: MicInput | LiveReplayFile,
        backing: BackingTrackFile | None,
        *,
        out_dir: Path,
        output_device: int | str | None = None,
        console: SessionConsole | None = None,
    ) -> None:
        self.cfg = cfg
        self.reference_source = reference
        self.live_source = live
        self.backing_source = backing
        self.out_dir = out_dir
        self.output_device = output_device
        self.console = console or SessionConsole()
        self.session_id = new_session_id()
        self._graph: AudioGraph | None = None
        self._stop_requested = False

    def request_stop(self) -> None:
        """Stop a running session from another thread. The record is still written."""
        self._stop_requested = True
        if self._graph is not None:
            self._graph.stop()

    def run(self) -> SessionRecord:
        cfg = self.cfg
        mx.random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        mode = "replay" if isinstance(self.live_source, LiveReplayFile) else "live"
        if isinstance(self.live_source, LiveReplayFile):
            import soundfile as sf

            rate = int(sf.info(str(self.live_source.path)).samplerate)
        else:
            rate = input_device_rate(self.live_source.device)

        reference = ReferencePCM(self.reference_source, rate)
        backing: BackingPCM | None = None
        if self.backing_source is not None:
            if mode == "replay":
                raise ValueError("replay sessions do not play a backing track")
            backing = BackingPCM(self.backing_source, output_device_rate(self.output_device))

        ref_backend, live_backend = build_backends(cfg)
        assert_same_recognizer(ref_backend.spec, live_backend.spec)
        ref_stream: _Stream[Reference] = _Stream(Reference, ref_backend, cfg, rate)
        live_stream: _Stream[Live] = _Stream(Live, live_backend, cfg, rate)
        spec = ref_backend.spec

        aligner = Aligner(cfg.alignment, spec.unit, reference_end_s=reference.duration_s)
        score = RunningScore(
            cfg.alignment.timing_fail_weight,
            reference.duration_s,
            graded_timing=cfg.alignment.graded_timing,
            tolerance_s=cfg.alignment.tolerance_s,
        )

        tail_s = (
            cfg.alignment.tolerance_s
            + cfg.alignment.max_lag_s
            + cfg.chunking.resolve_margin_s
            + cfg.chunking.hop_s
        )
        stop_after = reference.n_samples + int(tail_s * rate)
        graph = AudioGraph(
            reference,
            self.live_source,
            backing,
            output_device=self.output_device,
            drift_interval_s=cfg.drift_log_interval_s,
            stop_after_frames=stop_after if mode == "live" else None,
        )
        graph.assert_reference_has_no_output_route()
        self._graph = graph
        if self._stop_requested:
            graph.stop()

        self.console.header(spec, mode, graph.duplex, rate, reference.duration_s)
        session_dir = self.out_dir / self.session_id
        perf_path = (
            self.live_source.path
            if isinstance(self.live_source, LiveReplayFile)
            else session_dir / "performance.wav"
        )
        writer = (
            PerformanceWriter(perf_path, rate) if isinstance(self.live_source, MicInput) else None
        )

        pairs: list[PairRecord] = []
        started_at = datetime.now(UTC).isoformat()
        wall0 = time.monotonic()
        error: str | None = None
        drift_logged = 0
        transport_s = 0.0

        def process(block: CapturedBlock, last: bool) -> None:
            nonlocal drift_logged, transport_s
            if writer is not None:
                writer.write(block.live)
            ref_tokens = ref_stream.feed(block.reference, last)
            live_tokens = live_stream.feed(block.live, last)
            transport_s = (block.start_frame + len(block.live)) / rate
            self.console.resolved(ref_tokens, live_tokens, transport_s)
            aligner.add_reference(ref_tokens)
            aligner.add_live(live_tokens)
            if last:
                new_pairs = aligner.finish()
            else:
                new_pairs = aligner.advance(
                    ref_stream.recognizer.resolved_until, live_stream.recognizer.resolved_until
                )
            position_s = live_stream.recognizer.resolved_until.seconds
            for p in new_pairs:
                score.add(p)
                s = score.score(position_s)
                pairs.append(PairRecord.of(p, s, wall0))
                self.console.verdict(p, s)
            while drift_logged < len(graph.drift.samples):
                self.console.drift(graph.drift.samples[drift_logged])
                drift_logged += 1

        try:
            graph.start()
            pending: CapturedBlock | None = None
            while True:
                block = graph.blocks.get()
                if block is None:
                    break
                if pending is not None:
                    process(pending, last=False)
                pending = block
            if graph.error is not None:
                raise graph.error
            if pending is not None:
                process(pending, last=True)
            else:
                for p in aligner.finish():
                    score.add(p)
                    pairs.append(PairRecord.of(p, score.score(0.0), wall0))
        except KeyboardInterrupt:
            error = "interrupted"
        finally:
            if self._stop_requested and error is None:
                error = "stopped"
            graph.stop()
            if writer is not None:
                writer.close()

        final_position = max(transport_s, live_stream.recognizer.resolved_until.seconds)
        final_score = score.score(final_position)
        record = SessionRecord(
            session_id=self.session_id,
            started_at=started_at,
            mode=mode,
            reference_path=str(self.reference_source.path),
            backing_path=str(self.backing_source.path) if self.backing_source else None,
            performance_wav_path=str(perf_path),
            transport_rate=rate,
            reference_duration_s=reference.duration_s,
            recognizer=RecognizerRecord.of(spec),
            config=cfg,
            pairs=pairs,
            counts=CountsRecord.of(score),
            match_rate=score.match_rate,
            final_score=final_score,
            drift=DriftRecord(
                duplex=graph.duplex,
                input_rate=graph.drift.in_rate,
                output_rate=graph.drift.out_rate,
                samples=[
                    DriftSampleRecord(transport_s=s.transport_s, drift_s=s.drift_s)
                    for s in graph.drift.samples
                ],
                max_abs_s=graph.drift.max_abs_s,
                final_s=graph.drift.final_s,
            ),
            live_tokens_ignored_after_reference_end=aligner.ignored_after_end,
            session_duration_s=time.monotonic() - wall0,
            error=error,
        )
        record_path = session_dir / "session.json"
        record.write(record_path)
        self.console.final(final_score, score.match_rate, score.counts, str(record_path))
        return record


__all__ = ["Session", "Role"]
