"""Command line: `offbook devices | run | replay | check-models | schema`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from offbook.config import AlignmentConfig, ChunkingConfig, RecognizerConfig, SessionConfig

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")

_PHONEME_ALIGNMENT: dict[str, Any] = {"tolerance_s": 0.4, "max_lag_s": 1.5, "window_tokens": 48}


def _config(
    phoneme: bool,
    tolerance: float | None,
    max_lag: float | None,
    window_tokens: int | None,
    timing_weight: float | None,
    window_s: float | None,
    hop_s: float | None,
    margin_s: float | None,
    dtype: str,
    share_weights: bool,
    agree_s: float | None = None,
    edge_guard_s: float | None = None,
    record_transcripts: bool = False,
) -> SessionConfig:
    align: dict[str, Any] = dict(_PHONEME_ALIGNMENT) if phoneme else {}
    for k, v in (
        ("tolerance_s", tolerance),
        ("max_lag_s", max_lag),
        ("window_tokens", window_tokens),
        ("timing_fail_weight", timing_weight),
    ):
        if v is not None:
            align[k] = v
    chunk: dict[str, Any] = {
        k: v
        for k, v in (
            ("window_s", window_s),
            ("hop_s", hop_s),
            ("resolve_margin_s", margin_s),
            ("agree_s", agree_s),
            ("edge_guard_s", edge_guard_s),
        )
        if v is not None
    }
    return SessionConfig(
        chunking=ChunkingConfig(**chunk),
        alignment=AlignmentConfig(**align),
        recognizer=RecognizerConfig(
            impl="w2v2-phoneme" if phoneme else "parakeet",
            dtype=dtype,
            share_weights=share_weights,
        ),
        record_transcripts=record_transcripts,
    )


Phoneme = Annotated[bool, typer.Option("--phoneme", help="Use the wav2vec2 phoneme recognizer")]
Tolerance = Annotated[float | None, typer.Option(help="Timing tolerance window, seconds")]
MaxLag = Annotated[float | None, typer.Option(help="Extra lateness still paired, seconds")]
WindowTokens = Annotated[int | None, typer.Option(help="Alignment window, reference tokens")]
TimingWeight = Annotated[float | None, typer.Option(help="Weight of FAIL_TIMING vs lexical, 0-1")]
WindowS = Annotated[float | None, typer.Option(help="Recognizer window, seconds")]
HopS = Annotated[float | None, typer.Option(help="Recognizer hop, seconds")]
MarginS = Annotated[float | None, typer.Option(help="Recognizer resolve margin, seconds")]
AgreeS = Annotated[float | None, typer.Option(help="Agreement tolerance between decodes, s")]
EdgeS = Annotated[float | None, typer.Option(help="Left-edge guard of each window, seconds")]
Dtype = Annotated[str, typer.Option(help="float32 | bfloat16 | float16")]
Share = Annotated[bool, typer.Option("--share-weights", help="Share weights between instances")]
OutDir = Annotated[Path, typer.Option(help="Where session records are written")]
Quiet = Annotated[bool, typer.Option("--quiet", help="Only print the final result")]
Transcripts = Annotated[
    bool,
    typer.Option(
        "--record-transcripts",
        help="Write token text (lyrics) into session.json; off by default",
    ),
]


@app.command()
def devices() -> None:
    """List audio devices (index, name, channels, default rate)."""
    import sounddevice as sd

    typer.echo(str(sd.query_devices()))


@app.command()
def run(
    reference: Annotated[Path, typer.Argument(help="REFERENCE VOCAL file (never played)")],
    backing: Annotated[Path | None, typer.Option(help="Backing track → headphones")] = None,
    input_device: Annotated[str | None, typer.Option(help="Mic device index or name")] = None,
    output_device: Annotated[
        str | None, typer.Option(help="Headphone device index or name")
    ] = None,
    phoneme: Phoneme = False,
    tolerance: Tolerance = None,
    max_lag: MaxLag = None,
    window_tokens: WindowTokens = None,
    timing_weight: TimingWeight = None,
    window_s: WindowS = None,
    hop_s: HopS = None,
    margin_s: MarginS = None,
    agree_s: AgreeS = None,
    edge_guard_s: EdgeS = None,
    dtype: Dtype = "float32",
    share_weights: Share = False,
    out_dir: OutDir = Path("sessions"),
    quiet: Quiet = False,
    record_transcripts: Transcripts = False,
) -> None:
    """Score a live performance from the mic against the reference vocal."""
    from offbook.audio.sources import BackingTrackFile, MicInput, ReferenceVocalFile
    from offbook.session.console import SessionConsole
    from offbook.session.runner import Session

    cfg = _config(
        phoneme,
        tolerance,
        max_lag,
        window_tokens,
        timing_weight,
        window_s,
        hop_s,
        margin_s,
        dtype,
        share_weights,
        agree_s,
        edge_guard_s,
        record_transcripts,
    )
    Session(
        cfg,
        ReferenceVocalFile(reference),
        MicInput(_device(input_device)),
        BackingTrackFile(backing) if backing else None,
        out_dir=out_dir,
        output_device=_device(output_device),
        console=SessionConsole(quiet=quiet),
    ).run()


@app.command()
def replay(
    reference: Annotated[Path, typer.Argument(help="REFERENCE VOCAL file")],
    performance: Annotated[Path, typer.Argument(help="Recorded LIVE VOCAL .wav")],
    phoneme: Phoneme = False,
    tolerance: Tolerance = None,
    max_lag: MaxLag = None,
    window_tokens: WindowTokens = None,
    timing_weight: TimingWeight = None,
    window_s: WindowS = None,
    hop_s: HopS = None,
    margin_s: MarginS = None,
    agree_s: AgreeS = None,
    edge_guard_s: EdgeS = None,
    dtype: Dtype = "float32",
    share_weights: Share = False,
    out_dir: OutDir = Path("sessions"),
    quiet: Quiet = False,
    record_transcripts: Transcripts = False,
) -> None:
    """Re-score a recorded performance. No audio devices are opened."""
    from offbook.audio.sources import LiveReplayFile, ReferenceVocalFile
    from offbook.session.console import SessionConsole
    from offbook.session.runner import Session

    cfg = _config(
        phoneme,
        tolerance,
        max_lag,
        window_tokens,
        timing_weight,
        window_s,
        hop_s,
        margin_s,
        dtype,
        share_weights,
        agree_s,
        edge_guard_s,
        record_transcripts,
    )
    Session(
        cfg,
        ReferenceVocalFile(reference),
        LiveReplayFile(performance),
        None,
        out_dir=out_dir,
        console=SessionConsole(quiet=quiet),
    ).run()


@app.command()
def transcribe(
    reference: Annotated[Path, typer.Argument(help="REFERENCE VOCAL file (never played)")],
    full: Annotated[
        bool, typer.Option("--full", help="Also decode the whole file at once and report WER")
    ] = False,
    seconds: Annotated[float | None, typer.Option(help="Only the first N seconds")] = None,
    phoneme: Phoneme = False,
    window_s: WindowS = None,
    hop_s: HopS = None,
    margin_s: MarginS = None,
    agree_s: AgreeS = None,
    edge_guard_s: EdgeS = None,
    dtype: Dtype = "float32",
) -> None:
    """Diagnostic: transcribe one vocal through the streaming chunker, optionally vs one-shot."""
    from offbook.audio.sources import ReferenceVocalFile
    from offbook.session.transcribe import print_result, transcribe_reference

    cfg = _config(
        phoneme,
        None,
        None,
        None,
        None,
        window_s,
        hop_s,
        margin_s,
        dtype,
        False,
        agree_s,
        edge_guard_s,
    )
    result = transcribe_reference(cfg, ReferenceVocalFile(reference), full=full, seconds=seconds)
    print_result(result, reference)


@app.command()
def web(
    host: Annotated[str, typer.Option(help="Bind address; keep it local")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8765,
) -> None:
    """Serve the local control panel (file locations, devices, config, start/stop, readout)."""
    from offbook.web.server import serve

    typer.echo(f"off book control panel → http://{host}:{port}")
    serve(host, port)


@app.command("check-models")
def check_models() -> None:
    """Download the pinned model revisions and verify their hashes."""
    from offbook.asr.parakeet import ParakeetBackend
    from offbook.asr.w2v2_phoneme import W2v2PhonemeBackend

    for backend in (ParakeetBackend(ChunkingConfig()), W2v2PhonemeBackend(ChunkingConfig())):
        typer.echo(f"ok  {backend.spec.describe()}")


@app.command()
def schema(
    out: Annotated[Path, typer.Option(help="Output path")] = Path("schema/session.schema.json"),
) -> None:
    """Write the session record JSON schema."""
    from offbook.session.record import write_schema

    write_schema(out)
    typer.echo(f"wrote {out}")


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    app()
