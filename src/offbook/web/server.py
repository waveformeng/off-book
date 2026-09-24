"""Local control surface: file locations, devices, config, start/stop, live readout.

One session at a time. The session runs on a worker thread; its console events are
fanned out to browsers over server-sent events. Binds to localhost only by default —
this is a control panel for the machine the mic is plugged into, not a service.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from offbook.config import AlignmentConfig, RecognizerConfig, SessionConfig
from offbook.web.events import Event, EventConsole, WaveformBuffer

STATIC = Path(__file__).parent / "static"
AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".mp3", ".aiff", ".aif", ".ogg", ".opus", ".m4a"}


class StartRequest(BaseModel):
    mode: Literal["live", "replay"]
    reference: str = Field(description="REFERENCE VOCAL path (analysed, never played)")
    backing: str | None = Field(default=None, description="Backing track path (live only)")
    performance: str | None = Field(default=None, description="Recorded LIVE VOCAL (replay)")
    input_device: int | None = None
    output_device: int | None = None
    out_dir: str = "sessions"
    config: SessionConfig = SessionConfig()
    # Shown on the stage view's lower-third credits. Not part of the session record.
    title: str | None = Field(default=None, description="Song title, for the stage view")
    singer: str | None = Field(default=None, description="Performer's name, for the stage view")


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status: str = "idle"  # idle | loading | running | done | error
        self.error: str | None = None
        self.request: StartRequest | None = None
        self.session: Any = None
        self.thread: threading.Thread | None = None
        self.events: list[Event] = []  # history for late-joining browsers
        self.sink: queue.Queue[Event | None] = queue.Queue()
        self.result: dict[str, Any] | None = None
        self.started_wall: float | None = None
        # Bumped on every start. A browser that has been subscribed to /api/events since a
        # previous session (the stage view sits open all night) uses it to notice that the
        # history it was replaying has been replaced.
        self.generation = 0
        self.waveform = WaveformBuffer()

    def push(self, event: Event) -> None:
        self.events.append(event)


state = State()


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    # Ctrl-C: stop a running session cleanly so its record is written, then exit.
    with state.lock:
        session, thread = state.session, state.thread
    if session is not None:
        session.request_stop()
    if thread is not None and thread.is_alive():
        await anyio.to_thread.run_sync(thread.join)


app = FastAPI(title="off book", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# --- pages -----------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/stage")
def stage() -> FileResponse:
    """The performance view: the singer's waveform and words, the score as colour, the
    credits. Nothing from the reference stream is ever sent to it."""
    return FileResponse(STATIC / "stage.html")


# --- lookups ---------------------------------------------------------------------------


@app.get("/api/devices")
def devices() -> dict[str, Any]:
    import sounddevice as sd

    out: list[dict[str, Any]] = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            {
                "index": i,
                "name": d["name"],
                "inputs": d["max_input_channels"],
                "outputs": d["max_output_channels"],
                "rate": int(d["default_samplerate"]),
            }
        )
    default_in, default_out = sd.default.device
    return {"devices": out, "default_input": default_in, "default_output": default_out}


@app.get("/api/browse")
def browse(path: str = "~") -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        raise HTTPException(404, f"not a directory: {p}")
    dirs, files = [], []
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                dirs.append(child.name)
            elif child.suffix.lower() in AUDIO_SUFFIXES:
                files.append({"name": child.name, "size": child.stat().st_size})
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    return {"path": str(p), "parent": str(p.parent), "dirs": dirs, "files": files}


@app.get("/api/config/defaults")
def config_defaults() -> dict[str, Any]:
    return {
        "word": SessionConfig().model_dump(),
        "phoneme": SessionConfig(
            alignment=AlignmentConfig(tolerance_s=0.4, window_tokens=48),
            recognizer=RecognizerConfig(impl="w2v2-phoneme"),
        ).model_dump(),
    }


@app.get("/api/sessions")
def sessions(out_dir: str = "sessions") -> list[dict[str, Any]]:
    root = Path(out_dir)
    if not root.is_dir():
        return []
    out = []
    for rec in sorted(root.glob("*/session.json"), reverse=True):
        try:
            r = json.loads(rec.read_text())
        except (OSError, ValueError):
            continue
        out.append(
            {
                "session_id": r.get("session_id"),
                "started_at": r.get("started_at"),
                "mode": r.get("mode"),
                "final_score": r.get("final_score"),
                "match_rate": r.get("match_rate"),
                "recognizer": r.get("recognizer", {}).get("impl"),
                "reference_path": r.get("reference_path"),
                "performance_wav_path": r.get("performance_wav_path"),
                "record_path": str(rec),
                "error": r.get("error"),
            }
        )
    return out


@app.get("/api/sessions/{session_id}/record")
def session_record(session_id: str, out_dir: str = "sessions") -> FileResponse:
    rec = Path(out_dir) / session_id / "session.json"
    if not rec.is_file():
        raise HTTPException(404, "no such session")
    return FileResponse(rec, media_type="application/json", filename=f"{session_id}.json")


# --- execution -------------------------------------------------------------------------


def _validate(req: StartRequest) -> None:
    if not Path(req.reference).expanduser().is_file():
        raise HTTPException(400, f"reference vocal not found: {req.reference}")
    if req.mode == "live":
        if req.backing and not Path(req.backing).expanduser().is_file():
            raise HTTPException(400, f"backing track not found: {req.backing}")
        if req.performance:
            raise HTTPException(400, "a live session does not take a performance file")
    else:
        if not req.performance or not Path(req.performance).expanduser().is_file():
            raise HTTPException(400, f"performance file not found: {req.performance}")
        if req.backing:
            raise HTTPException(400, "replay sessions do not play a backing track")


def _run(req: StartRequest) -> None:
    from offbook.audio.sources import (
        BackingTrackFile,
        LiveReplayFile,
        MicInput,
        ReferenceVocalFile,
    )
    from offbook.session.runner import Session

    console = EventConsole(state.sink, quiet=False, waveform=state.waveform)
    live: MicInput | LiveReplayFile = (
        MicInput(req.input_device)
        if req.mode == "live"
        else LiveReplayFile(Path(req.performance or "").expanduser())
    )
    session = Session(
        req.config,
        ReferenceVocalFile(Path(req.reference).expanduser()),
        live,
        BackingTrackFile(Path(req.backing).expanduser()) if req.backing else None,
        out_dir=Path(req.out_dir),
        output_device=req.output_device,
        console=console,
    )
    with state.lock:
        state.session = session
        state.status = "loading"
    console.emit(
        "status",
        status="loading",
        session_id=session.session_id,
        mode=req.mode,
        title=req.title,
        singer=req.singer,
    )
    try:
        # The console header fires once the models are resident and the graph is up.
        _orig_header = console.header

        def header(*a: Any, **k: Any) -> None:
            with state.lock:
                state.status = "running"
            console.emit("status", status="running", session_id=session.session_id)
            _orig_header(*a, **k)

        console.header = header  # type: ignore[method-assign]
        record = session.run()
        with state.lock:
            state.status = "error" if record.error and record.error != "stopped" else "done"
            state.error = record.error
            state.result = {
                "session_id": record.session_id,
                "final_score": record.final_score,
                "match_rate": record.match_rate,
                "counts": record.counts.model_dump(),
                "record_path": str(Path(req.out_dir) / record.session_id / "session.json"),
                "performance_wav_path": record.performance_wav_path,
                "drift_max_abs_ms": record.drift.max_abs_s * 1e3,
                "model_hash": record.recognizer.model_hash,
                "error": record.error,
            }
        console.emit("status", status=state.status, session_id=session.session_id)
    except BaseException as e:  # noqa: BLE001 — anything the engine raises is shown in the UI
        with state.lock:
            state.status = "error"
            state.error = f"{type(e).__name__}: {e}"
        console.emit("status", status="error", error=state.error)
    finally:
        with state.lock:
            state.session = None


@app.post("/api/session/start")
def start(req: StartRequest) -> dict[str, Any]:
    _validate(req)
    with state.lock:
        if state.status in ("loading", "running"):
            raise HTTPException(409, "a session is already running")
        state.status = "loading"
        state.error = None
        state.result = None
        state.request = req
        state.events = []
        state.sink = queue.Queue()
        state.started_wall = time.time()
        state.generation += 1
        state.waveform = WaveformBuffer()
        state.thread = threading.Thread(target=_run, args=(req,), name="offbook-session")
        state.thread.start()
    return {"status": "loading"}


@app.post("/api/session/stop")
def stop() -> dict[str, Any]:
    with state.lock:
        session = state.session
        status = state.status
    if session is None or status not in ("loading", "running"):
        return {"status": status}
    session.request_stop()
    return {"status": "stopping"}


@app.get("/api/session/status")
def status() -> dict[str, Any]:
    with state.lock:
        return {
            "status": state.status,
            "error": state.error,
            "result": state.result,
            "request": state.request.model_dump() if state.request else None,
            "started_wall": state.started_wall,
            "generation": state.generation,
        }


@app.get("/api/events")
async def events(skip: int | None = None) -> StreamingResponse:
    """Server-sent events: history first, then live. `skip` names a generation (from
    `/api/session/status`) whose history the subscriber does not want — the stage view
    uses it so a page opened after a take has ended does not re-run that take."""

    async def gen() -> AsyncIterator[str]:
        sent = 0
        generation = state.generation
        first = True
        while True:
            # Drain the worker's queue into the shared history under the lock, then
            # replay anything this subscriber hasn't seen.
            with state.lock:
                if state.generation != generation:
                    generation, sent = state.generation, 0
                while True:
                    try:
                        ev = state.sink.get_nowait()
                    except queue.Empty:
                        break
                    if ev is not None:
                        state.push(ev)
                history = state.events
                if first and skip == generation:
                    sent = len(history)  # after the drain, so a trailing "done" is skipped too
                first = False
                pending = history[sent:]
                sent = len(history)
            for ev in pending:
                yield f"data: {json.dumps(ev)}\n\n"
            if not pending:
                yield ": keepalive\n\n"
                await anyio.sleep(0.15)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/stage/waveform")
async def stage_waveform() -> StreamingResponse:
    """Server-sent events: the LIVE VOCAL's envelope, batched per frame. No history — a
    subscriber gets what is captured from now on, and a new session restarts the ring."""

    async def gen() -> AsyncIterator[str]:
        seq = 0
        buf = state.waveform
        while True:
            if state.waveform is not buf:
                buf, seq = state.waveform, 0
            seq, frames = buf.since(seq)
            if frames:
                yield f"data: {json.dumps({'frames': frames})}\n\n"
            else:
                yield ": keepalive\n\n"
            await anyio.sleep(1 / 60)

    return StreamingResponse(gen(), media_type="text/event-stream")


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    # The event streams never end on their own; without a deadline, Ctrl-C would wait on
    # every open stage and control-panel tab before the process exits.
    uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=1)
