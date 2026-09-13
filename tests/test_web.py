"""The control panel's API, without models: validation, lookups, event framing."""

import queue
from pathlib import Path

from fastapi.testclient import TestClient

from offbook.web import server
from offbook.web.events import EventConsole

client = TestClient(server.app)


def test_index_and_defaults() -> None:
    assert client.get("/").status_code == 200
    d = client.get("/api/config/defaults").json()
    assert d["word"]["recognizer"]["impl"] == "parakeet"
    assert d["phoneme"]["recognizer"]["impl"] == "w2v2-phoneme"
    assert d["phoneme"]["alignment"]["window_tokens"] > d["word"]["alignment"]["window_tokens"]


def test_browse_lists_dirs_and_audio_only(tmp_path: Path, silence_wav: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.wav").write_bytes(b"")
    d = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    assert d["dirs"] == ["sub"]
    assert [f["name"] for f in d["files"]] == [silence_wav.name]
    assert client.get("/api/browse", params={"path": str(tmp_path / "nope")}).status_code == 404


def test_start_validates_files_and_modes(tmp_path: Path, silence_wav: Path) -> None:
    base = {"reference": str(silence_wav), "out_dir": str(tmp_path)}
    r = client.post("/api/session/start", json={"mode": "replay", **base})
    assert r.status_code == 400 and "performance" in r.json()["detail"]
    r = client.post(
        "/api/session/start", json={"mode": "live", "reference": str(tmp_path / "x.wav")}
    )
    assert r.status_code == 400 and "reference" in r.json()["detail"]
    r = client.post(
        "/api/session/start",
        json={
            "mode": "replay",
            "performance": str(silence_wav),
            "backing": str(silence_wav),
            **base,
        },
    )
    assert r.status_code == 400 and "backing" in r.json()["detail"]
    r = client.post("/api/session/start", json={"mode": "live", "performance": "p.wav", **base})
    assert r.status_code == 400


def test_sessions_listing_reads_records(tmp_path: Path) -> None:
    (tmp_path / "abc").mkdir()
    (tmp_path / "abc" / "session.json").write_text(
        '{"session_id":"abc","mode":"replay","final_score":42.0,"match_rate":0.42,'
        '"recognizer":{"impl":"parakeet"},"reference_path":"r.wav",'
        '"performance_wav_path":"p.wav","error":null}'
    )
    rows = client.get("/api/sessions", params={"out_dir": str(tmp_path)}).json()
    assert rows[0]["session_id"] == "abc" and rows[0]["final_score"] == 42.0
    r = client.get("/api/sessions/abc/record", params={"out_dir": str(tmp_path)})
    assert r.status_code == 200 and r.json()["session_id"] == "abc"
    assert (
        client.get("/api/sessions/zzz/record", params={"out_dir": str(tmp_path)}).status_code == 404
    )


def test_stop_when_idle_is_a_noop() -> None:
    assert client.post("/api/session/stop").json()["status"] in ("idle", "done", "error")


def test_event_console_emits_ticks_between_resolutions() -> None:
    from offbook.asr.base import Token
    from offbook.clock import ANALYSIS_RATE, TransportTime
    from offbook.roles import Reference

    sink: queue.Queue[dict[str, object] | None] = queue.Queue()
    c = EventConsole(sink)
    tok = Token(
        Reference, "x", TransportTime(0, ANALYSIS_RATE), TransportTime(1, ANALYSIS_RATE), 1.0, 0.0
    )
    for k in range(12):
        c.resolved([tok] if k == 6 else [], [], k * 0.064)
    sink.put(None)
    kinds = [e["kind"] for e in iter(sink.get_nowait, None) if e]
    assert kinds.count("resolved") == 1
    assert 2 <= kinds.count("tick") <= 3
