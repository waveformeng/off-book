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


def test_stage_page_and_font_are_served() -> None:
    assert client.get("/stage").status_code == 200
    assert "Kabel" in client.get("/stage").text
    assert client.get("/static/fonts/KabelBlack-Regular.woff").status_code == 200


def test_trace_decimates_evenly_and_keeps_endpoints() -> None:
    import numpy as np

    from offbook.web.events import trace

    block = np.linspace(-1, 1, 1024, dtype=np.float32).reshape(-1, 1)
    t = trace(block, points=256)
    assert len(t) == 256 and t[0] == -1.0 and t[-1] == 1.0
    assert t == sorted(t)
    assert trace(np.zeros((0, 1), dtype=np.float32)) == []
    assert len(trace(np.ones((7, 1), dtype=np.float32))) == 7  # short blocks pass through


def test_waveform_buffer_hands_out_only_new_frames() -> None:
    from offbook.web.events import WaveformBuffer

    b = WaveformBuffer(maxlen=3)
    seq, frames = b.since(0)
    assert seq == 0 and frames == []
    for i in range(5):
        b.push({"i": i})
    seq, frames = b.since(0)
    assert [f["i"] for f in frames] == [2, 3, 4] and seq == 5  # ring dropped 0 and 1
    b.push({"i": 5})
    seq, frames = b.since(seq)
    assert [f["i"] for f in frames] == [5] and seq == 6
    assert b.since(seq) == (6, [])


def test_event_console_feeds_live_audio_to_waveform_only() -> None:
    import numpy as np

    from offbook.web.events import WaveformBuffer

    sink: queue.Queue[dict[str, object] | None] = queue.Queue()
    wf = WaveformBuffer()
    c = EventConsole(sink, waveform=wf)
    c.live_audio(np.full((256, 1), 0.3, dtype=np.float32), 1.0)
    _, frames = wf.since(0)
    assert len(frames) == 1 and frames[0]["transport_s"] == 1.0
    assert frames[0]["pcm"] == [0.3] * 256
    assert sink.empty()  # never an event: the history must not carry audio
    EventConsole(sink).live_audio(np.zeros((256, 1), dtype=np.float32), 1.0)  # no buffer: no-op


def test_start_request_carries_stage_credits() -> None:
    req = server.StartRequest(mode="live", reference="r.wav", title="Africa", singer="Toto")
    assert req.title == "Africa" and req.singer == "Toto"
    assert server.StartRequest(mode="live", reference="r.wav").title is None


def test_record_omits_token_text_unless_asked() -> None:
    from offbook.asr.base import Token
    from offbook.clock import ANALYSIS_RATE, TransportTime
    from offbook.compare.verdict import TokenPair, Verdict
    from offbook.config import SessionConfig
    from offbook.roles import Live, Reference
    from offbook.session.record import PairRecord

    assert SessionConfig().record_transcripts is False
    ref = Token(
        Reference,
        "peanuts",
        TransportTime(0, ANALYSIS_RATE),
        TransportTime(1, ANALYSIS_RATE),
        1.0,
        5.0,
    )
    live = Token(
        Live, "peanuts", TransportTime(0, ANALYSIS_RATE), TransportTime(1, ANALYSIS_RATE), 0.9, 5.0
    )
    pair = TokenPair(ref, live, Verdict.MATCH)
    quiet = PairRecord.of(pair, 50.0, 0.0, text=False)
    assert quiet.reference is not None and quiet.live is not None
    assert quiet.reference.text is None and quiet.live.text is None
    assert quiet.verdict is Verdict.MATCH and quiet.live.confidence == 0.9
    assert "peanuts" not in quiet.model_dump_json()
    loud = PairRecord.of(pair, 50.0, 0.0, text=True)
    assert loud.reference is not None and loud.reference.text == "peanuts"
