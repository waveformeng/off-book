"""End-to-end replay with the real recognizers. Needs the pinned models and macOS `say`.

Runs only with OFFBOOK_MODEL_TESTS=1. Speech is synthesized at test time from a pangram —
not song lyrics — and never committed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OFFBOOK_MODEL_TESTS") != "1" or shutil.which("say") is None,
    reason="set OFFBOOK_MODEL_TESTS=1 on a Mac with the models downloaded",
)

REFERENCE = "the quick brown fox jumps over the lazy dog and then it runs away"
PERFORMANCE = "the quick brown fox jumps over the lazy cat and then it walks away"


def _say(text: str, path: Path) -> Path:
    subprocess.run(["say", "-o", str(path), "--data-format=LEI16@16000", text], check=True)
    return path


def _strip(record: dict[str, Any]) -> object:
    pairs = record["pairs"]
    assert isinstance(pairs, list)
    return [
        {k: v for k, v in p.items() if k != "resolved_wall_s"}
        | {
            side: (p[side] and {k: v for k, v in p[side].items() if k != "resolved_wall_s"})
            for side in ("reference", "live")
        }
        for p in pairs
    ]


@pytest.mark.parametrize("impl", ["parakeet", "w2v2-phoneme"])
def test_replay_is_deterministic_and_scores_identical_audio_100(impl: str, tmp_path: Path) -> None:
    from offbook.audio.sources import LiveReplayFile, ReferenceVocalFile
    from offbook.config import RecognizerConfig, SessionConfig
    from offbook.session.console import SessionConsole
    from offbook.session.runner import Session

    ref = _say(REFERENCE, tmp_path / "ref.wav")
    perf = _say(PERFORMANCE, tmp_path / "perf.wav")
    cfg = SessionConfig(recognizer=RecognizerConfig(impl=impl))

    def run(live: Path) -> dict[str, Any]:
        rec = Session(
            cfg,
            ReferenceVocalFile(ref),
            LiveReplayFile(live),
            None,
            out_dir=tmp_path / "sessions",
            console=SessionConsole(quiet=True),
        ).run()
        data: dict[str, Any] = json.loads(rec.model_dump_json())
        return data

    same = run(ref)
    assert same["final_score"] == 100.0, same["counts"]

    a, b = run(perf), run(perf)
    assert _strip(a) == _strip(b)
    assert a["final_score"] == b["final_score"] < 100.0
    assert a["recognizer"]["model_hash"] == b["recognizer"]["model_hash"]
