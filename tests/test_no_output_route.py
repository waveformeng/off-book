"""Constraint 2: the reference vocal has no route to an output device."""

from pathlib import Path

import numpy as np
import pytest
import sounddevice as sd

from offbook.audio.decode import BackingPCM, ReferencePCM
from offbook.audio.graph import AudioGraph, _BackingPlayer
from offbook.audio.guard import ReferenceOutputRouteError, UnauthorizedOutputError
from offbook.audio.sources import BackingTrackFile, LiveReplayFile, MicInput, ReferenceVocalFile


def test_reference_pcm_is_not_array_like(silence_wav: Path) -> None:
    ref = ReferencePCM(ReferenceVocalFile(silence_wav), 16_000)
    assert not hasattr(ref, "__array__")
    assert not hasattr(ref, "play")
    assert np.asarray(ref).dtype == object  # cannot be coerced into a sample buffer


def test_backing_player_rejects_reference(silence_wav: Path) -> None:
    ref = ReferencePCM(ReferenceVocalFile(silence_wav), 16_000)
    with pytest.raises(ReferenceOutputRouteError):
        _BackingPlayer(ref)  # type: ignore[arg-type]


def test_graph_assertion_catches_reference_on_output_path(silence_wav: Path) -> None:
    ref = ReferencePCM(ReferenceVocalFile(silence_wav), 16_000)
    backing = BackingPCM(BackingTrackFile(silence_wav), 16_000)
    graph = AudioGraph(ref, MicInput(), backing)
    graph.assert_reference_has_no_output_route()  # legitimate graph passes
    assert graph._player is not None
    graph._player.__dict__["backing"] = ref  # simulate a mis-wired output source
    with pytest.raises(ReferenceOutputRouteError):
        graph.assert_reference_has_no_output_route()


def test_replay_session_must_not_open_output(silence_wav: Path) -> None:
    ref = ReferencePCM(ReferenceVocalFile(silence_wav), 16_000)
    backing = BackingPCM(BackingTrackFile(silence_wav), 16_000)
    graph = AudioGraph(ref, LiveReplayFile(silence_wav), backing)
    with pytest.raises(ReferenceOutputRouteError):
        graph.assert_reference_has_no_output_route()


@pytest.mark.parametrize("entry", ["play", "OutputStream", "Stream", "RawOutputStream", "playrec"])
def test_sounddevice_outputs_are_guarded(entry: str) -> None:
    with pytest.raises(UnauthorizedOutputError):
        getattr(sd, entry)(np.zeros(16, dtype=np.float32), 16_000)


def test_live_tap_sees_the_live_block_alone(silence_wav: Path) -> None:
    ref = ReferencePCM(ReferenceVocalFile(silence_wav), 16_000)
    tapped: list[tuple[np.ndarray, float]] = []
    tap = lambda b, t: tapped.append((b, t))  # noqa: E731
    graph = AudioGraph(ref, MicInput(), None, blocksize=64, live_tap=tap)
    mic = np.full((64, 1), 0.5, dtype=np.float32)  # unmistakably not the (silent) reference
    graph._ingest(mic, 64)
    graph._ingest(mic, 64)
    assert [t for _, t in tapped] == [64 / 16_000, 128 / 16_000]
    assert all(b.shape == (64,) and np.all(b == 0.5) for b, _ in tapped)
    assert graph.blocks.qsize() == 2  # the tap is in addition to analysis, not instead of it
