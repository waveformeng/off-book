"""The audio graph. Three paths, kept apart:

    reference vocal → file → ReferencePCM → ReferencePacer → Frames[Reference]  (no output route)
    backing track   → file → BackingPCM   → output device                      (not analyzed)
    live vocal      → input device → Frames[Live] (+ performance .wav)

The mic input callback is the transport clock. On every block of N frames it advances
the reference pacer by exactly N frames, so the two analysis streams are sample-aligned
by construction. The backing track runs on the output device's clock; the DriftMonitor
measures it against the transport clock.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, final

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.typing import NDArray

from offbook.audio.decode import BackingPCM, ReferencePCM
from offbook.audio.drift import DriftMonitor
from offbook.audio.guard import ReferenceOutputRouteError, _opening_output
from offbook.audio.pacer import ReferencePacer
from offbook.audio.sources import LiveReplayFile, MicInput
from offbook.roles import Reference


@final
@dataclass(frozen=True, slots=True)
class CapturedBlock:
    """One mic block and the reference block paced against it. Same length, same clock."""

    live: NDArray[np.float32]
    reference: NDArray[np.float32]
    start_frame: int
    wall: float


class _BackingPlayer:
    """The only object that ever feeds an output device. Its only source is a BackingPCM."""

    def __init__(self, backing: BackingPCM) -> None:
        if not isinstance(backing, BackingPCM):
            raise ReferenceOutputRouteError(f"output source must be BackingPCM, got {backing!r}")
        self.backing = backing
        self.cursor = 0
        self.last_block: tuple[int, float | None] = (0, None)
        """(backing frame index, wall time) of the first frame of the latest output block."""

    def fill(self, outdata: NDArray[np.float32], frames: int, dac_time: float | None) -> None:
        end = min(self.cursor + frames, self.backing.n_frames)
        n = max(0, end - self.cursor)
        outdata[:n] = self.backing.samples[self.cursor : self.cursor + n]
        outdata[n:] = 0.0
        self.last_block = (self.cursor, dac_time)
        self.cursor += frames

    @property
    def done(self) -> bool:
        return self.cursor >= self.backing.n_frames


@final
class AudioGraph:
    def __init__(
        self,
        reference: ReferencePCM,
        live_source: MicInput | LiveReplayFile,
        backing: BackingPCM | None,
        *,
        output_device: int | str | None = None,
        blocksize: int = 1024,
        drift_interval_s: float = 1.0,
        stop_after_frames: int | None = None,
        live_tap: Callable[[NDArray[np.float32], float], None] | None = None,
    ) -> None:
        """`live_tap(block, end_s)` is called on the capture thread with each LIVE block the
        moment it is captured, ahead of the analysis queue. The stage waveform hangs off it:
        the analysis loop stalls for a decode every hop, so a trace fed from there freezes
        and jumps with the decoder instead of following the mic. Only the live block passes
        through the tap — the reference is paced after it and never reaches it."""
        self.reference = reference
        self.rate = reference.rate
        self.live_source = live_source
        self.blocksize = blocksize
        self._live_tap = live_tap
        self.blocks: queue.Queue[CapturedBlock | None] = queue.Queue()
        self._pacer = ReferencePacer(reference)
        self._mic_total = 0
        self._stop_after = stop_after_frames
        self._stopped = threading.Event()
        self._error: BaseException | None = None

        self._player = _BackingPlayer(backing) if backing is not None else None
        self._output_device = output_device
        self._in_stream: Any = None
        self._out_stream: Any = None
        self._replay_thread: threading.Thread | None = None

        out_rate = backing.rate if backing is not None else self.rate
        self.duplex = self._is_duplex()
        self.drift = DriftMonitor(self.rate, out_rate, drift_interval_s, self.duplex)

    # --- constraint 2 ------------------------------------------------------------------

    def output_sources(self) -> list[object]:
        return [self._player.backing] if self._player is not None else []

    def assert_reference_has_no_output_route(self) -> None:
        """Fail loudly if anything on an output path is, or holds, the reference vocal."""
        for src in self.output_sources():
            if src is self.reference:
                raise ReferenceOutputRouteError("output source is the reference object")
            if isinstance(src, ReferencePCM) or getattr(src, "role", None) is Reference:
                raise ReferenceOutputRouteError("reference vocal is registered as an output source")
            if not isinstance(src, BackingPCM):
                raise ReferenceOutputRouteError(f"unexpected output source {type(src).__name__}")
        if self._player is not None:
            for name, value in vars(self._player).items():
                if isinstance(value, ReferencePCM):
                    raise ReferenceOutputRouteError(f"backing player attribute {name} is reference")
        if isinstance(self.live_source, LiveReplayFile) and self._player is not None:
            raise ReferenceOutputRouteError("replay sessions must not open an output device")

    # --- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        self.assert_reference_has_no_output_route()
        if isinstance(self.live_source, LiveReplayFile):
            self._replay_thread = threading.Thread(
                target=self._replay_loop, name="offbook-replay", daemon=True
            )
            self._replay_thread.start()
            return
        if self.duplex and self._player is not None:
            with _opening_output():
                self._in_stream = sd.Stream(
                    device=(self.live_source.device, self._output_device),
                    samplerate=self.rate,
                    blocksize=self.blocksize,
                    dtype="float32",
                    channels=(1, 2),
                    callback=self._duplex_callback,
                )
            self._in_stream.start()
            return
        if self._player is not None:
            with _opening_output():
                self._out_stream = sd.OutputStream(
                    device=self._output_device,
                    samplerate=self._player.backing.rate,
                    blocksize=self.blocksize,
                    dtype="float32",
                    channels=2,
                    callback=self._output_callback,
                )
            self._out_stream.start()
        self._in_stream = sd.InputStream(
            device=self.live_source.device,
            samplerate=self.rate,
            blocksize=self.blocksize,
            dtype="float32",
            channels=1,
            callback=self._input_callback,
        )
        self._in_stream.start()

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                s.stop()
                s.close()
        if self._replay_thread is not None:
            self._replay_thread.join()
        self.blocks.put(None)

    @property
    def mic_frames(self) -> int:
        return self._mic_total

    @property
    def error(self) -> BaseException | None:
        return self._error

    # --- callbacks (PortAudio threads) ------------------------------------------------

    def _ingest(
        self, indata: NDArray[np.float32], frames: int, adc_time: float | None = None
    ) -> None:
        if self._stopped.is_set():
            return
        try:
            live = np.ascontiguousarray(indata[:frames, 0], dtype=np.float32).copy()
            start = self._mic_total
            self._mic_total += frames
            if self._live_tap is not None:
                self._live_tap(live, self._mic_total / self.rate)
            reference = self._pacer.advance(frames, self._mic_total)
            if self._player is not None:
                dac_frames, dac_time = self._player.last_block
            else:
                dac_frames, dac_time = start, None
            self.drift.observe(start, adc_time, dac_frames, dac_time)
            self.blocks.put(CapturedBlock(live, reference, start, time.monotonic()))
            if self._stop_after is not None and self._mic_total >= self._stop_after:
                threading.Thread(target=self.stop, daemon=True).start()
        except BaseException as e:  # noqa: BLE001 — surfaced to the session thread
            self._error = e
            self._stopped.set()
            self.blocks.put(None)
            raise sd.CallbackAbort from e

    def _input_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        self._ingest(indata, frames, float(time_info.inputBufferAdcTime))

    def _output_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        assert self._player is not None
        self._player.fill(outdata, frames, float(time_info.outputBufferDacTime))

    def _duplex_callback(
        self, indata: Any, outdata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        assert self._player is not None
        self._player.fill(outdata, frames, float(time_info.outputBufferDacTime))
        self._ingest(indata, frames, float(time_info.inputBufferAdcTime))

    # --- replay ------------------------------------------------------------------------

    def _is_duplex(self) -> bool:
        if isinstance(self.live_source, LiveReplayFile) or self._player is None:
            return False
        try:
            in_dev = sd.query_devices(self.live_source.device, "input")
            out_dev = sd.query_devices(self._output_device, "output")
        except (ValueError, sd.PortAudioError):
            return False
        return bool(in_dev["index"] == out_dev["index"]) and self._player.backing.rate == self.rate

    def _replay_loop(self) -> None:
        assert isinstance(self.live_source, LiveReplayFile)
        with sf.SoundFile(str(self.live_source.path)) as f:
            if f.samplerate != self.rate:
                self._error = ValueError(
                    f"replay file rate {f.samplerate} != transport rate {self.rate}"
                )
                self.blocks.put(None)
                return
            while not self._stopped.is_set():
                data = f.read(self.blocksize, dtype="float32", always_2d=True)
                if len(data) == 0:
                    break
                self._ingest(data.mean(axis=1, keepdims=True).astype(np.float32), len(data))
        if not self._stopped.is_set():
            self._stopped.set()
            self.blocks.put(None)


def input_device_rate(device: int | str | None) -> int:
    info = sd.query_devices(device, "input")
    return int(info["default_samplerate"])


def output_device_rate(device: int | str | None) -> int:
    info = sd.query_devices(device, "output")
    return int(info["default_samplerate"])
