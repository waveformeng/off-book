"""Audio sources. These are the only constructors for each stream role.

ReferenceVocalFile → ReferencePCM → Frames[Reference]      (analysis only, never output)
BackingTrackFile   → BackingPCM   → OutputBus              (output only, never analyzed)
MicInput           → Frames[Live] (+ performance .wav)
LiveReplayFile     → Frames[Live]                          (a recorded performance)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import final


@final
@dataclass(frozen=True, slots=True)
class ReferenceVocalFile:
    path: Path


@final
@dataclass(frozen=True, slots=True)
class BackingTrackFile:
    path: Path


@final
@dataclass(frozen=True, slots=True)
class MicInput:
    device: int | str | None = None
    """sounddevice input device index/name; None = system default."""


@final
@dataclass(frozen=True, slots=True)
class LiveReplayFile:
    path: Path
