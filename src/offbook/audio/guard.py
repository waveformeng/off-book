"""Belt and braces for constraint 2: only `offbook.audio.graph` may open an output stream.

Every sounddevice entry point that can reach an output device is wrapped at import. The
wrappers raise unless the graph is inside `_opening_output()`, and the graph itself only
ever opens an output whose source is a `BackingPCM`.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from typing import Any

import sounddevice as sd

_state = threading.local()


class ReferenceOutputRouteError(RuntimeError):
    pass


class UnauthorizedOutputError(RuntimeError):
    pass


def _wrap(name: str) -> None:
    original = getattr(sd, name)

    def guarded(*args: Any, **kwargs: Any) -> Any:
        if not getattr(_state, "allowed", False):
            raise UnauthorizedOutputError(
                f"sounddevice.{name} may only be opened by offbook.audio.graph"
            )
        return original(*args, **kwargs)

    guarded.__name__ = name
    setattr(sd, name, guarded)
    _originals[name] = original


_originals: dict[str, Callable[..., Any]] = {}
for _name in ("play", "OutputStream", "RawOutputStream", "Stream", "RawStream", "playrec"):
    _wrap(_name)


@contextlib.contextmanager
def _opening_output() -> Iterator[None]:
    _state.allowed = True
    try:
        yield
    finally:
        _state.allowed = False
