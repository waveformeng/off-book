"""Stream roles: the invariant that keeps the reference vocal and the live vocal apart.

`Reference` and `Live` are phantom types. They are never instantiated; they exist only
as type parameters and runtime tags. `Role` is a *constrained* TypeVar — exactly one of
the two, never a union — so `Frames[Reference]` and `Frames[Live]` are unrelated types
to mypy, and there is no signature that can accept "either".

Everything that carries audio or tokens through the engine is generic over `Role`. The
audio sources are the only constructors for each role (see `offbook.audio.sources`).
"""

from __future__ import annotations

from typing import TypeVar, final


@final
class Reference:
    """The REFERENCE VOCAL: the publisher's licensed vocal demo. Ground truth. Phantom type."""

    __slots__ = ()

    def __init__(self) -> None:
        raise TypeError("Reference is a phantom type and cannot be instantiated")


@final
class Live:
    """The LIVE VOCAL: the mic input. The thing being scored. Phantom type."""

    __slots__ = ()

    def __init__(self) -> None:
        raise TypeError("Live is a phantom type and cannot be instantiated")


Role = TypeVar("Role", Reference, Live)

RoleName = str


def role_name(role: type[Reference] | type[Live]) -> RoleName:
    if role is Reference:
        return "reference"
    if role is Live:
        return "live"
    raise TypeError(f"not a role: {role!r}")
