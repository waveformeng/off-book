"""Every line tagged ERR below must be a mypy error. test_roles_typing asserts on them."""

from typing import cast

from offbook.asr.base import Frames, Token
from offbook.compare.align import Aligner
from offbook.compare.verdict import TokenPair, Verdict
from offbook.roles import Live, Reference

ref_tok = cast(Token[Reference], None)
live_tok = cast(Token[Live], None)
ref_frames = cast(Frames[Reference], None)
aligner = cast(Aligner, None)

swapped_pair = TokenPair(live_tok, ref_tok, Verdict.MATCH)  # ERR swapped sides
live_as_ref: Token[Reference] = live_tok  # ERR assignment across roles
aligner.add_reference([live_tok])  # ERR live tokens into the reference side
aligner.add_live([ref_tok])  # ERR reference tokens into the live side
live_frames: Frames[Live] = ref_frames  # ERR frames across roles
either = cast(Token[Reference | Live], None)  # ERR no union role exists
