from offbook.asr.base import Token
from offbook.clock import ANALYSIS_RATE, TransportTime
from offbook.compare.align import Aligner, phoneme_distance
from offbook.compare.verdict import TokenPair, Verdict
from offbook.config import AlignmentConfig
from offbook.roles import Live, Reference

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()


def _ref(text: str, t: float) -> Token[Reference]:
    return Token(Reference, text, _tt(t), _tt(t + 0.2), 1.0, 0.0)


def _live(text: str, t: float) -> Token[Live]:
    return Token(Live, text, _tt(t), _tt(t + 0.2), 1.0, 0.0)


def _tt(t: float) -> TransportTime:
    return TransportTime.from_seconds(t, ANALYSIS_RATE)


def _run(
    ref: list[Token[Reference]], live: list[Token[Live]], cfg: AlignmentConfig | None = None
) -> list[TokenPair]:
    a = Aligner(cfg or AlignmentConfig(), "word")
    a.add_reference(ref)
    a.add_live(live)
    return a.finish()


def _verdicts(pairs: list[TokenPair]) -> list[tuple[str, str, Verdict]]:
    return [
        (p.reference.text if p.reference else "-", p.live.text if p.live else "-", p.verdict)
        for p in pairs
    ]


def test_identical_streams_all_match() -> None:
    ref = [_ref(w, 0.5 * i) for i, w in enumerate(WORDS)]
    live = [_live(w, 0.5 * i + 0.1) for i, w in enumerate(WORDS)]
    assert all(p.verdict is Verdict.MATCH for p in _run(ref, live))


def test_late_singer_does_not_cascade() -> None:
    ref = [_ref(w, 0.5 * i) for i, w in enumerate(WORDS)]
    live = [_live(w, 0.5 * i + 0.9) for i, w in enumerate(WORDS)]  # beyond tolerance, inside lag
    verdicts = _verdicts(_run(ref, live))
    assert all(v is Verdict.FAIL_TIMING for _, _, v in verdicts)
    assert [r for r, _, _ in verdicts] == WORDS


def test_within_tolerance_late_is_match() -> None:
    ref = [_ref(w, 0.5 * i) for i, w in enumerate(WORDS)]
    live = [_live(w, 0.5 * i + 0.5) for i, w in enumerate(WORDS)]  # a beat late at 120 bpm
    assert all(p.verdict is Verdict.MATCH for p in _run(ref, live))


def test_missed_substituted_inserted() -> None:
    ref = [_ref(w, 0.5 * i) for i, w in enumerate(WORDS)]
    live = []
    for i, w in enumerate(WORDS):
        if w == "delta":
            continue
        live.append(_live("zulu" if w == "golf" else w, 0.5 * i))
    live.append(_live("extra", 2.25))
    v = _verdicts(_run(ref, live))
    assert ("delta", "-", Verdict.FAIL_MISSED) in v
    assert ("golf", "zulu", Verdict.FAIL_LEXICAL) in v
    assert ("-", "extra", Verdict.FAIL_INSERTED) in v
    assert sum(1 for _, _, x in v if x is Verdict.MATCH) == 8


def test_tokens_wait_until_decidable() -> None:
    a = Aligner(AlignmentConfig(), "word")
    a.add_reference([_ref("alpha", 0.0), _ref("bravo", 0.5)])
    a.add_live([_live("alpha", 0.1)])
    assert a.advance(_tt(1.0), _tt(1.0)) == []  # live not resolved past alpha + tol + lag
    pairs = a.advance(_tt(3.0), _tt(2.3))
    assert _verdicts(pairs) == [("alpha", "alpha", Verdict.MATCH)]
    assert _verdicts(a.finish()) == [("bravo", "-", Verdict.FAIL_MISSED)]


def test_live_token_past_window_edge_is_not_inserted_early() -> None:
    cfg = AlignmentConfig(window_tokens=2)
    ref = [_ref(w, 0.5 * i) for i, w in enumerate(WORDS[:4])]
    live = [_live(w, 0.5 * i) for i, w in enumerate(WORDS[:4])]
    assert all(p.verdict is Verdict.MATCH for p in _run(ref, live, cfg))


def test_phoneme_distance() -> None:
    assert phoneme_distance("a", "a") == 0.0
    assert phoneme_distance("aɪ", "a") == 0.5
    assert phoneme_distance("k", "t") == 1.0


def test_repeated_tokens_pair_with_their_own_occurrence() -> None:
    # Identical streams full of repeats, denser than the tolerance: every token must MATCH.
    ref = [_ref("n", 0.1 * i) for i in range(30)]
    live = [_live("n", 0.1 * i) for i in range(30)]
    pairs = _run(ref, live, AlignmentConfig(window_tokens=12))
    assert [p.verdict for p in pairs] == [Verdict.MATCH] * 30
    assert all(p.dt_s == 0.0 for p in pairs)


def test_live_tokens_after_reference_end_are_ignored_not_inserted() -> None:
    ref = [_ref(w, 0.5 * i) for i, w in enumerate(WORDS[:4])]  # last reference token at 1.5 s
    live = [_live(w, 0.5 * i) for i, w in enumerate(WORDS[:4])]
    chatter = [_live(w, 6.0 + 0.3 * i) for i, w in enumerate(["so", "how", "did", "i", "do"])]
    a = Aligner(AlignmentConfig(), "word", reference_end_s=5.0)
    a.add_reference(ref)
    a.add_live(live + chatter)
    # Reference not finished yet: chatter must wait, not be decided.
    pairs = a.advance(_tt(4.0), _tt(8.0))
    assert all(p.verdict is Verdict.MATCH for p in pairs) and len(pairs) == 4
    assert a.finish() == []
    assert a.ignored_after_end == 5


def test_adlib_during_a_break_is_still_inserted_when_the_song_goes_on() -> None:
    ref = [_ref("alpha", 0.0), _ref("bravo", 0.5), _ref("charlie", 10.0)]
    live = [_live("alpha", 0.0), _live("bravo", 0.5), _live("yeah", 5.0), _live("charlie", 10.0)]
    a = Aligner(AlignmentConfig(), "word", reference_end_s=12.0)
    a.add_reference(ref[:2])
    a.add_live(live[:3])
    assert [p.verdict for p in a.advance(_tt(8.0), _tt(8.0))] == [Verdict.MATCH, Verdict.MATCH]
    a.add_reference(ref[2:])  # the next phrase arrives: the ad-lib is now decidable
    a.add_live(live[3:])
    v = _verdicts(a.advance(_tt(13.0), _tt(13.0)))
    assert ("-", "yeah", Verdict.FAIL_INSERTED) in v and ("charlie", "charlie", Verdict.MATCH) in v


def test_scoring_past_reference_end_can_be_enabled() -> None:
    a = Aligner(AlignmentConfig(score_past_reference_end=True), "word", reference_end_s=1.0)
    a.add_live([_live("late", 5.0)])
    assert _verdicts(a.finish()) == [("-", "late", Verdict.FAIL_INSERTED)]
