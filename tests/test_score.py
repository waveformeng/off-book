from offbook.asr.base import Token
from offbook.clock import ANALYSIS_RATE, TransportTime
from offbook.compare.score import RunningScore
from offbook.compare.verdict import TokenPair, Verdict
from offbook.roles import Live, Reference

T = TransportTime(0, ANALYSIS_RATE)
R = Token(Reference, "x", T, T, 1.0, 0.0)
L = Token(Live, "x", T, T, 1.0, 0.0)


def test_score_climbs_with_progress_and_lands_on_match_rate() -> None:
    s = RunningScore(timing_fail_weight=0.5, reference_duration_s=10.0)
    s.add(TokenPair(R, L, Verdict.MATCH))
    assert s.match_rate == 1.0
    assert s.score(2.5) == 25.0
    s.add(TokenPair(R, L, Verdict.FAIL_TIMING))  # worth 1 − 0.5
    s.add(TokenPair(R, None, Verdict.FAIL_MISSED))
    s.add(TokenPair(None, L, Verdict.FAIL_INSERTED))
    assert s.match_rate == 1.5 / 4
    assert s.score(10.0) == 37.5
    assert s.score(50.0) == 37.5  # progress is capped


def test_pair_invariants() -> None:
    import pytest

    with pytest.raises(ValueError):
        TokenPair(R, L, Verdict.FAIL_MISSED)
    with pytest.raises(ValueError):
        TokenPair(None, L, Verdict.MATCH)
    with pytest.raises(TypeError):
        TokenPair(L, R, Verdict.MATCH)  # type: ignore[arg-type]
