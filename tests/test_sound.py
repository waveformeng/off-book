import pytest

from offbook.compare.sound import metaphone


@pytest.mark.parametrize(
    "a,b",
    [
        ("for", "four"),
        ("there", "their"),
        ("right", "write"),
        ("your", "you're"),
        ("knight", "night"),
        ("sea", "see"),
        ("hear", "here"),
        ("bare", "bear"),
        ("to", "too"),
    ],
)
def test_homophones_share_a_key(a: str, b: str) -> None:
    assert metaphone(a) == metaphone(b) != ""


@pytest.mark.parametrize("a,b", [("won", "one"), ("whole", "hole"), ("two", "too")])
def test_known_misses_of_classic_metaphone(a: str, b: str) -> None:
    # Initial vowels and W/H-initial words are where the classic rules fall short.
    assert metaphone(a) != metaphone(b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("and", "at"),
        ("let", "the"),
        ("root", "fruit"),
        ("ball", "bug"),
        ("cat", "can"),
        ("it's", "is"),
        ("with", "to"),
        ("game", "gain"),
        ("me", "my"),
    ],
)
def test_different_words_differ(a: str, b: str) -> None:
    assert metaphone(a) != metaphone(b)
