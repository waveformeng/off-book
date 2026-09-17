from offbook.compare.normalize import normalize_phoneme, normalize_word


def test_words_are_lowercased_and_stripped() -> None:
    assert normalize_word(" Don't, ") == "don't"
    assert normalize_word("Café!") == "cafe"
    assert normalize_word("'tis") == "tis"


def test_breath_fillers_are_not_words() -> None:
    assert normalize_word("Uh") == ""
    assert normalize_word("hmm") == ""
    assert normalize_word("oh") == "oh"  # a real sung word stays


def test_phonemes_keep_their_marks() -> None:
    assert normalize_phoneme(" ɑː ") == "ɑː"
