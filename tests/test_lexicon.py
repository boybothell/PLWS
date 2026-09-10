from plws.lexicon import CORE, SO, THEREFORE, get_lexicon


def test_core_plus_but_so_therefore_is_single_token_words() -> None:
    words = get_lexicon("core_plus_but_so_therefore")
    assert words[: len(CORE)] == CORE
    for token in ("But", "but", "So", "so", "Therefore", "therefore"):
        assert token in words
    assert "Let me" not in words
    assert "let me" not in words
    assert all(" " not in token for token in words)


def test_so_and_therefore_are_single_word_lexicons() -> None:
    assert get_lexicon("so") == SO
    assert get_lexicon("therefore") == THEREFORE
    assert get_lexicon("so") == ("So", "so")
    assert get_lexicon("therefore") == ("Therefore", "therefore")


if __name__ == "__main__":
    test_core_plus_but_so_therefore_is_single_token_words()
    test_so_and_therefore_are_single_word_lexicons()
    print("ok")
