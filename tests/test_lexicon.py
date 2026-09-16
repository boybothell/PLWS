from plws.lexicon import (
    CORE,
    SO,
    THEREFORE,
    expand_bad_words,
    get_lexicon,
    suppress_bad_words,
    usable_bad_words,
)


def test_core_plus_alternative_adds_only_the_stem() -> None:
    words = get_lexicon("core_plus_alternative")
    assert words[: len(CORE)] == CORE
    for token in ("Alternative", "alternative"):
        assert token in words
    assert "another" not in words
    assert "Another" not in words
    assert "Let me" not in words
    banned = set(suppress_bad_words("core_plus_alternative"))
    for form in ("Alternative", "alternative", " Alternative", " alternative"):
        assert form in banned
    for form in ("another", "Another", " another"):
        assert form not in banned
    core_banned = set(suppress_bad_words("core"))
    assert "Alternative" not in core_banned
    assert "WAIT" in core_banned
    assert " WAIT" in core_banned


def test_core_plus_but_so_therefore_is_single_token_words() -> None:
    words = get_lexicon("core_plus_but_so_therefore")
    assert words[: len(CORE)] == CORE
    for token in ("But", "but", "So", "so", "Therefore", "therefore"):
        assert token in words
    assert "Let me" not in words
    assert "let me" not in words
    assert all(" " not in token for token in words)


def test_official_core_covers_observed_leaks() -> None:
    assert CORE == (
        "Wait",
        "wait",
        "等一下",
        "Alternatively",
        "alternatively",
        "Hmm",
        "hmm",
        "Hm",
        "hm",
    )
    for token in ("Hmmm", "hmmm", "Hmmmmm", "WAIT", "ALTERNATIVELY"):
        assert token not in CORE
    assert "HM" not in CORE
    banned = set(suppress_bad_words("core"))
    for form in (
        "WAIT",
        " WAIT",
        "Hmmm",
        " Hmmm",
        "ALTERNATIVELY",
        " ALTERNATIVELY",
        " alternatively",
    ):
        assert form in banned
    assert "HM" not in banned
    expanded = expand_bad_words(CORE)
    assert " alternatively" in expanded
    assert expanded[0] == CORE[0]


class _EmptyEncodeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if "等一下" in text:
            return []
        return [1]


def test_usable_bad_words_drops_empty_encodes() -> None:
    banned = suppress_bad_words("core")
    kept = usable_bad_words(banned, _EmptyEncodeTokenizer())
    assert "等一下" not in kept
    assert " 等一下" not in kept
    assert "Wait" in kept
    assert " Wait" in kept
    assert "WAIT" in kept


def test_llama_tokenizer_drops_dengyixia() -> None:
    from pathlib import Path

    model = Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Llama-8B")
    if not (model / "tokenizer.json").is_file() and not (model / "tokenizer.model").is_file():
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    kept = usable_bad_words(suppress_bad_words("core"), tokenizer)
    assert "等一下" not in kept
    assert " 等一下" not in kept
    assert "Wait" in kept
    assert tokenizer.encode("Wait", add_special_tokens=False)


def test_so_and_therefore_are_single_word_lexicons() -> None:
    assert get_lexicon("so") == SO
    assert get_lexicon("therefore") == THEREFORE
    assert get_lexicon("so") == ("So", "so")
    assert get_lexicon("therefore") == ("Therefore", "therefore")


if __name__ == "__main__":
    test_core_plus_but_so_therefore_is_single_token_words()
    test_official_core_covers_observed_leaks()
    test_core_plus_alternative_adds_only_the_stem()
    test_so_and_therefore_are_single_word_lexicons()
    test_usable_bad_words_drops_empty_encodes()
    test_llama_tokenizer_drops_dengyixia()
    print("ok")
