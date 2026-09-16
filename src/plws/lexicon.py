"""Frozen reflection lexicons used by PLWS-compatible analyses."""

from __future__ import annotations

WAIT: tuple[str, ...] = ("Wait", "wait", "等一下")
ALTERNATIVELY: tuple[str, ...] = (
    "Alternatively",
    "alternatively",
)
HMM: tuple[str, ...] = (
    "Hmm",
    "hmm",
    "Hm",
    "hm",
)
CORE: tuple[str, ...] = WAIT + ALTERNATIVELY + HMM
ALTERNATIVE: tuple[str, ...] = ("Alternative", "alternative")
CORE_PLUS_ALTERNATIVE: tuple[str, ...] = CORE + ALTERNATIVE
LET_ME: tuple[str, ...] = ("Let me", "let me")
BUT: tuple[str, ...] = ("But", "but")
SO: tuple[str, ...] = ("So", "so")
THEREFORE: tuple[str, ...] = ("Therefore", "therefore")
CORE_PLUS_BUT_SO_THEREFORE: tuple[str, ...] = CORE + BUT + SO + THEREFORE

SAFE: tuple[str, ...] = CORE + (
    "However",
    "however",
    "Maybe",
    "maybe",
    "Perhaps",
    "perhaps",
    "another way",
    "another approach",
    "another method",
    "double-check",
    "double check",
    "Hold on",
    "hold on",
    "换一种",
)

LEXICONS: dict[str, tuple[str, ...]] = {
    "alternatively": ALTERNATIVELY,
    "but": BUT,
    "core": CORE,
    "core_no_alternatively": WAIT + HMM,
    "core_no_hmm": WAIT + ALTERNATIVELY,
    "core_no_wait": ALTERNATIVELY + HMM,
    "core_plus_alternative": CORE_PLUS_ALTERNATIVE,
    "core_plus_but_so_therefore": CORE_PLUS_BUT_SO_THEREFORE,
    "core_plus_let_me": CORE + LET_ME,
    "hmm": HMM,
    "let_me": LET_ME,
    "safe": SAFE,
    "so": SO,
    "therefore": THEREFORE,
    "wait": WAIT,
}


def get_lexicon(name: str) -> tuple[str, ...]:
    """Return a named immutable lexicon."""

    try:
        return LEXICONS[name]
    except KeyError as error:
        choices = ", ".join(sorted(LEXICONS))
        raise ValueError(f"unknown lexicon {name!r}; choose one of: {choices}") from error


# Forms that actually leaked through the old CORE list. Official
# suppress covers these plus a leading-space copy.
OBSERVED_CORE_LEAKS: tuple[str, ...] = (
    "WAIT",
    "Hmmm",
    "hmmm",
    "Hmmmm",
    "hmmmm",
    "hmm",
    "Wait",
    "alternatively",
    "Alternatively",
)


def _hmm_length_leaks(min_m: int = 3, max_m: int = 12) -> tuple[str, ...]:
    """``Hmmm`` … ``H`` + 12 m's, both cases. ``HM`` is not included."""

    out: list[str] = []
    for n in range(min_m, max_m + 1):
        out.append("H" + "m" * n)
        out.append("h" + "m" * n)
    return tuple(out)


CORE_LEAK_EXTRAS: tuple[str, ...] = ("WAIT", "ALTERNATIVELY") + _hmm_length_leaks()


def expand_bad_words(words: tuple[str, ...] | list[str]) -> list[str]:
    """Exact strings plus a leading-space copy for vLLM ``bad_words``.

    BPE often emits `` alternatively`` as a different token from
    ``alternatively``. Do not add ``HM``: it collides with AM-HM.
    """
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        for candidate in (word, f" {word}" if not word.startswith(" ") else word):
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
    return out


def suppress_bad_words(name: str = "core") -> list[str]:
    """Official leftover ban list: named lexicon plus leakfix extras.

    ``core`` and ``core_*`` also ban ``WAIT`` / ``ALTERNATIVELY`` / ``Hmmm``
    (3–12 m's) and a leading-space copy of every string.
    """
    words = list(get_lexicon(name))
    if name == "core" or name.startswith("core"):
        seen = set(words)
        for extra in CORE_LEAK_EXTRAS:
            if extra in seen:
                continue
            seen.add(extra)
            words.append(extra)
    return expand_bad_words(words)


def usable_bad_words(words: list[str], tokenizer: object) -> list[str]:
    """Drop strings this tokenizer cannot encode.

    vLLM ``update_from_tokenizer`` indexes ``prompt_token_ids[0]``. An empty
    encode, as Llama does for ``等一下``, raises ``IndexError`` before any
    leftover continuation is generated.
    """

    encode = getattr(tokenizer, "encode")
    kept: list[str] = []
    for word in words:
        stripped = word.lstrip()
        if not stripped:
            continue
        bare = encode(stripped, add_special_tokens=False)
        spaced = encode(" " + stripped, add_special_tokens=False)
        if not bare or not spaced:
            continue
        kept.append(word)
    return kept
