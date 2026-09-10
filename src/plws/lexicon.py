"""Frozen reflection lexicons used by PLWS-compatible analyses."""

from __future__ import annotations

CORE: tuple[str, ...] = (
    "Wait",
    "wait",
    "Alternatively",
    "alternatively",
    "Hmm",
    "hmm",
    "Hm",
    "hm",
    "等一下",
)

WAIT: tuple[str, ...] = ("Wait", "wait", "等一下")
ALTERNATIVELY: tuple[str, ...] = ("Alternatively", "alternatively")
HMM: tuple[str, ...] = ("Hmm", "hmm", "Hm", "hm")
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
