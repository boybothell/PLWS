"""Online Answer Convergence on a sampled reasoning chain.

Probe each newly completed sentence with the same greedy boxed probe as the
offline runner. The probe is not written back. Stop the main chain when ten
consecutive extracted answers match, or when the think budget or ``</think>``
ends the chain.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from plws.answer_convergence import cumulative_sentence_prefixes

PROBE_SUFFIX = "\n</think>\n\\boxed"
THINK_CLOSE = "</think>"
CHUNK_TOKENS = 32
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*\s*$")


def take_reasoning_chunk(chunk: str) -> tuple[str, bool]:
    """Keep reasoning text and report whether ``</think>`` closed the chain."""

    if THINK_CLOSE in chunk:
        return chunk.split(THINK_CLOSE, 1)[0], True
    return chunk, False


def committed_prefixes(
    text: str,
    sentence_tokenize: Callable[[str], Iterable[str]],
    *,
    finished: bool,
) -> list[str]:
    """Return cumulative sentence prefixes that can be probed.

    The trailing fragment stays unprobed until the text ends on sentence
    punctuation or the main chain has stopped.
    """

    if not text.strip():
        return []
    prefixes = cumulative_sentence_prefixes(text, sentence_tokenize)
    if finished or _SENTENCE_END.search(text):
        return prefixes
    return prefixes[:-1]
