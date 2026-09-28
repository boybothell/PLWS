"""Pure helpers for the canonical Dynasor frozen-trajectory replay."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

EFFORT = "mid"
CHUNK_SIZE = 64
CERTAINTY_THRESHOLD = 3
PROBE_MAX_TOKENS = 20
UNCERTAIN_WORDS = ("wait", "hold", "but", "okay", "no", "hmm")
PROBE_SUFFIX = (
    "... Oh, I suddenly got the answer to the whole problem, "
    "**Final Answer**\n\n\\[ \\boxed{"
)


def probe_prompt_token_ids(
    base_ids: Sequence[int],
    reasoning_ids: Sequence[int],
    boundary: int,
    suffix_ids: Sequence[int],
) -> list[int]:
    """Build one probe as token ids so the next boundary extends this prefix.

    The probe suffix sits after the reasoning prefix. The following boundary
    therefore shares ``base_ids + reasoning_ids[:boundary]`` and not the suffix.
    """

    if boundary < 1 or boundary > len(reasoning_ids):
        raise ValueError(
            f"boundary must be in 1..{len(reasoning_ids)}, got {boundary}"
        )
    return [*base_ids, *reasoning_ids[:boundary], *suffix_ids]


def reusable_probe_state(state: object) -> dict[str, object] | None:
    """Return a saved probe history that can be resumed, or None."""

    if not isinstance(state, dict):
        return None
    answers = state.get("answers")
    certainties = state.get("certainties")
    if not isinstance(answers, list) or not isinstance(certainties, list):
        return None
    if len(answers) != len(certainties):
        return None
    if any(not isinstance(answer, str) for answer in answers):
        return None
    if any(not isinstance(certain, bool) for certain in certainties):
        return None
    try:
        probe_tokens = int(state.get("probe_tokens", 0))
        probe_count = int(state.get("probe_count", len(answers)))
    except (TypeError, ValueError):
        return None
    if probe_tokens < 0 or probe_count != len(answers):
        return None
    return {
        "answers": list(answers),
        "certainties": list(certainties),
        "probe_tokens": probe_tokens,
        "probe_count": probe_count,
    }


def token_chunk_boundaries(total_tokens: int, chunk_size: int = CHUNK_SIZE) -> tuple[int, ...]:
    """Return probe boundaries before the frozen trajectory's natural end."""

    total = int(total_tokens)
    chunk = int(chunk_size)
    if total < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total}")
    if chunk < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk}")
    return tuple(range(chunk, total, chunk))


def obtain_answer(probe_text: str) -> str:
    """Extract text before the first closing brace unmatched inside the probe."""

    stack: list[str] = []
    for index, character in enumerate(probe_text):
        if character == "{":
            stack.append(character)
        elif character == "}":
            if not stack:
                return probe_text[:index].strip()
            stack.pop()
    return ""


def is_certain_answer(
    probe_text: str,
    *,
    uncertain_words: Sequence[str] = UNCERTAIN_WORDS,
) -> bool:
    lowered = probe_text.lower()
    return not any(word.lower() in lowered for word in uncertain_words)


def normalize_gpqa_answer(answer: str) -> str:
    """Normalize a Dynasor GPQA probe to A/B/C/D or the empty string."""

    matches = re.findall(r"\b([A-D])\b", answer.upper())
    return matches[-1] if matches else ""


def should_early_exit(
    answers: Sequence[str],
    certainties: Sequence[bool],
    *,
    equivalent: Callable[[str, str], bool],
    threshold: int = CERTAINTY_THRESHOLD,
) -> bool:
    """Apply Dynasor's non-empty, certain, mathematically equivalent window."""

    if threshold < 1:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if len(answers) != len(certainties):
        raise ValueError("answers and certainties must have equal lengths")
    if len(answers) < threshold:
        return False
    window = list(answers[-threshold:])
    certain_window = certainties[-threshold:]
    if any(not answer for answer in window) or not all(certain_window):
        return False
    first = window[0]
    return all(equivalent(first, answer) for answer in window[1:])
