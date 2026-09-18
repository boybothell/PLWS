"""Canonical generation protocol shared by Full-CoT and PLWS."""

from __future__ import annotations

PROTOCOL_ID = "puma-fullcot-32k-v2"

# PUMA Step 1a generates at most 32K tokens from the original prompt.  If that
# budget ends before </think>, the official runner appends </think> and grants
# a separate answer-fix budget.
FULLCOT_GENERATION_TOKENS = 32768
TRUNCATED_ANSWER_FIX_TOKENS = 2048
PROMPT_RESERVE_TOKENS = 3072
PUMA_FINAL_REGENERATION_CAP = 4096
MAX_MODEL_LEN = (
    FULLCOT_GENERATION_TOKENS
    + PROMPT_RESERVE_TOKENS
    + TRUNCATED_ANSWER_FIX_TOKENS
)


def remaining_generation_tokens(prefix_tokens: int) -> int:
    """Return the official 32K generation budget left after a saved prefix."""

    return max(0, FULLCOT_GENERATION_TOKENS - int(prefix_tokens))


def puma_final_regeneration_tokens(
    prefix_tokens: int,
    *,
    requested_cap: int = PUMA_FINAL_REGENERATION_CAP,
    fullcot_generation_tokens: int = FULLCOT_GENERATION_TOKENS,
) -> int:
    """Cap PUMA final regeneration by the per-question host remainder."""

    prefix = int(prefix_tokens)
    cap = int(requested_cap)
    total = int(fullcot_generation_tokens)
    if prefix < 0:
        raise ValueError(f"prefix_tokens must be non-negative, got {prefix}")
    if cap < 1:
        raise ValueError(f"requested_cap must be positive, got {cap}")
    if total < 1:
        raise ValueError(
            f"fullcot_generation_tokens must be positive, got {total}"
        )
    return min(cap, max(0, total - prefix))
