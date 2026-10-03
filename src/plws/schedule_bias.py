"""Scheduled CORE logit bias on the post-lock continuation.

``t`` is the index of the continuation token about to be sampled. It starts
at 0 and counts only tokens generated after the lock prefix. It is not a
probe step and not a Full-CoT step.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence

HORIZON = 2048
PEAK = 10.0


def count_bias(n: int, rho: float, peak: float = PEAK) -> float:
    """Logit bias after ``n`` CORE sequences have already been completed.

    ``b(n) = -peak * (1 - rho**n)``. ``n`` counts sequences completed in the
    post-lock continuation before the token now being sampled. ``n == 0``
    leaves logits unchanged. Larger ``n`` moves the bias toward ``-peak``.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"post-lock count must be an int >= 0, got {n!r}")
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 0 < float(rho) < 1:
        raise ValueError(f"rho must be in (0, 1), got {rho!r}")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or peak <= 0:
        raise ValueError(f"peak must be positive, got {peak!r}")
    if n == 0:
        return 0.0
    return -float(peak) * (1.0 - float(rho) ** n)


def count_completed_sequences(
    token_ids: list[int], sequences: list[list[int]]
) -> int:
    """How many token positions complete at least one CORE sequence.

    A position counts once even when several sequences end there.
    """

    ends: set[int] = set()
    size = len(token_ids)
    for sequence in sequences:
        width = len(sequence)
        if width <= 0 or width > size:
            continue
        last = size - width + 1
        for start in range(last):
            if token_ids[start : start + width] == sequence:
                ends.add(start + width - 1)
    return len(ends)


class IncrementalSequenceCounter:
    """Count completed sequences without rescanning the whole continuation.

    Normal decoding only appends tokens. Each update therefore examines the
    newly appended end positions, making a full generation linear in its
    length. A rollback or changed suffix resets the counter and rescans once,
    which keeps the result correct for a non-append-only caller.
    """

    def __init__(self, sequences: list[list[int]]) -> None:
        by_last: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        seen: set[tuple[int, ...]] = set()
        for sequence in sequences:
            key = tuple(sequence)
            if not key or key in seen:
                continue
            seen.add(key)
            by_last[key[-1]].append(key)
        self._by_last = dict(by_last)
        self._tail_size = max((len(sequence) for sequence in seen), default=1)
        self._processed = 0
        self._count = 0
        self._tail: tuple[int, ...] = ()

    def update(self, token_ids: Sequence[int]) -> int:
        size = len(token_ids)
        overlap_start = max(0, self._processed - self._tail_size)
        old_tail = tuple(token_ids[overlap_start : self._processed])
        append_only = size >= self._processed and old_tail == self._tail
        if not append_only:
            self._processed = 0
            self._count = 0

        for end in range(self._processed, size):
            candidates = self._by_last.get(int(token_ids[end]), ())
            for sequence in candidates:
                width = len(sequence)
                if width <= end + 1 and tuple(
                    token_ids[end + 1 - width : end + 1]
                ) == sequence:
                    # Match count is per end position, not per surface form.
                    self._count += 1
                    break

        self._processed = size
        tail_start = max(0, size - self._tail_size)
        self._tail = tuple(token_ids[tail_start:size])
        return self._count


def continuation_bias(
    schedule: str,
    t: int,
    *,
    horizon: int = HORIZON,
    peak: float = PEAK,
) -> float:
    """Return the logit bias applied to a CORE token at continuation index ``t``.

    ``front`` starts at ``-peak`` and is 0 once ``t >= horizon``.
    ``back`` starts at 0, follows ``-peak * t / horizon`` until the horizon,
    then becomes negative infinity.
    """

    if schedule not in {"front", "back"}:
        raise ValueError(f"unknown bias schedule {schedule!r}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if peak <= 0:
        raise ValueError(f"peak must be positive, got {peak}")
    if t < 0:
        raise ValueError(f"continuation index must be >= 0, got {t}")
    if t >= horizon:
        return 0.0 if schedule == "front" else float("-inf")
    unit = t / horizon
    if schedule == "front":
        return -float(peak) * (1.0 - unit)
    return -float(peak) * unit


def penalized_token_ids(past: list[int], sequences: list[list[int]]) -> list[int]:
    """Token ids that would complete a banned string at this step.

    A one-token string is always included. A longer string is included only
    when its prefix equals the suffix of tokens already generated in this
    continuation. Each id is returned once.
    """

    found: dict[int, None] = {}
    past_ids = tuple(past)
    for sequence in sequences:
        if not sequence:
            continue
        if len(sequence) == 1:
            found.setdefault(sequence[0], None)
            continue
        if len(sequence) > len(past_ids) + 1:
            continue
        prefix = tuple(sequence[:-1])
        if past_ids[-len(prefix) :] == prefix:
            found.setdefault(sequence[-1], None)
    return list(found)


def encode_ban_sequences(words: list[str], tokenizer: object) -> list[list[int]]:
    """Encode CORE strings the same way vLLM ``bad_words`` does.

    A leading space is kept only when it changes the first token and does not
    change the number of tokens. Strings that already contain a leading space
    are stripped first, so ``" Hmmmm..."`` does not become an extra sequence
    the official ban never applies.
    """

    encode = getattr(tokenizer, "encode")
    seen: set[tuple[int, ...]] = set()
    sequences: list[list[int]] = []
    for word in words:
        unspaced: list[int] | None = None
        for add_prefix_space in (False, True):
            prompt = (" " if add_prefix_space else "") + word.lstrip()
            token_ids = [
                int(token_id) for token_id in encode(prompt, add_special_tokens=False)
            ]
            if not token_ids:
                continue
            if not add_prefix_space:
                unspaced = token_ids
                keep = True
            else:
                keep = (
                    unspaced is not None
                    and token_ids[0] != unspaced[0]
                    and len(token_ids) == len(unspaced)
                )
            if not keep:
                continue
            key = tuple(token_ids)
            if key in seen:
                continue
            seen.add(key)
            sequences.append(token_ids)
    return sequences


def sequence_fingerprint(sequences: list[list[int]]) -> str:
    """Stable id for the exact token sequences this run penalizes."""

    payload = json.dumps(sequences, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
