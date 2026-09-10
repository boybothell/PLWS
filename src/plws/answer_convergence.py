"""Pure Answer Convergence helpers.

The algorithm follows Liu and Wang (EMNLP 2025): split a complete reasoning
trace at sentence boundaries, greedily probe an answer after each cumulative
prefix, and stop after ``k`` consecutive identical extracted answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class ConvergenceState:
    last_answer: str | None = None
    repeat_count: int = 0


def cumulative_sentence_prefixes(
    text: str, sentence_tokenize: Callable[[str], Iterable[str]]
) -> list[str]:
    """Return cumulative prefixes while preserving the source text exactly."""

    sentences = list(sentence_tokenize(text))
    prefixes: list[str] = []
    end = 0
    for sentence in sentences:
        index = text.find(sentence, end)
        if index < 0:
            raise ValueError(f"sentence tokenizer output not found after offset {end}")
        end = index + len(sentence)
        prefixes.append(text[:end].strip())
    return prefixes


def observe_answer(
    state: ConvergenceState, answer: str, *, threshold: int
) -> tuple[ConvergenceState, bool]:
    """Update exact-answer consistency and report whether it converged."""

    if threshold < 1:
        raise ValueError("threshold must be positive")
    repeat_count = (
        state.repeat_count + 1 if answer == state.last_answer else 1
    )
    updated = ConvergenceState(answer, repeat_count)
    return updated, repeat_count >= threshold
