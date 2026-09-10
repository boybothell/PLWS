"""First same-answer window detection and H/M/L classification."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

K = 4
MSS = 10
TAU = 0.995
EPS = 0.03

AnswerT = TypeVar("AnswerT")
AnswerComparator = Callable[[AnswerT, AnswerT], bool]


class WindowLevel(str, Enum):
    """Confidence class for a same-answer window."""

    H = "H"
    M = "M"
    L = "L"


@dataclass(frozen=True, slots=True)
class Observation(Generic[AnswerT]):
    """One usable trial answer observed at a one-based reasoning step."""

    step: int
    answer: AnswerT
    confidence: float

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class AnswerWindow(Generic[AnswerT]):
    """A qualifying contiguous same-answer window."""

    observations: tuple[Observation[AnswerT], ...]

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("a window cannot be empty")

    @property
    def start_step(self) -> int:
        return self.observations[0].step

    @property
    def end_step(self) -> int:
        return self.observations[-1].step

    @property
    def answer(self) -> AnswerT:
        return self.observations[0].answer

    @property
    def confidences(self) -> tuple[float, ...]:
        return tuple(item.confidence for item in self.observations)

    def level(self, *, tau: float = TAU, eps: float = EPS) -> WindowLevel:
        return classify_window(self, tau=tau, eps=eps)


def _default_answers_equal(left: AnswerT, right: AnswerT) -> bool:
    return left == right


def first_same_answer_window(
    observations: Sequence[Observation[AnswerT]],
    *,
    k: int = K,
    mss: int = MSS,
    answers_equal: AnswerComparator[AnswerT] | None = None,
) -> AnswerWindow[AnswerT] | None:
    """Return the first qualifying contiguous same-answer window.

    A candidate is evaluated when its final step is at least ``mss``. Thus,
    with the defaults, observations from steps 7--10 can form the first
    window. A missing step or an answer change resets the current run.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if mss < 1:
        raise ValueError("mss must be >= 1")

    equal = answers_equal or _default_answers_equal
    run: list[Observation[AnswerT]] = []
    previous_step = 0

    for item in observations:
        if item.step <= previous_step:
            raise ValueError("observations must have strictly increasing steps")

        contiguous = bool(run) and item.step == run[-1].step + 1
        same_answer = contiguous and equal(item.answer, run[0].answer)
        run = [*run, item] if same_answer else [item]
        previous_step = item.step

        if len(run) >= k and item.step >= mss:
            return AnswerWindow(tuple(run[-k:]))

    return None


def classify_window(
    window: AnswerWindow[AnswerT] | Sequence[Observation[AnswerT]],
    *,
    tau: float = TAU,
    eps: float = EPS,
) -> WindowLevel:
    """Classify a same-answer window using the frozen PUMA H/M/L rule.

    H: first confidence >= tau and every later confidence >= first - eps.
    L: every confidence < tau.
    M: all remaining windows.
    """

    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")

    items = window.observations if isinstance(window, AnswerWindow) else tuple(window)
    if not items:
        raise ValueError("a window cannot be empty")

    confidences = tuple(item.confidence for item in items)
    first = confidences[0]
    if first >= tau and all(confidence >= first - eps for confidence in confidences[1:]):
        return WindowLevel.H
    if all(confidence < tau for confidence in confidences):
        return WindowLevel.L
    return WindowLevel.M
