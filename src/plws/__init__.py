"""Public PLWS API with attn_early_exit kept as a separate compatibility package."""

from .window import (
    EPS,
    K,
    MSS,
    TAU,
    AnswerComparator,
    AnswerWindow,
    Observation,
    WindowLevel,
    classify_window,
    first_same_answer_window,
)

__all__ = [
    "EPS",
    "K",
    "MSS",
    "TAU",
    "AnswerComparator",
    "AnswerWindow",
    "Observation",
    "WindowLevel",
    "classify_window",
    "first_same_answer_window",
]

__version__ = "0.1.0"
