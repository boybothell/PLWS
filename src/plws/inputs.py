"""Small production input helpers independent of analysis/report scripts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_LATEX = re.compile(r"\\(left|right|cdot|times|dfrac|tfrac|frac|mathrm|text)")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def normalize_answer(value: Any) -> str:
    text = _WHITESPACE.sub("", str(value or "").strip().lower())
    text = text.replace("dfrac", "frac").replace("tfrac", "frac")
    text = _LATEX.sub("", text)
    return text.replace("\\", "")


def same_answer(left: Any, right: Any) -> bool:
    normalized_left = normalize_answer(left)
    normalized_right = normalize_answer(right)
    return bool(normalized_left) and normalized_left == normalized_right


def usable_trial_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(trials, key=lambda item: int(item["stopped_len"])):
        answer = str(row.get("final_answer") or "")
        confidence = finite(row.get("confidence"))
        if answer and math.isfinite(confidence):
            rows.append(row)
    return rows


def answer_credit(
    answer: Any,
    ground_truth: Any,
    original_answer: Any,
    original_correct: bool,
) -> int:
    if same_answer(answer, ground_truth):
        return 1
    if same_answer(answer, original_answer):
        return int(original_correct)
    return 0
