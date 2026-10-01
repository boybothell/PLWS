"""Build a frozen GPQA diamond file from the official CSV.

``baselines/answer_only.py`` in idavidrein/gpqa stores the correct option as
text, then assigns A–D only after ``random.sample`` shuffles the four choices.
The gold label is that letter. This module follows that shuffle and freezes
the result so a later Python does not draw a new order.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

OFFICIAL_REPOSITORY = "https://github.com/idavidrein/gpqa"
OFFICIAL_SHUFFLE = "baselines/answer_only.py random.sample"
OFFICIAL_DATASET_ID = "modelscope/gpqa"
OFFICIAL_FILE = "gpqa_diamond.csv"
DATASET_NAME = "gpqa-diamond-official"
SHUFFLE_SEED = 42
LETTERS = "ABCD"


def order_choices(
    correct: str,
    incorrect: tuple[str, str, str],
    rng: random.Random,
) -> tuple[list[str], str]:
    """Shuffle one row the way ``generate_question_and_answers`` does.

    Index 4 is the correct text. Indices 1–3 are the incorrect texts, in the
    official column order. The returned letter is the slot that received index 4.
    """

    choice_order = rng.sample([1, 2, 3, 4], 4)
    by_index = {
        1: incorrect[0],
        2: incorrect[1],
        3: incorrect[2],
        4: correct,
    }
    ordered = [by_index[index] for index in choice_order]
    return ordered, LETTERS[choice_order.index(4)]


def render_question(stem: str, ordered: list[str]) -> str:
    choices = "\n".join(
        f"({LETTERS[index]}) {text}" for index, text in enumerate(ordered)
    )
    return f"{stem.rstrip()}\n\nChoices:\n{choices}"


def load_official_rows(csv_path: Path, *, seed: int = SHUFFLE_SEED) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if len(raw_rows) != 198:
        raise ValueError(f"official GPQA diamond must have 198 rows, got {len(raw_rows)}")
    rng = random.Random(seed)
    formatted: list[dict[str, str]] = []
    for row in raw_rows:
        record_id = str(row.get("Record ID", "")).strip()
        stem = str(row.get("Question", ""))
        correct = str(row.get("Correct Answer", ""))
        incorrect = tuple(str(row.get(f"Incorrect Answer {index}", "")) for index in (1, 2, 3))
        if not record_id or not stem.strip() or not correct.strip():
            raise ValueError(f"official GPQA row {record_id or '?'} is missing text")
        if any(not text.strip() for text in incorrect):
            raise ValueError(f"official GPQA row {record_id} is missing an incorrect choice")
        ordered, letter = order_choices(correct, incorrect, rng)
        if ordered[LETTERS.index(letter)] != correct:
            raise ValueError(f"official GPQA row {record_id} gold letter missed the correct text")
        formatted.append(
            {
                "question": render_question(stem, ordered),
                "answer": letter,
                "record_id": record_id,
            }
        )
    if len({row["record_id"] for row in formatted}) != len(formatted):
        raise ValueError("official GPQA Record ID values are not unique")
    return formatted


def letter_histogram(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["answer"] for row in rows))
