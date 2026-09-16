"""Detect and undo SentencePiece token-piece dumps (Ġ / Ċ / ĉ).

Llama-8B Full-CoT was saved as tokenizer pieces instead of decoded text.
PUMA splits steps on real newlines, so those cells collapsed to 1 step.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

NEWLINE_PIECE = "Ċ"
SPACE_PIECE = "Ġ"
TAB_PIECE = "ĉ"
TEXT_FIELDS = ("generated_text", "reasoning", "raw_response", "model_response")


def has_tokenizer_pieces(text: str | None) -> bool:
    if not text:
        return False
    return NEWLINE_PIECE in text or text.count(SPACE_PIECE) >= 3


def sanitize_tokenizer_pieces(text: str) -> str:
    if not has_tokenizer_pieces(text):
        return text
    return (
        text.replace(NEWLINE_PIECE, "\n")
        .replace(TAB_PIECE, "\t")
        .replace(SPACE_PIECE, " ")
    )


def row_has_tokenizer_pieces(row: Mapping[str, Any]) -> bool:
    return any(has_tokenizer_pieces(str(row.get(field) or "")) for field in TEXT_FIELDS)


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for field in TEXT_FIELDS:
        value = out.get(field)
        if isinstance(value, str) and has_tokenizer_pieces(value):
            out[field] = sanitize_tokenizer_pieces(value)
    steps = out.get("reasoning_steps")
    if isinstance(steps, list) and steps:
        fixed = [
            sanitize_tokenizer_pieces(step) if isinstance(step, str) else step
            for step in steps
        ]
        if any(has_tokenizer_pieces(str(step)) for step in steps if isinstance(step, str)):
            out["reasoning_steps"] = []
        else:
            out["reasoning_steps"] = fixed
    return out


def sanitize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [sanitize_row(dict(row)) for row in rows]
