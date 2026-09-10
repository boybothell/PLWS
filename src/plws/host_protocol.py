"""Validation helpers for canonical Full-CoT host trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)


def validate_fullcot_sample_meta(
    meta_path: Path,
    *,
    model_tag: str,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    if not meta_path.is_file():
        raise ValueError(f"missing canonical Full-CoT metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": PROTOCOL_ID,
        "model_tag": model_tag,
        "dataset": dataset,
        "seed": seed,
        "max_tokens": FULLCOT_GENERATION_TOKENS,
        "answer_fix_max_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
        "prompt_reserve_tokens": PROMPT_RESERVE_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "prompt_version": "default",
    }
    mismatches = [
        f"{key}={meta.get(key)!r} (required {value!r})"
        for key, value in required.items()
        if meta.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Full-CoT host protocol mismatch: " + ", ".join(mismatches)
        )
    return meta
