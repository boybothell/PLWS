#!/usr/bin/env python3
"""官方 PUMA Full-CoT 终答抽取，不引进 vLLM。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(PUMA))

from baselines.utils.math_util import (  # noqa: E402
    extract_multi_choice_answer,
    my_answer_extraction,
)

_GPQA = {"A", "B", "C", "D"}


def extract_final_answer(text: str, dataset: str = "") -> str:
    """对齐 PUMA `run_vllm.extract_answer`：先 last \\boxed，GPQA 再找 ANSWER: X。"""
    task = "gpqa" if "gpqa" in (dataset or "").lower() else "math"
    answer = my_answer_extraction(text or "")
    if task == "gpqa" and answer not in _GPQA:
        match = re.search(r"ANSWER\s*:\s*([A-D])", text or "")
        if match:
            answer = match.group(1)
    return answer or ""


def extract_from_saved(row: dict) -> tuple[str, str]:
    """从已写记录抽终答。有全文用全文；只有 200 字片段时，GPQA 抽不出字母再走选项回退。"""
    dataset = str(row.get("dataset") or "")
    if row.get("answer_text") is not None:
        blob = f"{row.get('think_tail') or ''}\n</think>\n\n{row.get('answer_text') or ''}"
        return extract_final_answer(blob, dataset), "full"
    blob = str(row.get("new_text") or "")
    answer = extract_final_answer(blob, dataset)
    if "gpqa" in dataset and answer not in _GPQA:
        fallback = extract_multi_choice_answer(blob)
        if fallback in _GPQA:
            return fallback, "snippet_choice"
        return answer, "snippet_miss"
    if answer:
        return answer, "snippet"
    return answer, "snippet_miss"
