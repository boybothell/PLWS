"""Lightweight answer extraction + gold check (PUMA grader if available)."""
from __future__ import annotations

import re
import sys
from pathlib import Path


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL = re.compile(
    r"(?:final answer|the answer is|answer is)\s*[:：]?\s*(.+)",
    re.IGNORECASE,
)


def extract_answer_candidate(text: str) -> str | None:
    if not text:
        return None
    boxes = _BOXED.findall(text)
    if boxes:
        return boxes[-1].strip()
    m = _FINAL.search(text)
    if m:
        cand = m.group(1).strip().split("\n")[0].strip()
        cand = cand.strip("$").strip("`").strip()
        return cand or None
    return None


def _load_puma_check():
    puma = Path(__file__).resolve().parents[2] / "PUMA"
    if str(puma) not in sys.path:
        sys.path.insert(0, str(puma))
    try:
        from puma.math_grader import check_is_correct  # type: ignore

        return check_is_correct
    except Exception:
        return None


_CHECK = None


def answers_equal(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    global _CHECK
    if _CHECK is None:
        _CHECK = _load_puma_check()
    if _CHECK is not None:
        try:
            return bool(_CHECK(str(pred), str(gold), timeout=True))
        except Exception:
            pass
    # fallback: normalized string
    a = re.sub(r"\s+", "", str(pred).strip().lower())
    b = re.sub(r"\s+", "", str(gold).strip().lower())
    return bool(a) and a == b
