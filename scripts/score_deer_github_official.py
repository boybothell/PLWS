#!/usr/bin/env python3
"""Score official GitHub DEER dumps with the same grader as PUMA."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(PUMA))
sys.path.insert(0, str(PUMA / "puma"))

from baselines.utils.math_util import my_answer_extraction  # noqa: E402
from math_grader import check_is_correct  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/plws")
BASES = (
    ROOT / "results/baselines/deer/github_official_greedy_16k_priority_7b_8b_14b",
    ROOT / "results/baselines/deer/github_official_greedy_16k_qwen3",
)
OUT = ROOT / "results/reports/deer_github_official.json"
BOXED = re.compile(r"^\\boxed\{(.+)\}$")

MODELS = (
    ("r1_7b", "7B", "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"),
    ("nemotron_8b", "8B", "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"),
    ("r1_14b", "14B", "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"),
    ("qwen3_4b", "Qwen3-4B", "/mnt/d/lsj/models/Qwen3-4B"),
    ("qwen3_8b", "Qwen3-8B", "/mnt/d/lsj/models/Qwen3-8B"),
)
DATASETS = (
    ("math", "MATH", "math-500", 500),
    ("olympiadbench", "OlympiadBench", "olympiadbench", 675),
    ("gpqa", "GPQA-Diamond", "gpqa-diamond", 198),
    ("aime", "AIME24", "aime24", 30),
    ("aime25", "AIME25", "aime25", 30),
)


def clean_gt(gt: str) -> str:
    gt = str(gt or "").strip()
    m = BOXED.fullmatch(gt)
    return m.group(1).strip() if m else gt


def extract_pred(text: str, dataset: str) -> str:
    pred = my_answer_extraction(text, dataset=dataset)
    if "gpqa" in dataset and pred not in ("A", "B", "C", "D"):
        m = re.search(r"ANSWER\s*:\s*([A-D])", text or "")
        if m:
            pred = m.group(1)
    return str(pred or "").strip()


def grade_one(args: tuple[int, str, str]) -> tuple[int, bool]:
    idx, pred, gt = args
    try:
        return idx, bool(check_is_correct(pred, gt))
    except Exception:
        return idx, False


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_jsonl(model: str, dataset: str) -> Path | None:
    hits: list[Path] = []
    for base in BASES:
        root = base / model / dataset
        if not root.is_dir():
            continue
        hits.extend(root.rglob("*.jsonl"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def score_file(path: Path, dataset: str, tokenizer) -> dict:
    rows = load_jsonl(path)
    texts = []
    for row in rows:
        responses = row.get("generated_responses") or []
        text = responses[0] if responses else row.get("generated_text") or ""
        texts.append(str(text or ""))
    gts = [clean_gt(r.get("gold_answer") or r.get("answer") or "") for r in rows]
    preds = [extract_pred(t, dataset) for t in texts]
    packed = [(i, p, g) for i, (p, g) in enumerate(zip(preds, gts))]
    if packed:
        with Pool(8) as pool:
            graded = dict(pool.map(grade_one, packed, chunksize=8))
    else:
        graded = {}
    ok = sum(1 for i in range(len(rows)) if graded.get(i))
    toks = [
        len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
        for text in texts
    ]
    n = len(rows)
    high = sum(int(r.get("high_prob") or 0) for r in rows)
    regular = sum(int(r.get("regular_end") or 0) for r in rows)
    too_long = sum(int(r.get("too_long") or 0) for r in rows)
    steps = [int(r.get("thinking_steps") or 0) for r in rows]
    return {
        "n": n,
        "ok": ok,
        "acc": (100.0 * ok / n) if n else None,
        "tok": (sum(toks) / n) if n else None,
        "high_prob": high,
        "high_prob_rate": (100.0 * high / n) if n else None,
        "regular_end": regular,
        "too_long": too_long,
        "mean_thinking_steps": (sum(steps) / n) if n else None,
        "path": str(path.relative_to(ROOT)),
    }


def main() -> None:
    tok_cache: dict[str, object] = {}
    cells: list[dict] = []
    for tag, zh, model_path in MODELS:
        if model_path not in tok_cache:
            print(f"load tokenizer {zh}", flush=True)
            tok_cache[model_path] = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        tokenizer = tok_cache[model_path]
        for deer_ds, dszh, grade_ds, expect in DATASETS:
            path = find_jsonl(tag, deer_ds)
            if path is None:
                cells.append(
                    {
                        "model": zh,
                        "dataset": dszh,
                        "expect": expect,
                        "complete": False,
                        "missing": True,
                    }
                )
                continue
            print(f"score {zh} {dszh} {path}", flush=True)
            one = score_file(path, grade_ds, tokenizer)
            one.update(
                {
                    "model": zh,
                    "dataset": dszh,
                    "expect": expect,
                    "complete": one["n"] == expect,
                    "missing": False,
                }
            )
            cells.append(one)
            print(
                f"  n={one['n']}/{expect} acc={one['acc']:.2f} tok={one['tok']:.0f} "
                f"early={one['high_prob_rate']:.1f}%",
                flush=True,
            )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "scripts/score_deer_github_official.py",
        "note": (
            "Official DEER GitHub: greedy, 16k, official prompt, one run. "
            "Not aligned with Full-CoT/PUMA/PLWS sampling cells."
        ),
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
