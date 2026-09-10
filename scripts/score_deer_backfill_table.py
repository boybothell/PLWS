#!/usr/bin/env python3
"""Score official DEER backfill cells with the same grader/tokenizer as PUMA."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(PUMA))
sys.path.insert(0, str(PUMA / "puma"))

from baselines.utils.math_util import my_answer_extraction  # noqa: E402
from math_grader import check_is_correct  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/plws")
BASE = ROOT / "results/baselines/deer/backfill"
OUT = ROOT / "results/baselines/deer/backfill/_table_cells.json"

MODELS = (
    ("r1_7b", "7B", "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"),
    ("nemotron_8b", "8B", "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"),
    ("r1_14b", "14B", "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"),
)
DATASETS = (
    ("math-500", "MATH", 500),
    ("olympiadbench", "OlympiadBench", 675),
    ("gpqa-diamond", "GPQA-Diamond", 198),
    ("aime24", "AIME24", 30),
    ("aime25", "AIME25", 30),
)
SEEDS = (0, 1, 42, 123)
BOXED = re.compile(r"^\\boxed\{(.+)\}$")


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
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_file(path: Path, dataset: str, tokenizer) -> dict:
    rows = load_jsonl(path)
    texts = [str(r.get("generated_text") or "") for r in rows]
    gts = [clean_gt(r.get("gold_answer") or r.get("answer") or "") for r in rows]
    preds = [extract_pred(t, dataset) for t in texts]
    tasks = list(enumerate(zip(preds, gts)))
    packed = [(i, p, g) for i, (p, g) in tasks]
    if packed:
        with Pool(8) as pool:
            graded = dict(pool.map(grade_one, packed, chunksize=8))
    else:
        graded = {}
    ok = sum(1 for i in range(len(rows)) if graded.get(i))
    toks = []
    trial = []
    for i, text in enumerate(texts):
        toks.append(len(tokenizer.encode(text, add_special_tokens=False)) if text else 0)
        trial.append(int(rows[i].get("num_trial_answer_tokens") or 0))
    n = len(rows)
    return {
        "n": n,
        "ok": ok,
        "acc": (100.0 * ok / n) if n else None,
        "tok": (sum(toks) / n) if n else None,
        "trial": (sum(trial) / n) if n else None,
    }


def main() -> None:
    tok_cache: dict[str, object] = {}
    cells: dict[str, dict] = {}
    for tag, zh, model_path in MODELS:
        if model_path not in tok_cache:
            print(f"load tokenizer {zh}", flush=True)
            tok_cache[model_path] = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        tokenizer = tok_cache[model_path]
        for dataset, dszh, expect in DATASETS:
            seed_rows = []
            for seed in SEEDS:
                path = BASE / tag / dataset / f"seed_{seed}" / "deer.jsonl"
                if not path.is_file():
                    continue
                print(f"score {zh} {dszh} seed={seed}", flush=True)
                one = score_file(path, dataset, tokenizer)
                one["seed"] = seed
                one["complete"] = one["n"] == expect
                seed_rows.append(one)
            cells[f"{tag}|{dataset}"] = {
                "model": zh,
                "dataset": dszh,
                "expect": expect,
                "seeds": seed_rows,
            }
    OUT.write_text(json.dumps(cells, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
