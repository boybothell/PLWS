#!/usr/bin/env python3
"""Old verbalized-distribution ask, dict format, probabilities must sum to 1.

Same 7B Olympiad k=4 window jobs. Closed think. Do not pass seed= to LLM().
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_LENS_DISABLE", "1")

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_judge import QWEN3_ASSISTANT, QWEN3_USER_OPEN, QWEN4B_MODEL  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import replay_default_dense_gate as dd  # noqa: E402
import report_first_lock_room as room  # noqa: E402

JOBS = AE / "results/hc_verbal/olympiadbench_s42/jobs.jsonl"
OUT = AE / "results/hc_verbal_sum1/olympiadbench_s42"
LETTERS = "ABCD"
DICT_RE = re.compile(r"\{[^{}]+\}")
PAIR_RE = re.compile(r"""['\"]?([A-D])['\"]?\s*[:=]\s*(-?[0-9]*\.?[0-9]+)""")


def build_user(question: str, reasoning: str, cands: list[str]) -> str:
    used = [LETTERS[i] for i in range(len(cands))] + ["D"]
    lines = [f"{letter}. {ans}" for letter, ans in zip(used[:-1], cands, strict=True)]
    lines.append("D. None of the above")
    listed = "\n".join(lines)
    keys = ", ".join(f'"{letter}": p' for letter in used)
    return (
        "You are a careful and impartial evaluator.\n\n"
        "Assign a probability from 0.0 to 1.0 to every listed candidate.\n"
        "The probabilities MUST sum to 1.0. This is one distribution, not "
        "separate scores.\n"
        "Put leftover probability on D if none of the other options is reliable.\n\n"
        "Give ONLY a Python dict from letter to probability, no other words.\n"
        "Do not copy candidate text into the values.\n"
        f"Format: {{{keys}}} with A+B+C+D=1.0.\n\n"
        f"[Problem]\n{question}\n\n"
        f"[Partial reasoning]\n{reasoning}\n\n"
        f"[Candidates]\n{listed}"
    )


def _raw_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x < 0:
        return None
    return x


def parse_scores(text: str, n_cands: int) -> dict[str, float] | None:
    used = [LETTERS[i] for i in range(n_cands)] + ["D"]
    blob = (text or "").strip()
    obj: Any = None
    match = DICT_RE.search(blob)
    raw = match.group(0) if match else blob
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(raw)
            break
        except Exception:
            continue
    found: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            letter = str(key).strip().upper()
            num = _raw_float(value)
            if letter in used and num is not None:
                found[letter] = num
    if any(letter not in found for letter in used):
        for letter, raw_v in PAIR_RE.findall(blob):
            num = _raw_float(raw_v)
            if num is not None:
                found[letter.upper()] = num
    if any(letter not in found for letter in used):
        return None
    total = sum(found[letter] for letter in used)
    if 1.5 < total <= 101.0 and all(found[letter] <= 100.0 for letter in used):
        found = {letter: found[letter] / 100.0 for letter in used}
        total = sum(found[letter] for letter in used)
    if abs(total - 1.0) > 0.02:
        return None
    if any(found[letter] > 1.0 + 1e-6 for letter in used):
        return None
    return {letter: found[letter] for letter in used}


def done_keys(path: Path) -> set[tuple[int, int, int]]:
    if not path.is_file():
        return set()
    out: set[tuple[int, int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") in {"ok", "too_long", "parse_fail", "no_prefix"}:
            out.add((int(row["question_idx"]), int(row["win_step"]), int(row["rel_step"])))
    return out


def load_prefixes(qis: set[int]) -> dict[tuple[int, int], dict[str, Any]]:
    path = room.dense_trial_path("r1_7b", "olympiadbench", 42)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in dd.load_json(path):
        qi = int(row["question_idx"])
        if qi not in qis:
            continue
        out[(qi, int(row["stopped_len"]))] = row
    return out


def slim(job: dict[str, Any]) -> dict[str, Any]:
    return {k: job[k] for k in job if k != "question"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, default=JOBS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    jobs = [job for job in jobs if int(job["question_idx"]) % args.num_shards == args.shard_id]
    jobs.sort(key=lambda x: (int(x["question_idx"]), int(x["win_step"]), int(x["rel_step"])))
    args.out = args.out or (OUT / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [
        job
        for job in jobs
        if (int(job["question_idx"]), int(job["win_step"]), int(job["rel_step"]))
        not in done_keys(args.out)
    ]
    print(
        f"hc-verbal-sum1 oly shard={args.shard_id}/{args.num_shards} "
        f"jobs={len(jobs)} pending={len(pending)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return

    prefixes = load_prefixes({int(job["question_idx"]) for job in pending})
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN4B_MODEL), trust_remote_code=True)
    llm = LLM(
        model=str(QWEN4B_MODEL),
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            prompts: list[str] = []
            ready: list[dict[str, Any]] = []
            for job in chunk:
                trial = prefixes.get((int(job["question_idx"]), int(job["decision_step"])))
                if trial is None or not str(trial.get("reasoning_prefix") or ""):
                    handle.write(
                        json.dumps({**slim(job), "status": "no_prefix"}, ensure_ascii=False) + "\n"
                    )
                    continue
                user = build_user(job["question"], str(trial["reasoning_prefix"]), job["cands"])
                chat = QWEN3_USER_OPEN + user + QWEN3_ASSISTANT
                n_tok = len(tokenizer.encode(chat, add_special_tokens=False))
                if n_tok + args.max_tokens > args.max_context:
                    handle.write(
                        json.dumps(
                            {**slim(job), "status": "too_long", "n_tok": n_tok},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                prompts.append(chat)
                ready.append(job)
            if not prompts:
                handle.flush()
                continue
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for job, out in zip(ready, outputs, strict=True):
                text = out.outputs[0].text if out.outputs else ""
                scores = parse_scores(text, len(job["cands"]))
                rec = slim(job)
                rec["text"] = (text or "").strip()[:160]
                if scores is None:
                    rec["status"] = "parse_fail"
                else:
                    rec["status"] = "ok"
                    rec["scores"] = scores
                    rec["p_self"] = scores["A"]
                    rec["p_none"] = scores["D"]
                    rec["pick"] = max(scores, key=scores.get)
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            handle.flush()
            print(
                f"shard{args.shard_id} wrote={wrote}/{len(pending)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    print(f"done shard={args.shard_id} wrote={wrote}", flush=True)


if __name__ == "__main__":
    main()
