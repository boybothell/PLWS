#!/usr/bin/env python3
"""DTSR (ACL 2026) thought-sufficiency score on first leftover windows.

Third-person prompt from the paper. Judge is Qwen3-4B with thinking closed.
Only leftover prefixes; labels come from leftover_after.
"""
from __future__ import annotations

import argparse
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

from score_confcal_v1 import nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_first_lock_room as room  # noqa: E402
import report_k4_hyps as hy  # noqa: E402
import report_k4_second_lock as sl  # noqa: E402
import report_leftover_after as after  # noqa: E402

JUDGE = Path("/mnt/d/lsj/models/Qwen3-4B")
OUT = AE / "results/leftover_dtsr"
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
PROMPT = (
    "Assess the confidence that the provided thought is sufficient to solve "
    "the question. Output only in the format: Confidence: <number>, where "
    "<number> is a real value between 0 and 100. Here, 100 means the thought "
    "is fully sufficient to reach the final answer directly without additional "
    "reasoning, and 0 means the thought is completely insufficient to do so.\n\n"
    "Question:\n{question}\n\nThought:\n{thought}"
)
CONF_RE = re.compile(r"(?i)confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)")
NUM_RE = re.compile(r"\b([0-9]{1,3}(?:\.[0-9]+)?)\b")


def parse_score(text: str) -> float:
    raw = (text or "").strip()
    match = CONF_RE.search(raw)
    if match is None:
        match = NUM_RE.search(raw)
    if match is None:
        return float("nan")
    value = float(match.group(1))
    if value > 100.0:
        return float("nan")
    return value


def load_jobs(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    xs = after.load_left(model, dataset, seed)
    if not xs:
        return []
    trials_path = room.dense_trial_path(model, dataset, seed)
    by: dict[int, list[dict[str, Any]]] = {}
    for row in dd.load_json(trials_path):
        by.setdefault(int(row["question_idx"]), []).append(row)
    jobs: list[dict[str, Any]] = []
    for rec in xs:
        trials = by.get(int(rec["question_idx"]))
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        wins = sl.same_windows(rows)
        left = next((win for win in wins if hy.leftover(win)), None)
        if left is None:
            continue
        left_row = rows[left["end"]]
        thought = str(left_row.get("reasoning_prefix") or "")
        question = str(left_row.get("question") or "")
        if not thought or not question:
            continue
        jobs.append(
            {
                "uid": f"{model}:{dataset}:{seed}:{rec['question_idx']}",
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "question_idx": int(rec["question_idx"]),
                "left_step": int(rec["left_step"]),
                "kind": rec["kind"],
                "confidence": rec.get("left_c"),
                "left_ok": rec["left_ok"],
                "wait_helps": rec["wait_helps"],
                "will_change": rec["will_change"],
                "never_high": rec["never_high"],
                "same_as_high": rec["same_as_high"],
                "question": question,
                "thought": thought,
            }
        )
    return jobs


def done_uids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") in {"ok", "too_long", "empty"}:
            out.add(str(row["uid"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-thought-tokens", type=int, default=12000)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.jobs and args.jobs.is_file():
        jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
        jobs = [job for job in jobs if job.get("model") == args.model_tag]
    else:
        jobs = []
        for dataset in DATASETS:
            jobs.extend(load_jobs(args.model_tag, dataset, 42))
    jobs.sort(key=lambda x: (x["dataset"], x["question_idx"]))
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_id]
    args.out = args.out or (OUT / f"{args.model_tag}_s42" / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if job["uid"] not in done_uids(args.out)]
    print(
        f"dtsr leftover {args.model_tag} shard={args.shard_id}/{args.num_shards} "
        f"jobs={len(jobs)} pending={len(pending)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(str(JUDGE), trust_remote_code=True)
    llm = LLM(
        model=str(JUDGE),
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=16)
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            prompts: list[str] = []
            metas: list[dict[str, Any]] = []
            for job in chunk:
                thought = job["thought"]
                thought_ids = tokenizer.encode(thought, add_special_tokens=False)
                truncated = False
                if len(thought_ids) > args.max_thought_tokens:
                    thought = tokenizer.decode(thought_ids[-args.max_thought_tokens :])
                    truncated = True
                user = PROMPT.format(question=job["question"], thought=thought)
                try:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if text.rstrip().endswith("<think>"):
                        text = text.rstrip() + "\n</think>\n\n"
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) >= args.max_context - 32:
                    metas.append({**job, "status": "too_long", "n_prompt": len(ids), "truncated": truncated})
                    continue
                prompts.append(text)
                metas.append({**job, "status": "ok", "n_prompt": len(ids), "truncated": truncated})
            if prompts:
                outs = llm.generate(prompts, params, use_tqdm=False)
            else:
                outs = []
            oi = 0
            for meta in metas:
                rec = {
                    "uid": meta["uid"],
                    "model": meta["model"],
                    "dataset": meta["dataset"],
                    "seed": meta["seed"],
                    "question_idx": meta["question_idx"],
                    "left_step": meta["left_step"],
                    "kind": meta["kind"],
                    "confidence": meta["confidence"],
                    "left_ok": meta["left_ok"],
                    "wait_helps": meta["wait_helps"],
                    "will_change": meta["will_change"],
                    "never_high": meta["never_high"],
                    "same_as_high": meta["same_as_high"],
                    "n_prompt": meta["n_prompt"],
                    "truncated": meta["truncated"],
                    "status": meta["status"],
                    "suff": float("nan"),
                    "raw": "",
                }
                if meta["status"] == "ok":
                    raw = outs[oi].outputs[0].text if outs[oi].outputs else ""
                    oi += 1
                    rec["raw"] = raw[:200]
                    rec["suff"] = parse_score(raw)
                    if rec["suff"] != rec["suff"]:
                        rec["status"] = "empty"
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                wrote += 1
            print(
                f"shard{args.shard_id} wrote={wrote}/{len(pending)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    print(f"done shard={args.shard_id} wrote={wrote}", flush=True)


if __name__ == "__main__":
    main()
