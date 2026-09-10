#!/usr/bin/env python3
"""4B teacher-force logp of each candidate answer. No letters, no verbal scores.

Same 7B Olympiad k=4 window jobs as the verbal runs. Closed think.
Do not pass seed= to LLM().
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
OUT = AE / "results/hc_ans_logp/olympiadbench_s42"
CUE = "Final answer: "


def build_user(question: str, reasoning: str) -> str:
    return (
        "You are a careful and impartial evaluator.\n\n"
        "Read the problem and the partial reasoning. Then write the final answer only.\n\n"
        f"[Problem]\n{question}\n\n"
        f"[Partial reasoning]\n{reasoning}"
    )


def pack_ids(tokenizer: Any, question: str, reasoning: str, answer: str) -> tuple[list[int], int] | None:
    body = QWEN3_USER_OPEN + build_user(question, reasoning) + QWEN3_ASSISTANT + CUE
    full = body + answer
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    prefix_len = 0
    for left, right in zip(body_ids, full_ids):
        if left != right:
            break
        prefix_len += 1
    if prefix_len == 0:
        prefix_len = len(body_ids)
    if prefix_len >= len(full_ids):
        return None
    return full_ids, prefix_len


def mean_target_logprob(output: Any, prefix_len: int) -> tuple[float, int]:
    ids = list(output.prompt_token_ids or [])
    logs = list(output.prompt_logprobs or [])
    values: list[float] = []
    for index in range(max(prefix_len, 1), min(len(ids), len(logs))):
        item = (logs[index] or {}).get(ids[index])
        if item is None:
            return float("nan"), 0
        values.append(float(item.logprob if hasattr(item, "logprob") else item))
    if not values:
        return float("nan"), 0
    return sum(values) / len(values), len(values)


def softmax(logps: list[float]) -> list[float] | None:
    if not logps or any(x != x or math.isinf(x) for x in logps):
        return None
    peak = max(logps)
    exps = [math.exp(x - peak) for x in logps]
    total = sum(exps)
    if total <= 0:
        return None
    return [x / total for x in exps]


def done_keys(path: Path) -> set[tuple[int, int, int]]:
    if not path.is_file():
        return set()
    out: set[tuple[int, int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") in {"ok", "too_long", "no_prefix", "empty_target", "logp_fail"}:
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
    parser.add_argument("--limit", type=int, default=0)
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
    if args.limit:
        pending = pending[: args.limit]
    print(
        f"hc-ans-logp oly shard={args.shard_id}/{args.num_shards} "
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
        max_logprobs=1,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            prompts: list[dict[str, list[int]]] = []
            meta: list[tuple[dict[str, Any], str, int]] = []
            for job in chunk:
                rec = slim(job)
                trial = prefixes.get((int(job["question_idx"]), int(job["decision_step"])))
                if trial is None or not str(trial.get("reasoning_prefix") or ""):
                    handle.write(json.dumps({**rec, "status": "no_prefix"}, ensure_ascii=False) + "\n")
                    continue
                cands = [str(x) for x in (job.get("cands") or []) if str(x)]
                if not cands:
                    handle.write(json.dumps({**rec, "status": "empty_target"}, ensure_ascii=False) + "\n")
                    continue
                packed: list[tuple[str, list[int], int]] = []
                too_long = False
                empty = False
                for ans in cands:
                    got = pack_ids(tokenizer, job["question"], str(trial["reasoning_prefix"]), ans)
                    if got is None:
                        empty = True
                        break
                    ids, prefix_len = got
                    if len(ids) + 1 > args.max_context:
                        too_long = True
                        break
                    packed.append((ans, ids, prefix_len))
                if empty:
                    handle.write(json.dumps({**rec, "status": "empty_target"}, ensure_ascii=False) + "\n")
                    continue
                if too_long:
                    handle.write(json.dumps({**rec, "status": "too_long"}, ensure_ascii=False) + "\n")
                    continue
                for ans, ids, prefix_len in packed:
                    prompts.append({"prompt_token_ids": ids})
                    meta.append((job, ans, prefix_len))
            outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            by_job: dict[tuple[int, int, int], list[tuple[str, float, int]]] = {}
            for (job, ans, prefix_len), output in zip(meta, outputs, strict=True):
                key = (int(job["question_idx"]), int(job["win_step"]), int(job["rel_step"]))
                mean_lp, n_tok = mean_target_logprob(output, prefix_len)
                by_job.setdefault(key, []).append((ans, mean_lp, n_tok))
            seen: set[tuple[int, int, int]] = set()
            for job, _ans, _prefix_len in meta:
                key = (int(job["question_idx"]), int(job["win_step"]), int(job["rel_step"]))
                if key in seen:
                    continue
                seen.add(key)
                rec = slim(job)
                scored = by_job.get(key) or []
                logps = [x[1] for x in scored]
                probs = softmax(logps)
                if not scored or probs is None:
                    rec["status"] = "logp_fail"
                    rec["mean_logp"] = logps
                else:
                    rec["status"] = "ok"
                    rec["cands"] = [x[0] for x in scored]
                    rec["mean_logp"] = logps
                    rec["n_tok"] = [x[2] for x in scored]
                    rec["probs"] = probs
                    rec["p_self"] = probs[0]
                    rec["logp_self"] = logps[0]
                    rec["pick"] = int(max(range(len(probs)), key=lambda i: (probs[i], -i)))
                    rec["margin"] = logps[0] - max(logps[1:], default=float("-inf"))
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
