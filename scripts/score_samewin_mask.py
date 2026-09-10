#!/usr/bin/env python3
"""Mask interference on every 4-step same-answer window (对窗 vs 错窗).

Teacher-force the window's boxed answer under the 7B solver prefix after
hiding one region.  Zero leftover filter: high / mix / low all enter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["VLLM_LENS_DISABLE"] = "1"

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_second_lock as sl
from pilot_counterfactual_support import append_answer, finite, mean_target_logprob
from score_confcal_keytoken import SOLVER
from score_confcal_v1 import nvidia_lib_path

OUT = AE / "results/samewin_mask"
MASKS = ("none", "hide_problem", "hide_cot", "hide_last", "hide_early", "hide_rand")
PLACE = "[MASK]"


def last_increment(current: str, prior: str) -> str:
    if prior and current.startswith(prior):
        return current[len(prior) :]
    return current


def variants(question: str, current: str, prior: str, rng: random.Random) -> dict[str, tuple[str, str]]:
    increment = last_increment(current, prior).lstrip()
    out: dict[str, tuple[str, str]] = {
        "none": (question, current),
        "hide_problem": (PLACE, current),
        "hide_cot": (question, PLACE),
        "hide_last": (question, prior if prior else PLACE),
        "hide_early": (question, increment if increment else PLACE),
    }
    n = len(increment)
    if prior and n > 0 and n < len(prior):
        start = rng.randrange(0, len(prior) - n + 1)
        masked = prior[:start] + (" " * n) + prior[start + n :] + increment
        out["hide_rand"] = (question, masked)
    else:
        out["hide_rand"] = (question, prior if prior else PLACE)
    return out


def load_jobs(dataset: str, seed: int) -> list[dict[str, Any]]:
    rg.K = 4
    rg.TAU = 0.995
    trials_path = room.dense_trial_path("r1_7b", dataset, seed)
    puma_path = dd.puma_stat_path("r1_7b", dataset, seed)
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    gp = low.gpath("r1_7b", dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    jobs: list[dict[str, Any]] = []
    for qi, trials in by.items():
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        info = official.get(qi) or {}
        g = gmap.get(qi) or {}
        last = max(trials, key=lambda x: int(x["stopped_len"]))
        gt = info.get("ground_truth") or g.get("ground_truth")
        original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
        orig_ok = (
            bool(info.get("original_correct"))
            if "original_correct" in info
            else bool(low.credit(original, gt, original, True))
        )
        index_by_step = {int(x["stopped_len"]): i for i, x in enumerate(rows)}
        for win in sl.same_windows(rows):
            step = int(win["step"])
            row = rows[win["end"]]
            prior = str(rows[win["end"] - 1].get("reasoning_prefix") or "") if win["end"] else ""
            jobs.append(
                {
                    "question_idx": qi,
                    "decision_step": step,
                    "index": index_by_step[step],
                    "kind": win["kind"],
                    "gold_ok": bool(low.credit(win["ans"], gt, original, orig_ok)),
                    "answer": str(win["ans"] or ""),
                    "geo_stored": finite(row.get("confidence")),
                    "question": str(row.get("question") or ""),
                    "current": str(row.get("reasoning_prefix") or ""),
                    "prior": prior,
                }
            )
    jobs.sort(key=lambda x: (x["question_idx"], x["decision_step"]))
    return jobs


def done_keys(path: Path) -> set[tuple[int, int]]:
    if not path.is_file():
        return set()
    out: set[tuple[int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            out.add((int(row["question_idx"]), int(row["decision_step"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--gpu-mem-util", type=float, default=0.88)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

    jobs = [
        job
        for job in load_jobs(args.dataset, args.seed)
        if job["question_idx"] % args.num_shards == args.shard_id
    ]
    args.out = args.out or (OUT / f"{args.dataset}_s{args.seed}" / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out)
    jobs = [job for job in jobs if (job["question_idx"], job["decision_step"]) not in done]
    counts = defaultdict(int)
    for job in jobs:
        counts[(job["kind"], job["gold_ok"])] += 1
    print(
        f"samewin-mask {args.dataset} shard={args.shard_id}/{args.num_shards} "
        f"pending={len(jobs)} mix={dict(counts)} -> {args.out}",
        flush=True,
    )
    if not jobs:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    llm = LLM(
        model=str(SOLVER),
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        max_logprobs=1,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    started = time.perf_counter()
    ok = 0
    with args.out.open("a") as handle:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start : start + args.batch_size]
            prompts: list[dict[str, list[int]]] = []
            meta: list[tuple[dict[str, Any], str, int]] = []
            for job in batch:
                rng = random.Random(int(job["question_idx"]) * 100003 + int(job["decision_step"]))
                for name, (question, reasoning) in variants(
                    job["question"], job["current"], job["prior"], rng
                ).items():
                    ids, prefix_len = append_answer(tokenizer, question, reasoning, job["answer"])
                    if len(ids) + 1 > args.max_context:
                        continue
                    prompts.append({"prompt_token_ids": ids})
                    meta.append((job, name, prefix_len))
            outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            packed: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
            for (job, name, prefix_len), output in zip(meta, outputs, strict=True):
                packed[(job["question_idx"], job["decision_step"])][name] = mean_target_logprob(
                    output, prefix_len
                )
            for job in batch:
                scores = packed.get((job["question_idx"], job["decision_step"])) or {}
                if "none" not in scores:
                    continue
                rec = {
                    "status": "ok",
                    "question_idx": job["question_idx"],
                    "decision_step": job["decision_step"],
                    "kind": job["kind"],
                    "gold_ok": job["gold_ok"],
                    "answer": job["answer"],
                    "geo_stored": job["geo_stored"],
                }
                base = scores["none"]
                rec["logp_none"] = base
                rec["geo_none"] = math.exp(base) if base == base else float("nan")
                for name in MASKS:
                    if name == "none":
                        continue
                    val = finite(scores.get(name))
                    rec[f"logp_{name}"] = val
                    rec[f"geo_{name}"] = math.exp(val) if val == val else float("nan")
                    rec[f"dlogp_{name}"] = base - val if val == val and base == base else float("nan")
                    rec[f"dgeo_{name}"] = (
                        rec["geo_none"] - rec[f"geo_{name}"] if val == val and base == base else float("nan")
                    )
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                ok += 1
            done_n = ok + len(done)
            print(
                f"shard{args.shard_id} wrote={ok} total_ok~{done_n} "
                f"elapsed={time.perf_counter()-started:.0f}s",
                flush=True,
            )
    print(f"done shard={args.shard_id} wrote={ok}", flush=True)


if __name__ == "__main__":
    main()
