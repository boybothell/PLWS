#!/usr/bin/env python3
"""Read first boxed-content token mass and second-best answer sequence.

Same first-per-kind windows as resample. Prefix ends at \\boxed.
Generate 24 tokens with top-20 logprobs for the branch; teacher-force
{old} and historical alts for sequence / first-token margins.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_LENS_DISABLE", "1")

AE = Path(__file__).resolve().parents[1]
PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(PUMA / "puma"))

from score_confcal_v1 import logprob_value, nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_first_lock_room as room  # noqa: E402
import score_samewin_reprobe as rp  # noqa: E402
from gen_trial_answers import (  # noqa: E402
    build_prompt,
    find_boxed_content_token_span,
)
from prompt_utils import get_confident_ending, get_task_type  # noqa: E402

MODEL = rp.MODEL
OUT = AE / "results/samewin_firsttok"


def attach_hist_alts(jobs: list[dict[str, Any]], dataset: str, seed: int) -> list[dict[str, Any]]:
    trials_path = room.dense_trial_path("r1_7b", dataset, seed)
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    for job in jobs:
        rows = rg.usable_rows(by[job["question_idx"]])
        alts: list[str] = []
        for row in rows:
            if int(row["stopped_len"]) >= job["decision_step"]:
                break
            ans = str(row.get("final_answer") or "")
            if ans and not rg.same(ans, job["old_answer"]) and not any(rg.same(ans, old) for old in alts):
                alts.append(ans)
        job["hist_alts"] = alts[:3]
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


def mapping(step: Any) -> dict[int, float]:
    if not isinstance(step, dict) or not step:
        return {}
    return {int(tok): logprob_value(item) for tok, item in step.items()}


def first_stats(token_ids: list[int], logprobs: list[Any], tokenizer: Any) -> dict[str, Any]:
    start, end = find_boxed_content_token_span(token_ids, tokenizer)
    if start >= end or start >= len(logprobs):
        return {
            "first_p": float("nan"),
            "first_entropy": float("nan"),
            "first_margin": float("nan"),
            "first_top2_p": float("nan"),
            "n_content": 0,
        }
    mp = mapping(logprobs[start])
    vals = sorted(mp.values(), reverse=True)
    chosen = token_ids[start]
    first_lp = mp.get(int(chosen), float("nan"))
    first_p = math.exp(first_lp) if first_lp == first_lp else float("nan")
    probs = [math.exp(v) for v in vals if v == v]
    ent = float(-sum(p * math.log(p) for p in probs if p > 0)) if probs else float("nan")
    margin = vals[0] - vals[1] if len(vals) >= 2 else float("nan")
    top2 = math.exp(vals[1]) if len(vals) >= 2 else float("nan")
    return {
        "first_p": first_p,
        "first_entropy": ent,
        "first_margin": margin,
        "first_top2_p": top2,
        "n_content": end - start,
        "first_tok": int(chosen),
    }


def tf_content(output: Any, prefix_len: int, tokenizer: Any) -> dict[str, float]:
    ids = list(output.prompt_token_ids or [])
    logs = list(output.prompt_logprobs or [])
    gen_ids = ids[prefix_len:]
    start, end = find_boxed_content_token_span(gen_ids, tokenizer)
    values: list[float] = []
    for i in range(start, end):
        pos = prefix_len + i
        if pos >= len(ids) or pos >= len(logs) or not logs[pos]:
            return {"first_logp": float("nan"), "mean_logp": float("nan")}
        item = logs[pos].get(ids[pos])
        if item is None:
            return {"first_logp": float("nan"), "mean_logp": float("nan")}
        values.append(logprob_value(item))
    if not values:
        return {"first_logp": float("nan"), "mean_logp": float("nan")}
    return {"first_logp": values[0], "mean_logp": float(sum(values) / len(values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--logprobs-k", type=int, default=20)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    jobs = attach_hist_alts(
        [j for j in rp.load_jobs(args.dataset, args.seed) if j["question_idx"] % args.num_shards == args.shard_id],
        args.dataset,
        args.seed,
    )
    args.out = args.out or (OUT / f"{args.dataset}_s{args.seed}" / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out)
    pending = [job for job in jobs if (job["question_idx"], job["decision_step"]) not in done]
    kinds = defaultdict(int)
    for job in pending:
        kinds[(job["kind"], job["gold_ok"], len(job["hist_alts"]))] += 1
    print(
        f"samewin-firsttok {args.dataset} shard={args.shard_id}/{args.num_shards} "
        f"pending={len(pending)} mix={dict(kinds)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ending = get_confident_ending(args.dataset, is_trial=True)
    task = get_task_type(args.dataset)
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_logprobs=args.logprobs_k,
    )
    gen_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs_k,
        detokenize=False,
    )
    tf_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            prefixes: list[str] = []
            ready: list[dict[str, Any]] = []
            for job in chunk:
                prompt = build_prompt(
                    tokenizer,
                    MODEL,
                    job["question"],
                    job["current"],
                    task,
                    "default",
                    True,
                    args.dataset,
                    confident_ending=ending,
                )
                n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
                if n_tok + args.max_tokens > args.max_context:
                    handle.write(
                        json.dumps(
                            {
                                "status": "too_long",
                                "question_idx": job["question_idx"],
                                "decision_step": job["decision_step"],
                            }
                        )
                        + "\n"
                    )
                    continue
                prefixes.append(prompt)
                ready.append(job)
            if not prefixes:
                continue
            gens = llm.generate(prefixes, gen_params, use_tqdm=False)
            tf_prompts: list[dict[str, list[int]]] = []
            tf_meta: list[tuple[int, str, int]] = []
            for i, job in enumerate(ready):
                prefix_ids = list(gens[i].prompt_token_ids)
                answers = [job["old_answer"], *job["hist_alts"]]
                for ans in answers:
                    cont = tokenizer.encode("{" + str(ans) + "}", add_special_tokens=False)
                    tf_prompts.append({"prompt_token_ids": prefix_ids + cont})
                    tf_meta.append((i, ans, len(prefix_ids)))
            tfs = llm.generate(tf_prompts, tf_params, use_tqdm=False) if tf_prompts else []
            by_i: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
            for (i, ans, prefix_len), out in zip(tf_meta, tfs, strict=True):
                by_i[i][ans] = tf_content(out, prefix_len, tokenizer)
            for i, (job, gen) in enumerate(zip(ready, gens, strict=True)):
                g0 = gen.outputs[0]
                stats = first_stats(list(g0.token_ids), list(g0.logprobs or []), tokenizer)
                old_tf = by_i[i].get(job["old_answer"]) or {}
                alt_mean = [by_i[i][a]["mean_logp"] for a in job["hist_alts"] if a in by_i[i]]
                alt_first = [by_i[i][a]["first_logp"] for a in job["hist_alts"] if a in by_i[i]]
                alt_mean = [x for x in alt_mean if x == x]
                alt_first = [x for x in alt_first if x == x]
                mean_old = old_tf.get("mean_logp", float("nan"))
                first_old = old_tf.get("first_logp", float("nan"))
                rec = {
                    "status": "ok",
                    "question_idx": job["question_idx"],
                    "decision_step": job["decision_step"],
                    "kind": job["kind"],
                    "gold_ok": job["gold_ok"],
                    "old_answer": job["old_answer"],
                    "geo_stored": job["geo_stored"],
                    "n_hist_alt": len(job["hist_alts"]),
                    "first_p": stats["first_p"],
                    "first_entropy": stats["first_entropy"],
                    "first_margin": stats["first_margin"],
                    "first_top2_p": stats["first_top2_p"],
                    "n_content": stats.get("n_content", 0),
                    "tf_first_logp": first_old,
                    "tf_mean_logp": mean_old,
                    "tf_geo": math.exp(mean_old) if mean_old == mean_old else float("nan"),
                    "seq_margin": (
                        mean_old - max(alt_mean) if mean_old == mean_old and alt_mean else float("nan")
                    ),
                    "first_tf_margin": (
                        first_old - max(alt_first) if first_old == first_old and alt_first else float("nan")
                    ),
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            handle.flush()
            print(
                f"shard{args.shard_id} wrote={wrote}/{len(pending)} elapsed={time.perf_counter()-started:.0f}s",
                flush=True,
            )
    print(f"done shard={args.shard_id} wrote={wrote}", flush=True)


if __name__ == "__main__":
    main()
