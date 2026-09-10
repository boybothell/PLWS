#!/usr/bin/env python3
"""4B teacher-force support probes: PMI, margin, evidence gain, wrapup.

Same calculations as 7B score_dense_solver_probes.py. Prefix is Qwen3 open
think stuffed with 7B reasoning. 4B does not generate. Cross-vocab JS is
not computed.
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

os.environ["VLLM_LENS_DISABLE"] = "1"

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from pilot_counterfactual_support import fast_eq, finite, mean_target_logprob  # noqa: E402
from score_confcal_judge import QWEN4B_MODEL  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402
from score_dense_4b_open_think import THINK_CLOSE, boxed, open_think_header  # noqa: E402
from score_dense_solver_probes import last_increment, mean_span_logprob  # noqa: E402

WRAP_CUE = "The final answer is "


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def done_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
        for row in load_jsonl(path)
        if row.get("status") == "ok"
    }


def unique_history_answers(history: list[dict[str, Any]], index: int, current: str) -> list[str]:
    answers: list[str] = []
    for row in history[:index]:
        answer = str(row.get("answer") or "")
        if answer and not fast_eq(answer, current) and not any(fast_eq(answer, old) for old in answers):
            answers.append(answer)
    return answers[-5:]


def think_prefix(tokenizer, question: str, reasoning: str) -> list[int]:
    header = open_think_header(tokenizer, question)
    return tokenizer.encode(header + reasoning + "\n", add_special_tokens=False)


def append_answer(tokenizer, question: str, reasoning: str, answer: str) -> tuple[list[int], int]:
    prefix = think_prefix(tokenizer, question, reasoning)
    target = tokenizer.encode(boxed(answer), add_special_tokens=False)
    return prefix + target, len(prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    os.environ["VLLM_LENS_DISABLE"] = "1"
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.candidates)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["question_idx"])].append(row)
    for seq in grouped.values():
        seq.sort(key=lambda row: int(row["decision_step"]))
    qis = sorted(grouped)
    keep = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs: list[dict[str, Any]] = []
    for qi in qis:
        if qi not in keep:
            continue
        for index, row in enumerate(grouped[qi]):
            jobs.append(
                {
                    "question_idx": qi,
                    "index": index,
                    "decision_step": int(row["decision_step"]),
                    "answer": str(row.get("answer") or ""),
                    "geo_conf": row.get("geo_conf"),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [job for job in jobs if (job["question_idx"], job["decision_step"], job["answer"]) not in done_keys(args.out)]
    if args.limit:
        jobs = jobs[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN4B_MODEL), trust_remote_code=True)
    print(
        f"4b-support shard={args.shard_id}/{args.num_shards} pending={len(jobs)} -> {args.out}",
        flush=True,
    )
    llm = LLM(
        model=str(QWEN4B_MODEL),
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        max_logprobs=1,
        seed=42,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    started = time.perf_counter()
    with args.out.open("a") as handle:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start : start + args.batch_size]
            prompts: list[dict[str, list[int]]] = []
            meta: list[tuple[dict[str, Any], str, int, int]] = []
            for job in batch:
                history = grouped[job["question_idx"]]
                row = history[job["index"]]
                question = str(row.get("question") or "")
                current = str(row.get("reasoning_prefix") or "")
                prior = str(history[job["index"] - 1].get("reasoning_prefix") or "") if job["index"] else ""
                increment = last_increment(current, prior).lstrip()
                alternatives = unique_history_answers(history, job["index"], job["answer"])
                variants = [
                    ("current", current, job["answer"]),
                    ("prior", prior, job["answer"]),
                    ("question_only", "", job["answer"]),
                    ("increment_only", increment, job["answer"]),
                ]
                variants.extend(("alternative", current, answer) for answer in alternatives)
                for kind, reasoning, answer in variants:
                    ids, prefix_len = append_answer(tokenizer, question, reasoning, answer)
                    if len(ids) + 1 > args.max_context:
                        continue
                    prompts.append({"prompt_token_ids": ids})
                    meta.append((job, kind, prefix_len, len(ids)))
                wrap_prefix = think_prefix(tokenizer, question, current)
                wrap_tail = tokenizer.encode(THINK_CLOSE + WRAP_CUE + boxed(job["answer"]), add_special_tokens=False)
                wrap_ids = wrap_prefix + wrap_tail
                if len(wrap_ids) + 1 <= args.max_context:
                    prompts.append({"prompt_token_ids": wrap_ids})
                    meta.append((job, "wrapup", len(wrap_prefix), len(wrap_ids)))
                if prior and increment:
                    cur_ids = think_prefix(tokenizer, question, current)
                    prior_ids = think_prefix(tokenizer, question, prior)
                    if len(cur_ids) + 1 <= args.max_context and len(prior_ids) < len(cur_ids):
                        prompts.append({"prompt_token_ids": cur_ids})
                        meta.append((job, "increment_span", len(prior_ids), len(cur_ids)))
            try:
                outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            except Exception as exc:
                for job in batch:
                    handle.write(
                        json.dumps({**job, "status": "forward_fail", "error": str(exc)[:200]}) + "\n"
                    )
                continue
            scores: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(lambda: defaultdict(list))
            for (job, kind, start_len, end_len), output in zip(meta, outputs, strict=True):
                key = (job["question_idx"], job["decision_step"], job["answer"])
                if kind == "increment_span":
                    scores[key]["increment_nll"] = -mean_span_logprob(output, start_len, end_len)
                elif kind == "wrapup":
                    scores[key]["wrapup_nll"] = -mean_span_logprob(output, start_len, end_len)
                elif kind == "alternative":
                    scores[key]["alternative"].append(mean_target_logprob(output, start_len))
                else:
                    scores[key][kind] = mean_target_logprob(output, start_len)
            for job in batch:
                key = (job["question_idx"], job["decision_step"], job["answer"])
                value = scores[key]
                current = finite(value.get("current"))
                prior = finite(value.get("prior"))
                question_only = finite(value.get("question_only"))
                increment_only = finite(value.get("increment_only"))
                alts = [x for x in (finite(item) for item in value.get("alternative", [])) if math.isfinite(x)]
                best_alt = max(alts) if alts else float("nan")
                handle.write(
                    json.dumps(
                        {
                            **job,
                            "status": "ok",
                            "current_logp": current,
                            "question_only_logp": question_only,
                            "reasoning_pmi": current - question_only if math.isfinite(current) and math.isfinite(question_only) else float("nan"),
                            "margin": current - best_alt if math.isfinite(current) and math.isfinite(best_alt) else float("nan"),
                            "evidence_gain": current - prior if math.isfinite(current) and math.isfinite(prior) else float("nan"),
                            "hist_forget": current - increment_only if math.isfinite(current) and math.isfinite(increment_only) else float("nan"),
                            "increment_nll": finite(value.get("increment_nll")),
                            "wrapup_nll": finite(value.get("wrapup_nll")),
                            "neg_increment_nll": -finite(value.get("increment_nll")),
                            "neg_wrapup_nll": -finite(value.get("wrapup_nll")),
                            "n_historical_alternatives": len(alts),
                        }
                    )
                    + "\n"
                )
            handle.flush()
            print(
                f"[{min(start + len(batch), len(jobs))}/{len(jobs)}] "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
