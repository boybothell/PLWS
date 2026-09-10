#!/usr/bin/env python3
"""7B vLLM teacher-force probes on every dense trial.

Signals (all causal, no future peek):
  current_logp, question_only_logp, reasoning_pmi, margin, evidence_gain
  increment_nll, wrapup_nll, hist_forget
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

from pilot_counterfactual_support import (  # noqa: E402
    append_answer,
    build_prefix,
    finite,
    mean_target_logprob,
    unique_history_answers,
)
from dump_dense_candidates import trial_path  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

WRAP_CUE = (
    "\n\nTime is up. Given the time I've spent and the approaches I've tried, "
    "I should stop thinking and now write summarization in one sentence.\n"
    "</think>\nThe final answer is "
)
OUT_DIR = AE / "results/confcal_judge/v2/dense_solver_probes"


def load_grouped(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(path.read_text()):
        grouped[int(row["question_idx"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["stopped_len"]))
    return grouped


def default_probe_out(model_tag: str, dataset: str, seed: int | None, shard_id: int) -> Path:
    if model_tag == "r1_7b" and dataset in {"math-500", "olympiadbench", "gpqa-diamond"}:
        return OUT_DIR / dataset / f"scores_shard{shard_id}.jsonl"
    stem = f"{dataset}_s{seed}" if seed is not None else dataset
    return OUT_DIR / model_tag / stem / f"scores_shard{shard_id}.jsonl"


def last_increment(current: str, prior: str) -> str:
    if prior and current.startswith(prior):
        return current[len(prior):]
    return current


def mean_span_logprob(output: Any, start: int, end: int) -> float:
    ids = output.prompt_token_ids or []
    logs = output.prompt_logprobs or []
    values: list[float] = []
    for index in range(max(start, 1), min(end, len(ids), len(logs))):
        item = (logs[index] or {}).get(ids[index])
        if item is None:
            return float("nan")
        values.append(float(item.logprob if hasattr(item, "logprob") else item))
    return float(sum(values) / len(values)) if values else float("nan")


def done_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    out = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                out.add((int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=Path, default=SOLVER)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--gpu-mem-util", type=float, default=0.88)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    seed = args.seed
    if seed is None and args.dataset.startswith("aime"):
        seed = 42
    trial = args.trial or trial_path(args.model_tag, args.dataset, seed if args.dataset.startswith("aime") else None)
    grouped = load_grouped(trial)
    jobs: list[dict[str, Any]] = []
    for qi, history in grouped.items():
        if qi % args.num_shards != args.shard_id:
            continue
        for index, row in enumerate(history):
            jobs.append(
                {
                    "dataset": args.dataset,
                    "question_idx": qi,
                    "index": index,
                    "decision_step": int(row["stopped_len"]),
                    "answer": str(row.get("final_answer") or ""),
                    "geo_conf": finite(row.get("confidence")),
                }
            )
    args.out = args.out or default_probe_out(args.model_tag, args.dataset, seed, args.shard_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out)
    jobs = [job for job in jobs if (job["question_idx"], job["decision_step"], job["answer"]) not in done]
    if not jobs:
        print(
            f"dense-solver {args.model_tag} {args.dataset} shard={args.shard_id}/{args.num_shards} pending=0 -> {args.out}",
            flush=True,
        )
        return
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    print(
        f"dense-solver {args.model_tag} {args.dataset} shard={args.shard_id}/{args.num_shards} "
        f"pending={len(jobs)} model={Path(args.model).name} -> {args.out}",
        flush=True,
    )
    llm = LLM(
        model=str(args.model),
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp,
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
                wrap_prefix = build_prefix(tokenizer, question, current)
                wrap_tail = tokenizer.encode(WRAP_CUE + f"\\boxed{{{job['answer']}}}", add_special_tokens=False)
                wrap_ids = wrap_prefix + wrap_tail
                if len(wrap_ids) + 1 <= args.max_context:
                    prompts.append({"prompt_token_ids": wrap_ids})
                    meta.append((job, "wrapup", len(wrap_prefix), len(wrap_ids)))
                if prior and increment:
                    cur_ids = build_prefix(tokenizer, question, current)
                    prior_ids = build_prefix(tokenizer, question, prior)
                    if len(cur_ids) + 1 <= args.max_context and len(prior_ids) < len(cur_ids):
                        prompts.append({"prompt_token_ids": cur_ids})
                        meta.append((job, "increment_span", len(prior_ids), len(cur_ids)))
            outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
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
                alts = [finite(x) for x in value.get("alternative", [])]
                alts = [x for x in alts if math.isfinite(x)]
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
