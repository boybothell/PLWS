#!/usr/bin/env python3
"""vLLM extraction of v1 H3/H4 judge signals on dense trial answers.

Highest-priority backend: okay-budget-vllm.  Set VLLM_LENS_DISABLE=1 before
importing vllm.  Closed-think only; H3 may emit a 0-100 integer, never a
judge rationale.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["VLLM_LENS_DISABLE"] = "1"

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE))
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_judge import (  # noqa: E402
    QWEN3_ASSISTANT,
    QWEN3_USER_OPEN,
    QWEN4B_MODEL,
    dense_trial_path,
    load_done,
    yes_no_token_ids,
)

V1_PREAMBLE = """You are a careful and impartial evaluator.

Determine whether a proposed final answer correctly answers the given problem.

The partial reasoning is supporting material only. It may be incomplete or
incorrect. Do not assume that either the reasoning or the proposed answer is
correct. Independently check the relevant facts, assumptions, and reasoning.

Judge only the correctness of the proposed final answer. Do not penalize it
because the partial reasoning is unfinished, brief, poorly written, or follows
a different approach.

For multiple-choice problems, evaluate the content of the selected option using
the choices exactly as presented. For open-ended problems, accept mathematically
or semantically equivalent answers.

[Problem]
"""
REASONING_HEADER = "\n\n[Partial reasoning]\n"
ANSWER_HEADER = "\n\n[Proposed final answer]\n"
DIRECT_SUFFIX = "\n\nIs the proposed final answer correct? Answer Yes or No."
RATING_SUFFIX = (
    "\n\nWhat is the probability that the proposed final answer is correct? "
    "Reply with one integer from 0 to 100 and nothing else."
)
OPTION_RE = re.compile(r"(?m)^\s*([A-Z])\.\s*(.+?)(?=^\s*[A-Z]\.\s|\Z)", re.DOTALL)
VERBAL_RE = re.compile(r"^\s*(\d{1,3})\s*%?\b")


def display_answer(question: str, answer: str) -> str:
    selected = answer.strip().upper()
    options = {letter: text.strip() for letter, text in OPTION_RE.findall(question)}
    if selected in options:
        return f"({selected}) {options[selected]}"
    return answer or "(none)"


def load_jobs(dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[int, dict[int, dict[str, Any]]]]:
    jobs: dict[tuple[int, int, str], dict[str, Any]] = {}
    trial_map: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for trial in json.loads(dense_trial_path(dataset, seed).read_text()):
        qi = int(trial["question_idx"])
        step = int(trial["stopped_len"])
        trial_map[qi][step] = trial
        answer = str(trial.get("final_answer") or "")
        jobs.setdefault(
            (qi, step, answer),
            {
                "question_idx": qi,
                "decision_step": step,
                "answer": answer,
                "geo_conf": float(trial.get("confidence") or float("nan")),
            },
        )
    return [jobs[key] for key in sorted(jobs)], trial_map


def logprob_value(item: Any) -> float:
    if hasattr(item, "logprob"):
        return float(item.logprob)
    return float(item)


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def h4_from_logprobs(logprobs: dict[int, Any] | None, yes_ids: list[int], no_ids: list[int]) -> dict[str, float]:
    mapping = {int(token): logprob_value(item) for token, item in (logprobs or {}).items()}
    lp_yes = logsumexp([mapping[token] for token in yes_ids if token in mapping])
    lp_no = logsumexp([mapping[token] for token in no_ids if token in mapping])
    p_yes = math.exp(lp_yes) if math.isfinite(lp_yes) else float("nan")
    p_no = math.exp(lp_no) if math.isfinite(lp_no) else float("nan")
    probs = [math.exp(value) for value in mapping.values()]
    returned = float(sum(probs))
    entropy = float(-sum(prob * math.log(prob) for prob in probs if prob > 0.0))
    ranked = sorted(mapping.values(), reverse=True)
    return {
        "yes": float(1.0 / (1.0 + math.exp(-(lp_yes - lp_no)))) if math.isfinite(lp_yes) and math.isfinite(lp_no) else float("nan"),
        "logp_yes": float(lp_yes),
        "logp_no": float(lp_no),
        "margin": float(lp_yes - lp_no) if math.isfinite(lp_yes) and math.isfinite(lp_no) else float("nan"),
        "yes_no_coverage": p_yes + p_no if math.isfinite(p_yes) and math.isfinite(p_no) else float("nan"),
        "vocab_entropy": entropy,
        "vocab_entropy_normalized": entropy / math.log(max(len(mapping), 2)),
        "top2_logit_margin": float(ranked[0] - ranked[1]) if len(ranked) >= 2 else float("nan"),
        "returned_mass": returned,
        "n_logprobs": len(mapping),
    }


def parse_verbal(text: str) -> tuple[int | None, str]:
    match = VERBAL_RE.match(text)
    value = int(match.group(1)) if match else None
    return (value if value is not None and 0 <= value <= 100 else None), text


def nvidia_lib_path() -> str:
    root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
    extra = ":".join(sorted(str(path) for path in (root / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir()))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logprobs-k", type=int, default=64)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    os.environ["VLLM_LENS_DISABLE"] = "1"
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir or (AE / "results/confcal_judge/v1" / f"{args.dataset}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"scores_shard{args.shard_id}.jsonl"
    jobs, trial_map = load_jobs(args.dataset, args.seed)
    question_ids = sorted({job["question_idx"] for job in jobs})
    mine = {qi for index, qi in enumerate(question_ids) if index % args.num_shards == args.shard_id}
    jobs = [job for job in jobs if job["question_idx"] in mine]
    if args.limit:
        jobs = jobs[: args.limit]
    done = load_done(scores_path)
    pending = [job for job in jobs if (job["question_idx"], job["decision_step"], job["answer"]) not in done]

    llm = LLM(
        model=str(QWEN4B_MODEL),
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        max_logprobs=max(args.logprobs_k, 20),
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN4B_MODEL), trust_remote_code=True)
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    stop_ids = [token for token in (tokenizer.eos_token_id, getattr(tokenizer, "eod_id", None)) if token is not None]
    for text in ("<|im_end|>",):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            stop_ids.append(encoded[0])
    # This vLLM build forbids combining logprobs=K with logprob_token_ids
    # unless K equals that list length.  Use top-k only; Yes/No are almost
    # always inside the top 64 for a Yes/No suffix.
    direct_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=args.logprobs_k,
        detokenize=False,
    )
    digit_ids: list[int] = []
    for digit in range(10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        if encoded and encoded[0] not in digit_ids:
            digit_ids.append(encoded[0])
    verbal_params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        stop=["\n"],
        stop_token_ids=stop_ids,
        allowed_token_ids=digit_ids + stop_ids,
        skip_special_tokens=False,
        include_stop_str_in_output=False,
    )
    meta = {
        "dataset": args.dataset,
        "seed": args.seed,
        "judge_model": str(QWEN4B_MODEL),
        "prompt_version": "v1",
        "backend": "vllm",
        "signals": ["direct_h4", "verbal_h3"],
        "execution": "vllm prefix-cache; closed think; H3 greedy 0-100; H4 top-k logprobs plus exact Yes/No ids",
        "logprobs_k": args.logprobs_k,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "n_jobs": len(jobs),
        "n_pending": len(pending),
    }
    (out_dir / f"meta_shard{args.shard_id}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{args.dataset} shard={args.shard_id}/{args.num_shards} jobs={len(jobs)} pending={len(pending)} backend=vllm", flush=True)

    ok = skipped = 0
    started = time.perf_counter()
    with scores_path.open("a") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            ready: list[dict[str, Any]] = []
            for job in batch:
                trial = trial_map[job["question_idx"]][job["decision_step"]]
                question = str(trial.get("question") or "")
                reasoning = str(trial.get("reasoning_prefix") or "")
                answer_for_judge = display_answer(question, job["answer"])
                body = (
                    f"{QWEN3_USER_OPEN}{V1_PREAMBLE}{question}{REASONING_HEADER}{reasoning}"
                    f"{ANSWER_HEADER}{answer_for_judge}"
                )
                body_tokens = len(tokenizer.encode(body, add_special_tokens=False))
                suffix_tokens = max(
                    len(tokenizer.encode(DIRECT_SUFFIX + QWEN3_ASSISTANT, add_special_tokens=False)),
                    len(tokenizer.encode(RATING_SUFFIX + QWEN3_ASSISTANT, add_special_tokens=False)),
                )
                if body_tokens + suffix_tokens > args.max_context:
                    handle.write(json.dumps({**job, "status": "too_long", "body_tokens": body_tokens}) + "\n")
                    skipped += 1
                    continue
                ready.append(
                    {
                        **job,
                        "answer_for_judge": answer_for_judge,
                        "body_tokens": body_tokens,
                        "direct_prompt": body + DIRECT_SUFFIX + QWEN3_ASSISTANT,
                        "verbal_prompt": body + RATING_SUFFIX + QWEN3_ASSISTANT,
                    }
                )
            if not ready:
                handle.flush()
                continue
            try:
                direct_out = llm.generate([row["direct_prompt"] for row in ready], direct_params, use_tqdm=False)
                verbal_out = llm.generate([row["verbal_prompt"] for row in ready], verbal_params, use_tqdm=False)
            except Exception as exc:
                for row in ready:
                    payload = {key: row[key] for key in ("question_idx", "decision_step", "answer", "geo_conf")}
                    payload.update({"status": "forward_fail", "error": str(exc)[:200]})
                    handle.write(json.dumps(payload) + "\n")
                skipped += len(ready)
                continue
            for row, direct, verbal in zip(ready, direct_out, verbal_out, strict=True):
                first = direct.outputs[0].logprobs[0] if direct.outputs and direct.outputs[0].logprobs else {}
                text = verbal.outputs[0].text if verbal.outputs else ""
                value, raw = parse_verbal(text)
                handle.write(
                    json.dumps(
                        {
                            "question_idx": row["question_idx"],
                            "decision_step": row["decision_step"],
                            "answer": row["answer"],
                            "geo_conf": row["geo_conf"],
                            "answer_for_judge": row["answer_for_judge"],
                            "status": "ok",
                            "body_tokens": row["body_tokens"],
                            "direct_h4": h4_from_logprobs(first, yes_ids, no_ids),
                            "verbal_h3": {"value": value, "text": raw},
                        }
                    )
                    + "\n"
                )
                ok += 1
            handle.flush()
            done_n = min(start + len(batch), len(pending))
            if done_n == len(batch) or done_n % (args.batch_size * 4) == 0 or done_n == len(pending):
                print(f"[{done_n}/{len(pending)}] ok={ok} skip={skipped} elapsed={time.perf_counter()-started:.0f}s", flush=True)
    print(f"done ok={ok} skip={skipped} -> {scores_path}", flush=True)


if __name__ == "__main__":
    main()
