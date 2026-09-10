#!/usr/bin/env python3
"""Universal Qwen3 Base/Instruct text re-evaluation and candidate ranking.

No cross-tokenizer distribution comparison is performed here.  Sequence
likelihood is normalized by UTF-8 byte length; candidate ranks use only
teacher-forced label log-probabilities and never generate a judge rationale.
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

from score_confcal_judge import dense_trial_path, load_done  # noqa: E402
from score_confcal_v1 import (  # noqa: E402
    ANSWER_HEADER,
    DIRECT_SUFFIX,
    QWEN3_ASSISTANT,
    QWEN3_USER_OPEN,
    REASONING_HEADER,
    V1_PREAMBLE,
    display_answer,
    nvidia_lib_path,
)

MODELS = {
    "base": Path("/mnt/d/lsj/models/Qwen3-4B-Base"),
    "instruct": Path("/mnt/d/lsj/models/Qwen3-4B"),
}
LABELS = tuple("ABCDEFGH")
NONE_CANDIDATE = "__NONE_OF_THE_ABOVE__"


def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def lse(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def token_logprob(item: Any) -> float:
    return float(item.logprob if hasattr(item, "logprob") else item)


def target_logprobs(output: Any, prefix_len: int) -> list[float]:
    """Actual-token logprobs for tokens appended after a known prefix."""
    values: list[float] = []
    prompt_ids = output.prompt_token_ids or []
    entries = output.prompt_logprobs or []
    for index in range(max(prefix_len, 1), min(len(entries), len(prompt_ids))):
        item = (entries[index] or {}).get(prompt_ids[index])
        if item is None:
            return []
        values.append(token_logprob(item))
    return values


def build_text_prompt(trial: dict[str, Any], mode: str) -> tuple[str, str, str]:
    question = str(trial.get("question") or "")
    reasoning = str(trial.get("reasoning_prefix") or "")
    answer = display_answer(question, str(trial.get("final_answer") or ""))
    # Base is deliberately given the same lexical content without an
    # instruction-chat wrapper; this isolates post-training rather than
    # pretending a Base checkpoint follows a chat protocol.
    body = f"{V1_PREAMBLE}{question}{REASONING_HEADER}{reasoning}{ANSWER_HEADER}{answer}"
    if mode == "instruct":
        # The answer remains inside the quoted user material.  Appending an
        # assistant header here would make the extracted span score that header
        # rather than the proposed answer.
        body = QWEN3_USER_OPEN + body
    return body, reasoning, answer


def span_logprob(logs: list[float], start: int, end: int, nbytes: int) -> dict[str, float]:
    piece = logs[max(start, 0) : max(end, 0)]
    if not piece:
        return {"sum": float("nan"), "per_token": float("nan"), "per_byte": float("nan")}
    return {
        "sum": sum(piece),
        "per_token": sum(piece) / len(piece),
        "per_byte": sum(piece) / max(nbytes, 1),
    }


def candidate_prompt(trial: dict[str, Any], mode: str, candidates: list[str]) -> str:
    question = str(trial.get("question") or "")
    reasoning = str(trial.get("reasoning_prefix") or "")
    listed = "\n".join(
        f"{LABELS[i]}. {'None of the above' if answer == NONE_CANDIDATE else display_answer(question, answer)}"
        for i, answer in enumerate(candidates)
    )
    text = (
        f"{V1_PREAMBLE}{question}{REASONING_HEADER}{reasoning}\n\n"
        f"[Candidate proposed answers]\n{listed}\n\n"
        "Which candidate is most likely correct? Reply with its letter only."
    )
    return QWEN3_USER_OPEN + text + QWEN3_ASSISTANT if mode == "instruct" else text + "\nAnswer:"


def fits(ids: list[int], max_context: int) -> bool:
    # vLLM reserves max_tokens (>=1) on top of the prompt.
    return len(ids) + 1 <= max_context


def generate_safe(llm: Any, prompts: list[dict[str, Any]], params: Any) -> list[Any]:
    if not prompts:
        return []
    try:
        return llm.generate(prompts, params, use_tqdm=False)
    except Exception as exc:
        print(f"batch generate failed ({exc}); retrying one-by-one", flush=True)
        outputs: list[Any] = []
        for prompt in prompts:
            try:
                outputs.append(llm.generate([prompt], params, use_tqdm=False)[0])
            except Exception as inner:
                print(f"item generate failed ({inner})", flush=True)
                outputs.append(None)
        return outputs


def write_row(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload) + "\n")
    handle.flush()


def unique_answers(history: list[dict[str, Any]], question: str, limit: int) -> list[str]:
    out: list[str] = []
    for trial in history:
        answer = str(trial.get("final_answer") or "")
        if answer and not any(answer.strip() == old.strip() for old in out):
            out.append(answer)
    # VPD's comparison must retain an explicit probability mass for unseen
    # alternatives; it prevents the single historic answer from receiving 1.0.
    return out[-max(limit - 1, 1):] + [NONE_CANDIDATE]


def load_trials(dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trial in json.loads(dense_trial_path(dataset, seed).read_text()):
        grouped[int(trial["question_idx"])].append(trial)
    jobs: list[dict[str, Any]] = []
    for qi, history in grouped.items():
        history.sort(key=lambda row: int(row["stopped_len"]))
        for index, trial in enumerate(history):
            jobs.append({"question_idx": qi, "decision_step": int(trial["stopped_len"]), "index": index})
    return sorted(jobs, key=lambda row: (row["question_idx"], row["decision_step"])), grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=MODELS, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not MODELS[args.mode].exists():
        raise FileNotFoundError(MODELS[args.mode])
    out_dir = args.out_dir or AE / "results/confcal_judge/v2/universal_v2b" / f"{args.dataset}_s{args.seed}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scores_shard{args.shard_id}.jsonl"
    jobs, grouped = load_trials(args.dataset, args.seed)
    qids = sorted(grouped)
    mine = {qi for index, qi in enumerate(qids) if index % args.num_shards == args.shard_id}
    jobs = [job for job in jobs if job["question_idx"] in mine]
    if args.limit:
        jobs = jobs[:args.limit]
    done = load_done(out_path)
    jobs = [job for job in jobs if (job["question_idx"], job["decision_step"], str(grouped[job["question_idx"]][job["index"]].get("final_answer") or "")) not in done]
    llm = LLM(model=str(MODELS[args.mode]), max_model_len=args.max_context, gpu_memory_utilization=args.gpu_mem_util, enable_prefix_caching=True, max_logprobs=1, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS[args.mode]), trust_remote_code=True)
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    started = time.perf_counter()
    with out_path.open("a") as handle:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start + args.batch_size]
            full_prompts, full_info = [], []
            for job in batch:
                trial = grouped[job["question_idx"]][job["index"]]
                prompt, reasoning, answer = build_text_prompt(trial, args.mode)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                base = {
                    "question_idx": job["question_idx"],
                    "decision_step": job["decision_step"],
                    "answer": str(trial.get("final_answer") or ""),
                    "geo_conf": finite(trial.get("confidence")),
                    "mode": args.mode,
                }
                if not fits(ids, args.max_context):
                    write_row(handle, {**base, "status": "too_long", "n_ids": len(ids)})
                    continue
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                reasoning_ids = tokenizer.encode(reasoning, add_special_tokens=False)
                prev = grouped[job["question_idx"]][job["index"] - 1] if job["index"] else None
                prev_len = len(tokenizer.encode(str(prev.get("reasoning_prefix") or ""), add_special_tokens=False)) if prev else 0
                answer_start = len(ids) - len(answer_ids)
                reasoning_end = answer_start
                reasoning_start = max(reasoning_end - len(reasoning_ids), 0)
                window_start = max(reasoning_end - min(len(reasoning_ids), 256), reasoning_start)
                step_start = min(reasoning_start + prev_len, reasoning_end)
                full_prompts.append({"prompt_token_ids": ids})
                full_info.append((job, trial, reasoning, answer, reasoning_start, step_start, window_start, reasoning_end, answer_start, len(ids), base))
            outputs = generate_safe(llm, full_prompts, params)
            candidate_inputs, candidate_meta = [], []
            for job, trial, _, answer, *_rest in full_info:
                history = grouped[job["question_idx"]][:job["index"] + 1]
                candidates = unique_answers(history, str(trial.get("question") or ""), args.candidate_limit)
                for label, candidate in zip(LABELS, candidates):
                    prefix = candidate_prompt(trial, args.mode, candidates)
                    ids = tokenizer.encode(prefix + label, add_special_tokens=False)
                    if not fits(ids, args.max_context):
                        continue
                    candidate_inputs.append({"prompt_token_ids": ids})
                    candidate_meta.append((job, answer, candidate, label, len(ids) - 1))
            candidate_outputs = generate_safe(llm, candidate_inputs, params)
            by_key: dict[tuple[int, int], list[tuple[str, str, float]]] = defaultdict(list)
            for meta, output in zip(candidate_meta, candidate_outputs, strict=True):
                job, answer, candidate, label, prefix_len = meta
                if output is None:
                    by_key[(job["question_idx"], job["decision_step"])].append((candidate, label, float("nan")))
                    continue
                logs = target_logprobs(output, prefix_len)
                by_key[(job["question_idx"], job["decision_step"])].append((candidate, label, logs[0] if logs else float("nan")))
            for info, output in zip(full_info, outputs, strict=True):
                job, trial, reasoning, answer, reasoning_start, step_start, window_start, reasoning_end, answer_start, n_ids, base = info
                if output is None:
                    write_row(handle, {**base, "status": "error", "n_ids": n_ids})
                    continue
                logs = target_logprobs(output, reasoning_start)
                shift = reasoning_start
                def local(start: int, end: int, nbytes: int) -> dict[str, float]:
                    return span_logprob(logs, start - shift, end - shift, nbytes)
                candidates = by_key[(job["question_idx"], job["decision_step"])]
                good = [value for _, _, value in candidates if math.isfinite(value)]
                current = next(
                    (value for candidate, _, value in candidates if candidate.strip() == str(trial.get("final_answer") or "").strip()),
                    float("nan"),
                )
                answer_span = local(answer_start, n_ids, len(answer.encode()))
                write_row(handle, {
                    **base, "status": "ok",
                    "answer_bytes": len(answer.encode()), "reasoning_bytes": len(reasoning.encode()),
                    "text_logprob_sum": answer_span["sum"],
                    "text_logprob_per_token": answer_span["per_token"],
                    "text_logprob_per_byte": answer_span["per_byte"],
                    "hb_answer": answer_span,
                    "hb_step": local(step_start, reasoning_end, max(len(reasoning.encode()), 1)),
                    "hb_window": local(window_start, reasoning_end, max(len(reasoning.encode()), 1)),
                    "hb_full": local(reasoning_start, reasoning_end, max(len(reasoning.encode()), 1)),
                    "candidate_logprob": current,
                    "candidate_prob": math.exp(current - lse(good)) if math.isfinite(current) and good else float("nan"),
                    "candidate_count": len(candidates),
                    "candidate_has_none": True,
                })
            print(f"[{min(start + len(batch), len(jobs))}/{len(jobs)}] elapsed={time.perf_counter()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
