#!/usr/bin/env python3
"""Qwen3-4B 开思考：每步把历史+当前试答列成 1-4，思考后再读编号概率。

不把 7B 思路塞进提示。解题模型是 7B 奥赛密探轨迹；4B 只闭集投票。
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

from score_confcal_v1 import logprob_value, nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import replay_rescue_R_gate as rg  # noqa: E402

JUDGE = "/mnt/d/lsj/models/Qwen3-4B"
OUT = AE / "results/hc_think"
NONE = "__NONE__"
DIGITS = ("1", "2", "3", "4")


def finite(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def logsumexp(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return float("nan")
    peak = max(xs)
    return peak + math.log(sum(math.exp(v - peak) for v in xs))


def done_keys(path: Path) -> set[tuple[int, int]]:
    if not path.is_file():
        return set()
    out: set[tuple[int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") in {"ok", "too_long"}:
            out.add((int(row["question_idx"]), int(row["decision_step"])))
    return out


def build_user(question: str, cands: list[str]) -> str:
    lines = []
    for i, ans in enumerate(cands, start=1):
        shown = "None of the above" if ans == NONE else ans
        lines.append(f"{i}. {shown}")
    listed = "\n".join(lines)
    return (
        "You are given a problem and candidate answers that already appeared as "
        "intermediate trials. Think about which candidate is most likely correct, "
        "then reply with only one digit (1-4).\n\n"
        f"[Problem]\n{question}\n\n[Candidates]\n{listed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--think-tokens", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    jobs = [job for job in jobs if int(job["question_idx"]) % args.num_shards == args.shard_id]
    args.out = args.out or (OUT / "olympiadbench_s42" / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if (job["question_idx"], job["decision_step"]) not in done_keys(args.out)]
    print(
        f"hc-think oly shard={args.shard_id}/{args.num_shards} "
        f"jobs={len(jobs)} pending={len(pending)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(JUDGE, trust_remote_code=True)
    digit_ids: dict[str, int] = {}
    for digit in DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if encoded:
            digit_ids[digit] = encoded[0]
    llm = LLM(
        model=JUDGE,
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        max_logprobs=16,
    )
    think_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.think_tokens,
        stop=["</think>"],
        include_stop_str_in_output=True,
    )
    pick_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=16, detokenize=False)
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            think_prompts: list[str] = []
            ready: list[dict[str, Any]] = []
            for job in chunk:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": build_user(job["question"], job["cands"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                n_tok = len(tokenizer.encode(chat, add_special_tokens=False))
                if n_tok + args.think_tokens + 8 > args.max_context:
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
                think_prompts.append(chat)
                ready.append({**job, "chat": chat})
            if not think_prompts:
                continue
            think_outs = llm.generate(think_prompts, think_params, use_tqdm=False)
            pick_prompts: list[str] = []
            pick_ready: list[dict[str, Any]] = []
            for job, out in zip(ready, think_outs, strict=True):
                think = out.outputs[0].text if out.outputs else ""
                closed = "</think>" in think
                prefix = job["chat"] + think
                if not closed:
                    prefix = prefix + "</think>\n\n"
                elif not prefix.endswith("\n"):
                    prefix = prefix + "\n"
                pick_prompts.append(prefix)
                pick_ready.append({**job, "think": think, "closed": closed, "think_n": len(think)})
            pick_outs = llm.generate(pick_prompts, pick_params, use_tqdm=False)
            for job, out in zip(pick_ready, pick_outs, strict=True):
                first = out.outputs[0].logprobs[0] if out.outputs and out.outputs[0].logprobs else {}
                mapping = {int(tok): logprob_value(item) for tok, item in (first or {}).items()}
                lps: list[float] = []
                for digit in DIGITS[: len(job["cands"])]:
                    tid = digit_ids.get(digit)
                    lps.append(mapping[tid] if tid is not None and tid in mapping else float("nan"))
                z = logsumexp(lps)
                probs = [math.exp(lp - z) if math.isfinite(lp) and math.isfinite(z) else float("nan") for lp in lps]
                if any(math.isfinite(p) for p in probs):
                    best_i = max(range(len(probs)), key=lambda i: probs[i] if math.isfinite(probs[i]) else -1.0)
                else:
                    best_i = 0
                pick = job["cands"][best_i]
                rec = {
                    "status": "ok",
                    "question_idx": job["question_idx"],
                    "decision_step": job["decision_step"],
                    "trial_answer": job["trial_answer"],
                    "geo_conf": job.get("geo_conf"),
                    "cands": job["cands"],
                    "pick": pick,
                    "pick_i": best_i + 1,
                    "pick_p": probs[best_i] if best_i < len(probs) else float("nan"),
                    "probs": probs,
                    "pick_is_trial": bool(pick != NONE and rg.same(pick, job["trial_answer"])),
                    "pick_is_none": pick == NONE,
                    "n_cands": len(job["cands"]),
                    "closed": job["closed"],
                    "think_n": job["think_n"],
                    "think_tail": (job["think"] or "")[-80:],
                }
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
