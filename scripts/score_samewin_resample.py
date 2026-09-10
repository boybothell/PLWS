#!/usr/bin/env python3
"""Same-prefix resample: no text change, PUMA sampling, 7B short trial.

First same-answer window of each kind per question. T=0.6 / top_p=0.95 / top_k=30.
"""
from __future__ import annotations

import argparse
import json
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

from score_confcal_v1 import nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import replay_rescue_R_gate as rg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import score_samewin_reprobe as rp  # noqa: E402
from gen_trial_answers import (  # noqa: E402
    build_prompt,
    compute_confidence,
    extract_first_braced_content,
    extract_logprob_from_step,
    find_boxed_content_token_span,
    parse_response,
)
from prompt_utils import get_confident_ending, get_task_type  # noqa: E402

MODEL = rp.MODEL
OUT = AE / "results/samewin_resample"


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


def parse_draw(gen: Any, tokenizer: Any, old: str, job: dict[str, Any], draw_id: int) -> dict[str, Any]:
    text = gen.text
    token_ids = list(gen.token_ids)
    logps = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []
    boxed = extract_first_braced_content(text)
    answer = boxed if boxed else parse_response(text)
    start, end = find_boxed_content_token_span(token_ids, tokenizer)
    use = logps[start:end] if start < end else logps
    return {
        "draw_id": draw_id,
        "new_answer": answer,
        "new_text": text[:200],
        "new_conf": compute_confidence(use, "geometric"),
        "keep": bool(rg.same(old, answer)),
        "new_gold_ok": bool(low.credit(answer, job["gt"], job["original"], job["orig_ok"])),
        "n_ans_tok": len(use),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-draw", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    jobs = [j for j in rp.load_jobs(args.dataset, args.seed) if j["question_idx"] % args.num_shards == args.shard_id]
    args.out = args.out or (OUT / f"{args.dataset}_s{args.seed}" / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out)
    pending = [job for job in jobs if (job["question_idx"], job["decision_step"]) not in done]
    kinds = defaultdict(int)
    for job in pending:
        kinds[(job["kind"], job["gold_ok"])] += 1
    print(
        f"samewin-resample {args.dataset} shard={args.shard_id}/{args.num_shards} "
        f"pending={len(pending)} n_draw={args.n_draw} T={args.temperature} mix={dict(kinds)} -> {args.out}",
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
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n_draw,
        max_tokens=args.max_tokens,
        logprobs=5,
    )
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            prompts: list[str] = []
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
                prompts.append(prompt)
                ready.append(job)
            if not prompts:
                continue
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for job, out in zip(ready, outputs, strict=True):
                draws = [
                    parse_draw(gen, tokenizer, job["old_answer"], job, i)
                    for i, gen in enumerate(out.outputs)
                ]
                keep_n = sum(1 for d in draws if d["keep"])
                rec = {
                    "status": "ok",
                    "question_idx": job["question_idx"],
                    "decision_step": job["decision_step"],
                    "kind": job["kind"],
                    "gold_ok": job["gold_ok"],
                    "old_answer": job["old_answer"],
                    "geo_stored": job["geo_stored"],
                    "temperature": args.temperature,
                    "n_draw": len(draws),
                    "draws": draws,
                    "keep_n": keep_n,
                    "keep_all": keep_n == len(draws) and bool(draws),
                    "split": any(not d["keep"] for d in draws),
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
