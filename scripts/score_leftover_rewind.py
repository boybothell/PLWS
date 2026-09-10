#!/usr/bin/env python3
"""剩窗回退一枪：丢掉原轨迹下一步尖思路，从剩窗前缀再写，不加换路句。

不交、不灌、不改提示。只看重写后答变不变、金标变不变。尖/不尖事后用 next_ent_mean 切。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_LENS_DISABLE", "1")

AE = Path(__file__).resolve().parents[1]
PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(PUMA / "puma"))

from score_confcal_keytoken import INSTRUCTION  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402
from score_leftover_jump import MODEL, done_uids, load_jobs  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import replay_rescue_R_gate as rg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
from gen_trial_answers import (  # noqa: E402
    build_prompt,
    compute_confidence,
    extract_first_braced_content,
    extract_logprob_from_step,
    find_boxed_content_token_span,
    parse_response,
)
from prompt_utils import get_confident_ending, get_task_type  # noqa: E402

OUT = AE / "results/leftover_rewind"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--think-tokens", type=int, default=1024)
    parser.add_argument("--trial-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.jobs and args.jobs.is_file():
        jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    else:
        jobs = load_jobs(args.model_tag)
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_id]
    args.out = args.out or (OUT / f"{args.model_tag}_s42" / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if job["uid"] not in done_uids(args.out)]
    print(
        f"leftover-rewind {args.model_tag} shard={args.shard_id}/{args.num_shards} "
        f"jobs={len(jobs)} pending={len(pending)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wait_id = tokenizer.encode("Wait", add_special_tokens=False)
    wait_id = wait_id[0] if len(wait_id) == 1 else tokenizer.convert_tokens_to_ids("Wait")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    think_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=30,
        max_tokens=args.think_tokens,
        stop=["</think>"],
        stop_token_ids=[wait_id],
        include_stop_str_in_output=False,
    )
    trial_params = SamplingParams(temperature=0.0, max_tokens=args.trial_tokens, logprobs=5)
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            think_prompts: list[str] = []
            ready: list[dict[str, Any]] = []
            for job in chunk:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{INSTRUCTION}\n{job['question']}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                text = chat + job["thought"]
                n_tok = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tok + args.think_tokens + args.trial_tokens + 64 > args.max_context:
                    handle.write(json.dumps({"uid": job["uid"], "status": "too_long"}) + "\n")
                    continue
                think_prompts.append(text)
                ready.append(job)
            if not think_prompts:
                continue
            think_outs = llm.generate(think_prompts, think_params, use_tqdm=False)
            trial_prompts: list[str] = []
            trial_ready: list[tuple[dict[str, Any], str, int]] = []
            for job, out in zip(ready, think_outs, strict=True):
                cont = out.outputs[0].text if out.outputs else ""
                new_thought = job["thought"] + cont
                ending = get_confident_ending(job["dataset"], is_trial=True)
                task = get_task_type(job["dataset"])
                trial = build_prompt(
                    tokenizer,
                    MODEL,
                    job["question"],
                    new_thought,
                    task,
                    "default",
                    True,
                    job["dataset"],
                    confident_ending=ending,
                )
                n_tok = len(tokenizer.encode(trial, add_special_tokens=False))
                if n_tok + args.trial_tokens > args.max_context:
                    handle.write(json.dumps({"uid": job["uid"], "status": "too_long", "cont_n": len(cont)}) + "\n")
                    continue
                trial_prompts.append(trial)
                trial_ready.append((job, cont, n_tok))
            if not trial_prompts:
                continue
            trial_outs = llm.generate(trial_prompts, trial_params, use_tqdm=False)
            for (job, cont, n_tok), out in zip(trial_ready, trial_outs, strict=True):
                gen = out.outputs[0]
                text = gen.text
                token_ids = list(gen.token_ids)
                logps = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []
                boxed = extract_first_braced_content(text)
                answer = boxed if boxed else parse_response(text)
                start, end = find_boxed_content_token_span(token_ids, tokenizer)
                use = logps[start:end] if start < end else logps
                rec = {
                    "status": "ok",
                    "uid": job["uid"],
                    "dataset": job["dataset"],
                    "question_idx": job["question_idx"],
                    "left_step": job["left_step"],
                    "kind": job["kind"],
                    "confidence": job.get("confidence"),
                    "left_ok": job["left_ok"],
                    "wait_helps": job["wait_helps"],
                    "will_change": job["will_change"],
                    "never_high": job["never_high"],
                    "same_as_high": job["same_as_high"],
                    "host_ok": job["host_ok"],
                    "old_answer": job["old_answer"],
                    "new_answer": answer,
                    "keep": bool(rg.same(job["old_answer"], answer)),
                    "new_gold_ok": bool(low.credit(answer, job["gt"], job["original"], job["orig_ok"])),
                    "new_conf": compute_confidence(use, "geometric"),
                    "next_ent_mean": job.get("next_ent_mean"),
                    "cont_n": len(cont),
                    "n_prompt": n_tok,
                    "new_text": (cont[-120:] + " | " + text[:80])[:200],
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
