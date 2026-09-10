#!/usr/bin/env python3
"""剩窗换路一枪：低把握同答窗后插入 ASAG jump，再写一段思路并探新试答。

不交、不灌。只看跳完答变不变、金标变不变。尖/不尖事后用 next_ent_mean 切。
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

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import report_first_lock_room as room  # noqa: E402
import report_k4_hyps as hy  # noqa: E402
import report_k4_second_lock as sl  # noqa: E402
import report_leftover_after as after  # noqa: E402
from gen_trial_answers import (  # noqa: E402
    build_prompt,
    compute_confidence,
    extract_first_braced_content,
    extract_logprob_from_step,
    find_boxed_content_token_span,
    parse_response,
)
from prompt_utils import get_confident_ending, get_task_type  # noqa: E402

MODEL = "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"
OUT = AE / "results/leftover_jump"
PIVOT = AE / "results/leftover_token_pivot"
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
JUMP = (
    "Wait, my previous reasoning is not correct. I should adopt a more concise "
    "and different approach to reexamine this problem.\n\n"
)


def load_pivot_ent(model: str) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for path in sorted(PIVOT.glob(f"{model}_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            try:
                out[(str(row["dataset"]), int(row["question_idx"]))] = float(row["next_ent_mean"])
            except Exception:
                continue
    return out


def load_jobs(model: str) -> list[dict[str, Any]]:
    ent = load_pivot_ent(model)
    jobs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        xs = after.load_left(model, dataset, 42)
        trials_path = room.dense_trial_path(model, dataset, seed=42)
        official_path = dd.puma_stat_path(model, dataset, 42)
        official = {int(r["question_idx"]): r for r in dd.load_json(official_path)} if official_path.is_file() else {}
        gp = low.gpath(model, dataset, 42)
        gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
        by: dict[int, list[dict[str, Any]]] = {}
        for row in dd.load_json(trials_path):
            by.setdefault(int(row["question_idx"]), []).append(row)
        for rec in xs:
            if rec.get("kind") != "low":
                continue
            trials = by.get(int(rec["question_idx"]))
            if not trials:
                continue
            rows = rg.usable_rows(trials)
            wins = sl.same_windows(rows)
            left = next((win for win in wins if hy.leftover(win)), None)
            if left is None:
                continue
            left_row = rows[left["end"]]
            thought = str(left_row.get("reasoning_prefix") or "")
            question = str(left_row.get("question") or "")
            if not thought or not question:
                continue
            info = official.get(int(rec["question_idx"])) or {}
            g = gmap.get(int(rec["question_idx"])) or {}
            last = max(trials, key=lambda x: int(x["stopped_len"]))
            gt = info.get("ground_truth") or g.get("ground_truth")
            original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
            orig_ok = (
                bool(info.get("original_correct"))
                if "original_correct" in info
                else bool(low.credit(original, gt, original, True))
            )
            jobs.append(
                {
                    "uid": f"{model}:{dataset}:42:{rec['question_idx']}",
                    "model": model,
                    "dataset": dataset,
                    "question_idx": int(rec["question_idx"]),
                    "left_step": int(rec["left_step"]),
                    "kind": rec["kind"],
                    "confidence": rec.get("left_c"),
                    "left_ok": rec["left_ok"],
                    "wait_helps": rec["wait_helps"],
                    "will_change": rec["will_change"],
                    "never_high": rec["never_high"],
                    "same_as_high": rec["same_as_high"],
                    "host_ok": rec["host_ok"],
                    "old_answer": str(left.get("ans") or ""),
                    "next_ent_mean": ent.get((dataset, int(rec["question_idx"]))),
                    "question": question,
                    "thought": thought,
                    "gt": gt,
                    "original": original,
                    "orig_ok": orig_ok,
                }
            )
    jobs.sort(key=lambda x: (x["dataset"], x["question_idx"]))
    return jobs


def done_uids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") in {"ok", "too_long"} and row.get("uid") is not None:
            out.add(str(row["uid"]))
    return out


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
        f"leftover-jump {args.model_tag} shard={args.shard_id}/{args.num_shards} "
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
                text = chat + job["thought"] + JUMP
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
                new_thought = job["thought"] + JUMP + cont
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
                    handle.write(
                        json.dumps(
                            {
                                "uid": job["uid"],
                                "status": "too_long",
                                "cont_n": len(cont),
                            }
                        )
                        + "\n"
                    )
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
