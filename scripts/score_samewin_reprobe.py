#!/usr/bin/env python3
"""Mask then re-probe: 7B PUMA trial ending, first same-answer window per question.

No extra judge. After hiding a region, generate a new boxed answer.
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

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import report_first_lock_room as room  # noqa: E402
import report_k4_second_lock as sl  # noqa: E402
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
OUT = AE / "results/samewin_reprobe"
PLACE = "[MASK]"
MASKS = ("none", "hide_problem", "hide_cot", "hide_last")


def last_increment(current: str, prior: str) -> str:
    if prior and current.startswith(prior):
        return current[len(prior) :]
    return current


def variants(question: str, current: str, prior: str) -> dict[str, tuple[str, str]]:
    return {
        "none": (question, current),
        "hide_problem": (PLACE, current),
        "hide_cot": (question, PLACE),
        "hide_last": (question, prior if prior.strip() else PLACE),
    }


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
        seen_kind: set[str] = set()
        for win in sl.same_windows(rows):
            if win["kind"] in seen_kind:
                continue
            seen_kind.add(win["kind"])
            row = rows[win["end"]]
            prior = str(rows[win["end"] - 1].get("reasoning_prefix") or "") if win["end"] else ""
            jobs.append(
                {
                    "question_idx": qi,
                    "decision_step": int(win["step"]),
                    "kind": win["kind"],
                    "gold_ok": bool(low.credit(win["ans"], gt, original, orig_ok)),
                    "old_answer": str(win["ans"] or ""),
                    "geo_stored": float(row.get("confidence") or float("nan")),
                    "question": str(row.get("question") or ""),
                    "current": str(row.get("reasoning_prefix") or ""),
                    "prior": prior,
                    "gt": gt,
                    "original": original,
                    "orig_ok": orig_ok,
                }
            )
    jobs.sort(key=lambda x: x["question_idx"])
    return jobs


def done_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.is_file():
        return set()
    out: set[tuple[int, int, str]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            out.add((int(row["question_idx"]), int(row["decision_step"]), str(row["mask"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    jobs = [j for j in load_jobs(args.dataset, args.seed) if j["question_idx"] % args.num_shards == args.shard_id]
    args.out = args.out or (OUT / f"{args.dataset}_s{args.seed}" / f"scores_shard{args.shard_id}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out)
    pending: list[tuple[dict[str, Any], str]] = []
    for job in jobs:
        for mask in MASKS:
            if (job["question_idx"], job["decision_step"], mask) not in done:
                pending.append((job, mask))
    kinds = defaultdict(int)
    for job in jobs:
        kinds[(job["kind"], job["gold_ok"])] += 1
    print(
        f"samewin-reprobe {args.dataset} shard={args.shard_id}/{args.num_shards} "
        f"q={len(jobs)} pending={len(pending)} mix={dict(kinds)} -> {args.out}",
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
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5)
    started = time.perf_counter()
    wrote = 0
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), args.batch_size):
            chunk = pending[begin : begin + args.batch_size]
            prompts: list[str] = []
            ready: list[tuple[dict[str, Any], str]] = []
            for job, mask in chunk:
                question, reasoning = variants(job["question"], job["current"], job["prior"])[mask]
                prompt = build_prompt(
                    tokenizer,
                    MODEL,
                    question,
                    reasoning,
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
                                "mask": mask,
                            }
                        )
                        + "\n"
                    )
                    continue
                prompts.append(prompt)
                ready.append((job, mask))
            if not prompts:
                continue
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for (job, mask), out in zip(ready, outputs, strict=True):
                gen = out.outputs[0]
                text = gen.text
                token_ids = list(gen.token_ids)
                logps = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []
                boxed = extract_first_braced_content(text)
                answer = boxed if boxed else parse_response(text)
                start, end = find_boxed_content_token_span(token_ids, tokenizer)
                use = logps[start:end] if start < end else logps
                keep = rg.same(job["old_answer"], answer)
                rec = {
                    "status": "ok",
                    "question_idx": job["question_idx"],
                    "decision_step": job["decision_step"],
                    "kind": job["kind"],
                    "gold_ok": job["gold_ok"],
                    "old_answer": job["old_answer"],
                    "geo_stored": job["geo_stored"],
                    "mask": mask,
                    "new_answer": answer,
                    "new_text": text[:200],
                    "new_conf": compute_confidence(use, "geometric"),
                    "keep": bool(keep),
                    "new_gold_ok": bool(low.credit(answer, job["gt"], job["original"], job["orig_ok"])),
                    "n_ans_tok": len(use),
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
