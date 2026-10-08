#!/usr/bin/env python3
"""Probe trial answers on the post-</think> suffix of saved generations.

OlympiadBench R1-7B seed 42 only. Thinking-side probes are left untouched;
this writes a separate post_think tree for fullcot and count-bias.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if "PLWS_ROOT" in __import__("os").environ:
    ROOT = Path(__import__("os").environ["PLWS_ROOT"]).resolve()
PUMA = ROOT / "tmp" / "PUMA" / "puma"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PUMA))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault(
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
    str(ROOT / "tmp" / "plws_step_probe" / "flashinfer_autotune"),
)

from gen_trial_answers import (  # noqa: E402
    build_prompt,
    compute_confidence,
    extract_first_braced_content,
    extract_logprob_from_step,
    find_boxed_content_token_span,
    parse_response,
)
from plws.grading import grade  # noqa: E402
from prompt_utils import get_task_type  # noqa: E402
from run_r1_7b_seed42 import continuation_ends, done_keys, load_jsonl  # noqa: E402
from separate_steps import separate_steps  # noqa: E402

MODEL = "/ssd/share/models/DeepSeek-R1-Distill-Qwen-7B"
SEED = 42
MAX_TOKENS = 30
DATASET = "olympiadbench"
JOBS = (
    ROOT
    / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b"
    / DATASET
    / f"seed_{SEED}"
    / "jobs/firstwin.jsonl"
)
FULLCOT_ANSWERS = ROOT / "samples/r1_7b" / DATASET / f"seed_{SEED}" / "answers.json"
COUNT_BIAS_SCORES = (
    ROOT
    / "results/runs/plws/count_bias/rho_0p98/r1_7b"
    / DATASET
    / f"seed_{SEED}"
)
OUT_ROOT = ROOT / "tmp/plws_step_probe/post_think"


def post_think_probes(protocol: str) -> list[dict]:
    jobs = load_jsonl(JOBS)
    if protocol == "fullcot":
        answers = json.loads(FULLCOT_ANSWERS.read_text())
        texts: dict[str, str] = {}
        for job in jobs:
            answer = answers[int(job["question_idx"]) - 1]
            if answer["question"].strip() != job["question"].strip():
                raise RuntimeError(f"question mismatch {job['uid']}")
            texts[job["uid"]] = answer["generated_text"]
        out_dir = OUT_ROOT / "fullcot" / "r1_7b" / DATASET / f"seed_{SEED}"
    elif protocol == "count-bias":
        texts = {}
        for path in sorted(COUNT_BIAS_SCORES.glob("shard_*.jsonl")):
            for row in load_jsonl(path):
                texts[row["uid"]] = row["generated_text"]
        out_dir = OUT_ROOT / "count_bias" / "rho_0p98" / "r1_7b" / DATASET / f"seed_{SEED}"
    else:
        raise SystemExit(f"unknown protocol {protocol}")

    found: list[dict] = []
    for job in jobs:
        key_uid = job["uid"]
        text = texts[key_uid]
        if "</think>" not in text:
            raise RuntimeError(f"missing </think> {key_uid}")
        think, post = text.split("</think>", 1)
        chunks = [chunk for chunk in separate_steps(post) if chunk.strip()]
        if not chunks:
            raise RuntimeError(f"empty post-</think> {key_uid}")
        ends = continuation_ends(post, chunks)
        gold = job.get("gt")
        if not gold:
            raise RuntimeError(f"missing gold {key_uid}")
        for rel, end in enumerate(ends, start=1):
            found.append(
                {
                    "uid": key_uid,
                    "dataset": DATASET,
                    "seed": SEED,
                    "question_idx": int(job["question_idx"]),
                    "question": job["question"],
                    "gt": gold,
                    "left_step": int(job["left_step"]),
                    "rel_step": rel,
                    "abs_step": int(job["left_step"]) + rel,
                    "phase": "post_think",
                    "protocol": protocol,
                    "reasoning_prefix": think + "</think>" + post[:end],
                    "out_dir": out_dir,
                }
            )
    return found


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--protocol",
        choices=("fullcot", "count-bias", "both"),
        default="both",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")

    protocols = ("fullcot", "count-bias") if args.protocol == "both" else (args.protocol,)
    indexed: list[dict] = []
    for protocol in protocols:
        indexed.extend(post_think_probes(protocol))

    pending: list[dict] = []
    seen: dict[str, set[tuple[str, int]]] = {}
    for index, row in enumerate(indexed):
        if index % args.num_shards != args.shard_id:
            continue
        directory: Path = row["out_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        seen_key = f"{row['protocol']}:{row['dataset']}:{row['seed']}"
        if seen_key not in seen:
            seen[seen_key] = done_keys(directory)
        if (row["uid"], row["rel_step"]) in seen[seen_key]:
            continue
        name = (
            "trials.jsonl"
            if args.num_shards == 1
            else f"trials.shard_{args.shard_id}.jsonl"
        )
        row["out"] = str(directory / name)
        pending.append(row)
    print(
        f"[post] shard={args.shard_id}/{args.num_shards} "
        f"pending={len(pending)}/{len(indexed)} protocols={','.join(protocols)}",
        flush=True,
    )
    if not pending or args.dry_run:
        print("[post] nothing pending" if not pending else "[post] dry-run", flush=True)
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from flashinfer_guard import apply_flashinfer_guard
    from vllm_shutdown import shutdown_llm

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "38000"))
    ready = []
    handles = {}
    for row in pending:
        task = get_task_type(row["dataset"])
        prompt = build_prompt(
            tokenizer,
            MODEL,
            row["question"],
            row["reasoning_prefix"],
            task,
            "default",
            True,
            row["dataset"],
        )
        n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
        out = Path(row["out"])
        if out not in handles:
            handles[out] = out.open("a", encoding="utf-8")
        if n_tok + MAX_TOKENS > max_len:
            rec = {
                "status": "too_long",
                "uid": row["uid"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "question_idx": row["question_idx"],
                "left_step": row["left_step"],
                "rel_step": row["rel_step"],
                "abs_step": row["abs_step"],
                "phase": "post_think",
                "protocol": row["protocol"],
                "n_prompt": n_tok,
            }
            handles[out].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles[out].flush()
            print(
                f"[post] too_long {row['uid']} t={row['rel_step']} n={n_tok}",
                flush=True,
            )
            continue
        row["prompt"] = prompt
        row["task"] = task
        ready.append(row)
    print(f"[post] generate {len(ready)} skip_long {len(pending) - len(ready)}", flush=True)
    if not ready:
        for handle in handles.values():
            handle.close()
        return

    llm_kwargs = {
        "model": MODEL,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "max_model_len": max_len,
        "gpu_memory_utilization": float(
            os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.78")
        ),
        "seed": SEED,
    }
    apply_flashinfer_guard(llm_kwargs, log=print)
    llm = LLM(**llm_kwargs)
    started = time.perf_counter()
    wrote = 0
    try:
        for begin in range(0, len(ready), 16):
            chunk = ready[begin : begin + 16]
            params = SamplingParams(
                temperature=0.6,
                max_tokens=MAX_TOKENS,
                top_p=0.95,
                top_k=30,
                logprobs=1,
                seed=SEED,
            )
            generated = llm.generate([row["prompt"] for row in chunk], params)
            for row, out in zip(chunk, generated, strict=True):
                gen = out.outputs[0]
                text = gen.text
                token_ids = list(gen.token_ids)
                logps = (
                    [extract_logprob_from_step(step) for step in gen.logprobs]
                    if gen.logprobs
                    else []
                )
                boxed = extract_first_braced_content(text)
                answer = boxed if boxed else parse_response(text)
                start, end = find_boxed_content_token_span(token_ids, tokenizer)
                use = logps[start:end] if start < end else logps
                ok, err = grade(answer, row["gt"])
                rec = {
                    "status": "ok",
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "question_idx": row["question_idx"],
                    "left_step": row["left_step"],
                    "rel_step": row["rel_step"],
                    "abs_step": row["abs_step"],
                    "phase": "post_think",
                    "protocol": row["protocol"],
                    "final_answer": answer,
                    "gt": row["gt"],
                    "gold_ok": ok,
                    "gold_error": err,
                    "confidence": compute_confidence(use, "geometric"),
                    "model_response": text,
                    "count_generated_tokens": len(token_ids),
                }
                handle = handles[Path(row["out"])]
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                wrote += 1
            print(
                f"[post] wrote={wrote}/{len(ready)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()
        shutdown_llm(llm)
    print(f"[post] done wrote={wrote}", flush=True)


if __name__ == "__main__":
    main()
