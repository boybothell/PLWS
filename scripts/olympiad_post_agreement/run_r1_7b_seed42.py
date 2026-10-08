#!/usr/bin/env python3
"""PLWS post-window trial answers for R1-7B seed 42.

t=0 is the same-answer window end and is not re-probed. Each later step is
the exact generated prefix through that continuation step, then the same
trial-answer setup as dense trials (no CORE ban, boxed ending, 30 tokens).
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
from separate_steps import separate_steps  # noqa: E402

MODEL = "/ssd/share/models/DeepSeek-R1-Distill-Qwen-7B"
SEED = 42
MAX_TOKENS = 30
OUT_ROOT = ROOT / "tmp" / "plws_step_probe" / "r1_7b"

CELLS = (
    {
        "dataset": "amc23",
        "seed": 42,
        "scores": (
            ROOT
            / "transfer/suppress_firstwin_all_models_20261005/legacy_amc_fallback/r1_7b/amc23/seed_42/amc23.jsonl",
        ),
        "jobs": ROOT
        / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b/amc23/seed_42/jobs/firstwin.jsonl",
    },
    {
        "dataset": "aime25",
        "seed": 42,
        "scores": (
            ROOT
            / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b/aime25/seed_42/scores/firstwin/shard_0.jsonl",
        ),
        "jobs": ROOT
        / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b/aime25/seed_42/jobs/firstwin.jsonl",
    },
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _norm_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    ends: list[int] = []
    index = 0
    size = len(text)
    while index < size:
        if text[index].isspace():
            nxt = index + 1
            while nxt < size and text[nxt].isspace():
                nxt += 1
            chars.append(" ")
            ends.append(nxt)
            index = nxt
        else:
            chars.append(text[index])
            ends.append(index + 1)
            index += 1
    return "".join(chars), ends


def _locate(cont: str, chunk: str, pos: int) -> tuple[int, int]:
    stripped = chunk.strip()
    if not stripped:
        return -1, 0
    direct_lengths: list[int] = []
    for length in (80, 60, 40, 24, len(stripped)):
        if 0 < length <= len(stripped) and length not in direct_lengths:
            direct_lengths.append(length)
    for length in direct_lengths:
        key = stripped[:length]
        start = cont.find(key, pos)
        if start >= 0:
            return start, len(key)
    folded, ends = _norm_map(cont)
    needle, _ = _norm_map(stripped)
    norm_from = len(ends)
    for index, end in enumerate(ends):
        start = ends[index - 1] if index else 0
        if start >= pos:
            norm_from = index
            break
    lengths: list[int] = []
    for length in (120, 80, 40, 24, 16, len(needle)):
        if 0 < length <= len(needle) and length not in lengths:
            lengths.append(length)
    for length in lengths:
        key = needle[:length].strip()
        if len(key) < 8:
            continue
        found = folded.find(key, norm_from)
        if found < 0:
            continue
        orig_start = ends[found - 1] if found else 0
        orig_end = ends[found + len(key) - 1]
        if orig_start < pos:
            continue
        return orig_start, orig_end - orig_start
    return -1, 0


def continuation_ends(cont: str, chunks: list[str]) -> list[int]:
    pos = 0
    starts: list[tuple[int, int]] = []
    for chunk in chunks:
        start, key_len = _locate(cont, chunk, pos)
        if start < 0:
            raise RuntimeError("continuation step is not in the generated text")
        starts.append((start, key_len))
        pos = start + 1
    ends: list[int] = []
    for index, (start, key_len) in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1][0]
            if end <= start:
                raise RuntimeError("continuation steps overlap")
        else:
            end = len(cont)
        ends.append(end)
    return ends


def probes_for(cell: dict) -> list[dict]:
    scores: dict[str, dict] = {}
    for path in cell["scores"]:
        for row in load_jsonl(path):
            scores[row["uid"]] = row
    found = []
    for job in load_jsonl(cell["jobs"]):
        rec = scores[job["uid"]]
        text = rec["generated_text"]
        thought = job["thought"]
        if not text.startswith(thought):
            raise RuntimeError(f"window prefix mismatch {job['uid']}")
        think = text.split("</think>", 1)[0]
        cont = think[len(thought) :]
        chunks = [chunk for chunk in separate_steps(cont) if chunk.strip()]
        ends = continuation_ends(cont, chunks)
        gold = job.get("gt")
        if not gold:
            raise RuntimeError(f"missing gold {job['uid']}")
        for rel, end in enumerate(ends, start=1):
            found.append(
                {
                    "uid": job["uid"],
                    "dataset": cell["dataset"],
                    "seed": int(cell["seed"]),
                    "question_idx": int(job["question_idx"]),
                    "question": job["question"],
                    "gt": gold,
                    "left_step": int(job["left_step"]),
                    "rel_step": rel,
                    "abs_step": int(job["left_step"]) + rel,
                    "reasoning_prefix": thought + cont[:end],
                }
            )
    return found


def done_keys(directory: Path) -> set[tuple[str, int]]:
    found: set[tuple[str, int]] = set()
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("trials*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") in {"ok", "too_long"} and row.get("uid"):
                found.add((str(row["uid"]), int(row["rel_step"])))
    return found


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="dataset:seed, repeatable. Default keeps the AIME/AMC seed-42 cells.",
    )
    parser.add_argument(
        "--protocol",
        choices=("firstwin", "count-bias"),
        default="firstwin",
    )
    parser.add_argument("--rho", default="0.98")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")
    rho_tag = "rho_" + args.rho.removeprefix("rho_").replace(".", "p")
    out_root = (
        ROOT / "tmp/plws_step_probe/count_bias" / rho_tag / "r1_7b"
        if args.protocol == "count-bias"
        else OUT_ROOT
    )
    if args.cell:
        cells = []
        for spec in args.cell:
            dataset, seed_text = spec.split(":")
            seed = int(seed_text)
            jobs = (
                ROOT
                / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b"
                / dataset
                / f"seed_{seed}"
                / "jobs/firstwin.jsonl"
            )
            if args.protocol == "count-bias":
                score_dir = (
                    ROOT
                    / "results/runs/plws/count_bias"
                    / rho_tag
                    / "r1_7b"
                    / dataset
                    / f"seed_{seed}"
                )
                scores = tuple(sorted(score_dir.glob("shard_*.jsonl")))
            else:
                scores = tuple(
                    sorted(
                        (
                            ROOT
                            / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b"
                            / dataset
                            / f"seed_{seed}"
                            / "scores/firstwin"
                        ).glob("shard_*.jsonl")
                    )
                )
            if not scores or not jobs.is_file():
                raise SystemExit(f"missing jobs or scores for {spec}")
            cells.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "scores": scores,
                    "jobs": jobs,
                }
            )
    else:
        cells = list(CELLS)

    pending: list[dict] = []
    seen: dict[str, set[tuple[str, int]]] = {}
    indexed: list[dict] = []
    for cell in cells:
        indexed.extend(probes_for(cell))
    for index, row in enumerate(indexed):
        if index % args.num_shards != args.shard_id:
            continue
        directory = out_root / row["dataset"] / f"seed_{row['seed']}"
        directory.mkdir(parents=True, exist_ok=True)
        seen_key = f"{row['dataset']}:{row['seed']}"
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
        f"[probe] shard={args.shard_id}/{args.num_shards} pending={len(pending)}/{len(indexed)}",
        flush=True,
    )
    if not pending or args.dry_run:
        print("[probe] nothing pending" if not pending else "[probe] dry-run", flush=True)
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
                "n_prompt": n_tok,
            }
            handles[out].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles[out].flush()
            print(
                f"[probe] too_long {row['uid']} t={row['rel_step']} n={n_tok}",
                flush=True,
            )
            continue
        row["prompt"] = prompt
        row["task"] = task
        ready.append(row)
    print(f"[probe] generate {len(ready)} skip_long {len(pending) - len(ready)}", flush=True)
    if not ready:
        for handle in handles.values():
            handle.close()
        return

    llm_kwargs = {
        "model": MODEL,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "max_model_len": max_len,
        "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.78")),
        "seed": SEED,
    }
    apply_flashinfer_guard(llm_kwargs, log=print)
    llm = LLM(**llm_kwargs)
    started = time.perf_counter()
    wrote = 0
    try:
        for begin in range(0, len(ready), 16):
            chunk = ready[begin : begin + 16]
            by_seed: dict[int, list] = {}
            for row in chunk:
                by_seed.setdefault(int(row["seed"]), []).append(row)
            outputs_by_id = {}
            for seed, rows in by_seed.items():
                params = SamplingParams(
                    temperature=0.6,
                    max_tokens=MAX_TOKENS,
                    top_p=0.95,
                    top_k=30,
                    logprobs=1,
                    seed=seed,
                )
                generated = llm.generate([row["prompt"] for row in rows], params)
                for row, out in zip(rows, generated, strict=True):
                    outputs_by_id[id(row)] = out
            for row in chunk:
                out = outputs_by_id[id(row)]
                gen = out.outputs[0]
                text = gen.text
                token_ids = list(gen.token_ids)
                logps = [
                    extract_logprob_from_step(step) for step in gen.logprobs
                ] if gen.logprobs else []
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
                f"[probe] wrote={wrote}/{len(ready)} elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()
        shutdown_llm(llm)
    print(f"[probe] done wrote={wrote}", flush=True)


if __name__ == "__main__":
    main()
