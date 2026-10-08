#!/usr/bin/env python3
"""Replay canonical -inf PLWS after its own </think>.

Keeps the saved thinking. Appends ``The final answer is \\boxed`` and generates
one token at a time until the outer closing brace, which stays in the output.
If that brace never arrives, generation stops at the question's original answer
budget (natural close: leftover of the 32K host budget; forced close: 2048).
The cue tokens count against that budget.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_LENS_DISABLE", "1")

ROOT = Path(os.environ.get("PLWS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

from plws.flashinfer_guard import apply_flashinfer_guard  # noqa: E402
from plws.grading import grade, require_grader  # noqa: E402
from plws.inputs import same_answer  # noqa: E402
from plws.piece_text import sanitize_tokenizer_pieces  # noqa: E402
from plws.protocol import (  # noqa: E402
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    TRUNCATED_ANSWER_FIX_TOKENS,
)
from plws.runtime import load_dotenv, model_path, nvidia_ld_library_path  # noqa: E402

load_dotenv(ROOT)
os.environ["LD_LIBRARY_PATH"] = nvidia_ld_library_path()
PUMA = Path(os.environ.get("PUMA_ROOT", ROOT / "tmp" / "PUMA")).resolve()
sys.path.insert(0, str(PUMA / "puma"))

CLOSE = "\n</think>\n\n"
CUE = "The final answer is \\boxed"
CELL = ROOT / "results/runs/plws/window_first/k_4/lexicon_core"
COUNT = ROOT / "results/runs/plws/count_bias/rho_0p98"


def boxed_close_keep_length(texts: list[str]) -> int | None:
    """Tokens to keep through the outer ``}``, or None if the box is not closed.

    A ``{`` and its matching ``}`` inside one token do not stop (``{27}``,
    ``{n}``). An empty ``{}`` does not stop. Nested inner braces do not stop.
    Content may sit in the opening token (``{n`` then ``}``) or in the closing
    token (``{`` then ``5}``). The token that produces the matching ``}`` is kept.
    """

    depth = 0
    opened_at: int | None = None
    content = False
    for index, text in enumerate(texts):
        for char in text:
            if char == "{":
                if depth == 0:
                    opened_at = index
                    content = False
                depth += 1
            elif char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and opened_at is not None and opened_at < index and content:
                    return index + 1
                if depth == 0:
                    opened_at = None
                    content = False
            elif depth > 0:
                content = True
    return None


def shutdown_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    client = getattr(engine, "engine_core", None)
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=10)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def jobs_by_uid(dataset: str, seed: int, model_tag: str) -> dict[str, dict[str, Any]]:
    path = CELL / model_tag / dataset / f"seed_{seed}" / "jobs" / "firstwin.jsonl"
    return {str(row["uid"]): row for row in load_jsonl(path)}


def legacy_score_rows(dataset: str, seed: int, model_tag: str) -> list[dict[str, Any]]:
    """AMC cells that live in leftover_suppress_toend, split by window kind."""

    root = (
        ROOT
        / "results/archive/legacy_layout/plws/leftover_suppress_toend"
    )
    rows: list[dict[str, Any]] = []
    for suffix in ("", "_mix", "_high"):
        folder = root / f"{model_tag}_s{seed}_suppress{suffix}"
        for path in sorted(folder.glob("scores_shard*.jsonl")):
            for row in load_jsonl(path):
                if row.get("status") == "ok" and row.get("dataset") in (None, dataset):
                    rows.append(row)
    return rows


def score_rows(
    dataset: str, seed: int, model_tag: str, kind: str = "firstwin"
) -> list[dict[str, Any]]:
    if kind == "count":
        folder = COUNT / model_tag / dataset / f"seed_{seed}"
        rows = []
        for path in sorted(folder.glob("shard_*.jsonl")):
            rows.extend(load_jsonl(path))
        return [row for row in rows if row.get("status") == "ok"]
    folder = CELL / model_tag / dataset / f"seed_{seed}" / "scores" / "firstwin"
    rows = []
    for path in sorted(folder.glob("shard_*.jsonl")):
        rows.extend(load_jsonl(path))
    ok = [row for row in rows if row.get("status") == "ok"]
    if ok:
        return ok
    return legacy_score_rows(dataset, seed, model_tag)


def done_uids(path: Path) -> set[str]:
    return {str(row["uid"]) for row in load_jsonl(path) if row.get("status") == "ok"}


def unclosed_uids(path: Path) -> set[str]:
    return {
        str(row["uid"])
        for row in load_jsonl(path)
        if row.get("status") == "ok" and not row.get("box_closed")
    }


def natural_close(row: dict[str, Any]) -> bool:
    if row.get("natural_close") is not None:
        return bool(row.get("natural_close"))
    # Legacy AMC rows omit the flag. A stop that did not hit the cap is the
    # model's own </think>.
    if row.get("hit_cap"):
        return False
    return str(row.get("finish_reason") or "") == "stop"


def original_answer_budget(row: dict[str, Any], close_tokens: int) -> int:
    stored = int(row.get("answer_budget") or 0)
    if stored > 0:
        return stored
    if natural_close(row):
        return (
            FULLCOT_GENERATION_TOKENS
            - int(row.get("n_left_tok") or 0)
            - int(row.get("n_cont_tok") or 0)
            - close_tokens
        )
    return TRUNCATED_ANSWER_FIX_TOKENS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--datasets", default="math-500,gpqa-diamond")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds. Empty means every seed on disk.")
    parser.add_argument(
        "--seed-override",
        action="append",
        default=[],
        help="dataset=seed,seed replaces --seeds for that dataset.",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help="Extra ceiling on generated tokens. 0 uses the original answer budget.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--only-unclosed",
        action="store_true",
        help="Replay rows whose saved boxed-close output never matched the outer brace.",
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Appended to seed_{n}.jsonl. Defaults to .fallback with --only-unclosed.",
    )
    parser.add_argument(
        "--score-kind",
        choices=("firstwin", "count"),
        default="firstwin",
        help="firstwin reads 窗后压 shards. count reads rho 0.98 随频次 shards.",
    )
    parser.add_argument(
        "--out-name",
        default="",
        help="Subdirectory under tmp/boxed_close. Default follows --score-kind.",
    )
    args = parser.parse_args()
    require_grader()

    from transformers import AutoTokenizer, GenerationConfig
    from prompt_utils import get_task_type
    from run_vllm import build_prompts_with_chat_template, extract_answer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from plws.deploy import configured_tensor_parallel

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    wanted_seeds = {int(item) for item in args.seeds.split(",") if item.strip()}
    seed_overrides: dict[str, set[int]] = {}
    for item in args.seed_override:
        name, _, raw = item.partition("=")
        seed_overrides[name.strip()] = {
            int(part) for part in raw.split(",") if part.strip()
        }
    weights = str(model_path(args.model_tag))
    tp = configured_tensor_parallel(args.model_tag)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    n_visible = len([item for item in visible.split(",") if item.strip()]) if visible else 0
    if n_visible != tp:
        raise SystemExit(
            f"GPU lane width {n_visible} != TP={tp} for {args.model_tag}; "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=True)
    config = GenerationConfig.from_pretrained(weights, trust_remote_code=True)
    temperature = float(getattr(config, "temperature", 0.6) or 0.6)
    top_p = float(getattr(config, "top_p", 0.95) or 0.95)
    top_k = getattr(config, "top_k", -1)
    if top_k is None:
        top_k = -1
    cue_tokens = len(tokenizer.encode(CUE, add_special_tokens=False))
    close_tokens = len(tokenizer.encode(CLOSE, add_special_tokens=False))
    eos_id = tokenizer.eos_token_id
    suffix = args.out_suffix
    if args.only_unclosed and not suffix:
        suffix = ".fallback"
    out_name = args.out_name or (
        "count_rho098" if args.score_kind == "count" else "firstwin_neginf"
    )
    out_root = ROOT / "tmp" / "boxed_close" / out_name / args.model_tag
    out_root.mkdir(parents=True, exist_ok=True)

    pending: list[dict[str, Any]] = []
    skipped_job = 0
    for dataset in datasets:
        if args.score_kind == "count":
            seed_root = COUNT / args.model_tag / dataset
        else:
            seed_root = CELL / args.model_tag / dataset
        if not seed_root.is_dir():
            raise SystemExit(f"missing cell {seed_root}")
        seeds_for = seed_overrides.get(dataset, wanted_seeds)
        for seed_dir in sorted(seed_root.glob("seed_*")):
            seed = int(seed_dir.name.split("_", 1)[1])
            if seeds_for and seed not in seeds_for:
                continue
            primary = out_root / dataset / f"seed_{seed}.jsonl"
            out = out_root / dataset / f"seed_{seed}{suffix}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            seen = done_uids(out)
            wanted = unclosed_uids(primary) if args.only_unclosed else None
            if wanted is not None and not wanted:
                continue
            jobs = jobs_by_uid(dataset, seed, args.model_tag)
            for row in score_rows(dataset, seed, args.model_tag, args.score_kind):
                uid = str(row["uid"])
                if wanted is not None and uid not in wanted:
                    continue
                if uid in seen:
                    continue
                job = jobs.get(uid)
                if job is None:
                    skipped_job += 1
                    print(f"skip score without job {uid}", flush=True)
                    continue
                pending.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "out": out,
                        "row": row,
                        "job": job,
                    }
                )
    print(
        f"boxed-close {args.model_tag} kind={args.score_kind} pending={len(pending)} "
        f"temp={temperature} top_p={top_p} top_k={top_k} cue_tokens={cue_tokens} "
        f"suffix={suffix!r} max_new={args.max_new} skipped_job={skipped_job}",
        flush=True,
    )
    if not pending:
        return

    # A later TP engine that reuses the shared MoE autotune cache hangs:
    # rank 0 cache-hits while the other ranks profile. Default off, same as
    # the 随频次 launcher. Set VLLM_ENABLE_FLASHINFER_AUTOTUNE=1 to force it.
    os.environ.setdefault("VLLM_ENABLE_FLASHINFER_AUTOTUNE", "0")
    os.environ.setdefault(
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        str(ROOT / "tmp" / "boxed_close" / "flashinfer_autotune"),
    )
    gpu_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
    max_batched = 32768
    if args.model_tag == "qwen3_30b_a3b":
        # MoE autotune asks for about 2 GiB after KV cache is reserved.
        # At 0.90 on 32 GiB cards that workspace does not fit.
        gpu_util = min(gpu_util, 0.78)
        max_batched = 8192
    print(
        f"engine {args.model_tag} gpu_util={gpu_util} max_batched={max_batched} tp={tp}",
        flush=True,
    )
    llm_kwargs: dict[str, Any] = {
        "model": weights,
        "trust_remote_code": True,
        "tensor_parallel_size": tp,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": gpu_util,
        "enable_prefix_caching": True,
        "max_num_seqs": 8,
        "max_num_batched_tokens": max_batched,
    }
    apply_flashinfer_guard(llm_kwargs, log=lambda msg: print(msg, flush=True))
    llm = LLM(**llm_kwargs)
    atexit.register(shutdown_llm, llm)
    started = time.perf_counter()
    wrote = 0
    for begin in range(0, len(pending), args.batch_size):
        chunk = pending[begin : begin + args.batch_size]
        ready: list[dict[str, Any]] = []
        handles = {item["out"]: item["out"].open("a", encoding="utf-8") for item in chunk}
        try:
            for item in chunk:
                row = item["row"]
                job = item["job"]
                text = str(row.get("generated_text") or "")
                if CLOSE not in text:
                    handles[item["out"]].write(
                        json.dumps({"uid": row["uid"], "status": "no_close", "dataset": item["dataset"], "seed": item["seed"]})
                        + "\n"
                    )
                    continue
                head = text.split(CLOSE, 1)[0]
                task = get_task_type(item["dataset"])
                chat = build_prompts_with_chat_template(
                    [job["question"]],
                    tokenizer,
                    weights,
                    task,
                    "default",
                )[0]
                prompt = chat + head + CLOSE + CUE
                n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
                n_prompt = n_tok
                budget = original_answer_budget(row, close_tokens)
                cap = min(max(0, budget - cue_tokens), MAX_MODEL_LEN - n_prompt)
                if args.max_new > 0:
                    cap = min(cap, args.max_new)
                if cap < 1:
                    handles[item["out"]].write(
                        json.dumps(
                            {
                                "uid": row["uid"],
                                "status": "too_long",
                                "dataset": item["dataset"],
                                "seed": item["seed"],
                                "n_prompt": n_prompt,
                                "answer_budget": budget,
                            }
                        )
                        + "\n"
                    )
                    continue
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                ready.append(
                    {
                        **item,
                        "task": task,
                        "n_prompt": n_prompt,
                        "prompt_ids": prompt_ids,
                        "cap": cap,
                        "answer_budget": budget,
                    }
                )
            if ready:
                generated: list[list[int]] = [[] for _ in ready]
                stopped = [False for _ in ready]
                closed_flags = [False for _ in ready]
                eos_flags = [False for _ in ready]
                step_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=int(top_k),
                    max_tokens=1,
                )
                limit = max(item["cap"] for item in ready)
                for step in range(limit):
                    active = [
                        index
                        for index, flag in enumerate(stopped)
                        if not flag and len(generated[index]) < ready[index]["cap"]
                    ]
                    if not active:
                        break
                    if step and step % 20 == 0:
                        print(
                            f"step {step} active={len(active)} "
                            f"max_gen={max(len(generated[index]) for index in active)}",
                            flush=True,
                        )
                    outputs = llm.generate(
                        [
                            TokensPrompt(
                                prompt_token_ids=ready[index]["prompt_ids"] + generated[index]
                            )
                            for index in active
                        ],
                        step_params,
                        use_tqdm=False,
                    )
                    for index, output in zip(active, outputs, strict=True):
                        gen = output.outputs[0] if output.outputs else None
                        piece_ids = list(gen.token_ids) if gen else []
                        finish = str(getattr(gen, "finish_reason", "") or "")
                        if eos_id is not None:
                            piece_ids = [token_id for token_id in piece_ids if token_id != eos_id]
                        if not piece_ids:
                            stopped[index] = True
                            eos_flags[index] = finish == "stop" or not gen
                            continue
                        generated[index].extend(piece_ids)
                        texts = [tokenizer.decode([token_id]) for token_id in generated[index]]
                        keep = boxed_close_keep_length(texts)
                        if keep is not None:
                            generated[index] = generated[index][:keep]
                            stopped[index] = True
                            closed_flags[index] = True
                        elif finish == "stop" or len(generated[index]) >= ready[index]["cap"]:
                            stopped[index] = True
                            eos_flags[index] = finish == "stop"
                for item, token_ids, closed, hit_eos in zip(
                    ready, generated, closed_flags, eos_flags, strict=True
                ):
                    piece = sanitize_tokenizer_pieces(
                        tokenizer.decode(token_ids, skip_special_tokens=True)
                    )
                    answer_text = CUE + piece
                    answer = extract_answer(answer_text, item["task"])
                    row = item["row"]
                    gold_ok, gold_error = grade(answer, row.get("gt") or item["job"].get("gt"))
                    record = {
                        "status": "ok",
                        "method": (
                            "count_rho098_boxed_close"
                            if args.score_kind == "count"
                            else "firstwin_neginf_boxed_close"
                        ),
                        "model": args.model_tag,
                        "dataset": item["dataset"],
                        "seed": item["seed"],
                        "uid": row["uid"],
                        "question_idx": row.get("question_idx"),
                        "old_answer": row.get("new_answer"),
                        "old_gold_ok": bool(row.get("new_gold_ok")),
                        "old_n_ans_tok": int(row.get("n_ans_tok") or 0),
                        "n_think_tok": int(row.get("n_think_tok") or 0),
                        "answer": answer,
                        "gold_ok": gold_ok,
                        "gold_error": gold_error,
                        "same_as_old": bool(same_answer(row.get("new_answer"), answer)),
                        "cue_tokens": cue_tokens,
                        "box_tokens": len(token_ids),
                        "answer_tokens": cue_tokens + len(token_ids),
                        "generated_tokens": len(token_ids),
                        "box_closed": bool(closed),
                        "finish_reason": "brace" if closed else ("eos" if hit_eos else "length"),
                        "answer_budget": item["answer_budget"],
                        "cap_tokens": item["cap"],
                        "n_prompt": item["n_prompt"],
                        "answer_text": answer_text[:400],
                        "answer_tail": answer_text[-200:],
                    }
                    handles[item["out"]].write(json.dumps(record, ensure_ascii=False) + "\n")
                    wrote += 1
        finally:
            for handle in handles.values():
                handle.flush()
                handle.close()
        print(
            f"[{wrote}/{len(pending)}] {time.perf_counter() - started:.0f}s",
            flush=True,
        )
    print(f"done {wrote} in {time.perf_counter() - started:.0f}s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
