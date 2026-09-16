#!/usr/bin/env python3
"""从锁点前缀按官方 Full-CoT 32K 总生成预算续写并压制反思词。

锁点前已经生成的 token 计入 32K。自然收到 </think> 后，终答只能使用
32K 内的剩余预算；只有 32K 用尽仍未收口时，才复刻官方 runner，强行
补 </think> 并给予单独的 2048-token answer-fix。
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_LENS_DISABLE", "1")

ROOT_HINT = Path(os.environ.get("PLWS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT_HINT / "src"))

from plws.artifacts import atomic_write_json, load_jsonl, utc_now  # noqa: E402
from plws.inputs import same_answer  # noqa: E402
from plws.lexicon import LEXICONS, suppress_bad_words, usable_bad_words  # noqa: E402
from plws.paths import PLWSPaths, datasets_for_jobs  # noqa: E402
from plws.runtime import model_path, nvidia_ld_library_path  # noqa: E402
from plws.protocol import (  # noqa: E402
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)
from plws.piece_text import sanitize_tokenizer_pieces  # noqa: E402
from plws.window import K  # noqa: E402

PATHS = PLWSPaths.discover(__file__)
AE = PATHS.root
PUMA = Path(os.environ.get("PUMA_ROOT", AE.parent / "PUMA")).resolve()


def shutdown_llm(llm: Any) -> None:
    """Explicitly stop vLLM before multiprocessing's exit finalizers run."""
    engine = getattr(llm, "llm_engine", None)
    client = getattr(engine, "engine_core", None)
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=10)


os.environ["LD_LIBRARY_PATH"] = nvidia_ld_library_path()
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(PUMA / "puma"))

MODELS = {
    tag: str(model_path(tag))
    for tag in (
        "r1_7b",
        "nemotron_8b",
        "r1_14b",
        "r1_1p5b",
        "r1_llama_8b",
        "r1_32b",
        "qwen3_4b",
        "qwen3_8b",
        "qwen3_30b_a3b",
        "qwq_32b",
        "qwen3_32b",
    )
}

from math_grader import check_is_correct  # noqa: E402
from prompt_utils import get_task_type  # noqa: E402

WAIT_RE = re.compile(r"\bwait\b|\balternatively\b|\bhmm+\b|等一下", re.I)


def request_seed(base: int | None, uid: str, phase: str) -> int | None:
    """Return a stable per-request seed for paired ablations."""

    if base is None:
        return None
    payload = f"{base}:{uid}:{phase}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("suppress", "free"), required=True)
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=FULLCOT_GENERATION_TOKENS,
        help="Full-CoT thinking budget. Default is the main-table 32K protocol.",
    )
    parser.add_argument(
        "--protocol-id",
        default=PROTOCOL_ID,
        help="Written into leftover scores. Default is puma-fullcot-32k-v2.",
    )
    parser.add_argument(
        "--think-tokens",
        type=int,
        default=0,
        help="Deprecated compatibility flag; the canonical protocol requires 0.",
    )
    parser.add_argument(
        "--answer-tokens",
        type=int,
        default=TRUNCATED_ANSWER_FIX_TOKENS,
        help="Answer-fix budget used only when the main generation is truncated.",
    )
    parser.add_argument("--trial-tokens", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="每批题数。<=0 表示该 shard 剩余题一次 generate()，跟官方 Full-CoT 一样连续批。",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Stable per-UID common random seed for paired ablations.",
    )
    parser.add_argument("--jobs", type=Path)
    parser.add_argument(
        "--dataset",
        help="Cell dataset. Required when --jobs is omitted; filters a legacy aggregate input.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        help="Compatibility override for the historical score directory layout.",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Score every job even if a reusable uid already exists (leak-fix reruns).",
    )
    parser.add_argument(
        "--isolated-output",
        action="store_true",
        help="Resume only from --out's directory; do not reuse canonical scores.",
    )
    parser.add_argument(
        "--run-kind",
        default="",
        help="firstwin is the PLWS cell. low/mix/high are leftover labels only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="输出目录用的 seed。不传则从 jobs 第一题读，再没有用 42。",
    )
    parser.add_argument(
        "--lexicon",
        choices=tuple(LEXICONS),
        default="core",
        help="core=Wait/Alternatively/Hmm；safe=再加 However/Maybe/another way/double-check/Hold on。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="Window size used to create these jobs and select the run path.",
    )
    args = parser.parse_args()
    if args.max_context < args.generation_tokens:
        parser.error(
            f"{args.protocol_id} requires --max-context >= "
            f"{args.generation_tokens}, got {args.max_context}"
        )
    if args.think_tokens != 0:
        parser.error(
            f"{args.protocol_id} has no independent continuation cap; "
            "--think-tokens must be 0"
        )
    if args.answer_tokens != TRUNCATED_ANSWER_FIX_TOKENS:
        parser.error(
            f"{args.protocol_id} requires --answer-tokens "
            f"{TRUNCATED_ANSWER_FIX_TOKENS}, got {args.answer_tokens}"
        )
    bad_words = suppress_bad_words(args.lexicon)
    kind = args.run_kind or "firstwin"

    jobs_source = args.jobs
    if jobs_source is None:
        if not args.dataset:
            parser.error("--dataset is required when --jobs is omitted")
        jobs_source = PATHS.resolve_jobs_path(
            args.model_tag,
            args.dataset,
            args.seed if args.seed is not None else 42,
            kind,
            k=args.k,
            lexicon=args.lexicon,
        )
    if not jobs_source.is_file():
        raise SystemExit(f"jobs file not found: {jobs_source}")
    all_jobs = load_jsonl(jobs_source)
    if args.dataset:
        all_jobs = [job for job in all_jobs if job.get("dataset") == args.dataset]
    try:
        datasets = list(
            datasets_for_jobs(
                all_jobs,
                requested=args.dataset,
                allow_mixed=args.out is not None or args.out_root is not None,
            )
        )
    except ValueError as error:
        parser.error(str(error))
    row_models = {str(job.get("model") or str(job.get("uid", "")).split(":")[0]) for job in all_jobs}
    if row_models and row_models != {args.model_tag}:
        parser.error(f"jobs model mismatch: expected {args.model_tag}, found {sorted(row_models)}")
    row_kinds = {str(job.get("kind") or "firstwin") for job in all_jobs}
    allowed = {kind, "firstwin", "low", "mix", "high"} if kind == "firstwin" else {kind}
    if row_kinds and not row_kinds <= allowed:
        parser.error(f"jobs tier mismatch: expected {kind}, found {sorted(row_kinds)}")
    row_ks = {int(job.get("k", K)) for job in all_jobs}
    if row_ks and row_ks != {args.k}:
        parser.error(f"jobs k mismatch: expected {args.k}, found {sorted(row_ks)}")

    jobs = all_jobs
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_id]
    if args.limit:
        jobs = jobs[: args.limit]
    kind_tag = f"_{args.run_kind}" if args.run_kind and args.run_kind not in {"", "firstwin", "low"} else ""
    if args.seed is not None:
        seed = int(args.seed)
    elif jobs and jobs[0].get("seed") is not None:
        seed = int(jobs[0]["seed"])
    else:
        seed = 42
    row_seeds = {int(job.get("seed", seed)) for job in all_jobs}
    allow_mixed_seeds = args.out is not None or args.out_root is not None
    if row_seeds and row_seeds != {seed} and not allow_mixed_seeds:
        parser.error(f"jobs seed mismatch: expected {seed}, found {sorted(row_seeds)}")
    if args.out is None:
        if args.out_root is not None:
            args.out = (
                args.out_root
                / f"{args.model_tag}_s{seed}_{args.mode}{kind_tag}"
                / f"scores_shard{args.shard_id}.jsonl"
            )
        else:
            args.out = PATHS.score_path(
                args.model_tag,
                datasets[0],
                seed,
                args.mode,
                kind,
                args.shard_id,
                k=args.k,
                lexicon=args.lexicon,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    read_dirs = {args.out.parent}
    if not args.isolated_output:
        reuse_kinds = ("firstwin", "low", "mix", "high") if kind == "firstwin" else (kind,)
        for dataset in datasets:
            for reuse_kind in reuse_kinds:
                read_dirs.update(
                    PATHS.score_read_dirs(
                        args.model_tag,
                        dataset,
                        seed,
                        args.mode,
                        reuse_kind,
                        k=args.k,
                        lexicon=args.lexicon,
                    )
                )
    resume_paths = [args.out]
    if not args.ignore_existing:
        resume_paths.extend(
            candidate
            for directory in read_dirs
            for candidate in (
                *sorted(directory.glob("shard_*.jsonl")),
                *sorted(directory.glob("scores_shard*.jsonl")),
                directory / "scores.jsonl",
            )
        )
    already = {
        str(row["uid"])
        for path in resume_paths
        for row in load_jsonl(path)
        if row.get("status") in {"ok", "too_long"}
        and row.get("uid")
        and row.get("protocol_id") == args.protocol_id
        and row.get("max_model_len") == args.max_context
        and int(row.get("truncated_answer_fix_tokens") or 0)
        == args.answer_tokens
        and int(row.get("fullcot_generation_tokens") or args.generation_tokens)
        == args.generation_tokens
    }
    pending = [job for job in jobs if job["uid"] not in already]
    artifact_id = args.out.stem
    manifest_path = args.out.parent / f"manifest_{artifact_id}.json"
    status_path = args.out.parent / f"status_{artifact_id}.json"
    manifest = {
        "schema_version": 1,
        "method": "plws",
        "experiment": "first_nonl" if kind == "first_nonl" else "window_first",
        "lock_policy": kind,
        "model": args.model_tag,
        "datasets": datasets,
        "seed": seed if len(row_seeds) == 1 else None,
        "seeds": sorted(row_seeds),
        "mode": args.mode,
        "kind": kind,
        "k": args.k,
        "lexicon": args.lexicon,
        "prompt_version": "default",
        "sampling_seed": args.sampling_seed,
        "protocol_id": args.protocol_id,
        "fullcot_generation_tokens": args.generation_tokens,
        "truncated_answer_fix_tokens": args.answer_tokens,
        "max_model_len": args.max_context,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "jobs": str(jobs_source),
        "output": str(args.out),
        "created_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    status_state = {"finished": False}

    def write_status(state: str, message: str | None = None) -> None:
        atomic_write_json(
            status_path,
            {
                "schema_version": 1,
                "state": state,
                "updated_at": utc_now(),
                "pending_at_start": len(pending),
                "message": message,
            },
        )

    def mark_interrupted() -> None:
        if not status_state["finished"]:
            write_status("failed", "process exited before successful completion")

    atexit.register(mark_interrupted)
    write_status("running")
    print(
        f"leftover-{args.mode} {args.model_tag} shard={args.shard_id}/{args.num_shards} "
        f"lexicon={args.lexicon} jobs={len(jobs)} pending={len(pending)} -> {args.out}",
        flush=True,
    )
    if not pending:
        write_status("succeeded", "nothing pending; protocol-valid scores already cover this shard")
        status_state["finished"] = True
        return

    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams
    from run_vllm import build_prompts_with_chat_template, extract_answer

    if args.model_tag not in MODELS:
        raise SystemExit(f"unknown model-tag {args.model_tag}")
    model_path = MODELS[args.model_tag]
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    tp_size = max(1, len(visible))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    filtered = usable_bad_words(bad_words, tokenizer)
    dropped = [word for word in bad_words if word not in set(filtered)]
    bad_words = filtered
    if dropped:
        print(
            f"{args.mode} {args.model_tag} drop unencodable bad_words={dropped}",
            flush=True,
        )
    generation_config = GenerationConfig.from_pretrained(
        model_path, trust_remote_code=True
    )
    temperature = getattr(generation_config, "temperature", 0.6)
    top_p = getattr(generation_config, "top_p", 0.95)
    top_k = getattr(generation_config, "top_k", -1)
    if top_k is None:
        top_k = -1
    manifest.update(
        {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "decoding_source": "model generation_config.json",
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        f"{args.mode} {args.model_tag} tp={tp_size} model={model_path} "
        f"max_model_len={args.max_context} temperature={temperature} "
        f"top_p={top_p} top_k={top_k}",
        flush=True,
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        max_model_len=args.max_context,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=True,
    )
    atexit.register(shutdown_llm, llm)
    think_base: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop": ["</think>"],
        "include_stop_str_in_output": False,
    }
    if args.mode == "suppress":
        think_base["bad_words"] = bad_words
    answer_base: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    close_tokens = len(
        tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
    )
    started = time.perf_counter()
    wrote = 0
    step = len(pending) if args.batch_size <= 0 else args.batch_size
    print(
        f"{args.mode} shard{args.shard_id} official-batch step={step} "
        f"pending={len(pending)} lexicon={args.lexicon} "
        f"bad_words={bad_words if args.mode == 'suppress' else []}",
        flush=True,
    )
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), step):
            chunk = pending[begin : begin + step]
            think_prompts: list[str] = []
            think_params: list[Any] = []
            ready: list[dict[str, Any]] = []
            prefix_toks: list[int] = []
            for job in chunk:
                task_type = get_task_type(job["dataset"])
                chat = build_prompts_with_chat_template(
                    [job["question"]],
                    tokenizer,
                    model_path,
                    task_type,
                    "default",
                )[0]
                thought = str(job.get("thought") or "")
                text = chat + thought
                n_tok = len(tokenizer.encode(text, add_special_tokens=False))
                n_left = len(tokenizer.encode(thought, add_special_tokens=False)) if thought else 0
                budget = max(0, args.generation_tokens - n_left)
                budget = min(budget, max(0, args.max_context - n_tok))
                required_context = n_tok + budget
                if budget < 1 or required_context > args.max_context:
                    handle.write(
                        json.dumps(
                            {
                                "uid": job["uid"],
                                "status": "too_long",
                                "mode": args.mode,
                                "experiment": job.get("experiment")
                                or (
                                    "first_nonl"
                                    if kind == "first_nonl"
                                    else "window_first"
                                ),
                                "lock_policy": job.get("lock_policy")
                                or job.get("lock")
                                or kind,
                                "cohort": job.get("cohort"),
                                "dataset": job.get("dataset"),
                                "question_idx": job.get("question_idx"),
                                "kind": job.get("kind") or kind,
                                "window_kind": job.get("window_kind")
                                or job.get("kind"),
                                "first_window_kind": job.get("first_window_kind"),
                                "first_window_step": job.get("first_window_step"),
                                "delay_steps": job.get("delay_steps"),
                                "n_left_tok": n_left,
                                "protocol_id": args.protocol_id,
                                "fullcot_generation_tokens": args.generation_tokens,
                                "truncated_answer_fix_tokens": args.answer_tokens,
                                "required_context": required_context,
                                "max_model_len": args.max_context,
                            }
                        )
                        + "\n"
                    )
                    continue
                think_prompts.append(text)
                think_seed = request_seed(
                    args.sampling_seed, str(job["uid"]), "think"
                )
                seed_args = {} if think_seed is None else {"seed": think_seed}
                think_params.append(
                    SamplingParams(
                        **think_base,
                        max_tokens=budget,
                        **seed_args,
                    )
                )
                ready.append(job)
                prefix_toks.append(n_left)
            if not think_prompts:
                continue
            print(
                f"{args.mode} shard{args.shard_id} think generate n={len(think_prompts)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
            think_outs = llm.generate(think_prompts, think_params, use_tqdm=True)
            answer_prompts: list[str] = []
            answer_ready: list[
                tuple[
                    dict[str, Any],
                    str,
                    int,
                    int,
                    int,
                    str,
                    bool,
                    str,
                    bool,
                    int,
                ]
            ] = []
            for job, prompt, out, n_left in zip(
                ready, think_prompts, think_outs, prefix_toks, strict=True
            ):
                gen0 = out.outputs[0] if out.outputs else None
                cont = gen0.text if gen0 else ""
                n_cont_tok = len(list(gen0.token_ids)) if gen0 else 0
                finish = str(getattr(gen0, "finish_reason", "") or "")
                stop_reason = str(getattr(gen0, "stop_reason", "") or "")
                hit_cap = finish == "length"
                natural_close = stop_reason == "</think>" or "</think>" in cont
                closed = prompt + cont
                if "</think>" not in closed:
                    closed = closed + "\n</think>\n\n"
                n_closed = len(tokenizer.encode(closed, add_special_tokens=False))
                if natural_close:
                    requested_answer_budget = (
                        args.generation_tokens
                        - n_left
                        - n_cont_tok
                        - close_tokens
                    )
                else:
                    requested_answer_budget = args.answer_tokens
                answer_budget = min(
                    requested_answer_budget,
                    args.max_context - n_closed,
                )
                if answer_budget < 1:
                    handle.write(
                        json.dumps(
                            {
                                "uid": job["uid"],
                                "status": "too_long",
                                "mode": args.mode,
                                "experiment": job.get("experiment")
                                or (
                                    "first_nonl"
                                    if kind == "first_nonl"
                                    else "window_first"
                                ),
                                "lock_policy": job.get("lock_policy")
                                or job.get("lock")
                                or kind,
                                "cohort": job.get("cohort"),
                                "dataset": job.get("dataset"),
                                "question_idx": job.get("question_idx"),
                                "kind": job.get("kind") or kind,
                                "window_kind": job.get("window_kind")
                                or job.get("kind"),
                                "first_window_kind": job.get("first_window_kind"),
                                "first_window_step": job.get("first_window_step"),
                                "delay_steps": job.get("delay_steps"),
                                "cont_n": len(cont),
                                "n_left_tok": n_left,
                                "n_cont_tok": n_cont_tok,
                                "finish_reason": finish,
                                "hit_cap": hit_cap,
                                "stop_reason": stop_reason,
                                "natural_close": natural_close,
                                "answer_budget": answer_budget,
                                "protocol_id": args.protocol_id,
                                "fullcot_generation_tokens": args.generation_tokens,
                                "truncated_answer_fix_tokens": args.answer_tokens,
                                "max_model_len": args.max_context,
                            }
                        )
                        + "\n"
                    )
                    continue
                answer_prompts.append(closed)
                answer_ready.append(
                    (
                        job,
                        cont,
                        n_closed,
                        n_left,
                        n_cont_tok,
                        finish,
                        hit_cap,
                        stop_reason,
                        natural_close,
                        answer_budget,
                    )
                )
            if not answer_prompts:
                continue
            answer_params = []
            for job, *_rest, answer_budget in answer_ready:
                answer_seed = request_seed(
                    args.sampling_seed, str(job["uid"]), "answer"
                )
                seed_args = {} if answer_seed is None else {"seed": answer_seed}
                answer_params.append(
                    SamplingParams(
                        **answer_base,
                        max_tokens=answer_budget,
                        **seed_args,
                    )
                )
            print(
                f"{args.mode} shard{args.shard_id} answer generate n={len(answer_prompts)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
            answer_outs = llm.generate(answer_prompts, answer_params, use_tqdm=True)
            for (
                job,
                cont,
                n_closed,
                n_left,
                n_cont_tok,
                finish,
                hit_cap,
                stop_reason,
                natural_close,
                answer_budget,
            ), out in zip(
                answer_ready, answer_outs, strict=True
            ):
                gen = out.outputs[0]
                text = sanitize_tokenizer_pieces(gen.text)
                token_ids = list(gen.token_ids)
                generated_text = f"{job.get('thought') or ''}{cont}\n</think>\n\n{text}"
                task_type = get_task_type(job["dataset"])
                answer = extract_answer(generated_text, task_type)
                try:
                    gold_ok = bool(check_is_correct(answer, job["gt"]))
                except Exception:
                    gold_ok = False
                rec = {
                    "status": "ok",
                    "mode": args.mode,
                    "uid": job["uid"],
                    "experiment": job.get("experiment")
                    or ("first_nonl" if kind == "first_nonl" else "window_first"),
                    "lock_policy": job.get("lock_policy") or job.get("lock") or kind,
                    "cohort": job.get("cohort"),
                    "dataset": job["dataset"],
                    "question_idx": job["question_idx"],
                    "left_step": job.get("left_step", 0),
                    "kind": job.get("kind") or "full",
                    "window_kind": job.get("window_kind") or job.get("kind"),
                    "first_window_kind": job.get("first_window_kind"),
                    "first_window_step": job.get("first_window_step"),
                    "delay_steps": job.get("delay_steps"),
                    "lexicon": args.lexicon,
                    "confidence": job.get("confidence"),
                    "left_ok": job.get("left_ok", False),
                    "wait_helps": job.get("wait_helps", False),
                    "will_change": job.get("will_change", False),
                    "never_high": job.get("never_high", True),
                    "same_as_high": job.get("same_as_high", False),
                    "host_ok": job.get("host_ok", False),
                    "old_answer": job.get("old_answer", ""),
                    "original_tokens": job.get("original_tokens"),
                    "new_answer": answer,
                    "keep": bool(same_answer(job["old_answer"], answer)),
                    "new_gold_ok": gold_ok,
                    "next_ent_mean": job.get("next_ent_mean"),
                    "cont_n": len(cont),
                    "n_left_tok": n_left,
                    "n_cont_tok": n_cont_tok,
                    "n_ans_tok": len(token_ids),
                    "n_think_tok": n_left + n_cont_tok,
                    "n_out_tok": n_cont_tok + len(token_ids),
                    "finish_reason": finish,
                    "hit_cap": hit_cap,
                    "stop_reason": stop_reason,
                    "natural_close": natural_close,
                    "answer_budget": answer_budget,
                    "protocol_id": args.protocol_id,
                    "fullcot_generation_tokens": args.generation_tokens,
                    "truncated_answer_fix_tokens": args.answer_tokens,
                    "max_model_len": args.max_context,
                    "n_wait": len(WAIT_RE.findall(cont)),
                    "has_wait": bool(WAIT_RE.search(cont)),
                    "n_prompt": n_closed,
                    "new_text": (cont[-80:] + " | " + text[:120])[:200],
                    "generated_text": generated_text,
                    "task_type": task_type,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"{args.mode} shard{args.shard_id} wrote={wrote}/{len(pending)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    write_status("succeeded", f"wrote {wrote} records")
    status_state["finished"] = True
    print(f"done mode={args.mode} shard={args.shard_id} wrote={wrote}", flush=True)
    shutdown_llm(llm)


if __name__ == "__main__":
    main()
