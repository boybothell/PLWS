#!/usr/bin/env python3
"""Canonical Dynasor replay on a frozen PUMA Full-CoT trajectory."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PUMA = Path(os.environ.get("PUMA_ROOT", ROOT.parent / "PUMA")).expanduser().resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PUMA))
sys.path.insert(0, str(PUMA / "puma"))

from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from math_grader import math_equal
from plws.deploy import deployment_manifest
from plws.grading import (
    GRADER_VERIFICATION_VERSION,
    must_grade,
    require_grader,
    verify_baseline_records,
)
from plws.dynasor import (
    CERTAINTY_THRESHOLD,
    CHUNK_SIZE,
    EFFORT,
    PROBE_MAX_TOKENS,
    PROBE_SUFFIX,
    is_certain_answer,
    normalize_gpqa_answer,
    obtain_answer,
    should_early_exit,
    token_chunk_boundaries,
)
from plws.host_protocol import validate_fullcot_sample_meta
from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)
from prompt_utils import get_task_type
from run_vllm import build_prompts_with_chat_template

UPSTREAM_REPOSITORY = "https://github.com/hao-ai-lab/Dynasor"
UPSTREAM_COMMIT = "0d2f1b93a60e031cfc7fb43843d2b736b225960c"
EXECUTION_MODE = "frozen_trajectory_offline_replay"
PROBE_TEMPERATURE = 0.6
PROBE_TOP_P = 0.95
PROBE_TOP_K = -1


def now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_records(record_dir: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not record_dir.is_dir():
        return records
    for path in record_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            records[int(record["question_idx"])] = record
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return records


def verify_loaded_records(
    records: dict[int, dict[str, Any]],
    samples: list[dict[str, Any]],
    record_dir: Path,
) -> None:
    """Verify and atomically update every resumable record before reuse."""

    for index in verify_baseline_records(records, samples):
        atomic_json(record_dir / f"{index:06d}.json", records[index])


def summarize(
    records: list[dict[str, Any]], *, wall_time_seconds: float
) -> dict[str, float | int]:
    n = len(records)
    return {
        "n": n,
        "accuracy": 100.0 * sum(row["correct"] for row in records) / n,
        "original_accuracy": 100.0
        * sum(row["original_correct"] for row in records)
        / n,
        "early_stop_rate": sum(row["stopped_early"] for row in records) / n,
        "avg_delivery_tokens": sum(row["delivery_tokens"] for row in records) / n,
        "avg_probe_tokens": sum(row["probe_trial_tokens"] for row in records) / n,
        "avg_total_method_tokens": sum(
            row["total_method_tokens"] for row in records
        )
        / n,
        "avg_replay_preparation_tokens": sum(
            row["replay_preparation_tokens"] for row in records
        )
        / n,
        "avg_probes": sum(row["num_probes"] for row in records) / n,
        "wall_time_seconds": round(wall_time_seconds, 3),
    }


def token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def shutdown_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    client = getattr(engine, "engine_core", None)
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--dataset-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--effort", default=EFFORT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--certainty-threshold",
        type=int,
        default=CERTAINTY_THRESHOLD,
    )
    parser.add_argument(
        "--probe-max-tokens",
        type=int,
        default=PROBE_MAX_TOKENS,
    )
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_method = {
        "effort": EFFORT,
        "chunk_size": CHUNK_SIZE,
        "certainty_threshold": CERTAINTY_THRESHOLD,
        "probe_max_tokens": PROBE_MAX_TOKENS,
    }
    actual_method = {
        "effort": args.effort,
        "chunk_size": args.chunk_size,
        "certainty_threshold": args.certainty_threshold,
        "probe_max_tokens": args.probe_max_tokens,
    }
    if actual_method != required_method:
        raise ValueError(
            f"canonical Dynasor requires {required_method}, got {actual_method}"
        )
    if args.max_model_len != MAX_MODEL_LEN:
        raise ValueError(
            f"{PROTOCOL_ID} requires max_model_len={MAX_MODEL_LEN}, "
            f"got {args.max_model_len}"
        )
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must be in (0, 1)")
    require_grader()
    if args.max_num_seqs < 1:
        raise ValueError("max-num-seqs must be positive")

    sample_path = args.sample.expanduser().resolve()
    sample_meta_path = sample_path.parent / "sample_meta.json"
    validate_fullcot_sample_meta(
        sample_meta_path,
        model_tag=args.model_tag,
        dataset=args.dataset,
        seed=args.seed,
    )
    dataset_path = args.dataset_file.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"missing pinned dataset: {dataset_path}")

    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"sample must be a non-empty JSON list: {sample_path}")
    if args.limit:
        samples = samples[: args.limit]
    required_fields = {
        "question",
        "ground_truth_answer",
        "generated_text",
        "reasoning",
    }
    for index, sample in enumerate(samples):
        missing = required_fields - sample.keys()
        if missing:
            raise ValueError(f"sample row {index} missing fields: {sorted(missing)}")
        if sample.get("dataset") != args.dataset:
            raise ValueError(
                f"sample row {index} dataset={sample.get('dataset')!r}, "
                f"expected {args.dataset!r}"
            )

    output_dir = args.output_dir.expanduser().resolve()
    record_dir = output_dir / "records"
    checkpoint_path = output_dir / "checkpoint.json"
    final_path = output_dir / "final_answers.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    if args.fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    config = GenerationConfig.from_pretrained(args.model, trust_remote_code=True)
    temperature = float(getattr(config, "temperature", 0.6) or 0.0)
    top_p = float(getattr(config, "top_p", 0.95) or 1.0)
    top_k = getattr(config, "top_k", -1)
    if top_k is None:
        top_k = -1
    top_k = int(top_k)
    tensor_parallel_size = len(
        [
            item
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item
        ]
    ) or 1

    identity = {
        "schema_version": 1,
        "method": "dynasor",
        "execution_mode": EXECUTION_MODE,
        "method_source": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
        },
        "protocol_id": PROTOCOL_ID,
        "model": str(Path(args.model).expanduser().resolve()),
        "model_tag": args.model_tag,
        "dataset": args.dataset,
        "dataset_file": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "seed": args.seed,
        "expected_records": len(samples),
        "prompt_version": "default",
        "fullcot_generation_tokens": FULLCOT_GENERATION_TOKENS,
        "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
        "prompt_reserve_tokens": PROMPT_RESERVE_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "delivery_sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        "probe_sampling": {
            "temperature": PROBE_TEMPERATURE,
            "top_p": PROBE_TOP_P,
            "top_k": PROBE_TOP_K,
            "max_tokens": args.probe_max_tokens,
        },
        "dynasor": {
            **required_method,
            "probe_suffix": PROBE_SUFFIX,
            "uncertainty_gate": "case-insensitive substring",
            "math_equivalence": "PUMA puma/math_grader.py::math_equal after plws.grading.require_grader",
            "gpqa_normalization": "last standalone A/B/C/D",
        },
        "sample": str(sample_path),
        "sample_sha256": file_sha256(sample_path),
        "sample_meta": str(sample_meta_path),
        "sample_meta_sha256": file_sha256(sample_meta_path),
        "deployment": deployment_manifest(
            args.model_tag,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            root=ROOT,
        ),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"manifest identity mismatch for {key}: "
                    f"{existing.get(key)!r} != {value!r}"
                )

    existing_records = load_records(record_dir)
    verify_loaded_records(existing_records, samples, record_dir)
    if len(existing_records) == len(samples) and final_path.is_file():
        ordered = [existing_records[index] for index in range(len(samples))]
        atomic_text(
            final_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        )
        previous_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {}
        )
        atomic_json(
            summary_path,
            summarize(
                ordered,
                wall_time_seconds=float(
                    previous_summary.get("wall_time_seconds") or 0.0
                ),
            ),
        )
        completed = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        atomic_json(
            manifest_path,
            {
                **completed,
                **identity,
                "grader_verification": GRADER_VERIFICATION_VERSION,
                "grader_verified_at": now(),
                "status": "succeeded",
                "completed_at": completed.get("completed_at") or now(),
                "summary": str(summary_path),
                "final_answers": str(final_path),
            },
        )
        print(f"[dynasor] already complete n={len(samples)}")
        return 0

    print(
        f"[dynasor] loading {args.model_tag} for "
        f"{args.dataset} seed={args.seed}"
    )
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    atexit.register(shutdown_llm, llm)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    task_type = get_task_type(args.dataset)
    base_prompts = build_prompts_with_chat_template(
        [str(sample["question"]) for sample in samples],
        tokenizer,
        args.model,
        task_type,
        "default",
    )
    reasoning_ids = [
        tokenizer.encode(
            str(sample["reasoning"]),
            add_special_tokens=False,
        )
        for sample in samples
    ]
    boundaries = [
        token_chunk_boundaries(len(token_ids), args.chunk_size)
        for token_ids in reasoning_ids
    ]
    original_token_counts = [
        token_count(tokenizer, str(sample["generated_text"]))
        for sample in samples
    ]

    checkpoint: dict[str, Any] = {}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    next_round = int(checkpoint.get("next_round", 0))
    states: dict[int, dict[str, Any]] = {
        int(key): value for key, value in checkpoint.get("states", {}).items()
    }
    for index in range(len(samples)):
        if index not in states and index not in existing_records:
            states[index] = {
                "answers": [],
                "certainties": [],
                "probe_tokens": 0,
                "probe_count": 0,
            }

    def write_result(
        index: int,
        *,
        answer: str,
        generated_text: str,
        prefix: str,
        prefix_tokens: int,
        stopped_early: bool,
        stop_boundary: int | None,
        state: dict[str, Any],
        probe_finish_reason: str | None = None,
        probe_stop_reason: Any = None,
    ) -> None:
        sample = samples[index]
        delivery_tokens = token_count(tokenizer, generated_text)
        probe_tokens = int(state["probe_tokens"])
        record = {
            "protocol_id": PROTOCOL_ID,
            "execution_mode": EXECUTION_MODE,
            "question_idx": index,
            "question": sample["question"],
            "ground_truth_answer": sample["ground_truth_answer"],
            "answer": answer,
            "generated_text": generated_text,
            "partial_reasoning": prefix,
            "correct": must_grade(answer, sample["ground_truth_answer"]),
            "original_correct": must_grade(
                sample.get("model_answer", ""),
                sample["ground_truth_answer"],
            ),
            "grader_verification": GRADER_VERIFICATION_VERSION,
            "grade_error": "",
            "stopped_early": stopped_early,
            "stop_token_boundary": stop_boundary,
            "reasoning_tokens": len(reasoning_ids[index]),
            "proposal_tokens_to_stop": prefix_tokens,
            "delivery_tokens": delivery_tokens,
            "probe_trial_tokens": probe_tokens,
            "total_method_tokens": delivery_tokens + probe_tokens,
            "replay_preparation_tokens": original_token_counts[index],
            "num_probes": int(state["probe_count"]),
            "probe_answers": list(state["answers"]),
            "probe_certainties": list(state["certainties"]),
            "probe_finish_reason": probe_finish_reason,
            "probe_stop_reason": probe_stop_reason,
        }
        atomic_json(record_dir / f"{index:06d}.json", record)
        existing_records[index] = record

    def write_fullcot(index: int, state: dict[str, Any]) -> None:
        sample = samples[index]
        write_result(
            index,
            answer=str(sample.get("model_answer", "")),
            generated_text=str(sample["generated_text"]),
            prefix="",
            prefix_tokens=original_token_counts[index],
            stopped_early=False,
            stop_boundary=None,
            state=state,
        )

    for index in list(states):
        if index in existing_records:
            del states[index]
        elif not boundaries[index]:
            write_fullcot(index, states[index])
            del states[index]

    sampling_params = SamplingParams(
        temperature=PROBE_TEMPERATURE,
        max_tokens=args.probe_max_tokens,
        top_p=PROBE_TOP_P,
        top_k=PROBE_TOP_K,
    )
    max_rounds = max((len(value) for value in boundaries), default=0)
    start = time.monotonic()
    for round_index in range(next_round, max_rounds):
        batch_indices = [
            index
            for index in sorted(states)
            if round_index < len(boundaries[index])
        ]
        if not batch_indices:
            continue
        prefixes: list[str] = []
        prompts: list[str] = []
        for index in batch_indices:
            boundary = boundaries[index][round_index]
            prefix = tokenizer.decode(
                reasoning_ids[index][:boundary],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompt = base_prompts[index] + prefix + PROBE_SUFFIX
            context_tokens = token_count(tokenizer, prompt)
            if context_tokens + args.probe_max_tokens > args.max_model_len:
                raise RuntimeError(
                    "Dynasor probe exceeds canonical context: "
                    f"question={index} boundary={boundary} "
                    f"context={context_tokens} probe={args.probe_max_tokens} "
                    f"max={args.max_model_len}"
                )
            prefixes.append(prefix)
            prompts.append(prompt)
        print(
            f"[dynasor] round={round_index + 1}/{max_rounds} "
            f"batch={len(prompts)} completed={len(existing_records)}"
        )
        outputs = llm.generate(prompts, sampling_params)
        for index, prefix, output in zip(batch_indices, prefixes, outputs):
            generated = output.outputs[0]
            probe_text = generated.text
            answer = obtain_answer(probe_text)
            if args.dataset == "gpqa-diamond":
                answer = normalize_gpqa_answer(answer)
            state = states[index]
            state["answers"].append(answer)
            state["certainties"].append(is_certain_answer(probe_text))
            state["probe_tokens"] = int(state["probe_tokens"]) + len(
                generated.token_ids
            )
            state["probe_count"] = int(state["probe_count"]) + 1
            boundary = boundaries[index][round_index]
            early = should_early_exit(
                state["answers"],
                state["certainties"],
                equivalent=lambda left, right: math_equal(left, right),
                threshold=args.certainty_threshold,
            )
            exhausted = round_index == len(boundaries[index]) - 1
            if early:
                delivery = (
                    prefix
                    + "\n\n... Oh, I have got the answer to the whole problem\n"
                    "**Final Answer:**\n\\[\n \\boxed{"
                    + answer
                    + "}\n\\]"
                )
                if token_count(tokenizer, delivery) > FULLCOT_GENERATION_TOKENS:
                    raise RuntimeError(
                        "Dynasor early-exit delivery exceeds 32K host budget: "
                        f"question={index} boundary={boundary}"
                    )
                write_result(
                    index,
                    answer=answer,
                    generated_text=delivery,
                    prefix=prefix,
                    prefix_tokens=boundary,
                    stopped_early=True,
                    stop_boundary=boundary,
                    state=state,
                    probe_finish_reason=getattr(
                        generated, "finish_reason", None
                    ),
                    probe_stop_reason=getattr(generated, "stop_reason", None),
                )
                del states[index]
            elif exhausted:
                write_fullcot(index, state)
                del states[index]

        atomic_json(
            checkpoint_path,
            {
                "updated_at": now(),
                "next_round": round_index + 1,
                "states": {str(key): value for key, value in states.items()},
            },
        )

    records = load_records(record_dir)
    if len(records) != len(samples):
        missing = sorted(set(range(len(samples))) - records.keys())
        raise RuntimeError(
            f"incomplete cell: {len(records)}/{len(samples)}, missing={missing}"
        )
    ordered = [records[index] for index in range(len(samples))]
    atomic_text(
        final_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
    )
    summary = summarize(
        ordered, wall_time_seconds=time.monotonic() - start
    )
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            **identity,
            "grader_verification": GRADER_VERIFICATION_VERSION,
            "grader_verified_at": now(),
            "status": "succeeded",
            "completed_at": now(),
            "summary": str(summary_path),
            "final_answers": str(final_path),
        },
    )
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    shutdown_llm(llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
