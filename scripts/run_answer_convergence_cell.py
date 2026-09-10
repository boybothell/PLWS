#!/usr/bin/env python3
"""Run one resumable Answer Convergence cell on an existing Full-CoT sample."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUMA = ROOT.parent / "PUMA"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PUMA))
sys.path.insert(0, str(PUMA / "puma"))

from nltk import sent_tokenize
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from baselines.utils.math_util import my_answer_extraction
from math_grader import check_is_correct
from plws.answer_convergence import (
    ConvergenceState,
    cumulative_sentence_prefixes,
    observe_answer,
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

OFFICIAL_REPOSITORY = "https://github.com/launchnlp/reasoning_earlystop"
OFFICIAL_CONSISTENCY_SOURCE_SHA = "24d4c856cc508bee5f98ba0b28aec375c2a50993"
ALGORITHM = "sentence prefixes; greedy answer probes; exact answer equality; k consecutive"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def shutdown_llm(llm: Any) -> None:
    """Explicitly stop vLLM before multiprocessing's exit finalizers run."""
    engine = getattr(llm, "llm_engine", None)
    client = getattr(engine, "engine_core", None)
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=10)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--probe-max-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threshold != 10:
        raise ValueError("canonical Answer Convergence requires threshold=10")
    if args.probe_max_tokens < 1:
        raise ValueError("probe-max-tokens must be positive")
    if args.max_model_len != MAX_MODEL_LEN:
        raise ValueError(
            f"{PROTOCOL_ID} requires max_model_len={MAX_MODEL_LEN}, "
            f"got {args.max_model_len}"
        )

    try:
        sent_tokenize("Preflight sentence.")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK punkt data is missing; install punkt and punkt_tab before queueing"
        ) from exc

    sample_path = args.sample.resolve()
    sample_meta_path = sample_path.parent / "sample_meta.json"
    validate_fullcot_sample_meta(
        sample_meta_path,
        model_tag=args.model_tag,
        dataset=args.dataset,
        seed=args.seed,
    )
    output_dir = args.output_dir.resolve()
    record_dir = output_dir / "records"
    checkpoint_path = output_dir / "checkpoint.json"
    final_path = output_dir / "final_answers.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    if args.fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)

    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"sample must be a non-empty JSON list: {sample_path}")
    if args.limit:
        samples = samples[: args.limit]

    required = {"question", "ground_truth_answer", "generated_text", "reasoning"}
    for index, sample in enumerate(samples):
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"sample row {index} missing fields: {sorted(missing)}")
        if sample.get("dataset") != args.dataset:
            raise ValueError(
                f"sample row {index} dataset={sample.get('dataset')!r}, "
                f"expected {args.dataset!r}"
            )

    identity = {
        "method": "answer_convergence",
        "model": str(Path(args.model).resolve()),
        "model_tag": args.model_tag,
        "dataset": args.dataset,
        "seed": args.seed,
        "threshold": args.threshold,
        "probe_temperature": 0.0,
        "probe_max_tokens": args.probe_max_tokens,
        "probe_sampling": {
            "temperature": 0.0,
            "max_tokens": args.probe_max_tokens,
            "stop": ["\n"],
            "presence_penalty": 1.0,
        },
        "protocol_id": PROTOCOL_ID,
        "fullcot_generation_tokens": FULLCOT_GENERATION_TOKENS,
        "prompt_reserve_tokens": PROMPT_RESERVE_TOKENS,
        "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "prompt_version": "default",
        "sample": str(sample_path),
        "sample_sha256": file_sha256(sample_path),
        "sample_meta": str(sample_meta_path),
        "sample_meta_sha256": file_sha256(sample_meta_path),
        "expected_records": len(samples),
        "algorithm": ALGORITHM,
        "trajectory_protocol": "PUMA-aligned Full-CoT sampling",
        "probe_prompt_protocol": "same base prompt as the reused Full-CoT trajectory",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_consistency_source_sha": OFFICIAL_CONSISTENCY_SOURCE_SHA,
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
    if len(existing_records) == len(samples) and final_path.is_file():
        print(f"[answer-convergence] already complete n={len(samples)}")
        return 0

    print(
        f"[answer-convergence] loading {args.model_tag} for "
        f"{args.dataset} seed={args.seed}"
    )
    tensor_parallel_size = len(
        [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    ) or 1
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.90,
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
    base_prompt_tokens = [
        token_count(tokenizer, prompt) for prompt in base_prompts
    ]

    prefixes: dict[int, list[str]] = {}
    for index, sample in enumerate(samples):
        prefixes[index] = cumulative_sentence_prefixes(
            str(sample["reasoning"]), sent_tokenize
        )

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
                "last_answer": None,
                "repeat_count": 0,
                "trial_tokens": 0,
                "trial_count": 0,
            }

    def write_result(
        index: int,
        *,
        answer: str,
        answer_text: str,
        answer_tokens: int,
        prefix: str,
        stopped_early: bool,
        stop_round: int,
        trial_tokens: int,
        trial_count: int,
        probe_finish_reason: str | None,
        probe_stop_reason: Any,
    ) -> None:
        sample = samples[index]
        prefix_tokens = token_count(tokenizer, prefix)
        forced_close_tokens = (
            token_count(tokenizer, "\n</think>\n") if prefix else 0
        )
        probe_suffix_tokens = (
            token_count(tokenizer, "\n</think>\n\\boxed") if prefix else 0
        )
        original_tokens = token_count(tokenizer, str(sample["generated_text"]))
        delivery_tokens = prefix_tokens + forced_close_tokens + answer_tokens
        record = {
            "protocol_id": PROTOCOL_ID,
            "fullcot_generation_tokens": FULLCOT_GENERATION_TOKENS,
            "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "question_idx": index,
            "question": sample["question"],
            "ground_truth_answer": sample["ground_truth_answer"],
            "answer": answer,
            "generated_text": answer_text,
            "partial_reasoning": prefix,
            "correct": bool(
                check_is_correct(str(answer), str(sample["ground_truth_answer"]))
            ),
            "original_correct": bool(
                check_is_correct(
                    str(sample.get("model_answer", "")),
                    str(sample["ground_truth_answer"]),
                )
            ),
            "stopped_early": stopped_early,
            "stop_sentence_index": stop_round,
            "total_sentences": len(prefixes[index]),
            "num_trial_answers": trial_count,
            "probe_finish_reason": probe_finish_reason,
            "probe_stop_reason": probe_stop_reason,
            "prefix_tokens": prefix_tokens,
            "forced_close_tokens": forced_close_tokens,
            "probe_suffix_tokens": probe_suffix_tokens,
            "base_prompt_tokens": base_prompt_tokens[index],
            "context_tokens_at_stop": (
                base_prompt_tokens[index] + prefix_tokens + probe_suffix_tokens
            ),
            "answer_tokens": answer_tokens,
            "delivery_tokens": delivery_tokens,
            "num_trial_answer_tokens": trial_tokens,
            "total_generated_tokens": delivery_tokens + trial_tokens,
            "original_tokens": original_tokens,
        }
        atomic_json(record_dir / f"{index:06d}.json", record)
        existing_records[index] = record

    for index in list(states):
        if index in existing_records:
            del states[index]
            continue
        if prefixes[index]:
            continue
        sample = samples[index]
        answer = str(sample.get("model_answer", ""))
        write_result(
            index,
            answer=answer,
            answer_text=str(sample["generated_text"]),
            answer_tokens=token_count(tokenizer, str(sample["generated_text"])),
            prefix="",
            stopped_early=False,
            stop_round=0,
            trial_tokens=0,
            trial_count=0,
            probe_finish_reason=None,
            probe_stop_reason=None,
        )
        del states[index]

    max_rounds = max((len(value) for value in prefixes.values()), default=0)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.probe_max_tokens,
        stop=["\n"],
        presence_penalty=1.0,
    )

    for round_index in range(next_round, max_rounds):
        batch_indices = [
            index
            for index in sorted(states)
            if index not in existing_records and round_index < len(prefixes[index])
        ]
        if not batch_indices:
            continue
        prompts = []
        for index in batch_indices:
            prompt = (
                base_prompts[index]
                + prefixes[index][round_index]
                + "\n</think>\n\\boxed"
            )
            context_tokens = token_count(tokenizer, prompt)
            if context_tokens + args.probe_max_tokens > args.max_model_len:
                raise RuntimeError(
                    "Answer Convergence probe exceeds canonical context: "
                    f"question={index} round={round_index + 1} "
                    f"context={context_tokens} probe={args.probe_max_tokens} "
                    f"max={args.max_model_len}"
                )
            prompts.append(prompt)
        print(
            f"[answer-convergence] round={round_index + 1}/{max_rounds} "
            f"batch={len(prompts)} completed={len(existing_records)}"
        )
        outputs = llm.generate(prompts, sampling_params)
        for index, output in zip(batch_indices, outputs):
            generated = output.outputs[0]
            answer_text = "\\boxed" + generated.text
            answer = my_answer_extraction(answer_text, dataset=args.dataset)
            state_data = states[index]
            state, converged = observe_answer(
                ConvergenceState(
                    last_answer=state_data["last_answer"],
                    repeat_count=int(state_data["repeat_count"]),
                ),
                str(answer),
                threshold=args.threshold,
            )
            state_data.update(
                {
                    "last_answer": state.last_answer,
                    "repeat_count": state.repeat_count,
                    "trial_tokens": int(state_data["trial_tokens"])
                    + len(generated.token_ids),
                    "trial_count": int(state_data["trial_count"]) + 1,
                }
            )
            exhausted = round_index == len(prefixes[index]) - 1
            if converged or exhausted:
                write_result(
                    index,
                    answer=str(answer),
                    answer_text=answer_text,
                    answer_tokens=token_count(tokenizer, answer_text),
                    prefix=prefixes[index][round_index],
                    stopped_early=converged and not exhausted,
                    stop_round=round_index + 1,
                    trial_tokens=int(state_data["trial_tokens"]),
                    trial_count=int(state_data["trial_count"]),
                    probe_finish_reason=getattr(generated, "finish_reason", None),
                    probe_stop_reason=getattr(generated, "stop_reason", None),
                )
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
        raise RuntimeError(f"incomplete cell: {len(records)}/{len(samples)}, missing={missing}")

    ordered = [records[index] for index in range(len(samples))]
    atomic_text(
        final_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
    )
    n = len(ordered)
    summary = {
        "n": n,
        "accuracy": 100.0 * sum(row["correct"] for row in ordered) / n,
        "original_accuracy": 100.0
        * sum(row["original_correct"] for row in ordered)
        / n,
        "avg_delivery_tokens": sum(row["delivery_tokens"] for row in ordered) / n,
        "avg_probe_tokens": sum(
            row["num_trial_answer_tokens"] for row in ordered
        )
        / n,
        "avg_total_generated_tokens": sum(
            row["total_generated_tokens"] for row in ordered
        )
        / n,
        "avg_original_tokens": sum(row["original_tokens"] for row in ordered) / n,
        "early_stop_rate": sum(row["stopped_early"] for row in ordered) / n,
        "avg_trial_answers": sum(row["num_trial_answers"] for row in ordered) / n,
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            **identity,
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
