#!/usr/bin/env python3
"""Dynamic Answer Convergence for the official GPQA letter shuffle.

The main chain uses the model generation_config. After each new sentence, a
greedy probe is appended and then discarded. Ten identical extracted answers
stop the chain. Only r1_1p5b and r1_llama_8b are accepted.
"""

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
PUMA = Path(os.environ.get("PUMA_ROOT", ROOT.parent / "PUMA")).expanduser().resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PUMA))
sys.path.insert(0, str(PUMA / "puma"))

import nltk
from nltk import sent_tokenize
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from baselines.utils.math_util import my_answer_extraction
from plws.answer_convergence import ConvergenceState, observe_answer
from plws.answer_convergence_dynamic import (
    CHUNK_TOKENS,
    PROBE_SUFFIX,
    THINK_CLOSE,
    committed_prefixes,
    take_reasoning_chunk,
)
from plws.deploy import deployment_manifest
from plws.gpqa_official import DATASET_NAME, SHUFFLE_SEED
from plws.grading import (
    GRADER_VERIFICATION_VERSION,
    must_grade,
    require_grader,
    verify_baseline_records,
)
from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)
from prompt_utils import get_task_type
from run_vllm import build_prompts_with_chat_template

ALLOWED_MODELS = ("r1_1p5b", "r1_llama_8b")
EXECUTION_MODE = "dynamic_sentence_probe"
ALGORITHM = (
    "online host sampling; sentence prefixes; greedy answer probes; "
    "exact answer equality; k consecutive; stop without finishing the tail"
)


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


def fresh_state() -> dict[str, Any]:
    return {
        "reasoning": "",
        "probed": 0,
        "last_answer": None,
        "repeat_count": 0,
        "trial_tokens": 0,
        "trial_count": 0,
        "main_tokens": 0,
        "closed": False,
    }


def host_sampling(model: str) -> dict[str, float | int]:
    config = GenerationConfig.from_pretrained(model, trust_remote_code=True)
    temperature = float(getattr(config, "temperature", 0.6) or 0.0)
    top_p = float(getattr(config, "top_p", 0.95) or 1.0)
    top_k = getattr(config, "top_k", -1)
    if top_k is None:
        top_k = -1
    return {"temperature": temperature, "top_p": top_p, "top_k": int(top_k)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dataset-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--probe-max-tokens", type=int, default=100)
    parser.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model_tag not in ALLOWED_MODELS:
        raise ValueError(
            f"dynamic official GPQA is limited to {ALLOWED_MODELS}, got {args.model_tag}"
        )
    if args.dataset != DATASET_NAME:
        raise ValueError(f"dataset must be {DATASET_NAME}, got {args.dataset}")
    if args.threshold != 10:
        raise ValueError("canonical Answer Convergence requires threshold=10")
    if args.chunk_tokens < 1 or args.probe_max_tokens < 1:
        raise ValueError("chunk-tokens and probe-max-tokens must be positive")
    if args.max_model_len != MAX_MODEL_LEN:
        raise ValueError(f"{PROTOCOL_ID} requires max_model_len={MAX_MODEL_LEN}")
    require_grader()
    try:
        sent_tokenize("Preflight sentence.")
    except LookupError as exc:
        raise RuntimeError("NLTK punkt data is missing") from exc

    dataset_path = args.dataset_file.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) != 198 and not args.limit:
        raise ValueError(f"official GPQA file must have 198 rows, got {len(rows)}")
    samples = [
        {
            "question": row["question"],
            "ground_truth_answer": row["answer"],
            "record_id": row["record_id"],
        }
        for row in rows
    ]

    output_dir = args.output_dir.resolve()
    record_dir = output_dir / "records"
    checkpoint_path = output_dir / "checkpoint.json"
    final_path = output_dir / "final_answers.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    if args.fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    sampling = host_sampling(args.model)
    tensor_parallel_size = len(
        [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    ) or 1
    identity = {
        "method": "answer_convergence",
        "execution_mode": EXECUTION_MODE,
        "model": str(Path(args.model).resolve()),
        "model_tag": args.model_tag,
        "dataset": args.dataset,
        "dataset_file": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "seed": args.seed,
        "threshold": args.threshold,
        "choice_order": {
            "source": "idavidrein/gpqa baselines/answer_only.py",
            "shuffle": "random.Random(42).sample",
            "shuffle_seed": SHUFFLE_SEED,
            "frozen_in_dataset_file": True,
        },
        "prompt_version": "default",
        "delivery_sampling": sampling,
        "probe_sampling": {
            "temperature": 0.0,
            "max_tokens": args.probe_max_tokens,
            "stop": ["\n"],
            "presence_penalty": 1.0,
        },
        "chunk_tokens": args.chunk_tokens,
        "protocol_id": PROTOCOL_ID,
        "fullcot_generation_tokens": FULLCOT_GENERATION_TOKENS,
        "prompt_reserve_tokens": PROMPT_RESERVE_TOKENS,
        "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "expected_records": len(samples),
        "algorithm": ALGORITHM,
        "official_repository": "https://github.com/launchnlp/reasoning_earlystop",
        "nltk_version": nltk.__version__,
        "sentence_tokenizer": "nltk.sent_tokenize:english",
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
        comparable = {
            key: value
            for key, value in identity.items()
            if key not in {"deployment"}
        }
        for key, value in comparable.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"manifest identity mismatch for {key}: "
                    f"{existing.get(key)!r} != {value!r}"
                )

    existing_records = load_records(record_dir)
    verify_baseline_records(existing_records, samples)
    if len(existing_records) == len(samples) and final_path.is_file():
        print(f"[answer-convergence-dynamic] already complete n={len(samples)}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    task_type = get_task_type(args.dataset)
    base_prompts = build_prompts_with_chat_template(
        [sample["question"] for sample in samples],
        tokenizer,
        args.model,
        task_type,
        "default",
    )
    base_tokens = [token_count(tokenizer, prompt) for prompt in base_prompts]
    too_long = [index for index, count in enumerate(base_tokens) if count > PROMPT_RESERVE_TOKENS]
    if too_long:
        raise RuntimeError(
            f"prompt exceeds {PROMPT_RESERVE_TOKENS} tokens: questions={too_long[:8]}"
        )

    checkpoint: dict[str, Any] = {}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    states: dict[int, dict[str, Any]] = {
        int(key): value for key, value in checkpoint.get("states", {}).items()
    }
    for index in range(len(samples)):
        if index not in existing_records and index not in states:
            states[index] = fresh_state()

    def save_checkpoint() -> None:
        atomic_json(
            checkpoint_path,
            {
                "updated_at": now(),
                "states": {str(key): value for key, value in states.items()},
            },
        )

    def write_result(
        index: int,
        *,
        answer: str,
        answer_text: str,
        prefix: str,
        stopped_early: bool,
        state: dict[str, Any],
        probe_finish_reason: str | None,
        probe_stop_reason: Any,
    ) -> None:
        prefix_tokens = token_count(tokenizer, prefix)
        answer_tokens = token_count(tokenizer, answer_text)
        forced_close_tokens = token_count(tokenizer, "\n</think>\n") if prefix else 0
        overshoot_tokens = max(
            0, token_count(tokenizer, state["reasoning"]) - prefix_tokens
        )
        delivery_tokens = prefix_tokens + forced_close_tokens + answer_tokens
        trial_tokens = int(state["trial_tokens"])
        record = {
            "protocol_id": PROTOCOL_ID,
            "execution_mode": EXECUTION_MODE,
            "question_idx": index,
            "record_id": samples[index]["record_id"],
            "question": samples[index]["question"],
            "ground_truth_answer": samples[index]["ground_truth_answer"],
            "answer": answer,
            "generated_text": answer_text,
            "partial_reasoning": prefix,
            "correct": must_grade(answer, samples[index]["ground_truth_answer"]),
            "original_correct": False,
            "grader_verification": GRADER_VERIFICATION_VERSION,
            "grade_error": "",
            "stopped_early": stopped_early,
            "num_trial_answers": int(state["trial_count"]),
            "probe_finish_reason": probe_finish_reason,
            "probe_stop_reason": probe_stop_reason,
            "prefix_tokens": prefix_tokens,
            "forced_close_tokens": forced_close_tokens,
            "answer_tokens": answer_tokens,
            "delivery_tokens": delivery_tokens,
            "num_trial_answer_tokens": trial_tokens,
            "overshoot_tokens": overshoot_tokens,
            "main_tokens": int(state["main_tokens"]),
            "total_generated_tokens": delivery_tokens + trial_tokens + overshoot_tokens,
            "base_prompt_tokens": base_tokens[index],
        }
        atomic_json(record_dir / f"{index:06d}.json", record)
        existing_records[index] = record
        states.pop(index, None)

    active = [index for index in range(len(samples)) if index not in existing_records]
    if not active:
        pass
    else:
        print(
            f"[answer-convergence-dynamic] loading {args.model_tag} "
            f"seed={args.seed} pending={len(active)}"
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
            enable_flashinfer_autotune=False,
            seed=args.seed,
        )
        atexit.register(shutdown_llm, llm)
        probe_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.probe_max_tokens,
            stop=["\n"],
            presence_penalty=1.0,
        )
        round_index = 0
        while active:
            pending = []
            for index in list(active):
                state = states[index]
                prefixes = committed_prefixes(
                    state["reasoning"],
                    sent_tokenize,
                    finished=bool(state["closed"]),
                )
                if int(state["probed"]) < len(prefixes):
                    pending.append((index, prefixes[int(state["probed"])]))
            if pending:
                prompts = []
                for index, prefix in pending:
                    prompt = base_prompts[index] + prefix + PROBE_SUFFIX
                    context_tokens = token_count(tokenizer, prompt)
                    if context_tokens + args.probe_max_tokens > args.max_model_len:
                        raise RuntimeError(
                            "Answer Convergence probe exceeds canonical context: "
                            f"question={index} context={context_tokens}"
                        )
                    prompts.append(prompt)
                outputs = llm.generate(prompts, probe_params)
                for (index, prefix), output in zip(pending, outputs):
                    generated = output.outputs[0]
                    answer_text = "\\boxed" + generated.text
                    answer = str(
                        my_answer_extraction(answer_text, dataset=args.dataset)
                    )
                    state = states[index]
                    updated, converged = observe_answer(
                        ConvergenceState(
                            last_answer=state["last_answer"],
                            repeat_count=int(state["repeat_count"]),
                        ),
                        answer,
                        threshold=args.threshold,
                    )
                    state["last_answer"] = updated.last_answer
                    state["repeat_count"] = updated.repeat_count
                    state["trial_tokens"] = int(state["trial_tokens"]) + len(
                        generated.token_ids
                    )
                    state["trial_count"] = int(state["trial_count"]) + 1
                    state["probed"] = int(state["probed"]) + 1
                    prefixes = committed_prefixes(
                        state["reasoning"],
                        sent_tokenize,
                        finished=bool(state["closed"]),
                    )
                    exhausted = bool(state["closed"]) and int(state["probed"]) >= len(
                        prefixes
                    )
                    if converged or exhausted:
                        write_result(
                            index,
                            answer=answer,
                            answer_text=answer_text,
                            prefix=prefix,
                            stopped_early=converged and not exhausted,
                            state=state,
                            probe_finish_reason=getattr(generated, "finish_reason", None),
                            probe_stop_reason=getattr(generated, "stop_reason", None),
                        )
                        active.remove(index)
                save_checkpoint()
                continue

            growing = [
                index
                for index in active
                if not states[index]["closed"]
                and int(states[index]["main_tokens"]) < FULLCOT_GENERATION_TOKENS
            ]
            if not growing:
                raise RuntimeError(
                    "dynamic chain made no progress: "
                    f"questions={sorted(active)[:8]}"
                )
            chunk = min(
                args.chunk_tokens,
                min(
                    FULLCOT_GENERATION_TOKENS - int(states[index]["main_tokens"])
                    for index in growing
                ),
            )
            prompts = [
                base_prompts[index] + states[index]["reasoning"] for index in growing
            ]
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    top_k=int(sampling["top_k"]),
                    max_tokens=chunk,
                    stop=[THINK_CLOSE],
                ),
            )
            for index, output in zip(growing, outputs):
                generated = output.outputs[0]
                piece, closed = take_reasoning_chunk(generated.text)
                state = states[index]
                state["reasoning"] += piece
                state["main_tokens"] = int(state["main_tokens"]) + len(generated.token_ids)
                if (
                    closed
                    or generated.finish_reason == "stop"
                    or int(state["main_tokens"]) >= FULLCOT_GENERATION_TOKENS
                    or not generated.token_ids
                ):
                    state["closed"] = True
            round_index += 1
            print(
                f"[answer-convergence-dynamic] chunk={round_index} "
                f"growing={len(growing)} done={len(existing_records)}"
            )
            save_checkpoint()

        shutdown_llm(llm)

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
        "avg_delivery_tokens": sum(row["delivery_tokens"] for row in ordered) / n,
        "avg_probe_tokens": sum(row["num_trial_answer_tokens"] for row in ordered) / n,
        "avg_overshoot_tokens": sum(row["overshoot_tokens"] for row in ordered) / n,
        "avg_total_generated_tokens": sum(row["total_generated_tokens"] for row in ordered) / n,
        "early_stop_rate": sum(row["stopped_early"] for row in ordered) / n,
        "avg_trial_answers": sum(row["num_trial_answers"] for row in ordered) / n,
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            **identity,
            "grader_verification": GRADER_VERIFICATION_VERSION,
            "grader_verified_at": now(),
            "completed_at": now(),
            "summary": str(summary_path),
            "final_answers": str(final_path),
        },
    )
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
