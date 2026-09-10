#!/usr/bin/env python3
"""Pilot solver-relative answer support on high-geo MATH trial steps.

This does not ask the solver to generate or verify an answer.  It
teacher-forces the already observed current answer and historical alternatives
under the same R1-7B prompt, then measures:

* margin: current-answer average log P minus best historical alternative;
* evidence_gain: current-answer average log P under current reasoning minus
  its score under the immediately preceding reasoning prefix;
* reasoning_pmi: current-answer score under current reasoning minus its score
  with the question but no observed reasoning.

The pilot deliberately balances high-geo G and non-G rows.  It is therefore
only a conditional-discrimination test, not an end-to-end early-stop result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ["VLLM_LENS_DISABLE"] = "1"
AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_judge import dense_trial_path  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

_WS = re.compile(r"\s+")


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def fast_eq(left: Any, right: Any) -> bool:
    a = _WS.sub("", str(left or "").strip().lower().replace("dfrac", "frac").replace("$", ""))
    b = _WS.sub("", str(right or "").strip().lower().replace("dfrac", "frac").replace("$", ""))
    return bool(a) and a == b


def load_refs(dataset: str) -> tuple[dict[int, str], dict[int, str]]:
    root = AE / f"results/dense_G_r1_7b/{dataset}"
    a_final = {
        int(row["question_idx"]): str(row.get("A_final") or "")
        for row in json.loads((root / "per_sample.json").read_text())
    }
    gold: dict[int, str] = {}
    for index, row in enumerate(json.loads((root / "dense_puma" / "answers.json").read_text())):
        gold[int(row.get("question_idx") or index + 1)] = str(row.get("ground_truth_answer") or "")
    return a_final, gold


def build_prefix(tokenizer: Any, question: str, reasoning: str) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION}\n{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(prompt + reasoning + "\n", add_special_tokens=False)


def answer_ids(tokenizer: Any, answer: str) -> list[int]:
    return tokenizer.encode(f"\\boxed{{{answer}}}", add_special_tokens=False)


def append_answer(tokenizer: Any, question: str, reasoning: str, answer: str) -> tuple[list[int], int]:
    prefix = build_prefix(tokenizer, question, reasoning)
    target = answer_ids(tokenizer, answer)
    return prefix + target, len(prefix)


def mean_target_logprob(output: Any, prefix_len: int) -> float:
    ids = output.prompt_token_ids or []
    logs = output.prompt_logprobs or []
    values: list[float] = []
    for index in range(max(prefix_len, 1), min(len(ids), len(logs))):
        item = (logs[index] or {}).get(ids[index])
        if item is None:
            return float("nan")
        values.append(float(item.logprob if hasattr(item, "logprob") else item))
    return float(np.mean(values)) if values else float("nan")


def load_trials(dataset: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(dense_trial_path(dataset, 42).read_text()):
        grouped[int(row["question_idx"])].append(row)
    for history in grouped.values():
        history.sort(key=lambda row: int(row["stopped_len"]))
    return grouped


def select_high_geo(
    grouped: dict[int, list[dict[str, Any]]],
    dataset: str,
    per_class: int,
) -> list[dict[str, Any]]:
    a_final, gold = load_refs(dataset)
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for qi, history in grouped.items():
        for index, row in enumerate(history):
            answer = str(row.get("final_answer") or "")
            item = {
                "question_idx": qi,
                "index": index,
                "decision_step": int(row["stopped_len"]),
                "answer": answer,
                "geo_conf": finite(row.get("confidence")),
                "is_g": int(fast_eq(answer, a_final.get(qi)) or fast_eq(answer, gold.get(qi))),
            }
            if not math.isfinite(item["geo_conf"]):
                continue
            (positives if item["is_g"] else negatives).append(item)
    positives.sort(key=lambda row: row["geo_conf"], reverse=True)
    negatives.sort(key=lambda row: row["geo_conf"], reverse=True)
    # The same question may contribute at most one row per class, preventing
    # long trajectories from dominating this conditional pilot.
    def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            if row["question_idx"] in seen:
                continue
            seen.add(row["question_idx"])
            out.append(row)
            if len(out) >= per_class:
                break
        return out
    return dedupe(positives) + dedupe(negatives)


def select_paired_high_geo(
    grouped: dict[int, list[dict[str, Any]]],
    dataset: str,
    pair_count: int,
) -> list[dict[str, Any]]:
    """Within each question, pair its top-geo non-G with the closest-geo G."""
    a_final, gold = load_refs(dataset)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for qi, history in grouped.items():
        positives: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        for index, row in enumerate(history):
            answer = str(row.get("final_answer") or "")
            item = {
                "question_idx": qi,
                "index": index,
                "decision_step": int(row["stopped_len"]),
                "answer": answer,
                "geo_conf": finite(row.get("confidence")),
                "is_g": int(fast_eq(answer, a_final.get(qi)) or fast_eq(answer, gold.get(qi))),
            }
            if not math.isfinite(item["geo_conf"]):
                continue
            (positives if item["is_g"] else negatives).append(item)
        if not positives or not negatives:
            continue
        negative = max(negatives, key=lambda item: item["geo_conf"])
        positive = min(positives, key=lambda item: abs(item["geo_conf"] - negative["geo_conf"]))
        pairs.append((positive, negative))
    pairs.sort(key=lambda pair: pair[1]["geo_conf"], reverse=True)
    chosen = pairs[:pair_count]
    print(
        f"paired questions={len(chosen)}; "
        f"mean_abs_geo_gap={np.mean([abs(a['geo_conf'] - b['geo_conf']) for a, b in chosen]):.6f}",
        flush=True,
    )
    return [item for pair in chosen for item in pair]


def unique_history_answers(history: list[dict[str, Any]], index: int, current: str) -> list[str]:
    answers: list[str] = []
    for row in history[:index]:
        answer = str(row.get("final_answer") or "")
        if answer and not fast_eq(answer, current) and not any(fast_eq(answer, old) for old in answers):
            answers.append(answer)
    return answers[-5:]


def run(args: argparse.Namespace) -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    grouped = load_trials(args.dataset)
    chosen = (
        select_paired_high_geo(grouped, args.dataset, args.per_class)
        if args.paired
        else select_high_geo(grouped, args.dataset, args.per_class)
    )
    if not chosen:
        raise RuntimeError("no pilot jobs selected")
    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    out = args.out or AE / "results/confcal_judge/v2/counterfactual_pilot" / f"{args.dataset}_highgeo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[int, int, str]] = set()
    if out.exists():
        with out.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done.add((int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])))
    chosen = [row for row in chosen if (row["question_idx"], row["decision_step"], row["answer"]) not in done]
    print(f"selected={len(chosen)} after_resume; output={out}", flush=True)
    llm = LLM(
        model=str(SOLVER),
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        max_logprobs=1,
        seed=42,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    started = time.perf_counter()
    with out.open("a") as handle:
        for start in range(0, len(chosen), args.batch_size):
            batch = chosen[start:start + args.batch_size]
            prompts: list[dict[str, list[int]]] = []
            meta: list[tuple[dict[str, Any], str, str]] = []
            for job in batch:
                history = grouped[job["question_idx"]]
                row = history[job["index"]]
                question = str(row.get("question") or "")
                current_reasoning = str(row.get("reasoning_prefix") or "")
                prior_reasoning = str(history[job["index"] - 1].get("reasoning_prefix") or "") if job["index"] else ""
                alternatives = unique_history_answers(history, job["index"], job["answer"])
                variants = [("current", current_reasoning, job["answer"]), ("prior", prior_reasoning, job["answer"]), ("question_only", "", job["answer"])]
                variants.extend(("alternative", current_reasoning, answer) for answer in alternatives)
                for kind, reasoning, answer in variants:
                    ids, prefix_len = append_answer(tokenizer, question, reasoning, answer)
                    if len(ids) + 1 > args.max_context:
                        continue
                    prompts.append({"prompt_token_ids": ids})
                    meta.append((job, kind, answer, prefix_len))
            outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            scores: dict[tuple[int, int, str], dict[str, list[float] | float]] = defaultdict(lambda: defaultdict(list))
            for (job, kind, answer, prefix_len), output in zip(meta, outputs, strict=True):
                key = (job["question_idx"], job["decision_step"], job["answer"])
                value = mean_target_logprob(output, prefix_len)
                if kind == "alternative":
                    scores[key]["alternative"].append(value)  # type: ignore[union-attr]
                else:
                    scores[key][kind] = value
            for job in batch:
                key = (job["question_idx"], job["decision_step"], job["answer"])
                value = scores[key]
                current = finite(value.get("current"))
                prior = finite(value.get("prior"))
                question_only = finite(value.get("question_only"))
                alternatives = [finite(x) for x in value.get("alternative", [])]  # type: ignore[arg-type]
                alternatives = [x for x in alternatives if math.isfinite(x)]
                best_alternative = max(alternatives) if alternatives else float("nan")
                payload = {
                    **job,
                    "status": "ok",
                    "current_logp_per_token": current,
                    "prior_logp_per_token": prior,
                    "question_only_logp_per_token": question_only,
                    "best_historical_logp_per_token": best_alternative,
                    "n_historical_alternatives": len(alternatives),
                    "margin": current - best_alternative if math.isfinite(current) and math.isfinite(best_alternative) else float("nan"),
                    "evidence_gain": current - prior if math.isfinite(current) and math.isfinite(prior) else float("nan"),
                    "reasoning_pmi": current - question_only if math.isfinite(current) and math.isfinite(question_only) else float("nan"),
                }
                handle.write(json.dumps(payload) + "\n")
            handle.flush()
            print(f"[{min(start + len(batch), len(chosen))}/{len(chosen)}] elapsed={time.perf_counter() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--out", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
