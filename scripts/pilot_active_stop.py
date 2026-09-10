#!/usr/bin/env python3
"""Training-free pilots that add information at a geo stop candidate.

Modes:
  select    Build question-disjoint, geo-matched G/non-G candidate pairs.
  challenge Continue the solver briefly with an adversarial checking prompt.
  forecast  Ask frozen Qwen3-4B whether the solver will revise shortly.
  analyze   Report the two pilots against the held-out future answer-change label.

The label is never included in either model prompt.  This is a pilot only:
all selected rows are first candidates of the frozen geo policy
(tau=.98, k=2, epsilon=.03, mss=10).
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

from pilot_counterfactual_support import fast_eq  # noqa: E402
from score_confcal_judge import QWEN3_ASSISTANT, QWEN3_USER_OPEN, dense_trial_path, yes_no_token_ids  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import QWEN4B_MODEL, display_answer, nvidia_lib_path  # noqa: E402

from attn_early_exit.answers import answers_equal  # noqa: E402

def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_refs(dataset: str) -> dict[int, dict[str, str]]:
    root = AE / f"results/dense_G_r1_7b/{dataset}"
    final = {
        int(row["question_idx"]): str(row.get("A_final") or "")
        for row in json.loads((root / "per_sample.json").read_text())
    }
    gold = {
        int(row.get("question_idx") or index + 1): str(row.get("ground_truth_answer") or "")
        for index, row in enumerate(json.loads((root / "dense_puma" / "answers.json").read_text()))
    }
    return {
        question_idx: {"A_final": answer, "ground_truth": gold.get(question_idx, "")}
        for question_idx, answer in final.items()
    }


def is_g_answer(answer: str, ref: dict[str, str]) -> bool:
    return bool(
        fast_eq(answer, ref.get("A_final", ""))
        or fast_eq(answer, ref.get("ground_truth", ""))
        or answers_equal(answer, ref.get("A_final", ""))
        or answers_equal(answer, ref.get("ground_truth", ""))
    )


def load_trials(dataset: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(dense_trial_path(dataset, 42).read_text()):
        grouped[int(row["question_idx"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["stopped_len"]))
    return grouped


def first_geo_candidate(
    rows: list[dict[str, Any]], *, threshold: float, k: int, epsilon: float, mss: int
) -> int | None:
    """Return index of frozen-policy candidate's final confirmation trial."""
    for start, row in enumerate(rows):
        score = finite(row.get("confidence"))
        window = rows[start : start + k]
        if score < threshold or len(window) < k:
            continue
        if int(window[-1]["stopped_len"]) < mss:
            continue
        answer = str(row.get("final_answer") or "")
        if all(
            fast_eq(str(other.get("final_answer") or ""), answer)
            and finite(other.get("confidence")) >= score - epsilon
            for other in window[1:]
        ):
            return start + k - 1
    return None


def future_changed(rows: list[dict[str, Any]], index: int, horizon: int) -> int:
    answer = str(rows[index].get("final_answer") or "")
    return int(
        any(
            not fast_eq(str(row.get("final_answer") or ""), answer)
            for row in rows[index + 1 : index + 1 + horizon]
        )
    )


def select(args: argparse.Namespace) -> None:
    grouped = load_trials(args.dataset)
    refs = load_refs(args.dataset)
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for qi, rows in grouped.items():
        index = first_geo_candidate(
            rows, threshold=args.threshold, k=args.k, epsilon=args.epsilon, mss=args.mss
        )
        if index is None:
            continue
        row = rows[index]
        answer = str(row.get("final_answer") or "")
        is_g = int(is_g_answer(answer, refs.get(qi, {})))
        item = {
            "dataset": args.dataset,
            "question_idx": qi,
            "index": index,
            "decision_step": int(row["stopped_len"]),
            "answer": answer,
            "geo_conf": finite(row.get("confidence")),
            "is_g": is_g,
            "future_change_horizon": args.horizon,
            "future_changed": future_changed(rows, index, args.horizon),
            "question": str(row.get("question") or ""),
            "reasoning_prefix": str(row.get("reasoning_prefix") or ""),
            "recent_answers": [
                str(previous.get("final_answer") or "")
                for previous in rows[max(0, index - args.history) : index + 1]
            ],
        }
        (positives if is_g else negatives).append(item)

    # Match each high-geo non-G candidate to one unused G candidate by geo.
    positives.sort(key=lambda row: row["geo_conf"])
    negatives.sort(key=lambda row: row["geo_conf"], reverse=True)
    used: set[int] = set()
    chosen: list[dict[str, Any]] = []
    for negative in negatives:
        options = [
            (abs(positive["geo_conf"] - negative["geo_conf"]), pos, positive)
            for pos, positive in enumerate(positives)
            if pos not in used
        ]
        if not options:
            break
        _, pos, positive = min(options)
        used.add(pos)
        pair_id = len(chosen) // 2
        chosen.extend([{**positive, "pair_id": pair_id}, {**negative, "pair_id": pair_id}])
        if pair_id + 1 >= args.pairs:
            break

    if args.all:
        chosen = [{**row, "pair_id": None, "candidate_rank": 0} for row in positives + negatives]
        chosen.sort(key=lambda row: (int(row["question_idx"]), int(row["decision_step"])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in chosen:
            handle.write(json.dumps(row) + "\n")
    if args.all:
        print(
            f"selected all-first rows={len(chosen)} "
            f"G={len(positives)} nonG={len(negatives)} -> {args.out}",
            flush=True,
        )
        return
    gaps = [
        abs(chosen[i]["geo_conf"] - chosen[i + 1]["geo_conf"])
        for i in range(0, len(chosen), 2)
    ]
    print(
        f"selected pairs={len(chosen)//2} rows={len(chosen)} "
        f"raw G={len(positives)} raw nonG={len(negatives)} "
        f"mean_abs_geo_gap={np.mean(gaps) if gaps else float('nan'):.6f} -> {args.out}",
        flush=True,
    )


def load_selected(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


DROP_FIELDS = ("reasoning_prefix", "question")


def slim_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in DROP_FIELDS}


def done_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def build_solver_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return (
        prompt
        + str(row["reasoning_prefix"])
        + "\n\nBefore submitting the current proposed answer "
        + f"\\boxed{{{row['answer']}}}, actively try to falsify it. "
        + "Check the most fragile assumption or computation using a different approach. "
        + "Then give the final answer only in the form \\boxed{...}."
    )


def extract_boxed(text: str) -> str | None:
    """Return last balanced ``\\boxed{...}``, retaining nested TeX braces."""
    starts = [match.end() for match in re.finditer(r"\\boxed\{", text)]
    values: list[str] = []
    for start in starts:
        depth = 1
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[start:end].strip())
                    break
    return values[-1] if values else None


def challenge(args: argparse.Namespace) -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_selected(args.selected)
    if args.num_shards > 1:
        rows = [row for row in rows if int(row["question_idx"]) % args.num_shards == args.shard_id]
    done = done_keys(args.out)
    pending = [
        row
        for row in rows
        if (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in done
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    llm = LLM(
        model=str(SOLVER),
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        seed=42,
    )
    challenge_params = SamplingParams(temperature=0.0, max_tokens=args.max_new, detokenize=True)
    final_params = SamplingParams(temperature=0.0, max_tokens=32, detokenize=True)
    started = time.perf_counter()
    with args.out.open("a") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            ready: list[dict[str, Any]] = []
            prompts: list[str] = []
            for row in batch:
                prompt = build_solver_prompt(tokenizer, row)
                tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                if tokens + args.max_new > args.max_context:
                    handle.write(json.dumps({**slim_row(row), "status": "too_long", "prompt_tokens": tokens}) + "\n")
                    continue
                ready.append({**row, "prompt_tokens": tokens})
                prompts.append(prompt)
            challenge_outputs = llm.generate(prompts, challenge_params, use_tqdm=False) if prompts else []
            challenge_texts = [
                output.outputs[0].text if output.outputs else ""
                for output in challenge_outputs
            ]
            final_prompts = [
                prompt
                + text
                + "\n\nFinal answer: \\boxed{"
                for prompt, text in zip(prompts, challenge_texts, strict=True)
            ]
            final_outputs = llm.generate(final_prompts, final_params, use_tqdm=False) if final_prompts else []
            for row, text, final in zip(ready, challenge_texts, final_outputs, strict=True):
                final_text = final.outputs[0].text if final.outputs else ""
                challenged = extract_boxed("\\boxed{" + final_text)
                handle.write(
                    json.dumps(
                        {
                            **slim_row(row),
                            "status": "ok",
                            "challenge_prompt": "falsify_then_boxed_v1",
                            "challenge_max_new": args.max_new,
                            "challenge_text": text,
                            "challenge_final_text": final_text,
                            "challenge_answer": challenged,
                            "challenge_parsed": challenged is not None,
                            "challenge_changed": (
                                int(not fast_eq(challenged, str(row["answer"])))
                                if challenged is not None
                                else None
                            ),
                        }
                    )
                    + "\n"
                )
            handle.flush()
            print(f"challenge [{min(start+len(batch), len(pending))}/{len(pending)}] elapsed={time.perf_counter()-started:.0f}s", flush=True)


FORECAST_SUFFIX = """\

[Recent solver trial answers]
{history}

[Current proposed answer]
{answer}

The solver will continue its own reasoning for a short time. Predict its
behavior, not whether the answer is objectively correct.

[Decision]
A. Keep the current proposed answer unchanged.
B. Revise the proposed answer.

Reply with A or B only."""


def forecast_prompt(row: dict[str, Any], context_chars: int) -> str:
    reasoning = str(row["reasoning_prefix"])[-context_chars:]
    history = "\n".join(f"{index + 1}. {answer}" for index, answer in enumerate(row["recent_answers"]))
    body = (
        f"{QWEN3_USER_OPEN}You are forecasting another solver's near-term behavior.\n\n"
        f"[Problem]\n{row['question']}\n\n"
        f"[Most recent partial reasoning]\n{reasoning}"
        + FORECAST_SUFFIX.format(history=history, answer=row["answer"])
    )
    return body + QWEN3_ASSISTANT


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def choice_token_ids(tokenizer: Any, letter: str) -> list[int]:
    ids: list[int] = []
    for text in (letter, f" {letter}", letter.lower(), f" {letter.lower()}"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def forecast(args: argparse.Namespace) -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_selected(args.selected)
    if args.num_shards > 1:
        rows = [
            row
            for row in rows
            if int(row["question_idx"]) % args.num_shards == args.shard_id
        ]
    done = done_keys(args.out)
    pending = [
        row
        for row in rows
        if (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in done
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN4B_MODEL), trust_remote_code=True)
    keep_ids, revise_ids = choice_token_ids(tokenizer, "A"), choice_token_ids(tokenizer, "B")
    llm = LLM(
        model=str(QWEN4B_MODEL),
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        max_logprobs=args.logprobs_k,
        seed=42,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=args.logprobs_k, detokenize=False)
    started = time.perf_counter()
    with args.out.open("a") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            ready: list[dict[str, Any]] = []
            prompts: list[str] = []
            for row in batch:
                prompt = forecast_prompt(row, args.context_chars)
                tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                if tokens + 1 > args.max_context:
                    handle.write(json.dumps({**slim_row(row), "status": "too_long", "prompt_tokens": tokens}) + "\n")
                    continue
                ready.append({**row, "prompt_tokens": tokens})
                prompts.append(prompt)
            outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            for row, output in zip(ready, outputs, strict=True):
                top = output.outputs[0].logprobs[0] if output.outputs and output.outputs[0].logprobs else {}
                mapping = {
                    int(token): float(item.logprob if hasattr(item, "logprob") else item)
                    for token, item in top.items()
                }
                keep = [mapping[token] for token in keep_ids if token in mapping]
                revise = [mapping[token] for token in revise_ids if token in mapping]
                lp_keep = logsumexp(keep) if keep else float("nan")
                lp_revise = logsumexp(revise) if revise else float("nan")
                p_change = (
                    1.0 / (1.0 + math.exp(-(lp_revise - lp_keep)))
                    if math.isfinite(lp_revise) and math.isfinite(lp_keep)
                    else float("nan")
                )
                handle.write(
                    json.dumps(
                        {
                            **slim_row(row),
                            "status": "ok",
                            "forecast_prompt": "future_change_choice_v2",
                            "forecast_context_chars": args.context_chars,
                            "forecast_p_change": p_change,
                            "forecast_logp_keep": lp_keep,
                            "forecast_logp_revise": lp_revise,
                        }
                    )
                    + "\n"
                )
            handle.flush()
            print(f"forecast [{min(start+len(batch), len(pending))}/{len(pending)}] elapsed={time.perf_counter()-started:.0f}s", flush=True)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y, score = y[mask], score[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    pos, neg = score[y == 1], score[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def analyze(args: argparse.Namespace) -> None:
    selected = {(int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])): row for row in load_selected(args.selected)}
    for name, path, column, higher_is_g in (
        ("challenge", args.challenge, "challenge_changed", False),
        ("forecast", args.forecast, "forecast_p_change", False),
    ):
        if not path.exists():
            print(f"{name}: missing {path}")
            continue
        merged = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
            if key in selected and row.get("status") == "ok":
                merged.append({**selected[key], **row})
        y = np.asarray([row["is_g"] for row in merged], dtype=int)
        raw = np.asarray(
            [float(row[column]) if row.get(column) is not None else float("nan") for row in merged]
        )
        score = raw if higher_is_g else -raw
        print(f"\n{name}: n={len(merged)} G={int(y.sum())} nonG={int((1-y).sum())}")
        print(f"  AUROC(G)={auc(y, score):.3f}; future-change AUROC={auc(np.asarray([row['future_changed'] for row in merged]), raw):.3f}")
        for value in (0, 1):
            subset = raw[y == value]
            print(f"  {'G' if value else 'nonG'}: mean={np.nanmean(subset):.3f} rate/change={np.nanmean(subset):.3f} n={len(subset)}")
        if name == "challenge":
            parsed = np.asarray([bool(row.get("challenge_parsed")) for row in merged])
            print(f"  boxed parse rate={parsed.mean():.3f}; unparsed rows are excluded only if status != ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    select_p = sub.add_parser("select")
    select_p.add_argument("--dataset", default="math-500")
    select_p.add_argument("--pairs", type=int, default=100)
    select_p.add_argument("--horizon", type=int, default=8)
    select_p.add_argument("--history", type=int, default=5)
    select_p.add_argument("--threshold", type=float, default=0.98)
    select_p.add_argument("--k", type=int, default=2)
    select_p.add_argument("--epsilon", type=float, default=0.03)
    select_p.add_argument("--mss", type=int, default=10)
    select_p.add_argument("--out", type=Path, required=True)
    select_p.add_argument("--all", action="store_true", help="Keep every first geo candidate, no pairing.")
    select_p.set_defaults(func=select)

    for mode, func in (("challenge", challenge), ("forecast", forecast)):
        child = sub.add_parser(mode)
        child.add_argument("--selected", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
        child.add_argument("--batch-size", type=int, default=8)
        child.add_argument("--max-context", type=int, default=32768)
        child.add_argument("--gpu-mem-util", type=float, default=0.85)
        child.add_argument("--shard-id", type=int, default=0)
        child.add_argument("--num-shards", type=int, default=1)
        if mode == "challenge":
            child.add_argument("--max-new", type=int, default=128)
        else:
            child.add_argument("--context-chars", type=int, default=4000)
            child.add_argument("--logprobs-k", type=int, default=64)
        child.set_defaults(func=func)

    analyze_p = sub.add_parser("analyze")
    analyze_p.add_argument("--selected", type=Path, required=True)
    analyze_p.add_argument("--challenge", type=Path, required=True)
    analyze_p.add_argument("--forecast", type=Path, required=True)
    analyze_p.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
