#!/usr/bin/env python3
"""Family-KeyToken extraction with a strict tokenizer-equivalence gate.

JS is computed only after the complete token->id map, special tokens, and
probe encodings match.  Different-vocabulary truncation is refused.

Phases:
  gate   : tokenizer equality only
  solver : teacher-force the R1-7B trajectory, store top-k logprobs
  small  : teacher-force the family-small model on the same token ids
  merge  : aligned top-k JS / KL / entropy and boxed-window aggregates
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["VLLM_LENS_DISABLE"] = "1"
AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_judge import dense_trial_path  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

SOLVER = Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B")
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def tokenizer_signature(tokenizer: object) -> dict[str, object]:
    vocab = tokenizer.get_vocab()  # type: ignore[attr-defined]
    special = getattr(tokenizer, "special_tokens_map_extended", {})
    return {"vocab": vocab, "special": special, "vocab_size": len(vocab)}


def assert_same_tokenizer(left_path: Path, right_path: Path, probes: list[str]) -> dict[str, int]:
    from transformers import AutoTokenizer

    left = AutoTokenizer.from_pretrained(left_path, trust_remote_code=True)
    right = AutoTokenizer.from_pretrained(right_path, trust_remote_code=True)
    ls, rs = tokenizer_signature(left), tokenizer_signature(right)
    if ls["vocab"] != rs["vocab"] or ls["special"] != rs["special"]:
        raise ValueError(
            "Refusing JS: token->id or special-token mapping differs. "
            "Use a tokenizer-identical family small model, not Qwen3."
        )
    for text in probes:
        if left.encode(text, add_special_tokens=False) != right.encode(text, add_special_tokens=False):
            raise ValueError(f"Refusing JS: encoding mismatch for probe {text[:80]!r}")
    return {"vocab_size": int(ls["vocab_size"]), "probe_count": len(probes)}


def fits(ids: list[int], max_context: int) -> bool:
    return len(ids) + 1 <= max_context


def generate_safe(llm: Any, prompts: list[dict[str, Any]], params: Any) -> list[Any]:
    if not prompts:
        return []
    try:
        return llm.generate(prompts, params, use_tqdm=False)
    except Exception as exc:
        print(f"batch generate failed ({exc}); retrying one-by-one", flush=True)
        outputs: list[Any] = []
        for prompt in prompts:
            try:
                outputs.append(llm.generate([prompt], params, use_tqdm=False)[0])
            except Exception as inner:
                print(f"item generate failed ({inner})", flush=True)
                outputs.append(None)
        return outputs


def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def token_logprob(item: Any) -> float:
    return float(item.logprob if hasattr(item, "logprob") else item)


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    return float(sum(p[key] * math.log((p[key] + eps) / (q.get(key, 0.0) + eps)) for key in p))


def js_from_topk(p_top: dict[int, float], q_top: dict[int, float]) -> dict[str, float]:
    """JS on the union of top-k atoms plus one leftover-mass bucket."""
    p_mass = {str(key): math.exp(value) for key, value in p_top.items()}
    q_mass = {str(key): math.exp(value) for key, value in q_top.items()}
    p_mass["__tail__"] = max(0.0, 1.0 - sum(p_mass.values()))
    q_mass["__tail__"] = max(0.0, 1.0 - sum(q_mass.values()))
    keys = set(p_mass) | set(q_mass)
    p = {key: p_mass.get(key, 0.0) for key in keys}
    q = {key: q_mass.get(key, 0.0) for key in keys}
    mid = {key: 0.5 * (p[key] + q[key]) for key in keys}
    entropy_p = -sum(mass * math.log(mass) for mass in p.values() if mass > 0.0)
    entropy_q = -sum(mass * math.log(mass) for mass in q.values() if mass > 0.0)
    return {
        "js": 0.5 * kl(p, mid) + 0.5 * kl(q, mid),
        "kl_pq": kl(p, q),
        "kl_qp": kl(q, p),
        "entropy_p": entropy_p,
        "entropy_q": entropy_q,
        "self_certainty_p": -entropy_p,
    }


def load_done(path: Path) -> set[tuple[int, int, str]]:
    """Resume keys only.  Do not parse the stored top-k token dumps."""
    done: set[tuple[int, int, str]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            head = line.split(', "tokens"', 1)[0]
            if not head.rstrip().endswith("}"):
                head = head.rstrip().rstrip(",") + "}"
            row = json.loads(head)
            done.add((int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])))
    return done


def load_jobs(dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trial in json.loads(dense_trial_path(dataset, seed).read_text()):
        grouped[int(trial["question_idx"])].append(trial)
    jobs: list[dict[str, Any]] = []
    for qi, history in grouped.items():
        history.sort(key=lambda row: int(row["stopped_len"]))
        for index, trial in enumerate(history):
            jobs.append({"question_idx": qi, "decision_step": int(trial["stopped_len"]), "index": index})
    return sorted(jobs, key=lambda row: (row["question_idx"], row["decision_step"])), grouped


def build_ids(tokenizer, trial: dict[str, Any]) -> tuple[list[int], int, int]:
    question = str(trial.get("question") or "")
    reasoning = str(trial.get("reasoning_prefix") or "")
    answer = str(trial.get("final_answer") or "")
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}\n{question}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}\n{question}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    completion = f"{reasoning}\n\\boxed{{{answer}}}"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    boxed_ids = tokenizer.encode(f"\\boxed{{{answer}}}", add_special_tokens=False)
    boxed_start = len(full_ids) - len(boxed_ids)
    if boxed_start < len(prompt_ids) or full_ids[boxed_start:] != boxed_ids:
        boxed_start = len(full_ids)
    return full_ids, len(prompt_ids), boxed_start


def extract_topk(output: Any, start: int) -> list[dict[str, Any]]:
    ids = output.prompt_token_ids or []
    entries = output.prompt_logprobs or []
    rows: list[dict[str, Any]] = []
    for index in range(max(start, 1), min(len(ids), len(entries))):
        mapping = {int(token): token_logprob(item) for token, item in (entries[index] or {}).items()}
        actual = int(ids[index])
        rows.append(
            {
                "id": actual,
                "lp": mapping.get(actual, float("nan")),
                "top": mapping,
            }
        )
    return rows


def mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def merge_row(solver_rows: list[dict[str, Any]], small_rows: list[dict[str, Any]], boxed_offset: int, window: int) -> dict[str, Any]:
    n = min(len(solver_rows), len(small_rows))
    js_rows: list[dict[str, float]] = []
    for index in range(n):
        metrics = js_from_topk(solver_rows[index]["top"], small_rows[index]["top"])
        metrics["p_big"] = math.exp(solver_rows[index]["lp"]) if math.isfinite(solver_rows[index]["lp"]) else float("nan")
        metrics["p_small"] = math.exp(small_rows[index]["lp"]) if math.isfinite(small_rows[index]["lp"]) else float("nan")
        js_rows.append(metrics)
    boxed = js_rows[max(boxed_offset, 0) :]
    recent = js_rows[max(0, len(js_rows) - window) :]
    summary = {
        "n_tokens": n,
        "n_boxed": len(boxed),
        "js_mean": mean([row["js"] for row in js_rows]),
        "js_max": max((row["js"] for row in js_rows), default=float("nan")),
        "p_big_boxed": mean([row["p_big"] for row in boxed]),
        "p_small_boxed": mean([row["p_small"] for row in boxed]),
        "p_big_all": mean([row["p_big"] for row in js_rows]),
        "p_small_all": mean([row["p_small"] for row in js_rows]),
        "self_certainty": mean([row["self_certainty_p"] for row in js_rows]),
        "window": [{"js": row["js"], "p_big": row["p_big"], "p_small": row["p_small"]} for row in recent],
    }
    return summary


def score_from_window(summary: dict[str, Any], theta: float, side: str) -> float:
    selected = [float(summary[f"p_{side}_boxed"])] if math.isfinite(finite(summary.get(f"p_{side}_boxed"))) else []
    for row in summary.get("window") or []:
        if finite(row.get("js")) > theta and math.isfinite(finite(row.get(f"p_{side}"))):
            selected.append(float(row[f"p_{side}"]))
    return mean(selected)


def run_gate(args: argparse.Namespace) -> None:
    probes = [
        "A mathematical answer is \\boxed{42}.",
        "Let x=1/3; hence θ=π/2.",
        "<|im_start|>user\nSolve this problem.<|im_end|>",
        INSTRUCTION,
    ]
    jobs, grouped = load_jobs(args.dataset, args.seed)
    for job in jobs[:16]:
        trial = grouped[job["question_idx"]][job["index"]]
        probes.append(str(trial.get("reasoning_prefix") or "")[:400])
        probes.append(str(trial.get("final_answer") or ""))
    result = {
        "solver_model": str(args.solver_model),
        "small_model": str(args.small_model),
        "tokenizer_gate": assert_same_tokenizer(args.solver_model, args.small_model, [text for text in probes if text]),
        "status": "passed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def run_forward(args: argparse.Namespace, model_path: Path) -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir or AE / "results/confcal_judge/v2/keytoken" / f"{args.dataset}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.phase}_shard{args.shard_id}.jsonl"
    print(f"loading jobs dataset={args.dataset} shard={args.shard_id} phase={args.phase}", flush=True)
    jobs, grouped = load_jobs(args.dataset, args.seed)
    qids = sorted(grouped)
    mine = {qi for index, qi in enumerate(qids) if index % args.num_shards == args.shard_id}
    jobs = [job for job in jobs if job["question_idx"] in mine]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"loading jobs dataset={args.dataset} shard={args.shard_id}", flush=True)
    done = load_done(out_path)
    print(f"resume skip={len(done)} remaining_before_filter={len(jobs)}", flush=True)
    jobs = [
        job
        for job in jobs
        if (job["question_idx"], job["decision_step"], str(grouped[job["question_idx"]][job["index"]].get("final_answer") or ""))
        not in done
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(args.solver_model), trust_remote_code=True)
    llm = LLM(
        model=str(model_path),
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        max_logprobs=args.logprobs_k,
        seed=args.seed,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.logprobs_k, detokenize=False)
    started = time.perf_counter()
    with out_path.open("a") as handle:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start : start + args.batch_size]
            prompts, meta = [], []
            for job in batch:
                trial = grouped[job["question_idx"]][job["index"]]
                ids, prompt_len, boxed_start = build_ids(tokenizer, trial)
                if not fits(ids, args.max_context):
                    handle.write(json.dumps({**job, "answer": trial.get("final_answer") or "", "status": "too_long", "n_ids": len(ids)}) + "\n")
                    continue
                prompts.append({"prompt_token_ids": ids})
                meta.append((job, trial, prompt_len, boxed_start, len(ids)))
            outputs = generate_safe(llm, prompts, params)
            for item, output in zip(meta, outputs, strict=True):
                job, trial, prompt_len, boxed_start, n_ids = item
                if output is None:
                    handle.write(json.dumps({**job, "answer": trial.get("final_answer") or "", "status": "error", "n_ids": n_ids}) + "\n")
                    continue
                rows = extract_topk(output, prompt_len)
                handle.write(
                    json.dumps(
                        {
                            "question_idx": job["question_idx"],
                            "decision_step": job["decision_step"],
                            "answer": str(trial.get("final_answer") or ""),
                            "geo_conf": finite(trial.get("confidence")),
                            "status": "ok",
                            "prompt_len": prompt_len,
                            "boxed_offset": max(boxed_start - prompt_len, 0),
                            "n_ids": n_ids,
                            "tokens": rows,
                        }
                    )
                    + "\n"
                )
            handle.flush()
            print(f"[{min(start + len(batch), len(jobs))}/{len(jobs)}] phase={args.phase} elapsed={time.perf_counter()-started:.0f}s", flush=True)


def load_phase(path: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            out[(int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))] = row
    return out


def run_merge(args: argparse.Namespace) -> None:
    out_dir = args.out_dir or AE / "results/confcal_judge/v2/keytoken" / f"{args.dataset}_s{args.seed}"
    scores_path = out_dir / f"scores_shard{args.shard_id}.jsonl"
    solver = load_phase(out_dir / f"solver_shard{args.shard_id}.jsonl")
    small = load_phase(out_dir / f"small_shard{args.shard_id}.jsonl")
    done = load_done(scores_path)
    keys = sorted(set(solver) & set(small) - done)
    with scores_path.open("a") as handle:
        for key in keys:
            left, right = solver[key], small[key]
            summary = merge_row(left["tokens"], right["tokens"], int(left.get("boxed_offset") or 0), args.window)
            payload = {
                "question_idx": key[0],
                "decision_step": key[1],
                "answer": key[2],
                "geo_conf": left.get("geo_conf"),
                "status": "ok",
                "keytoken": summary,
                "score_big_js0": score_from_window(summary, 0.0, "big"),
                "score_small_js0": score_from_window(summary, 0.0, "small"),
                "score_big_js025": score_from_window(summary, 0.25, "big"),
                "score_small_js025": score_from_window(summary, 0.25, "small"),
            }
            handle.write(json.dumps(payload) + "\n")
    print(f"merged {len(keys)} -> {scores_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("gate", "solver", "small", "merge"), required=True)
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver-model", type=Path, default=SOLVER)
    parser.add_argument("--small-model", type=Path, default=Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-1.5B"))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--logprobs-k", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "gate":
        args.out = args.out or AE / "results/confcal_judge/v2/keytoken" / f"{args.dataset}_s{args.seed}_tokenizer_gate.json"
        run_gate(args)
        return
    if args.phase == "solver":
        run_forward(args, args.solver_model)
        return
    if args.phase == "small":
        run_forward(args, args.small_model)
        return
    run_merge(args)


if __name__ == "__main__":
    main()
