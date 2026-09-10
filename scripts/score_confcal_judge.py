#!/usr/bin/env python3
"""Judge-mode confidence calibration on held local-K=4 windows.

The original solver prefix is masked. A fresh user turn quotes the problem
and partial reasoning as material; the model is not asked to continue the
CoT. Thinking is closed (`enable_thinking=False` / explicit </think>).

Conditions
  A : problem + reasoning          -> already sufficient to determine the answer?
  B : problem + reasoning + trial A -> is this proposed answer correct?

Readouts
  yesno  : next-token P(Yes) / (P(Yes)+P(No))   [primary]
  verbal : greedy 0-100 integer parsed to [0,1] [control]

Judges
  self   : DeepSeek-R1-Distill-Qwen-7B
  qwen4b : Qwen3-4B (same Instruct checkpoint as crosscal_verdict)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE))

CANDIDATES = AE / "results/fs_process_support_any_safe/candidates.jsonl"
SELF_MODEL = Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B")
QWEN4B_MODEL = Path("/mnt/d/lsj/models/Qwen3-4B")

PREAMBLE = (
    "You are given a problem and a partial reasoning transcript written by "
    "another model. Judge it as asked at the end. Do not solve the problem "
    "yourself.\n\n[Problem]\n"
)
REASONING_HEADER = "\n\n[Partial reasoning]\n"
A_YESNO = (
    "\n\n[Judgement] Is the reasoning above already sufficient to determine "
    'the final answer to the problem? Reply with a single word: "Yes" or "No".'
)
B_YESNO = (
    "\n\n[Proposed final answer] {answer}\n\n[Judgement] Is this proposed "
    'answer correct for the problem? Reply with a single word: "Yes" or "No".'
)
A_VERBAL = (
    "\n\n[Judgement] Based only on the reasoning above, what is the "
    "probability (0-100) that a correct final answer can already be "
    "determined? Reply with a single integer from 0 to 100 and nothing else."
)
B_VERBAL = (
    "\n\n[Proposed final answer] {answer}\n\n[Judgement] Based on the "
    "reasoning and the proposed answer, what is the probability (0-100) "
    "that this answer is correct? Reply with a single integer from 0 to 100 "
    "and nothing else."
)
QWEN3_USER_OPEN = "<|im_start|>user\n"
QWEN3_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
VERBAL_RE = re.compile(r"(\d{1,3})")


def dense_trial_path(dataset: str, seed: int) -> Path:
    root = AE / f"results/dense_G_r1_7b/{dataset}"
    candidates = (
        root / "dense_puma" / "trial_answers.json",
        root / f"seed_{seed}" / "dense_puma" / "trial_answers.json",
        root / f"s{seed}" / "dense_puma" / "trial_answers.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no dense trial_answers.json for dataset={dataset} seed={seed}; tried {candidates}"
    )


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def parse_verbal(text: str | None) -> float | None:
    """Accept only a leading 0-100 integer. Mid-sentence digits are not confidence."""
    if not text:
        return None
    match = re.match(r"^\s*(\d{1,3})\s*%?\b", text)
    if not match:
        return None
    value = int(match.group(1))
    if 0 <= value <= 100:
        return value / 100.0
    return None


def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    def first_ids(variants: list[str]) -> list[int]:
        ids: list[int] = []
        for text in variants:
            enc = tokenizer.encode(text, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return ids

    yes = first_ids(["Yes", " Yes", "yes", " yes", "YES"])
    no = first_ids(["No", " No", "no", " no", "NO"])
    overlap = set(yes) & set(no)
    if overlap:
        raise RuntimeError(f"Yes/No token id overlap: {overlap}")
    return yes, no


def chat_wrappers(tokenizer, judge: str) -> tuple[str, str]:
    """Return (user_open, assistant_header) so the body can stay in KV cache."""
    if judge == "qwen4b":
        return QWEN3_USER_OPEN, QWEN3_ASSISTANT
    marker = "<<<BODY>>>"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if marker not in text:
        raise RuntimeError("chat template did not preserve the body marker")
    user_open, assistant = text.split(marker, 1)
    stripped = assistant.rstrip()
    if stripped.endswith("<think>"):
        assistant = stripped + "\n</think>\n\n"
    elif "<think>" in assistant and "</think>" not in assistant:
        assistant = assistant + "</think>\n\n"
    return user_open, assistant


def held_questions(dataset: str, seed: int, model_id: str) -> set[int]:
    out: set[int] = set()
    with CANDIDATES.open() as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("model") == model_id
                and row.get("dataset") == dataset
                and int(row.get("seed", 42)) == seed
                and row.get("split") == "held"
            ):
                out.add(int(row["question_idx"]))
    return out


def load_jobs(dataset: str, seed: int, model_id: str, split: str) -> list[dict[str, Any]]:
    """Every dense trial step on held questions (trial-as-final, any-step G)."""
    keep = held_questions(dataset, seed, model_id) if split == "held" else None
    jobs: dict[tuple[int, int, str], dict[str, Any]] = {}
    for trial in json.loads(dense_trial_path(dataset, seed).read_text()):
        qi = int(trial["question_idx"])
        if keep is not None and qi not in keep:
            continue
        key = (qi, int(trial["stopped_len"]), str(trial.get("final_answer") or ""))
        if key in jobs:
            continue
        jobs[key] = {
            "question_idx": key[0],
            "decision_step": key[1],
            "answer": key[2],
            "split": "held" if keep is None or qi in keep else "",
            "geo_conf": float(trial.get("confidence") or float("nan")),
            "mean_conf_K": float(trial.get("confidence") or float("nan")),
        }
    return [jobs[key] for key in sorted(jobs)]


def load_done(path: Path) -> set[tuple[int, int, str]]:
    done: set[tuple[int, int, str]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(
                (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
            )
    return done


@torch.inference_mode()
def advance_cache(
    model,
    cache,
    full_ids: list[int],
    *,
    cache_len: int,
    input_device: torch.device,
    chunk_tokens: int,
    want_last_logits: bool = False,
):
    if cache is None:
        cache = DynamicCache(config=model.config)
        start = 0
    else:
        start = min(cache_len, len(full_ids))
        cache.crop(start)
    last_logits = None
    fed = 0
    ids = torch.tensor(full_ids, dtype=torch.long)
    for chunk_start in range(start, len(full_ids), chunk_tokens):
        chunk_end = min(len(full_ids), chunk_start + chunk_tokens)
        chunk = ids[chunk_start:chunk_end].unsqueeze(0).to(input_device)
        outputs = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        fed += chunk_end - chunk_start
        cache = outputs.past_key_values
        if want_last_logits and chunk_end == len(full_ids):
            last_logits = outputs.logits[0, -1].float()
        del outputs, chunk
    return cache, last_logits, fed


def verdict_from_logits(logits, yes_ids, no_ids) -> dict[str, float]:
    probs = F.softmax(logits, dim=-1)
    p_yes = float(probs[yes_ids].sum())
    p_no = float(probs[no_ids].sum())
    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "p_yes_norm": p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else float("nan"),
    }


@torch.inference_mode()
def greedy_continue(
    model,
    cache,
    last_logits,
    *,
    input_device: torch.device,
    max_new: int,
    stop_ids: set[int],
) -> tuple[list[int], Any]:
    tokens: list[int] = []
    logits = last_logits
    for _ in range(max_new):
        nxt = int(torch.argmax(logits))
        tokens.append(nxt)
        if nxt in stop_ids:
            break
        outputs = model(
            input_ids=torch.tensor([[nxt]], device=input_device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1].float()
        del outputs
    return tokens, cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="r1_7b")
    parser.add_argument("--judge", choices=("self", "qwen4b"), required=True)
    parser.add_argument("--split", default="held")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--verbal-tokens", type=int, default=8)
    parser.add_argument("--skip-verbal", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    model_path = SELF_MODEL if args.judge == "self" else QWEN4B_MODEL
    out_dir = args.out_dir or (
        AE / "results/confcal_judge" / args.judge / f"{args.dataset}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"scores_shard{args.shard_id}.jsonl"

    jobs = load_jobs(args.dataset, args.seed, args.model_id, args.split)
    qis = sorted({job["question_idx"] for job in jobs})
    my_qis = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [job for job in jobs if job["question_idx"] in my_qis]
    jobs.sort(key=lambda job: (job["question_idx"], job["decision_step"], job["answer"]))
    if args.limit:
        jobs = jobs[: args.limit]
    done = load_done(scores_path)
    pending = [job for job in jobs if (job["question_idx"], job["decision_step"], job["answer"]) not in done]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    user_open, assistant_header = chat_wrappers(tokenizer, args.judge)
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    stop_ids = set()
    for extra in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if extra is not None:
            stop_ids.add(int(extra))
    for text in ("\n", "\n\n"):
        enc = tokenizer.encode(text, add_special_tokens=False)
        if enc:
            stop_ids.add(enc[0])

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    trials = json.loads(dense_trial_path(args.dataset, args.seed).read_text())
    trial_map: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for trial in trials:
        trial_map[int(trial["question_idx"])][int(trial["stopped_len"])] = trial

    meta = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model_id": args.model_id,
        "judge": args.judge,
        "judge_model": str(model_path),
        "split": args.split,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "n_jobs": len(jobs),
        "n_pending": len(pending),
        "user_open": user_open,
        "assistant_header": assistant_header,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "prompts": {
            "preamble": PREAMBLE,
            "reasoning_header": REASONING_HEADER,
            "A_yesno": A_YESNO,
            "B_yesno": B_YESNO,
            "A_verbal": A_VERBAL,
            "B_verbal": B_VERBAL,
        },
    }
    (out_dir / f"meta_shard{args.shard_id}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"judge={args.judge} shard={args.shard_id}/{args.num_shards} "
        f"jobs={len(jobs)} pending={len(pending)} device={args.device}",
        flush=True,
    )

    body_cache = None
    body_ids_prev: list[int] = []
    prev_qi: int | None = None
    n_ok = n_skip = 0
    t0 = time.perf_counter()
    with scores_path.open("a") as handle:
        for num, job in enumerate(pending, 1):
            qi = job["question_idx"]
            step = job["decision_step"]
            trial = trial_map.get(qi, {}).get(step)
            if trial is None:
                n_skip += 1
                handle.write(json.dumps({**job, "judge": args.judge, "status": "missing_trial"}) + "\n")
                handle.flush()
                continue
            if qi != prev_qi:
                body_cache = None
                body_ids_prev = []
                prev_qi = qi
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

            question = str(trial.get("question") or "")
            reasoning = str(trial.get("reasoning_prefix") or "")
            body_text = f"{user_open}{PREAMBLE}{question}{REASONING_HEADER}{reasoning}"
            body_ids = tokenizer.encode(body_text, add_special_tokens=False)
            answer = job["answer"]
            branches = {
                "A_yesno": A_YESNO,
                "B_yesno": B_YESNO.format(answer=answer or "(none)"),
            }
            if not args.skip_verbal:
                branches["A_verbal"] = A_VERBAL
                branches["B_verbal"] = B_VERBAL.format(answer=answer or "(none)")
            branch_ids = {
                name: tokenizer.encode(text + assistant_header, add_special_tokens=False)
                for name, text in branches.items()
            }
            max_branch = max(len(ids) for ids in branch_ids.values())
            if len(body_ids) + max_branch + (0 if args.skip_verbal else args.verbal_tokens) > args.max_context:
                n_skip += 1
                handle.write(
                    json.dumps(
                        {
                            **job,
                            "judge": args.judge,
                            "status": "too_long",
                            "body_tokens": len(body_ids),
                        }
                    )
                    + "\n"
                )
                handle.flush()
                continue

            common = common_prefix_len(body_ids_prev, body_ids) if body_ids_prev else 0
            if body_ids_prev and common != len(body_ids_prev):
                body_cache = None
                common = 0
            try:
                body_cache, _, _ = advance_cache(
                    model,
                    body_cache,
                    body_ids,
                    cache_len=common if body_cache is not None else 0,
                    input_device=input_device,
                    chunk_tokens=args.chunk_tokens,
                )
                scores: dict[str, Any] = {}
                for name, ids in branch_ids.items():
                    body_cache, last_logits, _ = advance_cache(
                        model,
                        body_cache,
                        body_ids + ids,
                        cache_len=len(body_ids),
                        input_device=input_device,
                        chunk_tokens=args.chunk_tokens,
                        want_last_logits=True,
                    )
                    if name.endswith("yesno"):
                        scores[name] = verdict_from_logits(last_logits, yes_ids, no_ids)
                    else:
                        gen_ids, body_cache = greedy_continue(
                            model,
                            body_cache,
                            last_logits,
                            input_device=input_device,
                            max_new=args.verbal_tokens,
                            stop_ids=stop_ids,
                        )
                        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        scores[name] = parse_verbal(text)
                        scores[f"{name}_text"] = text.strip()[:40]
                    body_cache.crop(len(body_ids))
            except Exception as exc:
                n_skip += 1
                handle.write(
                    json.dumps(
                        {
                            **job,
                            "judge": args.judge,
                            "status": "forward_fail",
                            "error": str(exc)[:200],
                        }
                    )
                    + "\n"
                )
                handle.flush()
                body_cache = None
                body_ids_prev = []
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            body_ids_prev = body_ids
            n_ok += 1
            handle.write(
                json.dumps(
                    {
                        **job,
                        "judge": args.judge,
                        "status": "ok",
                        "body_tokens": len(body_ids),
                        **scores,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if num == 1 or num % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[{num}/{len(pending)}] ok={n_ok} skip={n_skip} "
                    f"{elapsed:.0f}s qi={qi} step={step}",
                    flush=True,
                )

    print(f"done ok={n_ok} skip={n_skip} -> {scores_path}", flush=True)


if __name__ == "__main__":
    main()
