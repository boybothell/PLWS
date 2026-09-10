#!/usr/bin/env python3
"""Prompt ablation for the frozen Qwen3-4B judge (pilot subset).

Variants per dense trial step (baseline P0 = existing "do not solve" scores):
  P1_verify : verify-framed preamble, thinking closed, direct Yes/No readout
  P1_ctx    : P1_verify + the solver's recent trial answers listed in context
  P2_think  : verify preamble, open <think>, greedy judge CoT (budgeted),
              then close think and read Yes/No

All variants store raw log-probs so post-hoc scaling can be fitted later.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE))
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_judge import (  # noqa: E402
    QWEN3_ASSISTANT,
    QWEN3_USER_OPEN,
    QWEN4B_MODEL,
    REASONING_HEADER,
    common_prefix_len,
    dense_trial_path,
    yes_no_token_ids,
)
from score_confcal_4b_variants import advance, load_done_dirs  # noqa: E402

VERIFY_PREAMBLE = (
    "You are given a problem and a partial reasoning transcript written by "
    "another model. Your job is to verify the proposed answer given at the "
    "end; you may check the key steps briefly before judging.\n\n[Problem]\n"
)
B_SUFFIX = (
    "\n\n[Proposed final answer] {answer}\n\n[Judgement] Is this proposed "
    'answer correct for the problem? Reply with a single word: "Yes" or "No".'
)
CTX_SUFFIX = (
    "\n\n[Solver's recent trial answers at earlier checkpoints] {recent}"
    "\n\n[Proposed final answer] {answer}\n\n[Judgement] Is this proposed "
    'answer correct for the problem? Reply with a single word: "Yes" or "No".'
)
THINK_OPEN = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
THINK_CLOSE = "\n</think>\n\n"


def yes_no_readout(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> dict[str, float]:
    logprobs = F.log_softmax(logits, dim=-1)
    lp_yes = torch.logsumexp(logprobs[yes_ids], dim=0)
    lp_no = torch.logsumexp(logprobs[no_ids], dim=0)
    margin = float(lp_yes - lp_no)
    return {
        "yes": float(torch.sigmoid(lp_yes - lp_no)),
        "logp_yes": float(lp_yes),
        "logp_no": float(lp_no),
        "margin": margin,
    }


@torch.inference_mode()
def greedy_think(
    model,
    cache,
    prefix_ids: list[int],
    suffix_ids: list[int],
    *,
    device,
    chunk_tokens: int,
    max_new: int,
    end_ids: list[int],
) -> tuple[Any, list[int], list[int]]:
    """Extend cache with suffix then greedily decode up to max_new tokens."""
    full = prefix_ids + suffix_ids
    cache, logits, _ = advance(
        model, cache, full, cache_len=len(prefix_ids), device=device, chunk_tokens=chunk_tokens
    )
    generated: list[int] = []
    ids = list(full)
    for _ in range(max_new):
        token = int(logits.argmax())
        generated.append(token)
        ids.append(token)
        if generated[-len(end_ids):] == end_ids:
            break
        out = model(
            input_ids=torch.tensor([[token]], device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1].float()
        del out
    return cache, ids, generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-questions", type=int, default=32)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=14336)
    parser.add_argument("--think-tokens", type=int, default=192)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or (
        AE / "results/confcal_judge/prompt_ablation" / f"{args.dataset}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"

    trials = json.loads(dense_trial_path(args.dataset, args.seed).read_text())
    by_q: dict[int, list[dict]] = defaultdict(list)
    for trial in trials:
        by_q[int(trial["question_idx"])].append(trial)
    qis = sorted(by_q)
    take = max(1, len(qis) // args.n_questions)
    chosen = qis[::take][: args.n_questions]

    jobs: list[dict[str, Any]] = []
    for qi in chosen:
        steps = sorted(by_q[qi], key=lambda t: int(t["stopped_len"]))
        seen: set[tuple[int, str]] = set()
        history: list[tuple[int, str]] = []
        for trial in steps:
            ans = str(trial.get("final_answer") or "")
            step = int(trial["stopped_len"])
            key = (step, ans)
            recent = history[-4:]
            if key not in seen:
                seen.add(key)
                jobs.append(
                    {
                        "question_idx": qi,
                        "decision_step": step,
                        "answer": ans,
                        "question": trial.get("question") or "",
                        "reasoning_prefix": trial.get("reasoning_prefix") or "",
                        "recent": list(recent),
                    }
                )
            history.append((step, ans))

    done = load_done_dirs([scores_path])
    pending = [j for j in jobs if (j["question_idx"], j["decision_step"], j["answer"]) not in done]
    print(f"questions={len(chosen)} jobs={len(jobs)} pending={len(pending)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(QWEN4B_MODEL, trust_remote_code=True)
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    think_end = tokenizer.encode("\n</think>", add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN4B_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device

    body_cache = None
    old_body: list[int] = []
    prev_qi = None
    ok = skipped = 0
    started = time.perf_counter()
    with scores_path.open("a") as out:
        for number, job in enumerate(pending, 1):
            qi, step, answer = job["question_idx"], job["decision_step"], job["answer"]
            if qi != prev_qi:
                body_cache, old_body, prev_qi = None, [], qi
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            body_text = (
                f"{QWEN3_USER_OPEN}{VERIFY_PREAMBLE}{job['question']}"
                f"{REASONING_HEADER}{job['reasoning_prefix']}"
            )
            body_ids = tokenizer.encode(body_text, add_special_tokens=False)
            recent = job["recent"]
            recent_text = (
                "; ".join(f"step {s}: {a or '(none)'}" for s, a in recent) if recent else "(none yet)"
            )
            branches = {
                "P1_verify": B_SUFFIX.format(answer=answer or "(none)") + QWEN3_ASSISTANT,
                "P1_ctx": CTX_SUFFIX.format(recent=recent_text, answer=answer or "(none)") + QWEN3_ASSISTANT,
            }
            branch_ids = {
                key: tokenizer.encode(text, add_special_tokens=False) for key, text in branches.items()
            }
            think_suffix = tokenizer.encode(
                B_SUFFIX.format(answer=answer or "(none)") + THINK_OPEN, add_special_tokens=False
            )
            if len(body_ids) + max(len(v) for v in branch_ids.values()) + args.think_tokens > args.max_context:
                out.write(json.dumps({
                    "question_idx": qi, "decision_step": step, "answer": answer,
                    "status": "too_long", "body_tokens": len(body_ids),
                }) + "\n")
                skipped += 1
                continue
            common = common_prefix_len(old_body, body_ids) if old_body else 0
            if old_body and common != len(old_body):
                body_cache, common = None, 0
            try:
                body_cache, _, _ = advance(
                    model, body_cache, body_ids,
                    cache_len=common if body_cache is not None else 0,
                    device=device, chunk_tokens=args.chunk_tokens,
                )
                scores: dict[str, Any] = {}
                for key, ids in branch_ids.items():
                    body_cache, logits, _ = advance(
                        model, body_cache, body_ids + ids, cache_len=len(body_ids),
                        device=device, chunk_tokens=args.chunk_tokens,
                    )
                    scores[key] = yes_no_readout(logits, yes_ids, no_ids)
                    body_cache.crop(len(body_ids))
                body_cache, full_ids, generated = greedy_think(
                    model, body_cache, body_ids, think_suffix,
                    device=device, chunk_tokens=args.chunk_tokens,
                    max_new=args.think_tokens, end_ids=think_end,
                )
                if generated[-len(think_end):] != think_end:
                    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
                else:
                    close_ids = tokenizer.encode("\n\n", add_special_tokens=False)
                body_cache, logits, _ = advance(
                    model, body_cache, full_ids + close_ids, cache_len=len(full_ids),
                    device=device, chunk_tokens=args.chunk_tokens,
                )
                scores["P2_think"] = {
                    **yes_no_readout(logits, yes_ids, no_ids),
                    "think_tokens": len(generated),
                    "think_text": tokenizer.decode(generated)[-600:],
                }
                body_cache.crop(len(body_ids))
            except Exception as exc:
                out.write(json.dumps({
                    "question_idx": qi, "decision_step": step, "answer": answer,
                    "status": "forward_fail", "error": str(exc)[:200],
                }) + "\n")
                skipped += 1
                body_cache, old_body = None, []
                continue
            old_body = body_ids
            ok += 1
            out.write(json.dumps({
                "question_idx": qi, "decision_step": step, "answer": answer,
                "status": "ok", "body_tokens": len(body_ids), **scores,
            }) + "\n")
            out.flush()
            if number == 1 or number % 10 == 0:
                print(
                    f"[{number}/{len(pending)}] ok={ok} skip={skipped} "
                    f"{time.perf_counter()-started:.0f}s qi={qi} step={step}",
                    flush=True,
                )
    print(f"done ok={ok} skip={skipped} -> {scores_path}", flush=True)


if __name__ == "__main__":
    main()
