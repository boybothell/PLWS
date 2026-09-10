#!/usr/bin/env python3
"""Training-free confidence readouts from a frozen Qwen3-4B judge.

The extractor is deliberately separate from score_confcal_judge.py so existing
baseline scores remain immutable.  It scores every dense trial on either the
calibration or held question split and stores all primitive signals:

* B Yes/No final-layer and last-layer-ensemble (EAGLE-YN);
* 0--9 final-layer and layer-ensemble expected scores (EAGLE-10bin);
* Correct/Incorrect/Insufficient-evidence masses;
* cautious/neutral/optimistic Yes/No branches;
* verifier surprise (CMP) and uncertainty (CME) over answer tokens.
"""
from __future__ import annotations

import argparse
import json
import math
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

from score_confcal_judge import (  # noqa: E402
    B_YESNO,
    PREAMBLE,
    QWEN3_ASSISTANT,
    QWEN3_USER_OPEN,
    REASONING_HEADER,
    QWEN4B_MODEL,
    common_prefix_len,
    dense_trial_path,
    load_done,
    yes_no_token_ids,
)

THREE_WAY = (
    "\n\n[Proposed final answer] {answer}\n\n[Judgement] Which best describes the "
    "proposed answer given the problem and partial reasoning? Reply with one word: "
    '"Correct", "Incorrect", or "Insufficient".'
)
SCORE_10 = (
    "\n\n[Proposed final answer] {answer}\n\n[Judgement] Rate the probability that "
    "this answer is correct from 0 (certainly incorrect) to 9 (certainly correct). "
    "Reply with one digit only."
)
STEER = {
    "cautious": (
        "\n\n[Proposed final answer] {answer}\n\n[Judgement] Be very cautious: only "
        'say "Yes" if the answer is decisively correct from the supplied material. '
        'Reply with one word: "Yes" or "No".'
    ),
    "neutral": B_YESNO,
    "optimistic": (
        "\n\n[Proposed final answer] {answer}\n\n[Judgement] Give the proposed answer "
        'the benefit of reasonable doubt when the supplied material supports it. Reply '
        'with one word: "Yes" or "No".'
    ),
}
CMP_PREFIX = "\n\n[Proposed final answer]\n"


def load_jobs(dataset: str, seed: int) -> list[dict[str, Any]]:
    jobs: dict[tuple[int, int, str], dict[str, Any]] = {}
    for trial in json.loads(dense_trial_path(dataset, seed).read_text()):
        key = (
            int(trial["question_idx"]),
            int(trial["stopped_len"]),
            str(trial.get("final_answer") or ""),
        )
        jobs.setdefault(
            key,
            {
                "question_idx": key[0],
                "decision_step": key[1],
                "answer": key[2],
                "geo_conf": float(trial.get("confidence") or float("nan")),
            },
        )
    return [jobs[k] for k in sorted(jobs)]


def load_done_dirs(paths: list[Path]) -> set[tuple[int, int, str]]:
    done: set[tuple[int, int, str]] = set()
    for path in paths:
        if path.is_file():
            done |= load_done(path)
            continue
        if not path.is_dir():
            continue
        for shard in path.glob("scores*.jsonl"):
            done |= load_done(shard)
    return done


def class_token_ids(
    tokenizer, labels: list[str], *, include_spaced: bool = True
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for label in labels:
        ids: list[int] = []
        variants = [label, label.lower()]
        if include_spaced:
            variants += [f" {label}", f" {label.lower()}"]
        for text in variants:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokens and tokens[0] not in ids:
                ids.append(tokens[0])
        if not ids:
            raise RuntimeError(f"no token ids for {label!r}")
        out[label] = ids
    overlap: dict[int, list[str]] = defaultdict(list)
    for label, ids in out.items():
        for token_id in ids:
            overlap[token_id].append(label)
    bad = {token_id: names for token_id, names in overlap.items() if len(names) > 1}
    if bad:
        raise RuntimeError(f"overlapping class token ids: {bad}")
    return out


def normalized_classes(logits: torch.Tensor, ids: dict[str, list[int]]) -> dict[str, float]:
    probs = F.softmax(logits, dim=-1)
    masses = {label: probs[token_ids].sum() for label, token_ids in ids.items()}
    denom = sum(masses.values())
    return {
        label: float(value / denom) if float(denom) > 0 else float("nan")
        for label, value in masses.items()
    }


@torch.inference_mode()
def eagle_score(
    hidden_states: tuple[torch.Tensor, ...],
    lm_head,
    final_norm,
    ids: dict[str, list[int]],
    n_layers: int,
) -> dict[str, float]:
    """Average logits from final n transformer layers, then normalize classes."""
    states = hidden_states[-n_layers:]
    # Qwen exposes intermediate residual streams before its final RMSNorm.  The
    # ordinary LM head sees that norm, so apply it to every logit-lens layer.
    logits = torch.stack([lm_head(final_norm(state[0, -1])).float() for state in states]).mean(0)
    return normalized_classes(logits, ids)


@torch.inference_mode()
def advance(
    model,
    cache: DynamicCache | None,
    ids: list[int],
    *,
    cache_len: int,
    device: torch.device,
    chunk_tokens: int,
    output_hidden_states: bool = False,
) -> tuple[DynamicCache, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
    if cache is None:
        cache = DynamicCache(config=model.config)
        start = 0
    else:
        start = min(cache_len, len(ids))
        cache.crop(start)
    last_logits = None
    last_hidden = None
    for lo in range(start, len(ids), chunk_tokens):
        hi = min(len(ids), lo + chunk_tokens)
        chunk = torch.tensor([ids[lo:hi]], device=device)
        outputs = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            output_hidden_states=output_hidden_states and hi == len(ids),
        )
        cache = outputs.past_key_values
        if hi == len(ids):
            last_logits = outputs.logits[0, -1].float()
            last_hidden = outputs.hidden_states if output_hidden_states else None
        del outputs, chunk
    return cache, last_logits, last_hidden


@torch.inference_mode()
def cmp_cme(
    model,
    cache: DynamicCache,
    body_ids: list[int],
    prefix_ids: list[int],
    answer_ids: list[int],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Teacher-force proposed-answer tokens under the frozen verifier."""
    if not answer_ids:
        return {"cmp": float("nan"), "cme": float("nan"), "n_tokens": 0}
    # A single non-cached postfix call exposes every answer-token prediction.
    cache.crop(len(body_ids))
    postfix = torch.tensor([prefix_ids + answer_ids], device=device)
    outputs = model(
        input_ids=postfix,
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    pred_start = max(len(prefix_ids) - 1, 0)
    pred_logits = outputs.logits[0, pred_start : pred_start + len(answer_ids)].float()
    if len(pred_logits) != len(answer_ids):
        # A zero-token prefix is not expected but preserve a safe failure mode.
        return {"cmp": float("nan"), "cme": float("nan"), "n_tokens": 0}
    target = torch.tensor(answer_ids, device=device)
    nll = F.cross_entropy(pred_logits, target, reduction="none")
    probs = F.softmax(pred_logits, dim=-1)
    entropy = -(probs * F.log_softmax(pred_logits, dim=-1)).sum(-1)
    del outputs, postfix, pred_logits, probs
    return {
        "cmp": float(torch.exp(nll.mean())),
        "cme": float(entropy.mean()),
        "n_tokens": len(answer_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="r1_7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--eagle-layers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or (
        AE / "results/confcal_judge/qwen4b_variants" / f"{args.dataset}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"scores_shard{args.shard_id}.jsonl"
    jobs = load_jobs(args.dataset, args.seed)
    qis = sorted({job["question_idx"] for job in jobs})
    mine = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [job for job in jobs if job["question_idx"] in mine]
    if args.limit:
        jobs = jobs[: args.limit]
    done = load_done_dirs(
        [
            scores_path,
            out_dir,
            AE / "results/confcal_judge/qwen4b_variants" / f"{args.dataset}_s{args.seed}_cal",
        ]
    )
    pending = [job for job in jobs if (job["question_idx"], job["decision_step"], job["answer"]) not in done]

    tokenizer = AutoTokenizer.from_pretrained(QWEN4B_MODEL, trust_remote_code=True)
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yn_ids = {"yes": yes_ids, "no": no_ids}
    three_ids = class_token_ids(tokenizer, ["Correct", "Incorrect", "Insufficient"])
    # Qwen has distinct unspaced digit tokens but one shared leading-space token.
    # The prompt requires one digit only, so score its direct vocabulary completion.
    digit_ids = class_token_ids(tokenizer, [str(i) for i in range(10)], include_spaced=False)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN4B_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    layer_count = int(getattr(model.config, "num_hidden_layers", 0))
    if not 1 <= args.eagle_layers <= layer_count:
        raise ValueError(f"--eagle-layers must be in [1,{layer_count}]")

    trials = json.loads(dense_trial_path(args.dataset, args.seed).read_text())
    trial_map: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for trial in trials:
        trial_map[int(trial["question_idx"])][int(trial["stopped_len"])] = trial

    meta = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model_id": args.model_id,
        "split": "all",
        "judge_model": str(QWEN4B_MODEL),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "n_jobs": len(jobs),
        "n_pending": len(pending),
        "eagle_layers": args.eagle_layers,
        "signals": ["B_yesno", "B_3way", "EAGLE_yn", "EAGLE_10bin", "SteerConf", "CMP", "CME"],
    }
    (out_dir / f"meta_shard{args.shard_id}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"split=all shard={args.shard_id}/{args.num_shards} jobs={len(jobs)} pending={len(pending)}", flush=True)

    body_cache = None
    old_body: list[int] = []
    prev_qi = None
    ok = skipped = 0
    started = time.perf_counter()
    with scores_path.open("a") as out:
        for number, job in enumerate(pending, 1):
            qi, step, answer = job["question_idx"], job["decision_step"], job["answer"]
            trial = trial_map.get(qi, {}).get(step)
            if trial is None:
                out.write(json.dumps({**job, "status": "missing_trial"}) + "\n")
                skipped += 1
                continue
            if qi != prev_qi:
                body_cache, old_body, prev_qi = None, [], qi
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            body_text = (
                f"{QWEN3_USER_OPEN}{PREAMBLE}{trial.get('question') or ''}"
                f"{REASONING_HEADER}{trial.get('reasoning_prefix') or ''}"
            )
            body_ids = tokenizer.encode(body_text, add_special_tokens=False)
            branch_text = {
                "B_yesno": B_YESNO.format(answer=answer or "(none)"),
                "B_3way": THREE_WAY.format(answer=answer or "(none)"),
                "EAGLE_10bin": SCORE_10.format(answer=answer or "(none)"),
                **{f"SteerConf_{key}": value.format(answer=answer or "(none)") for key, value in STEER.items()},
            }
            branch_ids = {
                key: tokenizer.encode(text + QWEN3_ASSISTANT, add_special_tokens=False)
                for key, text in branch_text.items()
            }
            cmp_prefix = tokenizer.encode(CMP_PREFIX, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer or "(none)", add_special_tokens=False)
            max_suffix = max([len(ids) for ids in branch_ids.values()] + [len(cmp_prefix) + len(answer_ids)])
            if len(body_ids) + max_suffix > args.max_context:
                out.write(json.dumps({**job, "status": "too_long", "body_tokens": len(body_ids)}) + "\n")
                skipped += 1
                continue
            common = common_prefix_len(old_body, body_ids) if old_body else 0
            if old_body and common != len(old_body):
                body_cache, common = None, 0
            try:
                body_cache, _, _ = advance(
                    model, body_cache, body_ids, cache_len=common if body_cache is not None else 0,
                    device=device, chunk_tokens=args.chunk_tokens
                )
                scores: dict[str, Any] = {}
                for key, ids in branch_ids.items():
                    body_cache, logits, hidden = advance(
                        model, body_cache, body_ids + ids, cache_len=len(body_ids), device=device,
                        chunk_tokens=args.chunk_tokens, output_hidden_states=key in {"B_yesno", "EAGLE_10bin"}
                    )
                    if key == "B_yesno":
                        scores[key] = normalized_classes(logits, yn_ids)
                        scores["EAGLE_yn"] = eagle_score(
                            hidden, model.lm_head, model.model.norm, yn_ids, args.eagle_layers
                        )
                    elif key == "B_3way":
                        scores[key] = normalized_classes(logits, three_ids)
                    elif key == "EAGLE_10bin":
                        final = normalized_classes(logits, digit_ids)
                        eagle = eagle_score(
                            hidden, model.lm_head, model.model.norm, digit_ids, args.eagle_layers
                        )
                        scores[key] = {
                            "final_expected": sum(int(k) * v for k, v in final.items()) / 9.0,
                            "eagle_expected": sum(int(k) * v for k, v in eagle.items()) / 9.0,
                            "final_distribution": final,
                            "eagle_distribution": eagle,
                        }
                    else:
                        scores[key] = normalized_classes(logits, yn_ids)
                    body_cache.crop(len(body_ids))
                steer = [scores[f"SteerConf_{key}"]["yes"] for key in STEER]
                scores["SteerConf"] = {
                    "mean": float(sum(steer) / len(steer)),
                    "min": float(min(steer)),
                    "max": float(max(steer)),
                    "spread": float(max(steer) - min(steer)),
                }
                scores["CMP_CME"] = cmp_cme(
                    model, body_cache, body_ids, cmp_prefix, answer_ids, device=device
                )
                body_cache.crop(len(body_ids))
            except Exception as exc:
                out.write(json.dumps({**job, "status": "forward_fail", "error": str(exc)[:200]}) + "\n")
                skipped += 1
                body_cache, old_body = None, []
                continue
            old_body = body_ids
            ok += 1
            out.write(json.dumps({**job, "status": "ok", "body_tokens": len(body_ids), **scores}) + "\n")
            out.flush()
            if number == 1 or number % 10 == 0:
                print(f"[{number}/{len(pending)}] ok={ok} skip={skipped} {time.perf_counter()-started:.0f}s qi={qi} step={step}", flush=True)
    print(f"done ok={ok} skip={skipped} -> {scores_path}", flush=True)


if __name__ == "__main__":
    main()
