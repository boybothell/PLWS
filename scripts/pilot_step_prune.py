#!/usr/bin/env python3
"""Step-entropy prune-and-reanswer on first geo-stop.

Paper static test: drop the lowest-entropy steps, then ask for the answer.
Hypothesis: true G is fixed by the high-entropy skeleton; false plateaus
collapse when their fluent (low-entropy) support is removed.
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

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE))
sys.path.insert(0, str(AE / "scripts"))

from attn_early_exit.answers import answers_equal  # noqa: E402
from pilot_active_stop import extract_boxed  # noqa: E402
from pilot_counterfactual_support import fast_eq  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

ANSWER_CUE = "\n</think>\nThe final answer is \\boxed{"
KAPPAS = (0.5, 0.8)
MAX_NEW = 32


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def done_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
        for row in load_jsonl(path)
        if row.get("status") == "ok"
    }


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def nanmean(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def compress_steps(steps: list[Any], ents: list[float], kappa: float) -> tuple[str, int]:
    n = len(steps)
    drop_n = min(max(int(round(kappa * n)), 0), n - 1)
    order = np.argsort(np.asarray(ents, dtype=float))
    drop = set(int(i) for i in order[:drop_n] if math.isfinite(ents[int(i)]))
    if len(drop) >= n:
        drop = set(int(i) for i in order[: n - 1])
    parts = [("[SKIP]" if i in drop else step.text) for i, step in enumerate(steps)]
    return "\n\n".join(parts), len(drop)


def score_cmd(args: argparse.Namespace) -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from attn_early_exit.steps import split_steps_by_delim

    rows = load_jsonl(args.candidates)
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [row for row in rows if int(row["question_idx"]) in keep]
    jobs.sort(key=lambda row: (int(row["question_idx"]), int(row["decision_step"])))
    if args.limit:
        jobs = jobs[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in jobs
        if (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in done_keys(args.out)
    ]

    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(SOLVER),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    print(
        f"step-prune shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"device={device} -> {args.out}",
        flush=True,
    )

    @torch.inference_mode()
    def token_entropy(ids: list[int]) -> list[float]:
        cache = DynamicCache(config=model.config)
        ents = [float("nan")] * len(ids)
        prev_logits = None
        for lo in range(0, len(ids), args.chunk_tokens):
            hi = min(len(ids), lo + args.chunk_tokens)
            chunk = torch.tensor([ids[lo:hi]], device=device)
            out = model(input_ids=chunk, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = out.logits[0].float()
            logp = torch.log_softmax(logits, dim=-1)
            token_ent = -(logp.exp() * logp).sum(-1)
            if lo > 0 and prev_logits is not None:
                prev_logp = torch.log_softmax(prev_logits, dim=-1)
                ents[lo] = float(-(prev_logp.exp() * prev_logp).sum().item())
            for offset in range(hi - lo - 1):
                ents[lo + offset + 1] = float(token_ent[offset].item())
            prev_logits = logits[-1].detach()
            del out, chunk, logits, logp, token_ent
        return ents

    @torch.inference_mode()
    def boxed_from(ids: list[int]) -> tuple[str, str | None]:
        if len(ids) + MAX_NEW > args.max_context:
            return "", None
        input_ids = torch.tensor([ids], device=device)
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
        del out, input_ids
        return text, extract_boxed("\\boxed{" + text)

    ok = skipped = 0
    started = time.perf_counter()
    with args.out.open("a") as handle:
        for number, row in enumerate(pending, 1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTION}\n{row.get('question') or ''}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            reason = str(row.get("reasoning_prefix") or "")
            reason_ids = tokenizer.encode(reason, add_special_tokens=False)
            avail = args.max_context - len(prompt_ids) - 1
            slim = {key: row[key] for key in row if key != "reasoning_prefix"}
            if avail < 32 or not reason_ids:
                handle.write(json.dumps({**slim, "status": "too_short"}) + "\n")
                skipped += 1
                continue
            truncated = 0
            if len(reason_ids) > avail:
                reason_ids = reason_ids[-avail:]
                reason = tokenizer.decode(reason_ids, skip_special_tokens=False)
                truncated = 1
            ids = prompt_ids + reason_ids
            try:
                ents = token_entropy(ids)
                steps = split_steps_by_delim(tokenizer, ids, len(prompt_ids), delim="\n\n", only_think=True)
                if len(steps) < 2:
                    steps = split_steps_by_delim(tokenizer, ids, len(prompt_ids), delim="\n", only_think=True)
                step_ents = [nanmean(ents[step.start : step.end]) for step in steps]
            except Exception as exc:
                handle.write(json.dumps({**slim, "status": "forward_fail", "error": str(exc)[:200]}) + "\n")
                skipped += 1
                continue
            if len(steps) < 2 or not any(math.isfinite(v) for v in step_ents):
                handle.write(json.dumps({**slim, "status": "few_steps", "n_steps": len(steps)}) + "\n")
                skipped += 1
                continue
            payload = {
                **slim,
                "status": "ok",
                "prompt_len": len(prompt_ids),
                "seq_len": len(ids),
                "truncated": truncated,
                "n_steps": len(steps),
                "last_step_ent": step_ents[-1],
            }
            current = str(row.get("answer") or "")
            cue_ids = tokenizer.encode(ANSWER_CUE, add_special_tokens=False)
            try:
                for kappa in KAPPAS:
                    compressed, n_drop = compress_steps(steps, step_ents, kappa)
                    body = tokenizer.encode(compressed, add_special_tokens=False)
                    gen_ids = prompt_ids + body + cue_ids
                    text, parsed = boxed_from(gen_ids)
                    same = int(
                        parsed is not None
                        and (fast_eq(parsed, current) or answers_equal(parsed, current))
                    )
                    tag = f"k{int(kappa * 100)}"
                    payload.update(
                        {
                            f"{tag}_drop": n_drop,
                            f"{tag}_text": text[:240],
                            f"{tag}_answer": parsed,
                            f"{tag}_parsed": int(parsed is not None),
                            f"{tag}_same": same if parsed is not None else 0,
                            f"{tag}_changed": int(parsed is not None and not same),
                        }
                    )
            except Exception as exc:
                handle.write(json.dumps({**slim, "status": "gen_fail", "error": str(exc)[:200]}) + "\n")
                skipped += 1
                continue
            handle.write(json.dumps(payload) + "\n")
            ok += 1
            handle.flush()
            if number % 8 == 0 or number == len(pending):
                print(
                    f"[{number}/{len(pending)}] ok={ok} skip={skipped} "
                    f"{time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    print(f"done ok={ok} skip={skipped} -> {args.out}", flush=True)


def analyze_cmd(args: argparse.Namespace) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    def auc(y, s):
        y, s = np.asarray(y), np.asarray(s)
        if len(y) < 4 or y.min() == y.max():
            return float("nan")
        return float(roc_auc_score(y, s))

    def tpr_at(y, s, target=0.05):
        y, s = np.asarray(y), np.asarray(s)
        if len(y) < 4 or y.min() == y.max():
            return float("nan")
        fpr, tpr, _ = roc_curve(y, s)
        usable = tpr[fpr <= target]
        return float(usable[-1]) if len(usable) else 0.0

    rows = [row for path in args.scores for row in load_jsonl(path) if row.get("status") == "ok"]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[str(row.get("dataset") or "unknown")].append(row)
    lines = [
        "# Step-entropy prune-and-reanswer on first geo-stop",
        "",
        "`same` = compressed skeleton still yields the current trial answer.",
        "Higher `same` / lower `changed` is treated as more G-like.",
        "",
    ]
    for dataset, items in by.items():
        y = np.asarray([int(row.get("is_g") or 0) for row in items], dtype=int)
        raw_geo = np.asarray([finite(row.get("geo_conf")) for row in items])
        lines += [
            f"## {dataset} (n={len(items)}, G={int(y.sum())}, nonG={int((1 - y).sum())})",
            "",
            "| signal | AUROC | TPR@1% | TPR@5% | G p50 / rate | nonG p50 / rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        def same_of(row: dict[str, Any], tag: str) -> float:
            parsed = row.get(f"{tag}_answer")
            current = row.get("answer")
            if parsed is None:
                return float("nan")
            return float(fast_eq(parsed, current) or answers_equal(str(parsed), str(current)))

        for row in items:
            row["k50_same"] = same_of(row, "k50")
            row["k80_same"] = same_of(row, "k80")
            row["k50_changed"] = 1.0 - row["k50_same"] if math.isfinite(row["k50_same"]) else float("nan")
            row["k80_changed"] = 1.0 - row["k80_same"] if math.isfinite(row["k80_same"]) else float("nan")

        for name, invert in (
            ("geo_conf", False),
            ("k50_same", False),
            ("k80_same", False),
            ("k50_changed", True),
            ("k80_changed", True),
            ("k50_parsed", False),
            ("k80_parsed", False),
        ):
            raw = np.asarray([finite(row.get(name)) for row in items])
            mask = np.isfinite(raw)
            if mask.sum() < 8:
                continue
            scores = -raw[mask] if invert else raw[mask]
            g = raw[mask & (y == 1)]
            n = raw[mask & (y == 0)]
            lines.append(
                f"| {name}{'-inv' if invert else ''} | {auc(y[mask], scores):.3f} | "
                f"{tpr_at(y[mask], scores, 0.01):.3f} | {tpr_at(y[mask], scores, 0.05):.3f} | "
                f"{float(np.mean(g)) if len(g) else float('nan'):.3f} | "
                f"{float(np.mean(n)) if len(n) else float('nan'):.3f} |"
            )
        same80 = np.asarray([finite(row.get("k80_same")) for row in items])
        combo = raw_geo + 0.05 * np.nan_to_num(same80, nan=0.0)
        mask = np.isfinite(raw_geo)
        lines.append(
            f"| geo+0.05*k80_same | {auc(y[mask], combo[mask]):.3f} | "
            f"{tpr_at(y[mask], combo[mask], 0.01):.3f} | {tpr_at(y[mask], combo[mask], 0.05):.3f} | "
            f"{float(np.median(combo[mask & (y == 1)])):.4f} | "
            f"{float(np.median(combo[mask & (y == 0)])):.4f} |"
        )
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    score = sub.add_parser("score")
    score.add_argument("--candidates", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--shard-id", type=int, default=0)
    score.add_argument("--num-shards", type=int, default=1)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--max-context", type=int, default=16384)
    score.add_argument("--chunk-tokens", type=int, default=1024)
    score.add_argument("--limit", type=int, default=0)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--scores", type=Path, nargs="+", required=True)
    analyze.add_argument("--out", type=Path, default=AE / "tables/probe_step_prune.md")
    args = parser.parse_args()
    if args.cmd == "score":
        score_cmd(args)
    else:
        analyze_cmd(args)


if __name__ == "__main__":
    main()
