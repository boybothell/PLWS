#!/usr/bin/env python3
"""Think Clearly </think>-attention peakiness on first geo-stop.

Inject the paper's forced-summary cue and read attention from </think>
onto the existing reasoning prefix. Hypothesis: true G is peaked on a
few load-bearing tokens; false plateaus are scattered.
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
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

CUE = (
    "\n\nTime is up. Given the time I've spent and the approaches I've tried, "
    "I should stop thinking and now write summarization in one sentence.\n"
    "</think>"
)


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


def peakiness(weights) -> dict[str, float]:
    values = weights.float().clamp_min(0)
    total = float(values.sum().item())
    if total <= 0 or values.numel() == 0:
        return {
            "entropy": float("nan"),
            "norm_entropy": float("nan"),
            "top10": float("nan"),
            "max": float("nan"),
            "herfindahl": float("nan"),
            "neg_entropy": float("nan"),
            "neg_norm_entropy": float("nan"),
        }
    probs = values / total
    logp = probs.clamp_min(1e-12).log()
    entropy = float(-(probs * logp).sum().item())
    n = max(int(probs.numel()), 2)
    norm_ent = entropy / math.log(n)
    topk = min(10, n)
    top10 = float(probs.topk(topk).values.sum().item())
    herfindahl = float((probs * probs).sum().item())
    return {
        "entropy": entropy,
        "norm_entropy": norm_ent,
        "top10": top10,
        "max": float(probs.max().item()),
        "herfindahl": herfindahl,
        "neg_entropy": -entropy,
        "neg_norm_entropy": -norm_ent,
    }


def score_cmd(args: argparse.Namespace) -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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
    model = AutoModelForCausalLM.from_pretrained(
        str(SOLVER),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    cue_ids = tokenizer.encode(CUE, add_special_tokens=False)
    think_id = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_id) != 1:
        raise RuntimeError(f"</think> is not a single token: {think_id}")
    if cue_ids[-1] != think_id[0]:
        raise RuntimeError(f"cue does not end with </think>: {cue_ids[-8:]}")
    print(
        f"think-attn shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"device={device} cue={len(cue_ids)} -> {args.out}",
        flush=True,
    )

    @torch.inference_mode()
    def prefill(ids: list[int]) -> Any:
        cache = DynamicCache(config=model.config)
        for lo in range(0, len(ids), args.chunk_tokens):
            hi = min(len(ids), lo + args.chunk_tokens)
            chunk = torch.tensor([ids[lo:hi]], device=device)
            out = model(input_ids=chunk, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            del out, chunk
        return cache

    @torch.inference_mode()
    def cue_attention(cache, prompt_len: int, prefix_len: int) -> dict[str, float]:
        chunk = torch.tensor([cue_ids], device=device)
        out = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            output_attentions=True,
        )
        # last query = </think>; kv = prefix + cue
        layers = []
        for layer_attn in out.attentions:
            # (1, heads, q, kv)
            layers.append(layer_attn[0, :, -1, :].float().mean(0).cpu())
        del out, chunk
        stacked = torch.stack(layers, dim=0)  # (L, kv)
        last = stacked[-1]
        mean_last4 = stacked[-4:].mean(0)
        reason = slice(prompt_len, prefix_len)
        last_stats = peakiness(last[reason])
        mean_stats = peakiness(mean_last4[reason])
        last_total = float(last[:prefix_len].sum().item())
        prompt_mass = float(last[:prompt_len].sum().item())
        reason_mass = float(last[reason].sum().item())
        return {
            "n_reason": prefix_len - prompt_len,
            "prompt_frac": prompt_mass / last_total if last_total > 0 else float("nan"),
            "reason_frac": reason_mass / last_total if last_total > 0 else float("nan"),
            **{f"last_{key}": value for key, value in last_stats.items()},
            **{f"l4_{key}": value for key, value in mean_stats.items()},
        }

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
            reason_ids = tokenizer.encode(str(row.get("reasoning_prefix") or ""), add_special_tokens=False)
            avail = args.max_context - len(prompt_ids) - len(cue_ids) - 1
            slim = {key: row[key] for key in row if key != "reasoning_prefix"}
            if avail < 32:
                handle.write(json.dumps({**slim, "status": "too_short"}) + "\n")
                skipped += 1
                continue
            truncated = 0
            if len(reason_ids) > avail:
                reason_ids = reason_ids[-avail:]
                truncated = 1
            prefix_ids = prompt_ids + reason_ids
            try:
                cache = prefill(prefix_ids)
                feats = cue_attention(cache, len(prompt_ids), len(prefix_ids))
            except Exception as exc:
                handle.write(json.dumps({**slim, "status": "forward_fail", "error": str(exc)[:200]}) + "\n")
                skipped += 1
                continue
            handle.write(
                json.dumps(
                    {
                        **slim,
                        "status": "ok",
                        "prompt_len": len(prompt_ids),
                        "seq_len": len(prefix_ids),
                        "truncated": truncated,
                        **feats,
                    }
                )
                + "\n"
            )
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
    signals = [
        "geo_conf",
        "last_neg_entropy",
        "last_neg_norm_entropy",
        "last_top10",
        "last_max",
        "last_herfindahl",
        "l4_neg_entropy",
        "l4_neg_norm_entropy",
        "l4_top10",
        "prompt_frac",
        "reason_frac",
        "last_entropy",
        "last_norm_entropy",
    ]
    invert = {"last_entropy", "last_norm_entropy"}
    lines = [
        "# Think Clearly </think>-attention on first geo-stop",
        "",
        "Higher score = more G-like, except raw entropy. Forced-summary cue;",
        "attention is from `</think>` onto the reasoning prefix.",
        "",
    ]
    for dataset, items in by.items():
        y = np.asarray([int(row.get("is_g") or 0) for row in items], dtype=int)
        raw_geo = np.asarray([finite(row.get("geo_conf")) for row in items])
        raw_peak = np.asarray([finite(row.get("last_neg_norm_entropy")) for row in items])
        lines += [
            f"## {dataset} (n={len(items)}, G={int(y.sum())}, nonG={int((1 - y).sum())})",
            "",
            "| signal | AUROC | TPR@1% | TPR@5% | G p50 | nonG p50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in signals:
            raw = np.asarray([finite(row.get(name)) for row in items])
            mask = np.isfinite(raw)
            if mask.sum() < 8:
                continue
            scores = -raw[mask] if name in invert else raw[mask]
            g = raw[mask & (y == 1)]
            n = raw[mask & (y == 0)]
            lines.append(
                f"| {name}{'-inv' if name in invert else ''} | {auc(y[mask], scores):.3f} | "
                f"{tpr_at(y[mask], scores, 0.01):.3f} | {tpr_at(y[mask], scores, 0.05):.3f} | "
                f"{float(np.median(g)) if len(g) else float('nan'):.4f} | "
                f"{float(np.median(n)) if len(n) else float('nan'):.4f} |"
            )
        combo = raw_geo + 0.05 * np.nan_to_num(raw_peak, nan=0.0)
        mask = np.isfinite(raw_geo)
        lines.append(
            f"| geo+0.05*negH | {auc(y[mask], combo[mask]):.3f} | "
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
    score.add_argument("--max-context", type=int, default=8192)
    score.add_argument("--chunk-tokens", type=int, default=1024)
    score.add_argument("--limit", type=int, default=0)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--scores", type=Path, nargs="+", required=True)
    analyze.add_argument("--out", type=Path, default=AE / "tables/probe_think_attn.md")
    args = parser.parse_args()
    if args.cmd == "score":
        score_cmd(args)
    else:
        analyze_cmd(args)


if __name__ == "__main__":
    main()
