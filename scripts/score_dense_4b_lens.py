#!/usr/bin/env python3
"""4B last-layer answer confidence and mid-layer logit lens.

Same recipe as 7B score_first_af_lens.py, but the reader is Qwen3-4B.
Prefix = open think, stuffed with 7B reasoning; boxed tokens are scored
with 4B's own tokenizer. 4B does not generate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_v1 import nvidia_lib_path  # noqa: E402
from score_dense_4b_open_think import boxed, open_think_header  # noqa: E402
from score_confcal_judge import QWEN4B_MODEL  # noqa: E402

# 36-layer Qwen3-4B; same relative depths as 7B's 8/14/20/27 on 28 layers.
LAYERS = (8, 17, 26, 35)


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.candidates)
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [row for row in rows if int(row["question_idx"]) in keep]
    jobs.sort(key=lambda row: (int(row["question_idx"]), int(row["decision_step"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in jobs
        if (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in done_keys(args.out)
    ]
    if args.limit:
        pending = pending[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN4B_MODEL), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN4B_MODEL),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    n_layers = int(model.config.num_hidden_layers)
    usable = [layer for layer in LAYERS if layer < n_layers]
    print(
        f"4b-lens shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"layers={n_layers} lens={usable} -> {args.out}",
        flush=True,
    )
    started = time.perf_counter()
    prev_qi = None
    prev_h = None
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            header = open_think_header(tokenizer, str(row.get("question") or ""))
            prefix = tokenizer.encode(
                header + str(row.get("reasoning_prefix") or "") + "\n",
                add_special_tokens=False,
            )
            answer = tokenizer.encode(boxed(str(row.get("answer") or "")), add_special_tokens=False)
            ids = prefix + answer
            if len(ids) + 1 > args.max_context:
                overflow = len(ids) + 1 - args.max_context
                if overflow >= len(prefix):
                    handle.write(
                        json.dumps(
                            {
                                "question_idx": int(row["question_idx"]),
                                "decision_step": int(row["decision_step"]),
                                "answer": row.get("answer"),
                                "geo_conf": row.get("geo_conf"),
                                "status": "too_long",
                            }
                        )
                        + "\n"
                    )
                    continue
                ids = prefix[overflow:] + answer
                prefix = prefix[overflow:]
            tensor = torch.tensor([ids], device=device)
            with torch.inference_mode():
                out = model(tensor, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states
            start = len(prefix)
            end = len(ids)
            if end <= start:
                handle.write(
                    json.dumps(
                        {
                            "question_idx": int(row["question_idx"]),
                            "decision_step": int(row["decision_step"]),
                            "answer": row.get("answer"),
                            "geo_conf": row.get("geo_conf"),
                            "status": "no_answer",
                        }
                    )
                    + "\n"
                )
                continue
            target = tensor[0, start:end]
            logits = out.logits[0, start - 1 : end - 1].float()
            last_logp = torch.log_softmax(logits, dim=-1)
            gather = last_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            last_mean = float(gather.mean().item())
            last_min = float(gather.min().item())
            conf4b = float(math.exp(last_mean))
            ent = float((-(last_logp.exp() * last_logp).sum(-1)).mean().item())
            last_h = hidden[-1][0, start - 1].float()
            ans_h = hidden[-1][0, end - 1].float()
            cos = float(torch.nn.functional.cosine_similarity(last_h, ans_h, dim=0).item())
            qi = int(row["question_idx"])
            step_cos = (
                float(torch.nn.functional.cosine_similarity(last_h, prev_h, dim=0).item())
                if prev_qi == qi and prev_h is not None
                else float("nan")
            )
            prev_qi = qi
            prev_h = last_h.detach()
            lens: dict[str, float] = {}
            best = float("-inf")
            best_layer = float("nan")
            for layer in usable:
                if layer + 1 >= len(hidden):
                    continue
                h = model.model.norm(hidden[layer + 1][0, start - 1 : end - 1])
                lp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
                score = float(lp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item())
                lens[f"lens_l{layer}"] = score
                if score > best:
                    best = score
                    best_layer = float(layer)
            mid = usable[1] if len(usable) > 1 else usable[0]
            handle.write(
                json.dumps(
                    {
                        "question_idx": qi,
                        "decision_step": int(row["decision_step"]),
                        "answer": row.get("answer"),
                        "geo_conf": row.get("geo_conf"),
                        "is_g": row.get("is_g"),
                        "is_g_window": row.get("is_g_window"),
                        "status": "ok",
                        "n_tokens": len(ids),
                        "conf4b": conf4b,
                        "last_mean_logp": last_mean,
                        "last_min_logp": last_min,
                        "ans_entropy": ent,
                        "neg_ans_entropy": -ent,
                        "hidden_norm": float(last_h.norm().item()),
                        "prefix_ans_cos": cos,
                        "hidden_nn_cos": step_cos,
                        "lens_best": best if best > float("-inf") else float("nan"),
                        "lens_best_layer": best_layer,
                        "lens_rise": last_mean - lens.get(f"lens_l{mid}", float("nan")),
                        **lens,
                    }
                )
                + "\n"
            )
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
