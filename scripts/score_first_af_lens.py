#!/usr/bin/env python3
"""Dense 7B mid-layer logit-lens and step-to-step hidden cosine.

Last-layer boxed logp is a byproduct (already exists as current_logp).
New scores: lens_l8/l14/l20, hidden_nn_cos, prefix_ans_cos.
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

from layer_grid import parse_layers  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

LAYERS = (8, 14, 20, 27)


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
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=SOLVER)
    parser.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer indices. Empty = dense grid (every 2 or 4 layers, plus quarters).",
    )
    parser.add_argument("--device-map", default="", help="e.g. auto for 2-GPU 30B/32B")
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
    if not pending:
        print(f"dense-lens shard={args.shard_id}/{args.num_shards} pending=0 -> {args.out}", flush=True)
        return
    model_path = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    load_kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    if args.device_map:
        n_vis = torch.cuda.device_count()
        load_kw["device_map"] = args.device_map
        load_kw["max_memory"] = {i: "36GiB" for i in range(n_vis)}
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kw).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    head_device = model.lm_head.weight.device
    n_layers = int(model.config.num_hidden_layers)
    layer_ids = parse_layers(args.layers, n_layers)
    print(
        f"dense-lens shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"model={model_path.name} layers={n_layers} lens={layer_ids} -> {args.out}",
        flush=True,
    )
    started = time.perf_counter()
    prev_qi = None
    prev_h = None
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix = tokenizer.encode(prompt + str(row["reasoning_prefix"]) + "\n", add_special_tokens=False)
            answer = tokenizer.encode(f"\\boxed{{{row['answer']}}}", add_special_tokens=False)
            ids = prefix + answer
            if len(ids) + 1 > args.max_context:
                overflow = len(ids) + 1 - args.max_context
                if overflow >= len(prefix):
                    handle.write(json.dumps({**{k: v for k, v in row.items() if k != "reasoning_prefix"}, "status": "too_long"}) + "\n")
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
                handle.write(json.dumps({**{k: v for k, v in row.items() if k != "reasoning_prefix"}, "status": "no_answer"}) + "\n")
                continue
            logits = out.logits[0, start - 1 : end - 1].float()
            target = tensor[0, start:end].to(logits.device)
            last_logp = torch.log_softmax(logits, dim=-1)
            gather = last_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            last_mean = float(gather.mean().item())
            last_min = float(gather.min().item())
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
            for layer in layer_ids:
                if layer + 1 >= len(hidden):
                    continue
                h = model.model.norm(hidden[layer + 1][0, start - 1 : end - 1].to(head_device))
                lp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
                score = float(lp.gather(-1, target.to(lp.device).unsqueeze(-1)).squeeze(-1).mean().item())
                lens[f"lens_l{layer}"] = score
                if score > best:
                    best = score
                    best_layer = float(layer)
            slim = {key: value for key, value in row.items() if key not in {"reasoning_prefix", "question"}}
            slim.update(
                {
                    "status": "ok",
                    "last_mean_logp": last_mean,
                    "last_min_logp": last_min,
                    "ans_entropy": ent,
                    "neg_ans_entropy": -ent,
                    "hidden_norm": float(last_h.norm().item()),
                    "prefix_ans_cos": cos,
                    "hidden_nn_cos": step_cos,
                    "lens_best": best if best > float("-inf") else float("nan"),
                    "lens_best_layer": best_layer,
                    "lens_rise": last_mean - lens.get("lens_l14", float("nan")),
                    **lens,
                }
            )
            handle.write(json.dumps(slim) + "\n")
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
