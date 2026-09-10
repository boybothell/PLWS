#!/usr/bin/env python3
"""ASAG-style attention entropy at selected PUMA steps. Last 4 layers, all heads."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(AE / "scripts"))

from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402
from score_dense_internal_metrics import _rotary_fns, load_jsonl, states_device  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import torch


def done_tag_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["question_idx"]), int(row["decision_step"]), str(row.get("tag") or ""))
        for row in load_jsonl(path)
        if row.get("status") in {"ok", "partial", "too_long", "oom", "no_answer"}
    }


def done_step_h(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in load_jsonl(path):
        if row.get("status") != "ok":
            continue
        out[(int(row["question_idx"]), int(row["decision_step"]))] = {
            "attn_H": row.get("attn_H"),
            "attn_H_mean": row.get("attn_H_mean"),
            "n_query": row.get("n_query"),
            "n_layers_used": row.get("n_layers_used"),
            "last4": row.get("last4"),
        }
    return out


def layer_entropy(
    model, hidden_in: torch.Tensor, layer_idx: int, query_pos: list[int]
) -> tuple[float, float]:
    apply_rope, repeat = _rotary_fns()
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    states = layer.input_layernorm(hidden_in.unsqueeze(0).to(states_device(layer)))
    seq = int(states.shape[1])
    if seq == 0 or not query_pos:
        return float("nan"), float("nan")
    pos = [p for p in query_pos if 0 <= p < seq]
    if not pos:
        return float("nan"), float("nan")
    position_ids = torch.arange(seq, device=states.device).view(1, -1)
    query = attn.q_proj(states).view(1, seq, -1, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(states).view(1, seq, -1, attn.head_dim).transpose(1, 2)
    cos, sin = model.model.rotary_emb(states, position_ids)
    query, key = apply_rope(query, key, cos, sin)
    key = repeat(key, attn.num_key_value_groups)
    q = query[0, :, pos, :].float()
    k = key[0].float()
    scores = torch.einsum("hqd,hsd->hqs", q, k) * float(attn.scaling)
    key_idx = torch.arange(seq, device=scores.device).view(1, 1, seq)
    q_idx = torch.tensor(pos, device=scores.device).view(1, -1, 1)
    scores = scores.masked_fill(key_idx > q_idx, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    log_w = weights.clamp_min(1e-12).log()
    row = -(weights * log_w).sum(dim=-1)
    logk = math.log(max(seq, 2))
    per_head = row.sum(dim=-1) / logk
    mean_head = row.mean(dim=-1) / logk
    return float(per_head.sum().item()), float(mean_head.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=SOLVER)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.candidates)
    qis = sorted({int(r["question_idx"]) for r in rows})
    keep = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [r for r in rows if int(r["question_idx"]) in keep]
    jobs.sort(key=lambda r: (int(r["question_idx"]), int(r["decision_step"]), str(r.get("tag") or "")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    already = done_tag_keys(args.out)
    cached = done_step_h(args.out)
    pending = [
        r
        for r in jobs
        if (int(r["question_idx"]), int(r["decision_step"]), str(r.get("tag") or "")) not in already
    ]
    print(
        f"asag-entropy shard={args.shard_id}/{args.num_shards} pending={len(pending)}/{len(jobs)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    last4 = list(range(max(0, n_layers - 4), n_layers))
    device = model.get_input_embeddings().weight.device
    started = time.perf_counter()
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
                    handle.write(json.dumps({**row, "reasoning_prefix": "", "status": "too_long"}) + "\n")
                    continue
                prefix = prefix[overflow:]
                ids = prefix + answer
            step_key = (int(row["question_idx"]), int(row["decision_step"]))
            if step_key in cached:
                slim = {k: v for k, v in row.items() if k != "reasoning_prefix"}
                slim.update({"status": "ok", **cached[step_key]})
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                continue
            tensor = torch.tensor([ids], device=device)
            start = len(prefix)
            end = len(ids)
            if end <= start:
                handle.write(json.dumps({**row, "reasoning_prefix": "", "status": "no_answer"}) + "\n")
                continue
            n_ans = end - start
            thought_pos = list(range(max(0, start - n_ans), start))
            boxed_pos = list(range(start, end))
            query_pos = thought_pos + boxed_pos
            try:
                with torch.inference_mode():
                    out = model(tensor, output_hidden_states=True, use_cache=False)
                    hidden = out.hidden_states
                    h_sum = 0.0
                    h_mean = 0.0
                    n_ok = 0
                    for layer_idx in last4:
                        # hidden[layer_idx] = input to layer layer_idx
                        hs, hm = layer_entropy(model, hidden[layer_idx][0], layer_idx, query_pos)
                        if hs == hs:
                            h_sum += hs
                            h_mean += hm
                            n_ok += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                handle.write(json.dumps({**row, "reasoning_prefix": "", "status": "oom"}) + "\n")
                continue
            slim = {k: v for k, v in row.items() if k != "reasoning_prefix"}
            slim.update(
                {
                    "status": "ok" if n_ok == len(last4) else "partial",
                    "attn_H": h_sum,
                    "attn_H_mean": h_mean / n_ok if n_ok else float("nan"),
                    "n_query": len(query_pos),
                    "n_layers_used": n_ok,
                    "last4": last4,
                }
            )
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
            handle.flush()
            if slim.get("status") == "ok":
                cached[step_key] = {
                    "attn_H": slim["attn_H"],
                    "attn_H_mean": slim["attn_H_mean"],
                    "n_query": slim["n_query"],
                    "n_layers_used": slim["n_layers_used"],
                    "last4": slim["last4"],
                }
            if index % 10 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
