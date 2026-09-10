#!/usr/bin/env python3
"""One HF forward per leftover window: question attention + Wait layer contrast."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(AE / "scripts"))

from score_asag_entropy import layer_entropy  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402
from score_dense_internal_metrics import last_layer_weights, load_jsonl, lookback_bundle  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import torch

WAIT_RE = re.compile(r"\bWait\b")


def done_keys(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(row["question_idx"])
        for row in load_jsonl(path)
        if row.get("status") in {"ok", "partial", "too_long", "oom", "no_answer"}
    }


def last_wait_end(prefix: str) -> int | None:
    matches = list(WAIT_RE.finditer(prefix))
    if not matches:
        return None
    return matches[-1].end()


def entropy_on_keys(model, hidden_in: torch.Tensor, layer_idx: int, query_pos: list[int], key_idx: list[int]) -> float:
    from score_dense_internal_metrics import _rotary_fns, states_device

    apply_rope, repeat = _rotary_fns()
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    states = layer.input_layernorm(hidden_in.unsqueeze(0).to(states_device(layer)))
    seq = int(states.shape[1])
    pos = [p for p in query_pos if 0 <= p < seq]
    keys = [k for k in key_idx if 0 <= k < seq]
    if not pos or not keys:
        return float("nan")
    position_ids = torch.arange(seq, device=states.device).view(1, -1)
    query = attn.q_proj(states).view(1, seq, -1, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(states).view(1, seq, -1, attn.head_dim).transpose(1, 2)
    cos, sin = model.model.rotary_emb(states, position_ids)
    query, key = apply_rope(query, key, cos, sin)
    key = repeat(key, attn.num_key_value_groups)
    q = query[0, :, pos, :].float()
    k = key[0].float()
    scores = torch.einsum("hqd,hsd->hqs", q, k) * float(attn.scaling)
    key_ids = torch.arange(seq, device=scores.device).view(1, 1, seq)
    q_ids = torch.tensor(pos, device=scores.device).view(1, -1, 1)
    scores = scores.masked_fill(key_ids > q_ids, torch.finfo(scores.dtype).min)
    keep = torch.zeros(seq, dtype=torch.bool, device=scores.device)
    keep[keys] = True
    scores = scores.masked_fill(~keep.view(1, 1, seq), torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    log_w = weights.clamp_min(1e-12).log()
    row = -(weights * log_w).sum(dim=-1)
    logk = math.log(max(len(keys), 2))
    return float((row.mean(dim=-1) / logk).mean().item())


def layer_next_logp(model, hidden, layer: int, pos: int, tok: int) -> float:
    if layer + 1 >= len(hidden) or pos < 0:
        return float("nan")
    head_device = model.lm_head.weight.device
    span = hidden[layer + 1][0, pos].to(head_device)
    logits = model.lm_head(model.model.norm(span)).float()
    if tok < 0 or tok >= logits.numel():
        return float("nan")
    return float(torch.log_softmax(logits, dim=-1)[tok].item())


def span_mean_logp(model, hidden, layer: int, start: int, end: int, ids: torch.Tensor) -> float:
    if layer + 1 >= len(hidden) or end <= start:
        return float("nan")
    head_device = model.lm_head.weight.device
    span = hidden[layer + 1][0, start - 1 : end - 1].to(head_device)
    if span.shape[0] == 0:
        return float("nan")
    logp = torch.log_softmax(model.lm_head(model.model.norm(span)).float(), dim=-1)
    target = ids[start:end].to(logp.device)
    return float(logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item())


def nan_bundle() -> dict[str, float]:
    return {k: float("nan") for k in ("lb_q", "lb_cot", "lb_ans", "lb_rev", "lb_recency5", "lookback_ratio")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=SOLVER)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.candidates)
    qis = sorted({int(r["question_idx"]) for r in rows})
    keep = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [r for r in rows if int(r["question_idx"]) in keep]
    jobs.sort(key=lambda r: int(r["question_idx"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    already = done_keys(args.out)
    pending = [r for r in jobs if int(r["question_idx"]) not in already]
    print(
        f"leftover-attn shard={args.shard_id}/{args.num_shards} pending={len(pending)}/{len(jobs)} -> {args.out}",
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
    half = n_layers // 2
    last = n_layers - 1
    last4 = list(range(max(0, n_layers - 4), n_layers))
    device = model.get_input_embeddings().weight.device

    def first_id(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0]) if ids else -1

    stop_id = first_id("</think>")
    wait_id = first_id("Wait")
    started = time.perf_counter()
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_text = str(row["reasoning_prefix"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prefix = tokenizer.encode(prompt + prefix_text + "\n", add_special_tokens=False)
            answer = tokenizer.encode(f"\\boxed{{{row['answer']}}}", add_special_tokens=False)
            ids = prefix + answer
            prompt_len = len(prompt_ids)
            wait_cut = last_wait_end(prefix_text)
            wait_pos = None
            if wait_cut is not None:
                wait_ids = tokenizer.encode(prompt + prefix_text[:wait_cut], add_special_tokens=False)
                wait_pos = len(wait_ids) - 1
            overflow = 0
            if len(ids) + 1 > args.max_context:
                overflow = len(ids) + 1 - args.max_context
                if overflow >= len(prefix):
                    handle.write(json.dumps({**{k: v for k, v in row.items() if k != "reasoning_prefix"}, "status": "too_long"}) + "\n")
                    continue
                ids = prefix[overflow:] + answer
                prefix = prefix[overflow:]
                prompt_len = max(0, prompt_len - overflow)
                if wait_pos is not None:
                    wait_pos -= overflow
                    if wait_pos < 0:
                        wait_pos = None
            tensor = torch.tensor([ids], device=device)
            start = len(prefix)
            end = len(ids)
            slim = {k: v for k, v in row.items() if k not in {"reasoning_prefix", "question"}}
            if end <= start:
                slim["status"] = "no_answer"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                continue
            try:
                with torch.inference_mode():
                    out = model(tensor, output_hidden_states=True, use_cache=False)
                    hidden = out.hidden_states
                    n_ans = end - start
                    thought_pos = list(range(max(0, start - n_ans), start))
                    boxed_pos = list(range(start, end))
                    q_keys = list(range(min(prompt_len, end)))
                    q_ent = 0.0
                    n_ok = 0
                    for layer_idx in last4:
                        val = entropy_on_keys(model, hidden[layer_idx][0], layer_idx, thought_pos + boxed_pos, q_keys)
                        if val == val:
                            q_ent += val
                            n_ok += 1
                    h_sum = 0.0
                    n_h = 0
                    for layer_idx in last4:
                        hs, _hm = layer_entropy(model, hidden[layer_idx][0], layer_idx, thought_pos + boxed_pos)
                        if hs == hs:
                            h_sum += hs
                            n_h += 1
                    try:
                        attn_end = last_layer_weights(model, hidden[-2][0], end - 1)
                        lb = lookback_bundle(attn_end, prompt_len, start, end, [])
                    except Exception:
                        lb = nan_bundle()
                    try:
                        attn_th = last_layer_weights(model, hidden[-2][0], max(0, start - 1))
                        th = lookback_bundle(attn_th, prompt_len, start, start, [])
                    except Exception:
                        th = nan_bundle()
                    last_mean = span_mean_logp(model, hidden, last, start, end, tensor[0])
                    half_mean = span_mean_logp(model, hidden, half, start, end, tensor[0])
                    wait_feat = {
                        "wait_pos": wait_pos,
                        "wait_lb_q": float("nan"),
                        "wait_lookback_ratio": float("nan"),
                        "wait_lb_cot": float("nan"),
                        "wait_lens_rise": float("nan"),
                        "wait_next_logp": float("nan"),
                        "wait_half_logp": float("nan"),
                        "wait_stop_margin": float("nan"),
                        "wait_q_entropy": float("nan"),
                    }
                    if wait_pos is not None and 0 <= wait_pos < end:
                        try:
                            attn_w = last_layer_weights(model, hidden[-2][0], wait_pos)
                            wlb = lookback_bundle(attn_w, prompt_len, wait_pos + 1, wait_pos + 1, [])
                            wait_feat["wait_lb_q"] = wlb["lb_q"]
                            wait_feat["wait_lookback_ratio"] = wlb["lookback_ratio"]
                            wait_feat["wait_lb_cot"] = wlb["lb_cot"]
                        except Exception:
                            pass
                        nxt = wait_pos + 1
                        if nxt < end:
                            tok = int(tensor[0, nxt].item())
                            last_lp = layer_next_logp(model, hidden, last, wait_pos, tok)
                            half_lp = layer_next_logp(model, hidden, half, wait_pos, tok)
                            wait_feat["wait_next_logp"] = last_lp
                            wait_feat["wait_half_logp"] = half_lp
                            wait_feat["wait_lens_rise"] = last_lp - half_lp if last_lp == last_lp and half_lp == half_lp else float("nan")
                        head_device = model.lm_head.weight.device
                        nxt_logp = torch.log_softmax(
                            model.lm_head(model.model.norm(hidden[-1][0, wait_pos].to(head_device))).float(),
                            dim=-1,
                        )
                        stop_lp = float(nxt_logp[stop_id].item()) if 0 <= stop_id < nxt_logp.numel() else float("nan")
                        wait_lp = float(nxt_logp[wait_id].item()) if 0 <= wait_id < nxt_logp.numel() else float("nan")
                        wait_feat["wait_stop_margin"] = stop_lp - wait_lp if stop_lp == stop_lp and wait_lp == wait_lp else float("nan")
                        wait_feat["wait_q_entropy"] = entropy_on_keys(
                            model, hidden[-2][0], last, [wait_pos], q_keys
                        )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                slim["status"] = "oom"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            slim.update(
                {
                    "status": "ok" if n_ok == len(last4) else "partial",
                    "lb_q": lb["lb_q"],
                    "lb_cot": lb["lb_cot"],
                    "lookback_ratio": lb["lookback_ratio"],
                    "thought_lb_q": th["lb_q"],
                    "thought_lookback_ratio": th["lookback_ratio"],
                    "q_entropy": q_ent / n_ok if n_ok else float("nan"),
                    "attn_H": h_sum if n_h else float("nan"),
                    "lens_rise": last_mean - half_mean if last_mean == last_mean and half_mean == half_mean else float("nan"),
                    "last_mean_logp": last_mean,
                    "prompt_len": prompt_len,
                    "seq_len": end,
                    "half_layer": half,
                    "n_layers_used": n_ok,
                    **wait_feat,
                }
            )
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 5 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
