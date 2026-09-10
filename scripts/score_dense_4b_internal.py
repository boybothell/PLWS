#!/usr/bin/env python3
"""4B DoLA / Lookback / EigenScore on stuffed 7B reasoning.

Same calculations as 7B score_dense_internal_metrics.py. Stop-margin is
skipped (already extracted). 4B does not generate.
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

from score_confcal_judge import QWEN4B_MODEL  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402
from score_dense_4b_open_think import boxed, open_think_header  # noqa: E402
from score_dense_internal_metrics import (  # noqa: E402
    EIGEN_K,
    REVISE_STRINGS,
    done_keys,
    eigenscore,
    find_spans,
    js_divergence,
    load_jsonl,
    lookback_bundle,
)

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

LAYERS = (8, 17, 26, 35)


def last_layer_weights(model, hidden_in: torch.Tensor, query_pos: int) -> torch.Tensor:
    layer = model.model.layers[-1]
    attn = layer.self_attn
    states = layer.input_layernorm(hidden_in.unsqueeze(0))
    seq = states.shape[1]
    position_ids = torch.arange(seq, device=states.device).view(1, -1)
    shape = (1, seq, -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(states).view(shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(states).view(shape)).transpose(1, 2)
    cos, sin = model.model.rotary_emb(states, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = repeat_kv(key, attn.num_key_value_groups)
    scores = torch.einsum("hd,hsd->hs", query[0, :, query_pos, :], key[0]) * attn.scaling
    banned = torch.arange(seq, device=scores.device) > query_pos
    scores = scores.masked_fill(banned, torch.finfo(scores.dtype).min)
    return torch.softmax(scores.float(), dim=-1).mean(dim=0)


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
    revise_pats = [tokenizer.encode(text, add_special_tokens=False) for text in REVISE_STRINGS]
    print(
        f"4b-internal shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"layers={n_layers} dola={usable} -> {args.out}",
        flush=True,
    )
    started = time.perf_counter()
    prev_qi = None
    last_hiddens: list[torch.Tensor] = []
    mid_hiddens: list[torch.Tensor] = []
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            header = open_think_header(tokenizer, str(row.get("question") or ""))
            prompt_ids = tokenizer.encode(header, add_special_tokens=False)
            prefix = tokenizer.encode(header + str(row.get("reasoning_prefix") or "") + "\n", add_special_tokens=False)
            answer = tokenizer.encode(boxed(str(row.get("answer") or "")), add_special_tokens=False)
            ids = prefix + answer
            prompt_len = len(prompt_ids)
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
                prompt_len = max(0, prompt_len - overflow)
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
            content_start = start + 2 if end - start > 3 else start
            content_end = end - 1 if end - start > 3 else end
            if content_end <= content_start:
                content_start, content_end = start, end
            first_tok = int(tensor[0, content_start].item())
            pre_logits = []
            layer_logp = {}
            layer_mean = {}
            top1_match = {}
            jsds = {}
            last_log = None
            target = tensor[0, content_start:content_end]
            for layer in usable:
                if layer + 1 >= len(hidden):
                    continue
                lens_h = model.model.norm(hidden[layer + 1][0, content_start - 1 : content_end - 1])
                logp = torch.log_softmax(model.lm_head(lens_h).float(), dim=-1)
                first_logp = logp[0]
                pre_logits.append((layer, first_logp))
                layer_logp[layer] = float(first_logp[first_tok].item())
                layer_mean[layer] = float(logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item())
                top1_match[layer] = int(int(first_logp.argmax().item()) == first_tok)
                if layer == usable[-1]:
                    last_log = first_logp
            if last_log is not None:
                for layer, logp in pre_logits:
                    jsds[layer] = js_divergence(last_log, logp)
            emerge = next((layer for layer in usable if top1_match.get(layer)), float("nan"))
            if jsds:
                pre_layer = max(jsds, key=jsds.get)
                dola_score = layer_logp.get(usable[-1], float("nan")) - layer_logp.get(pre_layer, float("nan"))
                dola_jsd = jsds[pre_layer]
            else:
                pre_layer = float("nan")
                dola_score = float("nan")
                dola_jsd = float("nan")
            last_h = hidden[-1][0, end - 1].float().cpu()
            mid_h = hidden[18][0, end - 1].float().cpu() if len(hidden) > 18 else last_h
            qi = int(row["question_idx"])
            if prev_qi != qi:
                last_hiddens = []
                mid_hiddens = []
            last_hiddens.append(last_h)
            mid_hiddens.append(mid_h)
            last_hiddens = last_hiddens[-EIGEN_K:]
            mid_hiddens = mid_hiddens[-EIGEN_K:]
            prev_qi = qi
            revise = []
            for pattern in revise_pats:
                revise.extend(find_spans(ids, pattern))
            revise = sorted(set(revise))
            try:
                attn_end = last_layer_weights(model, hidden[-2][0], end - 1)
                attn_pre = last_layer_weights(model, hidden[-2][0], start - 1)
                lb = lookback_bundle(attn_end, prompt_len, start, end, revise)
                pre_lb = lookback_bundle(attn_pre, prompt_len, start, start, revise)
                pre_lb = {
                    "pre_lb_q": pre_lb["lb_q"],
                    "pre_lb_recency5": pre_lb["lb_recency5"],
                    "pre_lookback_ratio": pre_lb["lookback_ratio"],
                }
            except Exception:
                lb = {key: float("nan") for key in ("lb_q", "lb_cot", "lb_ans", "lb_rev", "lb_recency5", "lookback_ratio")}
                pre_lb = {f"pre_{key}": float("nan") for key in ("lb_q", "lb_recency5", "lookback_ratio")}
            early = usable[0]
            handle.write(
                json.dumps(
                    {
                        "question_idx": qi,
                        "decision_step": int(row["decision_step"]),
                        "answer": row.get("answer"),
                        "geo_conf": row.get("geo_conf"),
                        "status": "ok",
                        **{f"dola_logp_l{layer}": layer_logp.get(layer, float("nan")) for layer in usable},
                        **{f"dola_mean_l{layer}": layer_mean.get(layer, float("nan")) for layer in usable},
                        **{f"dola_jsd_l{layer}": jsds.get(layer, float("nan")) for layer in usable},
                        **{f"dola_top1_l{layer}": top1_match.get(layer, 0) for layer in usable},
                        "dola_mean_rise": layer_mean.get(usable[-1], float("nan")) - layer_mean.get(early, float("nan")),
                        "dola_jsd_max": dola_jsd,
                        "dola_pre_layer": float(pre_layer) if pre_layer == pre_layer else float("nan"),
                        "dola_score": dola_score,
                        "emerge_layer": float(emerge) if emerge == emerge else float("nan"),
                        "neg_emerge": (-float(emerge)) if emerge == emerge else float("nan"),
                        "layer_agree": float(sum(top1_match.get(layer, 0) for layer in usable) / max(len(usable), 1)),
                        "eigen_k2": eigenscore(last_hiddens[-2:]),
                        "eigen_k4": eigenscore(last_hiddens),
                        "neg_eigen_k2": -eigenscore(last_hiddens[-2:]),
                        "neg_eigen_k4": -eigenscore(last_hiddens),
                        "mid_neg_eigen_k4": -eigenscore(mid_hiddens),
                        **lb,
                        **pre_lb,
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
