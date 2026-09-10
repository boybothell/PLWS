#!/usr/bin/env python3
"""Training-free internal metrics on every dense trial (one HF forward).

DoLA (ICLR 2024): pre-box layer JSD / contrast / emergence / top-1 agree.
Stop-margin: next-token logp(</think>) - logp(Wait) after boxed.
Lookback (EMNLP 2024): last-layer last-query mass on question / answer / Wait.
EigenScore (INSIDE, ICLR 2024): window LogDet of recent last-token hiddens.
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

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

import torch
LAYERS = (8, 14, 20, 27)
EIGEN_K = 4
REVISE_STRINGS = ("Wait", "wait", "Alternatively", "alternatively")


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


def find_spans(ids: list[int], pattern: list[int]) -> list[int]:
    if not pattern:
        return []
    hits = []
    width = len(pattern)
    for index in range(len(ids) - width + 1):
        if ids[index : index + width] == pattern:
            hits.extend(range(index, index + width))
    return hits


def js_divergence(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p = log_p.exp()
    q = log_q.exp()
    mid = 0.5 * (p + q)
    log_m = mid.clamp_min(1e-12).log()
    return float((0.5 * ((p * (log_p - log_m)).sum() + (q * (log_q - log_m)).sum())).item())


def eigenscore(vectors: list[torch.Tensor], alpha: float = 1e-3) -> float:
    if len(vectors) < 2:
        return float("nan")
    matrix = torch.stack([torch.nn.functional.normalize(item.float(), dim=0) for item in vectors], dim=0)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    gram = (matrix @ matrix.T) / max(matrix.shape[1], 1)
    gram = gram + alpha * torch.eye(gram.shape[0], dtype=gram.dtype)
    try:
        values = torch.linalg.eigvalsh(gram).clamp_min(1e-12)
        return float(values.log().mean().item())
    except Exception:
        return float("nan")


def _rotary_fns():
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

        return apply_rotary_pos_emb, repeat_kv
    except Exception:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        return apply_rotary_pos_emb, repeat_kv


def last_layer_weights(model, hidden_in: torch.Tensor, query_pos: int) -> torch.Tensor:
    apply_rope, repeat = _rotary_fns()
    layer = model.model.layers[-1]
    attn = layer.self_attn
    states = layer.input_layernorm(hidden_in.unsqueeze(0).to(states_device(layer)))
    seq = states.shape[1]
    position_ids = torch.arange(seq, device=states.device).view(1, -1)
    query = attn.q_proj(states).view(1, seq, -1, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(states).view(1, seq, -1, attn.head_dim).transpose(1, 2)
    cos, sin = model.model.rotary_emb(states, position_ids)
    query, key = apply_rope(query, key, cos, sin)
    key = repeat(key, attn.num_key_value_groups)
    scores = torch.einsum("hd,hsd->hs", query[0, :, query_pos, :], key[0]) * attn.scaling
    banned = torch.arange(seq, device=scores.device) > query_pos
    scores = scores.masked_fill(banned, torch.finfo(scores.dtype).min)
    return torch.softmax(scores.float(), dim=-1).mean(dim=0)


def states_device(layer) -> torch.device:
    return next(layer.parameters()).device


def mass(weights: torch.Tensor, index: torch.Tensor) -> float:
    if weights.numel() == 0:
        return float("nan")
    if index.numel() == 0:
        return 0.0
    keep = index[(index >= 0) & (index < weights.numel())]
    if keep.numel() == 0:
        return 0.0
    return float(weights[keep].sum().item())


def lookback_bundle(weights: torch.Tensor, prompt_len: int, start: int, end: int, revise: list[int]) -> dict[str, float]:
    seq = int(weights.numel())
    q_idx = torch.arange(min(prompt_len, seq), device=weights.device)
    cot_idx = torch.arange(min(prompt_len, seq), min(start, seq), device=weights.device)
    ans_idx = torch.arange(min(start, seq), min(end, seq), device=weights.device)
    rev_idx = torch.tensor(revise, device=weights.device, dtype=torch.long) if revise else torch.empty(0, dtype=torch.long, device=weights.device)
    rec_lo = max(end - 6, 0)
    rec_idx = torch.arange(rec_lo, min(end - 1, seq), device=weights.device)
    q_mass = mass(weights, q_idx)
    gen_mass = mass(weights, torch.arange(min(prompt_len, seq), seq, device=weights.device))
    denom = q_mass + gen_mass
    return {
        "lb_q": q_mass,
        "lb_cot": mass(weights, cot_idx),
        "lb_ans": mass(weights, ans_idx),
        "lb_rev": mass(weights, rev_idx),
        "lb_recency5": mass(weights, rec_idx),
        "lookback_ratio": (q_mass / denom) if denom > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", type=Path, default=SOLVER)
    parser.add_argument("--device-map", default="", help="e.g. auto for 2-GPU 30B/32B")
    parser.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer indices. Empty = dense grid (every 2 or 4 layers, plus quarters).",
    )
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
    if not pending:
        print(f"dense-internal shard={args.shard_id}/{args.num_shards} pending=0 -> {args.out}", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    load_kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    if args.device_map:
        n_vis = torch.cuda.device_count()
        load_kw["device_map"] = args.device_map
        load_kw["max_memory"] = {i: "36GiB" for i in range(n_vis)}
        model = AutoModelForCausalLM.from_pretrained(str(args.model), **load_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(args.model), **load_kw).to(args.device)
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    layer_ids = parse_layers(args.layers, n_layers)
    device = model.get_input_embeddings().weight.device
    head_device = model.lm_head.weight.device
    def first_id(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0]) if ids else -1

    stop_id = first_id("</think>")
    wait_id = first_id("Wait")
    alt_id = first_id("Alternatively")
    revise_pats = [tokenizer.encode(text, add_special_tokens=False) for text in REVISE_STRINGS]
    vocab = int(model.lm_head.weight.shape[0])
    print(
        f"dense-internal shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"layers={layer_ids} model={Path(args.model).name} "
        f"stop={stop_id} wait={wait_id} vocab={vocab} -> {args.out}",
        flush=True,
    )
    started = time.perf_counter()
    prev_qi = None
    last_hiddens: list[torch.Tensor] = []
    mid_hiddens: list[torch.Tensor] = []
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prefix = tokenizer.encode(prompt + str(row["reasoning_prefix"]) + "\n", add_special_tokens=False)
            answer = tokenizer.encode(f"\\boxed{{{row['answer']}}}", add_special_tokens=False)
            ids = prefix + answer
            prompt_len = len(prompt_ids)
            if len(ids) + 1 > args.max_context:
                overflow = len(ids) + 1 - args.max_context
                if overflow >= len(prefix):
                    handle.write(json.dumps({**{k: v for k, v in row.items() if k != "reasoning_prefix"}, "status": "too_long"}) + "\n")
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
                    handle.write(json.dumps({**{k: v for k, v in row.items() if k != "reasoning_prefix"}, "status": "no_answer"}) + "\n")
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
            for layer in layer_ids:
                if layer + 1 >= len(hidden):
                    continue
                span = hidden[layer + 1][0, content_start - 1 : content_end - 1].to(head_device)
                lens_h = model.model.norm(span)
                lens_logits = model.lm_head(lens_h).float()
                logp = torch.log_softmax(lens_logits, dim=-1)
                first_logp = logp[0]
                pre_logits.append((layer, first_logp))
                layer_logp[layer] = float(first_logp[first_tok].item())
                gathered = logp.gather(-1, target.to(logp.device).unsqueeze(-1)).squeeze(-1)
                layer_mean[layer] = float(gathered.mean().item())
                top1_match[layer] = int(int(first_logp.argmax().item()) == first_tok)
                if layer == layer_ids[-1]:
                    last_log = first_logp
            if last_log is not None:
                for layer, logp in pre_logits:
                    jsds[layer] = js_divergence(last_log, logp)
            emerge = next((layer for layer in layer_ids if top1_match.get(layer)), float("nan"))
            if jsds:
                pre_layer = max(jsds, key=jsds.get)
                dola_score = layer_logp.get(layer_ids[-1], float("nan")) - layer_logp.get(pre_layer, float("nan"))
                dola_jsd = jsds[pre_layer]
            else:
                pre_layer = float("nan")
                dola_score = float("nan")
                dola_jsd = float("nan")
            nxt = out.logits[0, -1].float()
            nxt_logp = torch.log_softmax(nxt, dim=-1)
            stop_logp = float(nxt_logp[stop_id].item()) if 0 <= stop_id < nxt_logp.numel() else float("nan")
            wait_logp = float(nxt_logp[wait_id].item()) if 0 <= wait_id < nxt_logp.numel() else float("nan")
            alt_logp = float(nxt_logp[alt_id].item()) if 0 <= alt_id < nxt_logp.numel() else float("nan")
            cont_ids = [idx for idx in (wait_id, alt_id) if 0 <= idx < nxt_logp.numel()]
            cont = torch.logsumexp(nxt_logp[cont_ids], dim=0) if cont_ids else nxt_logp.new_tensor(float("nan"))
            last_h = hidden[-1][0, end - 1].float().cpu()
            mid_h = hidden[n_layers // 2][0, end - 1].float().cpu() if len(hidden) > n_layers // 2 else last_h
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
            except Exception:
                lb = {key: float("nan") for key in ("lb_q", "lb_cot", "lb_ans", "lb_rev", "lb_recency5", "lookback_ratio")}
                pre_lb = {f"pre_{key}": float("nan") for key in ("lb_q", "lb_recency5", "lookback_ratio")}
            else:
                pre_lb = {
                    "pre_lb_q": pre_lb["lb_q"],
                    "pre_lb_recency5": pre_lb["lb_recency5"],
                    "pre_lookback_ratio": pre_lb["lookback_ratio"],
                }
            slim = {key: value for key, value in row.items() if key not in {"reasoning_prefix", "question"}}
            slim.update(
                {
                    "status": "ok",
                    **{f"dola_logp_l{layer}": layer_logp.get(layer, float("nan")) for layer in layer_ids},
                    **{f"dola_mean_l{layer}": layer_mean.get(layer, float("nan")) for layer in layer_ids},
                    **{f"dola_jsd_l{layer}": jsds.get(layer, float("nan")) for layer in layer_ids},
                    **{f"dola_top1_l{layer}": top1_match.get(layer, 0) for layer in layer_ids},
                    "dola_mean_rise": layer_mean.get(layer_ids[-1], float("nan")) - layer_mean.get(layer_ids[0], float("nan")),
                    "dola_jsd_max": dola_jsd,
                    "dola_pre_layer": float(pre_layer) if pre_layer == pre_layer else float("nan"),
                    "dola_score": dola_score,
                    "emerge_layer": float(emerge) if emerge == emerge else float("nan"),
                    "neg_emerge": (-float(emerge)) if emerge == emerge else float("nan"),
                    "layer_agree": float(sum(top1_match.get(layer, 0) for layer in layer_ids) / len(layer_ids)),
                    "stop_logp": stop_logp,
                    "wait_logp": wait_logp,
                    "alt_logp": alt_logp,
                    "stop_margin": stop_logp - wait_logp,
                    "stop_margin_alt": stop_logp - alt_logp,
                    "stop_vs_cont": stop_logp - float(cont.item()),
                    "eigen_k2": eigenscore(last_hiddens[-2:]),
                    "eigen_k4": eigenscore(last_hiddens),
                    "neg_eigen_k2": -eigenscore(last_hiddens[-2:]),
                    "neg_eigen_k4": -eigenscore(last_hiddens),
                    "mid_neg_eigen_k4": -eigenscore(mid_hiddens),
                    **lb,
                    **pre_lb,
                }
            )
            handle.write(json.dumps(slim) + "\n")
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
