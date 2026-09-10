#!/usr/bin/env python3
"""剩窗一次前向：近 K 步 EigenScore / 谱熵 + 末层 vs 一半深下一词 JSD/KL。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_keytoken import INSTRUCTION  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402
from score_dense_internal_metrics import eigenscore, js_divergence  # noqa: E402
from score_leftover_waithelp import MODELS, done_keys, load_jsonl  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()


def take(hidden_row, lo: int, hi: int) -> list:
    lo = max(0, lo)
    hi = max(lo + 1, hi)
    return [hidden_row[i].detach().float().cpu() for i in range(lo, hi)]


def spectrum_entropy(vectors: list, alpha: float = 1e-3) -> float:
    import torch

    if len(vectors) < 2:
        return float("nan")
    matrix = torch.stack([torch.nn.functional.normalize(item.float(), dim=0) for item in vectors], dim=0)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    gram = (matrix @ matrix.T) / max(matrix.shape[1], 1)
    gram = gram + alpha * torch.eye(gram.shape[0], dtype=gram.dtype)
    try:
        values = torch.linalg.eigvalsh(gram).clamp_min(1e-12)
        p = values / values.sum()
        return float((-(p * p.log())).sum().item())
    except Exception:
        return float("nan")


def layer_logp(model, hidden, layer_plus: int, pos: int):
    import torch

    if layer_plus < 0 or layer_plus >= len(hidden) or pos < 0:
        return None
    head = model.lm_head.weight.device
    vec = hidden[layer_plus][0, pos].to(head)
    return torch.log_softmax(model.lm_head(model.model.norm(vec)).float(), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.candidates)
    uids = sorted({str(r["uid"]) for r in rows})
    keep = {uid for i, uid in enumerate(uids) if i % args.num_shards == args.shard_id}
    jobs = [r for r in rows if str(r["uid"]) in keep]
    jobs.sort(key=lambda r: (str(r["dataset"]), int(r["seed"]), int(r["question_idx"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [r for r in jobs if str(r["uid"]) not in done_keys(args.out)]
    print(
        f"eigen-dola {args.model_tag} shard={args.shard_id}/{args.num_shards} "
        f"pending={len(pending)}/{len(jobs)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return

    model_path = MODELS[args.model_tag]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    n_layers = int(model.config.num_hidden_layers)
    half = n_layers // 2
    qwen3 = args.model_tag.startswith("qwen3")

    started = time.perf_counter()
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            kw = dict(
                tokenize=False,
                add_generation_prompt=True,
            )
            if qwen3:
                kw["enable_thinking"] = True
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                **kw,
            )
            prefix = tokenizer.encode(prompt + str(row["reasoning_prefix"]) + "\n", add_special_tokens=False)
            answer = tokenizer.encode(f"\\boxed{{{row['answer']}}}", add_special_tokens=False)
            ids = prefix + answer
            if len(ids) + 1 > args.max_context:
                overflow = len(ids) + 1 - args.max_context
                if overflow >= len(prefix):
                    slim = {k: v for k, v in row.items() if k not in {"reasoning_prefix", "question"}}
                    slim["status"] = "too_long"
                    handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                    handle.flush()
                    continue
                ids = prefix[overflow:] + answer
                prefix = prefix[overflow:]
            tensor = torch.tensor([ids], device=device)
            start = len(prefix)
            end = len(ids)
            slim = {k: v for k, v in row.items() if k not in {"reasoning_prefix", "question"}}
            if end <= 2:
                slim["status"] = "no_answer"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            try:
                with torch.inference_mode():
                    out = model(tensor, output_hidden_states=True, use_cache=False)
                    hidden = out.hidden_states
                    last_row = hidden[-1][0]
                    mid_row = hidden[half + 1][0] if half + 1 < len(hidden) else last_row
                    pre_end = max(1, start)
                    boxed_end = end
                    feats = {
                        "eigen_pre_k4": eigenscore(take(last_row, pre_end - 4, pre_end)),
                        "eigen_pre_k8": eigenscore(take(last_row, pre_end - 8, pre_end)),
                        "eigen_pre_k16": eigenscore(take(last_row, pre_end - 16, pre_end)),
                        "eigen_box_k4": eigenscore(take(last_row, boxed_end - 4, boxed_end)),
                        "eigen_box_k8": eigenscore(take(last_row, boxed_end - 8, boxed_end)),
                        "h_pre_k8": spectrum_entropy(take(last_row, pre_end - 8, pre_end)),
                        "h_box_k8": spectrum_entropy(take(last_row, boxed_end - 8, boxed_end)),
                        "mid_eigen_pre_k8": eigenscore(take(mid_row, pre_end - 8, pre_end)),
                    }
                    pos = max(0, start - 1)
                    log_last = layer_logp(model, hidden, len(hidden) - 1, pos)
                    log_half = layer_logp(model, hidden, half + 1, pos)
                    if log_last is not None and log_half is not None:
                        p = log_last.exp()
                        feats["dola_jsd"] = js_divergence(log_last, log_half)
                        feats["dola_kl_lh"] = float((p * (log_last - log_half)).sum().item())
                        feats["dola_kl_hl"] = float((log_half.exp() * (log_half - log_last)).sum().item())
                    else:
                        feats["dola_jsd"] = feats["dola_kl_lh"] = feats["dola_kl_hl"] = float("nan")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                slim["status"] = "oom"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            slim.update({"status": "ok", "half_layer": half, "seq_len": end, **feats})
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
