#!/usr/bin/env python3
"""第一扇剩窗一次前向：hidden + 停边距 / 下一步熵 / 当前答 logp / 和别的试答第一 token 差。"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_keytoken import INSTRUCTION  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

MODELS = {
    "r1_7b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"),
    "nemotron_8b": Path("/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"),
    "r1_14b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"),
    "qwen3_4b": Path("/mnt/d/lsj/models/Qwen3-4B"),
    "qwen3_8b": Path("/mnt/d/lsj/models/Qwen3-8B"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["uid"])
        for row in load_jsonl(path)
        if row.get("status") in {"ok", "too_long", "oom", "no_answer"}
    }


def first_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if ids else -1


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
        f"waithelp {args.model_tag} shard={args.shard_id}/{args.num_shards} "
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
    stop_id = first_id(tokenizer, "</think>")
    wait_id = first_id(tokenizer, "Wait")
    qwen3 = args.model_tag.startswith("qwen3")

    last_vecs: list[np.ndarray] = []
    pre_vecs: list[np.ndarray] = []
    if args.out.exists():
        n_ok = sum(1 for row in load_jsonl(args.out) if row.get("status") == "ok")
        for name, bucket in (("hidden_last", last_vecs), ("hidden_pre", pre_vecs)):
            npy = args.out.with_name(f"{name}_{args.out.stem}.npy")
            if npy.exists():
                bucket.extend(list(np.load(npy)))
        if not (len(last_vecs) == len(pre_vecs) == n_ok):
            last_vecs, pre_vecs = [], []
            args.out.write_text("")
            pending = jobs
            print("hidden/jsonl mismatch; restarting shard", flush=True)

    started = time.perf_counter()
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            if qwen3:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prefix_text = str(row["reasoning_prefix"])
            prefix = tokenizer.encode(prompt + prefix_text + "\n", add_special_tokens=False)
            answer = tokenizer.encode(f"\\boxed{{{row['answer']}}}", add_special_tokens=False)
            ids = prefix + answer
            overflow = 0
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
            if end <= start:
                slim["status"] = "no_answer"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            try:
                with torch.inference_mode():
                    out = model(tensor, output_hidden_states=True, use_cache=False)
                    hidden = out.hidden_states
                    last = hidden[-1][0, end - 1].float().cpu().numpy().astype(np.float16)
                    pre = hidden[-1][0, max(0, start - 1)].float().cpu().numpy().astype(np.float16)
                    head_device = model.lm_head.weight.device
                    pre_h = hidden[-1][0, max(0, start - 1)].to(head_device)
                    nxt = torch.log_softmax(model.lm_head(model.model.norm(pre_h)).float(), dim=-1)
                    p = torch.softmax(nxt, dim=-1)
                    next_entropy = float((-(p * (p.clamp_min(1e-12).log())).sum()).item())
                    stop_lp = float(nxt[stop_id].item()) if 0 <= stop_id < nxt.numel() else float("nan")
                    wait_lp = float(nxt[wait_id].item()) if 0 <= wait_id < nxt.numel() else float("nan")
                    stop_margin = (
                        stop_lp - wait_lp if stop_lp == stop_lp and wait_lp == wait_lp else float("nan")
                    )
                    span = hidden[-1][0, start - 1 : end - 1].to(head_device)
                    logp = torch.log_softmax(model.lm_head(model.model.norm(span)).float(), dim=-1)
                    target = tensor[0, start:end].to(logp.device)
                    last_mean = float(logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item())
                    cur_first = first_id(tokenizer, f"\\boxed{{{row['answer']}}}")
                    cur_lp = float(nxt[cur_first].item()) if 0 <= cur_first < nxt.numel() else float("nan")
                    alt_lps = []
                    for alt in row.get("alts") or []:
                        aid = first_id(tokenizer, f"\\boxed{{{alt}}}")
                        if 0 <= aid < nxt.numel():
                            alt_lps.append(float(nxt[aid].item()))
                    alt_margin = (
                        cur_lp - max(alt_lps)
                        if cur_lp == cur_lp and alt_lps
                        else float("nan")
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                slim["status"] = "oom"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            last_vecs.append(last)
            pre_vecs.append(pre)
            slim.update(
                {
                    "status": "ok",
                    "hidden_idx": len(last_vecs) - 1,
                    "stop_margin": stop_margin,
                    "next_entropy": next_entropy,
                    "last_mean_logp": last_mean,
                    "cur_first_logp": cur_lp,
                    "alt_margin": alt_margin,
                    "seq_len": end,
                }
            )
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)

    np.save(
        args.out.with_name(f"hidden_last_{args.out.stem}.npy"),
        np.stack(last_vecs) if last_vecs else np.zeros((0, 1), np.float16),
    )
    np.save(
        args.out.with_name(f"hidden_pre_{args.out.stem}.npy"),
        np.stack(pre_vecs) if pre_vecs else np.zeros((0, 1), np.float16),
    )
    print(f"done n={len(last_vecs)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
