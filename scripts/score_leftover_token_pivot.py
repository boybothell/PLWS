#!/usr/bin/env python3
"""剩窗后新思路 token：逐步熵 / top-1 / 是否走了第二候选。一次前向，不存 hidden。"""
from __future__ import annotations

import argparse
import json
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
CHUNK = 128
MAX_NEW = 768
DROP_KEYS = {"leftover_prefix", "next_prefix", "break_prefix", "question"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["uid"])
        for row in load_jsonl(path)
        if row.get("status") in {"ok", "too_long", "oom", "no_span", "align_fail"}
    }


def prompt_text(tokenizer, question: str, qwen3: bool) -> str:
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n{question}"}]
    if qwen3:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def encode_prefix(tokenizer, prompt: str, prefix: str) -> list[int]:
    return tokenizer.encode(prompt + prefix + "\n", add_special_tokens=False)


def common_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def summarize(ent: np.ndarray, p1: np.ndarray, not_top1: np.ndarray) -> dict[str, float]:
    if ent.size == 0:
        nan = float("nan")
        return {
            "n": 0.0,
            "ent_first": nan,
            "ent_last": nan,
            "ent_max": nan,
            "ent_mean": nan,
            "ent_p90": nan,
            "min_p1": nan,
            "mean_p1": nan,
            "frac_not_top1": nan,
            "n_ent_ge2": 0.0,
            "max_pos": nan,
        }
    i_max = int(ent.argmax())
    return {
        "n": float(ent.size),
        "ent_first": float(ent[0]),
        "ent_last": float(ent[-1]),
        "ent_max": float(ent.max()),
        "ent_mean": float(ent.mean()),
        "ent_p90": float(np.quantile(ent, 0.90)),
        "min_p1": float(p1.min()),
        "mean_p1": float(p1.mean()),
        "frac_not_top1": float(not_top1.mean()),
        "n_ent_ge2": float((ent >= 2.0).sum()),
        "max_pos": float(i_max / max(ent.size - 1, 1)),
    }


def prefix_stats(d: dict[str, float], tag: str) -> dict[str, float]:
    return {f"{tag}_{k}": v for k, v in d.items()}


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
        f"token_pivot {args.model_tag} shard={args.shard_id}/{args.num_shards} "
        f"pending={len(pending)}/{len(jobs)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(str(MODELS[args.model_tag]), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODELS[args.model_tag]),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    head_device = model.lm_head.weight.device
    qwen3 = args.model_tag.startswith("qwen3")

    started = time.perf_counter()
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            prompt = prompt_text(tokenizer, str(row["question"]), qwen3)
            left_ids = encode_prefix(tokenizer, prompt, str(row["leftover_prefix"]))
            next_ids = encode_prefix(tokenizer, prompt, str(row["next_prefix"]))
            break_text = str(row.get("break_prefix") or "")
            full_ids = (
                encode_prefix(tokenizer, prompt, break_text)
                if break_text
                else next_ids
            )
            left_n = common_len(left_ids, full_ids)
            next_n = common_len(next_ids, full_ids)
            slim = {k: v for k, v in row.items() if k not in DROP_KEYS}
            if left_n < max(8, int(0.8 * min(len(left_ids), len(full_ids)))):
                slim["status"] = "align_fail"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            end = min(len(full_ids), left_n + MAX_NEW)
            if end <= left_n:
                slim["status"] = "no_span"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            ids = full_ids[:end]
            overflow = 0
            if len(ids) + 1 > args.max_context:
                overflow = len(ids) + 1 - args.max_context
                if overflow >= left_n:
                    slim["status"] = "too_long"
                    handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                    handle.flush()
                    continue
                ids = ids[overflow:]
                left_n -= overflow
                next_n -= overflow
                end = len(ids)
            tensor = torch.tensor([ids], device=device)
            try:
                with torch.inference_mode():
                    out = model(tensor, output_hidden_states=True, use_cache=False)
                    hidden = out.hidden_states[-1][0]
                    ents: list[float] = []
                    p1s: list[float] = []
                    not1: list[float] = []
                    p2s: list[float] = []
                    for lo in range(left_n, end, CHUNK):
                        hi = min(end, lo + CHUNK)
                        span = hidden[lo - 1 : hi - 1].to(head_device)
                        logits = model.lm_head(model.model.norm(span)).float()
                        logp = torch.log_softmax(logits, dim=-1)
                        p = logp.exp()
                        ent = -(p * logp).sum(-1)
                        topv, topi = p.topk(2, dim=-1)
                        target = tensor[0, lo:hi].to(topi.device)
                        ents.extend(ent.cpu().tolist())
                        p1s.extend(topv[:, 0].cpu().tolist())
                        p2s.extend(topv[:, 1].cpu().tolist())
                        not1.extend((topi[:, 0] != target).float().cpu().tolist())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                slim["status"] = "oom"
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            ent_a = np.asarray(ents, dtype=np.float64)
            p1_a = np.asarray(p1s, dtype=np.float64)
            n1_a = np.asarray(not1, dtype=np.float64)
            p2_a = np.asarray(p2s, dtype=np.float64)
            next_cut = max(0, min(len(ent_a), next_n - left_n))
            next_stats = summarize(ent_a[:next_cut], p1_a[:next_cut], n1_a[:next_cut])
            all_stats = summarize(ent_a, p1_a, n1_a)
            last32 = summarize(ent_a[-32:], p1_a[-32:], n1_a[-32:])
            i_max = int(ent_a.argmax()) if ent_a.size else 0
            slim.update(prefix_stats(next_stats, "next"))
            slim.update(prefix_stats(all_stats, "span"))
            slim.update(prefix_stats(last32, "last32"))
            slim.update(
                {
                    "status": "ok",
                    "seq_len": end,
                    "align_left": left_n,
                    "p2_at_max": float(p2_a[i_max]) if p2_a.size else float("nan"),
                    "p1_at_max": float(p1_a[i_max]) if p1_a.size else float("nan"),
                    "gap_at_max": (
                        float(p1_a[i_max] - p2_a[i_max]) if p1_a.size else float("nan")
                    ),
                }
            )
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)

    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
