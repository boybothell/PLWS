#!/usr/bin/env python3
"""Dump last-token hiddens for every dense trial (one HF forward).

Vectors (float16, aligned with ok jsonl rows):
  hidden_last  — last layer, last boxed token
  hidden_pre   — last layer, token before \\boxed
  hidden_mid   — layer 14, last boxed token
"""
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

from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()


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
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

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
    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(SOLVER),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    print(f"hidden-dump shard={args.shard_id}/{args.num_shards} pending={len(pending)} -> {args.out}", flush=True)

    last_vecs: list[np.ndarray] = []
    pre_vecs: list[np.ndarray] = []
    mid_vecs: list[np.ndarray] = []
    if args.out.exists():
        n_ok = sum(1 for row in load_jsonl(args.out) if row.get("status") == "ok")
        for name, bucket in (
            ("hidden_last", last_vecs),
            ("hidden_pre", pre_vecs),
            ("hidden_mid", mid_vecs),
        ):
            npy = args.out.with_name(args.out.stem.replace("scores", name) + ".npy")
            if npy.exists():
                bucket.extend(list(np.load(npy)))
        if not (len(last_vecs) == len(pre_vecs) == len(mid_vecs) == n_ok):
            last_vecs, pre_vecs, mid_vecs = [], [], []
            args.out.write_text("")
            print("hidden/jsonl mismatch; restarting shard", flush=True)

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
            last = hidden[-1][0, end - 1].float().cpu().numpy().astype(np.float16)
            pre = hidden[-1][0, start - 1].float().cpu().numpy().astype(np.float16)
            mid = hidden[15][0, end - 1].float().cpu().numpy().astype(np.float16) if len(hidden) > 15 else last
            last_vecs.append(last)
            pre_vecs.append(pre)
            mid_vecs.append(mid)
            slim = {key: value for key, value in row.items() if key not in {"reasoning_prefix", "question"}}
            slim["status"] = "ok"
            slim["hidden_idx"] = len(last_vecs) - 1
            handle.write(json.dumps(slim) + "\n")
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    stem = args.out.stem.replace("scores", "")
    np.save(args.out.with_name(f"hidden_last{stem}.npy"), np.stack(last_vecs) if last_vecs else np.zeros((0, 1), np.float16))
    np.save(args.out.with_name(f"hidden_pre{stem}.npy"), np.stack(pre_vecs) if pre_vecs else np.zeros((0, 1), np.float16))
    np.save(args.out.with_name(f"hidden_mid{stem}.npy"), np.stack(mid_vecs) if mid_vecs else np.zeros((0, 1), np.float16))
    print(f"done n={len(last_vecs)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
