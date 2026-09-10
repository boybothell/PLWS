#!/usr/bin/env python3
"""Re-probe leftover windows after closing think. Same geo-conf, new suffix."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(PUMA / "puma"))

os.environ.setdefault("VLLM_LENS_DISABLE", "1")

from gen_trial_answers import (  # noqa: E402
    build_prompt,
    compute_confidence,
    extract_first_braced_content,
    extract_logprob_from_step,
    find_boxed_content_token_span,
    parse_response,
)
from prompt_utils import get_task_type  # noqa: E402

COMMIT = "</think>\n\nThe final answer is \\boxed"
MODEL = "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"


def done_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            out.add((int(row["question_idx"]), int(row["decision_step"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=30)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    qis = sorted({int(r["question_idx"]) for r in rows})
    keep = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [r for r in rows if int(r["question_idx"]) in keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    already = done_keys(args.out)
    pending = [r for r in jobs if (int(r["question_idx"]), int(r["decision_step"])) not in already]
    print(f"commit-probe shard={args.shard_id}/{args.num_shards} pending={len(pending)}/{len(jobs)}", flush=True)
    if not pending:
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")),
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens, logprobs=5, seed=42)
    prompts = []
    ready = []
    with args.out.open("a") as handle:
        for row in pending:
            prompt = build_prompt(
                tokenizer,
                MODEL,
                row["question"],
                row["reasoning_prefix"],
                get_task_type(args.dataset),
                "default",
                True,
                args.dataset,
                confident_ending=COMMIT,
            )
            n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
            if n_tok + args.max_tokens > int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")):
                handle.write(json.dumps({**row, "reasoning_prefix": "", "status": "too_long"}) + "\n")
                continue
            prompts.append(prompt)
            ready.append(row)
        started = time.perf_counter()
        wrote = 0
        for begin in range(0, len(ready), 32):
            chunk_rows = ready[begin : begin + 32]
            chunk_prompts = prompts[begin : begin + 32]
            outputs = llm.generate(chunk_prompts, params)
            for row, out in zip(chunk_rows, outputs, strict=True):
                gen = out.outputs[0]
                text = gen.text
                token_ids = list(gen.token_ids)
                logps = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []
                boxed = extract_first_braced_content(text)
                answer = boxed if boxed else parse_response(text)
                start, end = find_boxed_content_token_span(token_ids, tokenizer)
                use = logps[start:end] if start < end else logps
                slim = {k: v for k, v in row.items() if k != "reasoning_prefix"}
                slim.update(
                    {
                        "status": "ok",
                        "commit_answer": answer,
                        "commit_text": text[:200],
                        "commit_conf": compute_confidence(use, "geometric"),
                        "n_ans_tok": len(use),
                        "ending": COMMIT,
                    }
                )
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                wrote += 1
            handle.flush()
            print(f"[{wrote}/{len(ready)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done {wrote} in {time.perf_counter() - started:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
