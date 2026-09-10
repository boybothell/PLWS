#!/usr/bin/env python3
"""Head-to-head: HF mid-layer lens vs vllm-lens on the same 7B trials.

Does not set VLLM_LENS_DISABLE, so the vllm-lens plugin can capture residuals.
Uses GPU 6 by default. Does not touch the running extract queues.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

# Those imports set the kill switch; this script must turn the plugin back on.
os.environ.pop("VLLM_LENS_DISABLE", None)

LAYERS = (8, 14, 20, 27)
CAND = AE / "results/confcal_judge/v2/dense_candidates/math-500.jsonl"


def pick_rows(limit: int, max_ids: int, min_ids: int = 0) -> list[dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    chosen: list[dict] = []
    seen_q: dict[int, int] = {}
    for line in CAND.read_text().splitlines():
        row = json.loads(line)
        qi = int(row["question_idx"])
        if seen_q.get(qi, 0) >= 2:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix = tokenizer.encode(prompt + str(row["reasoning_prefix"]) + "\n", add_special_tokens=False)
        answer = tokenizer.encode(f"\\boxed{{{row['answer']}}}", add_special_tokens=False)
        ids = prefix + answer
        if len(ids) + 1 > max_ids or len(ids) < min_ids:
            continue
        row["_ids"] = ids
        row["_prefix_len"] = len(prefix)
        chosen.append(row)
        seen_q[qi] = seen_q.get(qi, 0) + 1
        if len(chosen) >= limit:
            break
    return chosen


def lens_from_hidden(hidden, norm, lm_head, ids: list[int], prefix_len: int) -> dict[str, float]:
    start, end = prefix_len, len(ids)
    target = torch.tensor(ids[start:end], device=hidden[LAYERS[0] + 1].device)
    out: dict[str, float] = {}
    for layer in LAYERS:
        h = norm(hidden[layer + 1][0, start - 1 : end - 1])
        logp = torch.log_softmax(lm_head(h).float(), dim=-1)
        gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        out[f"lens_l{layer}"] = float(gathered.mean().item())
        out[f"dola_l{layer}"] = float(logp[0, target[0]].item())
    return out


def run_hf(rows: list[dict]) -> tuple[list[dict], float, list[dict]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    load_t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(SOLVER),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda()
    model.eval()
    load_s = time.perf_counter() - load_t
    print(f"HF load {load_s:.1f}s", flush=True)
    scores = []
    slices = []
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            ids = row["_ids"]
            tensor = torch.tensor([ids], device="cuda")
            out = model(tensor, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states
            score = lens_from_hidden(hidden, model.model.norm, model.lm_head, ids, row["_prefix_len"])
            score["key"] = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
            scores.append(score)
            keep = {}
            start, end = row["_prefix_len"], len(ids)
            for layer in LAYERS:
                keep[layer] = hidden[layer + 1][0, start - 1 : end - 1].float().cpu()
            slices.append(keep)
    score_s = time.perf_counter() - started
    print(f"HF score {score_s:.1f}s  {len(rows) / score_s:.2f} items/s", flush=True)
    del model
    torch.cuda.empty_cache()
    return scores, score_s, slices


def run_vllm(rows: list[dict]) -> tuple[list[torch.Tensor], list[int], float]:
    os.environ.pop("VLLM_LENS_DISABLE", None)
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    from vllm import LLM, SamplingParams

    load_t = time.perf_counter()
    llm = LLM(
        model=str(SOLVER),
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_logprobs=1,
        seed=42,
    )
    print(f"vLLM load {time.perf_counter() - load_t:.1f}s", flush=True)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
        extra_args={"output_residual_stream": list(LAYERS)},
    )
    prompts = [{"prompt_token_ids": row["_ids"]} for row in rows]
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    score_s = time.perf_counter() - started
    print(f"vLLM score {score_s:.1f}s  {len(rows) / score_s:.2f} items/s", flush=True)
    streams = []
    seqs = []
    for output in outputs:
        rs = output.activations["residual_stream"]
        streams.append(rs.float().cpu())
        seqs.append(len(output.prompt_token_ids or row_ids(output)))
    del llm
    torch.cuda.empty_cache()
    return streams, seqs, score_s


def row_ids(output) -> list[int]:
    return list(output.prompt_token_ids or [])


def project_vllm(streams: list[torch.Tensor], rows: list[dict]) -> list[dict]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(SOLVER),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    norm = model.model.norm
    lm_head = model.lm_head
    scores = []
    with torch.inference_mode():
        for rs, row in zip(streams, rows, strict=True):
            # rs: (n_layer, seq, hidden) or (1, seq, hidden) for one layer
            if rs.dim() == 2:
                rs = rs.unsqueeze(0)
            start, end = row["_prefix_len"], len(row["_ids"])
            target = torch.tensor(row["_ids"][start:end])
            score = {"key": (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))}
            for index, layer in enumerate(LAYERS):
                layer_h = rs[index] if rs.shape[0] == len(LAYERS) else rs[0]
                span = layer_h[start - 1 : end - 1].to(dtype=next(norm.parameters()).dtype)
                h = norm(span)
                logp = torch.log_softmax(lm_head(h).float(), dim=-1)
                gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                score[f"lens_l{layer}"] = float(gathered.mean().item())
                score[f"dola_l{layer}"] = float(logp[0, target[0]].item())
            scores.append(score)
    del model
    return scores


def main() -> None:
    limit = int(os.environ.get("N", "12"))
    max_ids = int(os.environ.get("MAX_IDS", "1536"))
    min_ids = int(os.environ.get("MIN_IDS", "0"))
    rows = pick_rows(limit, max_ids, min_ids)
    print(
        f"n={len(rows)} min_ids={min_ids} max_ids={max_ids} "
        f"lens={[len(row['_ids']) for row in rows]}",
        flush=True,
    )
    if len(rows) < 4:
        raise SystemExit("not enough short trials")
    hf_scores, hf_s, hf_slices = run_hf(rows)
    streams, seqs, vllm_s = run_vllm(rows)
    print("residual shapes", [tuple(s.shape) for s in streams[:2]], "seq", seqs[:2], flush=True)
    vllm_scores = project_vllm(streams, rows)

    print("\n=== hidden match (cosine of boxed-span residual) ===", flush=True)
    cos_all = []
    for hf_keep, rs, row in zip(hf_slices, streams, rows, strict=True):
        if rs.dim() == 2:
            rs = rs.unsqueeze(0)
        start, end = row["_prefix_len"], len(row["_ids"])
        for index, layer in enumerate(LAYERS):
            a = hf_keep[layer]
            b = rs[index][start - 1 : end - 1]
            n = min(a.shape[0], b.shape[0])
            if n == 0:
                continue
            cos = torch.nn.functional.cosine_similarity(a[:n].flatten(), b[:n].flatten(), dim=0)
            cos_all.append(float(cos))
    print(
        f"mean cosine={sum(cos_all) / len(cos_all):.6f}  "
        f"min={min(cos_all):.6f}  n={len(cos_all)}",
        flush=True,
    )

    print("\n=== score match (same norm+lm_head) ===", flush=True)
    diffs: dict[str, list[float]] = {f"{kind}_l{layer}": [] for kind in ("lens", "dola") for layer in LAYERS}
    for left, right in zip(hf_scores, vllm_scores, strict=True):
        for key in diffs:
            diffs[key].append(abs(left[key] - right[key]))
    for key, values in diffs.items():
        print(
            f"{key:12s}  mean|Δ|={sum(values) / len(values):.5f}  "
            f"max|Δ|={max(values):.5f}",
            flush=True,
        )
    print(
        f"\nthroughput  HF={len(rows) / hf_s:.2f}/s  vLLM={len(rows) / vllm_s:.2f}/s  "
        f"speedup={hf_s / vllm_s:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
