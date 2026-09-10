#!/usr/bin/env python3
"""Sample full CoT with plain vLLM (no lens). Save generations.jsonl for HF replay.

Run with okay-budget-vllm's .venv + LD_LIBRARY_PATH (see projects/attn-early-exit/AGENTS.md).

IMPORTANT: that venv installs vllm-lens, which auto-loads via vllm.general_plugins and
forces enforce_eager=True (CUDA Graphs OFF). We set VLLM_LENS_DISABLE=1 before importing
vllm so sampling keeps CUDA Graphs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Must be set before `import vllm` (plugin registers on import).
os.environ["VLLM_LENS_DISABLE"] = "1"

# Fallback ids (overridden from tokenizer when possible)
THINK_END_ID_QWEN3 = 151668
THINK_END_ID_R1 = 151649  # DeepSeek-R1-Distill-Qwen </think>

DECODE = {
    "greedy": dict(temperature=0.0, top_p=1.0, top_k=0),
    "qwen3": dict(temperature=0.6, top_p=0.95, top_k=20),
    "r1": dict(temperature=0.6, top_p=0.95, top_k=0),  # R1 generation_config
}


def resolve_think_end_id(tokenizer) -> int:
    ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    # multi-token fallback by model name heuristics
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "deepseek" in name or "r1" in name:
        return THINK_END_ID_R1
    return THINK_END_ID_QWEN3


def build_think_prompt(tokenizer, problem: str, force_okay: bool) -> str:
    messages = [{"role": "user", "content": problem}]
    try:
        base = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        base = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if force_okay:
        # Avoid double <think> if template already opened it (R1/QwQ).
        if base.rstrip().endswith("<think>"):
            return base.rstrip() + "\nOkay"
        return base + "<think>\nOkay"
    return base


def load_math(n: int, problems_csv: str | None):
    if problems_csv:
        import csv

        problems, answers = [], []
        with open(problems_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                problems.append(row["problem"])
                answers.append(row.get("answer"))
                if len(problems) >= n:
                    break
        return problems, answers

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [ex["problem"] for ex in ds][:n]
    answers = [ex.get("answer") for ex in ds][:n]
    return problems, answers


def think_len(gen_ids: list[int], think_end_id: int) -> tuple[bool, int]:
    for i, t in enumerate(gen_ids):
        if t == think_end_id:
            return True, i
    return False, len(gen_ids)


def _inspect_cudagraph(llm) -> dict:
    """Read enforce_eager / cudagraph_mode from the live engine if possible."""
    info: dict = {
        "VLLM_LENS_DISABLE": os.environ.get("VLLM_LENS_DISABLE"),
        "enforce_eager": None,
        "cudagraph_mode": None,
        "compilation_mode": None,
    }
    try:
        eng = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        vcfg = getattr(eng, "vllm_config", None) if eng is not None else None
        if vcfg is not None:
            mcfg = getattr(vcfg, "model_config", None)
            ccfg = getattr(vcfg, "compilation_config", None)
            if mcfg is not None:
                info["enforce_eager"] = bool(getattr(mcfg, "enforce_eager", None))
            if ccfg is not None:
                info["cudagraph_mode"] = str(getattr(ccfg, "cudagraph_mode", None))
                info["compilation_mode"] = str(getattr(ccfg, "mode", None))
    except Exception as e:  # noqa: BLE001
        info["inspect_error"] = f"{type(e).__name__}: {e}"
    return info


def _cudagraph_disabled(graph_info: dict) -> bool:
    if graph_info.get("enforce_eager") is True:
        return True
    mode = str(graph_info.get("cudagraph_mode") or "")
    # e.g. "CUDAGraphMode.NONE" / "<CUDAGraphMode.NONE: 0>"
    if "NONE" in mode.upper() and "PIECEWISE" not in mode.upper() and "FULL" not in mode.upper():
        return True
    # bare "0" / "None"
    if mode in ("0", "None", "none", ""):
        # empty/None may mean inspect failed — not a hard fail
        return False
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/d/lsj/models/Qwen3-4B")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--problems-csv", default=None)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--decode", choices=list(DECODE), default="qwen3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="debug only: force eager (disables CUDA Graphs)",
    )
    p.add_argument("--force-okay", action="store_true", help="append <think>\\nOkay like okay-budget")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # Re-assert before import (in case caller exported something else).
    os.environ["VLLM_LENS_DISABLE"] = "1"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    dec = DECODE[args.decode]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    think_end_id = resolve_think_end_id(tok)
    problems, answers = load_math(args.n, args.problems_csv)

    prompts = [build_think_prompt(tok, q, args.force_okay) for q in problems]
    prompt_id_lens = [len(tok(pr, add_special_tokens=False)["input_ids"]) for pr in prompts]

    # Explicit: CUDA Graphs ON unless --enforce-eager (debug).
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
        enforce_eager=bool(args.enforce_eager),
    )
    graph_info = _inspect_cudagraph(llm)
    print(json.dumps({"vllm_graph": graph_info}, indent=2), flush=True)
    if not args.enforce_eager and _cudagraph_disabled(graph_info):
        raise RuntimeError(
            "CUDA Graphs still disabled after LLM init: "
            f"{graph_info}. Likely vllm-lens still active — "
            "ensure VLLM_LENS_DISABLE=1 is set before importing vllm "
            "(and that okay-budget-vllm/.venv is used with that env)."
        )
    sp_kwargs = dict(
        max_tokens=args.max_new_tokens,
        temperature=dec["temperature"],
        top_p=dec["top_p"],
        seed=args.seed,
    )
    if dec["top_k"] and dec["top_k"] > 0:
        sp_kwargs["top_k"] = dec["top_k"]
    if dec["temperature"] <= 0:
        sp_kwargs["temperature"] = 0.0
        sp_kwargs["top_p"] = 1.0
    params = SamplingParams(**sp_kwargs)

    outs = llm.generate(prompts, params)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "n": args.n,
        "max_new_tokens": args.max_new_tokens,
        "decode": args.decode,
        "decode_params": dec,
        "seed": args.seed,
        "force_okay": args.force_okay,
        "problems_csv": args.problems_csv,
        "enforce_eager": bool(args.enforce_eager),
        "VLLM_LENS_DISABLE": os.environ.get("VLLM_LENS_DISABLE"),
        "vllm_graph": graph_info,
        "think_end_id": think_end_id,
        "max_model_len": args.max_model_len,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    path = args.out / "generations.jsonl"
    n_exit = 0
    with path.open("w", encoding="utf-8") as f:
        for i, (out, plen, ans) in enumerate(
            tqdm(list(zip(outs, prompt_id_lens, answers)), desc="write")
        ):
            gen_ids = list(out.outputs[0].token_ids)
            exited, tlen = think_len(gen_ids, think_end_id)
            n_exit += int(exited)
            # Reconstruct full_ids = prompt + gen for HF teacher forcing
            prompt_ids = tok(prompts[i], add_special_tokens=False)["input_ids"]
            # vLLM may tokenize slightly differently; prefer engine prompt_token_ids if present
            pt = getattr(out, "prompt_token_ids", None)
            if pt is not None:
                prompt_ids = list(pt)
                plen = len(prompt_ids)
            full_ids = prompt_ids + gen_ids
            rec = {
                "idx": i,
                "problem": problems[i],
                "answer": ans,
                "prompt_len": plen,
                "gen_ids": gen_ids,
                "full_ids": full_ids,
                "generation": tok.decode(gen_ids, skip_special_tokens=False),
                "n_gen": len(gen_ids),
                "think_len": tlen,
                "exited": exited,
                "hit_max_new": len(gen_ids) >= args.max_new_tokens,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "n": len(problems),
        "n_exited": n_exit,
        "path": str(path),
    }
    (args.out / "sample_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # Allow `python -m attn_early_exit.sample_vllm` from repo root
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    main()
