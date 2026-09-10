#!/usr/bin/env python3
"""vLLM Wait-only next-token dump. No hidden states, no half-depth.

After the stuffed boxed trial, read the next-token distribution:

  stop_logp       = logp(</think>)
  wait_logp       = logp(Wait)
  alt_logp        = logp(Alternatively)
  stop_margin     = stop - wait
  stop_margin_alt = stop - alt
  stop_vs_cont    = stop - logsumexp(wait, alt)
  next_entropy    = entropy of the returned next-token logprobs
  next_top5       = top-5 tokens at that position

Writes to dense_puma_wait (PUMA chat header). Does not touch HF lens / internal
or the old once-prompt files in dense_wait_once. Skip only keys already in this out file.
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

os.environ["VLLM_LENS_DISABLE"] = "1"

AE = Path(__file__).resolve().parents[1]
PUMA = AE.parent / "PUMA"
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_keytoken import INSTRUCTION  # noqa: E402
from score_confcal_v1 import logprob_value, logsumexp, nvidia_lib_path  # noqa: E402

# PUMA-aligned Wait goes here so once-prompt rows in dense_wait_once are not skipped.
WAIT_ROOT = Path(os.environ.get("WAIT_ROOT") or AE / "results/confcal_judge/v2/dense_puma_wait")


def row_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (int(row["question_idx"]), int(row["decision_step"]), str(row.get("answer") or ""))


def scored_wait(row: dict[str, Any]) -> bool:
    status = row.get("status")
    if status in {"too_long", "oom", "no_answer"}:
        return True
    return bool(status == "ok" and "stop_margin" in row)


def load_wait_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    out: set[tuple[int, int, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if scored_wait(row):
                out.add(row_key(row))
    return out


def twin_internal(lens_out: Path) -> Path:
    text = str(lens_out)
    if "/dense_lens/" in text:
        return Path(text.replace("/dense_lens/", "/dense_internal/"))
    if "/dense_wait_once/" in text:
        return Path(text.replace("/dense_wait_once/", "/dense_internal/"))
    if "/dense_wait/" in text:
        return Path(text.replace("/dense_wait/", "/dense_internal/"))
    return lens_out


def twin_lens(wait_out: Path) -> Path:
    text = str(wait_out)
    if "/dense_wait_once/" in text:
        return Path(text.replace("/dense_wait_once/", "/dense_lens/"))
    if "/dense_wait/" in text:
        return Path(text.replace("/dense_wait/", "/dense_lens/"))
    return wait_out


def wait_done_keys(wait_out: Path) -> set[tuple[int, int, str]]:
    done = load_wait_keys(wait_out)
    if wait_out.parent.is_dir():
        for sibling in wait_out.parent.glob("scores_shard*.jsonl"):
            if sibling.resolve() != wait_out.resolve():
                done |= load_wait_keys(sibling)
    # PUMA re-run must not inherit once-prompt Wait from lens/internal twins.
    if "dense_puma_wait" in str(wait_out):
        return done
    done |= load_wait_keys(twin_internal(wait_out))
    done |= load_wait_keys(twin_lens(wait_out))
    return done


def wait_out_path(tag: str, dataset: str, seed: int | None, shard_id: int) -> Path:
    stem = f"{dataset}_s{seed}" if seed is not None else dataset
    if tag == "r1_7b" and seed is None and dataset in {"math-500", "olympiadbench", "gpqa-diamond"}:
        return WAIT_ROOT / dataset / f"scores_shard{shard_id}.jsonl"
    return WAIT_ROOT / tag / stem / f"scores_shard{shard_id}.jsonl"


def first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if ids else -1


def mapping_from_logprobs(logprobs: dict[int, Any] | None) -> dict[int, float]:
    return {int(token): logprob_value(item) for token, item in (logprobs or {}).items()}


def token_logp(mapping: dict[int, float], token_id: int) -> float:
    if token_id < 0:
        return float("nan")
    value = mapping.get(token_id)
    return float(value) if value is not None else float("nan")


def next_entropy(mapping: dict[int, float]) -> float:
    probs = [math.exp(value) for value in mapping.values() if math.isfinite(value)]
    return float(-sum(prob * math.log(prob) for prob in probs if prob > 0.0)) if probs else float("nan")


def generate_safe(llm: Any, prompts: list[str], params: Any) -> list[Any]:
    try:
        return llm.generate(prompts, params, use_tqdm=False)
    except Exception as exc:  # noqa: BLE001
        print(f"batch generate failed ({exc}); retrying one-by-one", flush=True)
        outputs: list[Any] = []
        for prompt in prompts:
            try:
                outputs.append(llm.generate([prompt], params, use_tqdm=False)[0])
            except Exception as inner:  # noqa: BLE001
                outputs.append(inner)
        return outputs


def first_logprobs(output: Any) -> dict[int, Any]:
    if isinstance(output, Exception) or not getattr(output, "outputs", None):
        return {}
    logs = output.outputs[0].logprobs
    return logs[0] if logs else {}


def topk_tokens(tokenizer, mapping: dict[int, float], k: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:k]
    out = []
    for token_id, logp in ranked:
        out.append(
            {
                "id": int(token_id),
                "token": tokenizer.decode([token_id], skip_special_tokens=False),
                "logp": float(logp),
            }
        )
    return out


def build_once_prompt(tokenizer, question: str, reasoning: str, answer: str) -> str:
    header = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION}\n{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return header + reasoning + "\n" + f"\\boxed{{{answer}}}"


def build_puma_prompt(tokenizer, model_name: str, dataset: str, question: str, reasoning: str, answer: str) -> str:
    sys.path.insert(0, str(PUMA))
    from puma.prompt_utils import build_base_prompt, get_task_type  # noqa: E402

    base = build_base_prompt(tokenizer, model_name, question, get_task_type(dataset), "default")
    return base + "<think>" + reasoning + "\n" + f"\\boxed{{{answer}}}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--prompt-mode", choices=("once", "puma"), default="once")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--logprobs-k", type=int, default=20)
    args = parser.parse_args()

    os.environ["VLLM_LENS_DISABLE"] = "1"
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for index, qi in enumerate(qis) if index % args.num_shards == args.shard_id}
    jobs = [row for row in rows if int(row["question_idx"]) in keep]
    jobs.sort(key=lambda row: (int(row["question_idx"]), int(row["decision_step"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = wait_done_keys(args.out)
    pending = [row for row in jobs if row_key(row) not in done]
    if not pending:
        print(
            f"wait-vllm shard={args.shard_id}/{args.num_shards} pending=0 -> {args.out}",
            flush=True,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    stop_id = first_token_id(tokenizer, "</think>")
    wait_id = first_token_id(tokenizer, "Wait")
    alt_id = first_token_id(tokenizer, "Alternatively")
    wait_sp_id = first_token_id(tokenizer, " Wait")
    alt_sp_id = first_token_id(tokenizer, " Alternatively")
    probe_ids = list(
        dict.fromkeys(idx for idx in (stop_id, wait_id, alt_id, wait_sp_id, alt_sp_id) if idx >= 0)
    )
    dataset = args.dataset or str(pending[0].get("dataset") or "")
    model_name = str(args.model)

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=True,
        disable_custom_all_reduce=True,
        max_logprobs=max(args.logprobs_k, len(probe_ids), 5),
        seed=42,
    )
    # This vLLM build requires logprobs == len(logprob_token_ids) when both are set.
    # Top-k dump and the Wait/stop ids are therefore two separate 1-token reads.
    params_top = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=max(args.logprobs_k, 5),
        detokenize=False,
    )
    params_probe = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(probe_ids),
        logprob_token_ids=probe_ids,
        detokenize=False,
    )

    print(
        f"wait-vllm shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"model={Path(model_name).name} mode={args.prompt_mode} tp={args.tp} "
        f"stop={stop_id} wait={wait_id} alt={alt_id} -> {args.out}",
        flush=True,
    )

    ok = skipped = 0
    started = time.perf_counter()
    with args.out.open("a") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            ready: list[dict[str, Any]] = []
            for row in batch:
                question = str(row.get("question") or "")
                reasoning = str(row.get("reasoning_prefix") or "")
                answer = str(row.get("answer") or "")
                slim = {key: value for key, value in row.items() if key not in {"reasoning_prefix", "question"}}
                if not answer:
                    slim["status"] = "no_answer"
                    handle.write(json.dumps(slim) + "\n")
                    skipped += 1
                    continue
                if args.prompt_mode == "puma":
                    prompt = build_puma_prompt(tokenizer, model_name, dataset, question, reasoning, answer)
                else:
                    prompt = build_once_prompt(tokenizer, question, reasoning, answer)
                n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
                if n_tok + 1 > args.max_context:
                    slim["status"] = "too_long"
                    slim["n_tok"] = n_tok
                    handle.write(json.dumps(slim) + "\n")
                    skipped += 1
                    continue
                ready.append({"row": row, "slim": slim, "prompt": prompt, "n_tok": n_tok, "answer": answer})
            if not ready:
                handle.flush()
                continue
            prompts = [item["prompt"] for item in ready]
            top_out = generate_safe(llm, prompts, params_top)
            probe_out = generate_safe(llm, prompts, params_probe)
            for item, top, probe in zip(ready, top_out, probe_out, strict=True):
                slim = item["slim"]
                slim["n_tok"] = item["n_tok"]
                slim["backend"] = "vllm_wait"
                slim["prompt_mode"] = args.prompt_mode
                if isinstance(top, Exception) and isinstance(probe, Exception):
                    slim["status"] = "forward_fail"
                    slim["error"] = str(probe)[:200]
                    handle.write(json.dumps(slim) + "\n")
                    skipped += 1
                    continue
                mapping = mapping_from_logprobs(first_logprobs(top))
                mapping.update(mapping_from_logprobs(first_logprobs(probe)))
                if not mapping:
                    slim["status"] = "forward_fail"
                    slim["error"] = "empty next-token logprobs"
                    handle.write(json.dumps(slim) + "\n")
                    skipped += 1
                    continue
                output = probe if not isinstance(probe, Exception) else top
                stop_logp = token_logp(mapping, stop_id)
                wait_logp = token_logp(mapping, wait_id)
                alt_logp = token_logp(mapping, alt_id)
                wait_sp = token_logp(mapping, wait_sp_id)
                alt_sp = token_logp(mapping, alt_sp_id)
                cont_vals = [value for value in (wait_logp, alt_logp) if math.isfinite(value)]
                cont = logsumexp(cont_vals) if cont_vals else float("nan")
                greedy = output.outputs[0].token_ids[0] if output.outputs and output.outputs[0].token_ids else -1
                slim.update(
                    {
                        "status": "ok",
                        "stop_logp": stop_logp,
                        "wait_logp": wait_logp,
                        "alt_logp": alt_logp,
                        "wait_logp_sp": wait_sp,
                        "alt_logp_sp": alt_sp,
                        "stop_margin": stop_logp - wait_logp
                        if math.isfinite(stop_logp) and math.isfinite(wait_logp)
                        else float("nan"),
                        "stop_margin_alt": stop_logp - alt_logp
                        if math.isfinite(stop_logp) and math.isfinite(alt_logp)
                        else float("nan"),
                        "stop_vs_cont": stop_logp - cont if math.isfinite(stop_logp) and math.isfinite(cont) else float("nan"),
                        "next_entropy": next_entropy(mapping),
                        "next_top5": topk_tokens(tokenizer, mapping, 5),
                        "next_token_id": int(greedy),
                        "next_token": tokenizer.decode([greedy], skip_special_tokens=False) if greedy >= 0 else "",
                        "n_logprobs": float(len(mapping)),
                        "returned_mass": float(
                            sum(math.exp(value) for value in mapping.values() if math.isfinite(value))
                        ),
                    }
                )
                handle.write(json.dumps(slim) + "\n")
                ok += 1
            handle.flush()
            done_n = min(start + len(batch), len(pending))
            if done_n == len(batch) or done_n % (args.batch_size * 4) == 0 or done_n == len(pending):
                print(
                    f"[{done_n}/{len(pending)}] ok={ok} skip={skipped} "
                    f"elapsed={time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    print(f"done ok={ok} skip={skipped} -> {args.out}", flush=True)
    if ok == 0 and pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
