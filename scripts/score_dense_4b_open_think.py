#!/usr/bin/env python3
"""Open-think 4B teacher-force stop-margin. 4B does not generate thinking.

Qwen3-4B chat is left in thinking role: user = problem, assistant opens
<think>, then the 7B reasoning prefix and this trial's \\boxed{answer} are
teacher-forced. Next-token readout uses 4B's own tokenizer:

  stop_margin = logp(</think>) - logp(Wait)
  stop_vs_cont = logp(</think>) - logsumexp(Wait, Alternatively)

Optional contrast: judge-framed user, same stuffed think, force </think>,
then Direct-style Yes/No. This is not a 4B-written CoT.

vLLM only. Cross-vocab JS is not computed.
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
sys.path.insert(0, str(AE))
sys.path.insert(0, str(AE / "scripts"))

from score_confcal_judge import QWEN4B_MODEL, yes_no_token_ids  # noqa: E402
from score_confcal_v1 import h4_from_logprobs, logprob_value, nvidia_lib_path  # noqa: E402

THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"
DIRECT_USER = (
    "[Problem]\n{question}\n\n[Proposed final answer]\n{answer}\n\n"
    "Is the proposed final answer correct? Answer Yes or No."
)


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if math.isfinite(number) else float("-inf")


def logsumexp(values: list[float]) -> float:
    finite_vals = [value for value in values if math.isfinite(value)]
    if not finite_vals:
        return float("-inf")
    peak = max(finite_vals)
    return peak + math.log(sum(math.exp(value - peak) for value in finite_vals))


def first_token_ids(tokenizer, variants: list[str]) -> list[int]:
    ids: list[int] = []
    for text in variants:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def mapping_from_logprobs(logprobs: dict[int, Any] | None) -> dict[int, float]:
    return {int(token): logprob_value(item) for token, item in (logprobs or {}).items()}


def lp_of(mapping: dict[int, float], ids: list[int]) -> float:
    return logsumexp([mapping[token] for token in ids if token in mapping])


def stop_bundle(mapping: dict[int, float], stop_ids: list[int], wait_ids: list[int], alt_ids: list[int]) -> dict[str, float]:
    stop_logp = lp_of(mapping, stop_ids)
    wait_logp = lp_of(mapping, wait_ids)
    alt_logp = lp_of(mapping, alt_ids)
    cont = logsumexp([wait_logp, alt_logp])
    return {
        "stop_logp": stop_logp,
        "wait_logp": wait_logp,
        "alt_logp": alt_logp,
        "stop_margin": stop_logp - wait_logp if math.isfinite(stop_logp) and math.isfinite(wait_logp) else float("nan"),
        "stop_margin_alt": stop_logp - alt_logp if math.isfinite(stop_logp) and math.isfinite(alt_logp) else float("nan"),
        "stop_vs_cont": stop_logp - cont if math.isfinite(stop_logp) and math.isfinite(cont) else float("nan"),
        "n_logprobs": float(len(mapping)),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_done(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    done: set[tuple[int, int, str]] = set()
    for row in load_jsonl(path):
        done.add((int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])))
    return done


def open_think_header(tokenizer, user_content: str) -> str:
    """Official Qwen3 generation prompt, then force an open <think> block.

    apply_chat_template(..., enable_thinking=True) currently ends at
    `<|im_start|>assistant\\n` and does not insert <think>. Closed-think
    (`enable_thinking=False`) inserts an empty think and must not be used.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if "</think>" in text:
        raise RuntimeError("open-think header unexpectedly closed the think block")
    if text.endswith(THINK_OPEN) or text.endswith("<think>\n"):
        return text
    if text.endswith("<|im_start|>assistant\n"):
        return text + THINK_OPEN
    raise RuntimeError(f"unexpected open-think header suffix: {text[-80:]!r}")


def boxed(answer: str) -> str:
    return f"\\boxed{{{answer}}}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["VLLM_LENS_DISABLE"] = "1"
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.candidates)
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for index, qi in enumerate(qis) if index % args.num_shards == args.shard_id}
    jobs = [row for row in rows if int(row["question_idx"]) in keep]
    jobs.sort(key=lambda row: (int(row["question_idx"]), int(row["decision_step"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in jobs
        if (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in load_done(args.out)
    ]
    if args.limit:
        pending = pending[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN4B_MODEL), trust_remote_code=True)
    stop_ids = first_token_ids(tokenizer, ["</think>"])
    wait_ids = first_token_ids(tokenizer, ["Wait", " Wait"])
    alt_ids = first_token_ids(tokenizer, ["Alternatively", " Alternatively"])
    if len(stop_ids) != 1:
        raise RuntimeError(f"</think> is not a single 4B token: {stop_ids}")
    probe_ids = list(dict.fromkeys(stop_ids + wait_ids + alt_ids))
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yn_ids = list(dict.fromkeys(yes_ids + no_ids))

    llm = LLM(
        model=str(QWEN4B_MODEL),
        trust_remote_code=True,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        max_logprobs=max(len(probe_ids), len(yn_ids), 20),
        seed=args.seed,
    )
    stop_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(probe_ids),
        logprob_token_ids=probe_ids,
        detokenize=False,
    )
    yn_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(yn_ids),
        logprob_token_ids=yn_ids,
        detokenize=False,
    )
    meta = {
        "judge_model": str(QWEN4B_MODEL),
        "prompt_version": "open_think_teacherforce_v1",
        "backend": "vllm",
        "signals": ["stop_margin", "stop_margin_nl", "closed_direct_h4"],
        "token_ids": {"stop": stop_ids, "wait": wait_ids, "alt": alt_ids, "yes": yes_ids, "no": no_ids},
        "execution": (
            "Qwen3 enable_thinking=True + explicit <think>; 7B reasoning and boxed "
            "are stuffed; 4B does not generate think tokens; next-token stop-margin "
            "plus optional forced </think> Yes/No"
        ),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_jobs": len(jobs),
        "n_pending": len(pending),
    }
    args.out.with_name(f"meta_shard{args.shard_id}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"4b-open-think shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"stop={stop_ids} wait={wait_ids} alt={alt_ids} -> {args.out}",
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
                boxed_ans = boxed(answer)
                header = open_think_header(tokenizer, question)
                judge_header = open_think_header(
                    tokenizer, DIRECT_USER.format(question=question, answer=answer)
                )
                stop_prompt = header + reasoning + "\n" + boxed_ans
                stop_nl_prompt = stop_prompt + "\n"
                yn_prompt = judge_header + reasoning + "\n" + boxed_ans + THINK_CLOSE
                n_stop = len(tokenizer.encode(stop_nl_prompt, add_special_tokens=False))
                n_yn = len(tokenizer.encode(yn_prompt, add_special_tokens=False))
                if max(n_stop, n_yn) + 1 > args.max_context:
                    handle.write(
                        json.dumps(
                            {
                                "question_idx": int(row["question_idx"]),
                                "decision_step": int(row["decision_step"]),
                                "answer": answer,
                                "geo_conf": row.get("geo_conf"),
                                "status": "too_long",
                                "n_stop": n_stop,
                                "n_yn": n_yn,
                            }
                        )
                        + "\n"
                    )
                    skipped += 1
                    continue
                ready.append(
                    {
                        "row": row,
                        "answer": answer,
                        "stop_prompt": stop_prompt,
                        "stop_nl_prompt": stop_nl_prompt,
                        "yn_prompt": yn_prompt,
                        "n_stop": n_stop,
                        "n_yn": n_yn,
                    }
                )
            if not ready:
                handle.flush()
                continue
            try:
                stop_out = llm.generate([item["stop_prompt"] for item in ready], stop_params, use_tqdm=False)
                stop_nl_out = llm.generate([item["stop_nl_prompt"] for item in ready], stop_params, use_tqdm=False)
                yn_out = llm.generate([item["yn_prompt"] for item in ready], yn_params, use_tqdm=False)
            except Exception as exc:
                for item in ready:
                    row = item["row"]
                    handle.write(
                        json.dumps(
                            {
                                "question_idx": int(row["question_idx"]),
                                "decision_step": int(row["decision_step"]),
                                "answer": item["answer"],
                                "geo_conf": row.get("geo_conf"),
                                "status": "forward_fail",
                                "error": str(exc)[:200],
                            }
                        )
                        + "\n"
                    )
                skipped += len(ready)
                continue
            for item, stop, stop_nl, yn in zip(ready, stop_out, stop_nl_out, yn_out, strict=True):
                row = item["row"]
                stop_map = mapping_from_logprobs(stop.outputs[0].logprobs[0] if stop.outputs and stop.outputs[0].logprobs else {})
                nl_map = mapping_from_logprobs(
                    stop_nl.outputs[0].logprobs[0] if stop_nl.outputs and stop_nl.outputs[0].logprobs else {}
                )
                yn_first = yn.outputs[0].logprobs[0] if yn.outputs and yn.outputs[0].logprobs else {}
                after = stop_bundle(stop_map, stop_ids, wait_ids, alt_ids)
                after_nl = stop_bundle(nl_map, stop_ids, wait_ids, alt_ids)
                closed = h4_from_logprobs(yn_first, yes_ids, no_ids)
                handle.write(
                    json.dumps(
                        {
                            "question_idx": int(row["question_idx"]),
                            "decision_step": int(row["decision_step"]),
                            "answer": item["answer"],
                            "geo_conf": row.get("geo_conf"),
                            "is_g": row.get("is_g"),
                            "is_g_window": row.get("is_g_window"),
                            "status": "ok",
                            "n_stop": item["n_stop"],
                            "n_yn": item["n_yn"],
                            **after,
                            **{f"{key}_nl": value for key, value in after_nl.items()},
                            "closed_yes": closed["yes"],
                            "closed_margin": closed["margin"],
                            "closed_logp_yes": closed["logp_yes"],
                            "closed_logp_no": closed["logp_no"],
                            "closed_yes_no_coverage": closed["yes_no_coverage"],
                        }
                    )
                    + "\n"
                )
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


if __name__ == "__main__":
    main()
