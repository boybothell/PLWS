#!/usr/bin/env python3
"""s1 budget forcing：模型自己吐 </think> 时拦下，末尾接 Wait，再继续想。

NUM_IGNORE=2 就是论文 Table 11 的 2x Wait。接到 R1 上结束符是 </think>。
空 thinking，题干 / gold / 上下文和官方 Full-CoT 同一套。不禁词。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_LENS_DISABLE", "1")

AE = Path(__file__).resolve().parents[1]
PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")


def _nvidia_lib_path() -> str:
    root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
    extra = ":".join(
        sorted(str(path) for path in (root / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


os.environ["LD_LIBRARY_PATH"] = _nvidia_lib_path()
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(PUMA / "puma"))

from score_confcal_keytoken import INSTRUCTION  # noqa: E402
from score_leftover_jump import done_uids  # noqa: E402

MODELS = {
    "r1_7b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B",
    "nemotron_8b": "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_14b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B",
}

from math_grader import check_is_correct  # noqa: E402
from prompt_utils import get_task_type  # noqa: E402
from run_vllm import extract_answer  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402

OUT = AE / "results/s1_2x_wait"
JOBS = AE / "results/fromstart_core/r1_7b_s42/jobs.jsonl"
PAPER_DS = ("math-500", "gpqa-diamond", "aime24", "aime25")
WAIT_RE = re.compile(r"\bwait\b|\balternatively\b|\bhmm+\b|等一下", re.I)
IGNORE_STR = "Wait"


def _chat(tokenizer: Any, model_tag: str, question: str) -> str:
    tmpl = {"tokenize": False, "add_generation_prompt": True}
    if model_tag.startswith("qwen3"):
        tmpl["enable_thinking"] = True
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION}\n{question}"}],
        **tmpl,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--num-ignore", type=int, default=2, help="s1 的 NUM_IGNORE。2 = 2x Wait。")
    parser.add_argument("--ignore-str", default=IGNORE_STR)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--think-tokens", type=int, default=0)
    parser.add_argument("--answer-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-mem-util", type=float, default=0.88)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=Path, default=JOBS)
    parser.add_argument("--out-root", type=Path, default=OUT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--src",
        choices=("scratch", "official", "window"),
        default="scratch",
        help="scratch=空前缀重采；official/window=接已有思考，立刻 2× Wait。",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(PAPER_DS),
        help="逗号分隔。默认论文三摊 + AIME25。空=jobs 里全部。",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    keep = {x.strip() for x in args.datasets.split(",") if x.strip()}
    if keep:
        jobs = [job for job in jobs if job.get("dataset") in keep]
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_id]
    if args.limit:
        jobs = jobs[: args.limit]
    if args.seed is not None:
        seed = int(args.seed)
    elif jobs and jobs[0].get("seed") is not None:
        seed = int(jobs[0]["seed"])
    else:
        seed = 42
    src_tag = "2x" if args.src == "scratch" else f"{args.src}{args.num_ignore}x"
    args.out = args.out or (
        args.out_root / f"{args.model_tag}_s{seed}_{src_tag}" / f"scores_shard{args.shard_id}.jsonl"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    already = set()
    for path in args.out.parent.glob("scores_shard*.jsonl"):
        already |= done_uids(path)
    pending = [job for job in jobs if job["uid"] not in already]
    print(
        f"s1-{args.src}-{args.num_ignore}x {args.model_tag} shard={args.shard_id}/{args.num_shards} "
        f"ignore={args.ignore_str!r} jobs={len(jobs)} pending={len(pending)} -> {args.out}",
        flush=True,
    )
    if not pending:
        return
    if args.model_tag not in MODELS:
        raise SystemExit(f"unknown model-tag {args.model_tag}")
    model_path = MODELS[args.model_tag]
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    tp_size = max(1, len(visible))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(
        f"s1 {args.model_tag} tp={tp_size} model={model_path} max_model_len={args.max_context}",
        flush=True,
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
    )
    think_kw: dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 30,
        "stop": ["</think>"],
        "include_stop_str_in_output": False,
    }
    answer_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=30,
        max_tokens=args.answer_tokens,
    )
    reserve = args.answer_tokens + 128
    started = time.perf_counter()
    wrote = 0
    step = len(pending) if args.batch_size <= 0 else args.batch_size
    print(
        f"s1 shard{args.shard_id} step={step} pending={len(pending)} "
        f"num_ignore={args.num_ignore} datasets={sorted(keep)}",
        flush=True,
    )
    with args.out.open("a") as handle:
        for begin in range(0, len(pending), step):
            chunk = pending[begin : begin + step]
            states: list[dict[str, Any]] = []
            for job in chunk:
                chat = _chat(tokenizer, args.model_tag, job["question"])
                n_tok = len(tokenizer.encode(chat, add_special_tokens=False))
                room = args.max_context - n_tok - reserve
                if room < 16:
                    handle.write(
                        json.dumps({"uid": job["uid"], "status": "too_long", "mode": "s1_force", "n_left_tok": 0})
                        + "\n"
                    )
                    continue
                prefix = str(job.get("thought") or "") if args.src != "scratch" else ""
                if prefix:
                    prefix = prefix.split("</think>", 1)[0]
                n_left = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0
                room_left = room - n_left
                if room_left < 8:
                    handle.write(
                        json.dumps(
                            {
                                "uid": job["uid"],
                                "status": "too_long",
                                "mode": "s1_force",
                                "src": args.src,
                                "n_left_tok": n_left,
                            }
                        )
                        + "\n"
                    )
                    continue
                budget = room_left if args.think_tokens <= 0 else min(args.think_tokens, room_left)
                states.append(
                    {
                        "job": job,
                        "chat": chat,
                        "thought": prefix,
                        "n_left_tok": n_left,
                        "remaining": budget,
                        "n_injected": 0,
                        "n_stops": 0,
                        "hit_cap": False,
                        "finish": "stop" if prefix else "",
                    }
                )
            if not states:
                continue

            def run_think(active: list[dict[str, Any]], min_tokens: int, label: str) -> None:
                prompts = [s["chat"] + s["thought"] for s in active]
                params = [
                    SamplingParams(**think_kw, max_tokens=max(1, s["remaining"]), min_tokens=min_tokens)
                    for s in active
                ]
                print(
                    f"s1 shard{args.shard_id} {label} n={len(active)} "
                    f"elapsed={time.perf_counter() - started:.0f}s",
                    flush=True,
                )
                outs = llm.generate(prompts, params, use_tqdm=True)
                for state, out in zip(active, outs, strict=True):
                    gen = out.outputs[0] if out.outputs else None
                    text = gen.text if gen else ""
                    n_tok = len(list(gen.token_ids)) if gen else 0
                    finish = str(getattr(gen, "finish_reason", "") or "")
                    state["thought"] += text
                    state["remaining"] = max(0, state["remaining"] - n_tok)
                    state["finish"] = finish
                    if finish == "length":
                        state["hit_cap"] = True
                    elif finish == "stop":
                        state["n_stops"] += 1

            if args.src == "scratch":
                run_think(states, 0, "think0")
            for ign in range(args.num_ignore):
                more = [s for s in states if s["finish"] == "stop" and s["remaining"] > 0]
                if not more:
                    break
                for state in more:
                    state["thought"] += args.ignore_str
                    state["n_injected"] += 1
                run_think(more, 1, f"wait{ign + 1}")

            answer_prompts: list[str] = []
            ready: list[dict[str, Any]] = []
            for state in states:
                closed = state["chat"] + state["thought"]
                if "</think>" not in closed:
                    closed = closed + "\n</think>\n\n"
                n_closed = len(tokenizer.encode(closed, add_special_tokens=False))
                if n_closed + args.answer_tokens > args.max_context:
                    handle.write(
                        json.dumps(
                            {
                                "uid": state["job"]["uid"],
                                "status": "too_long",
                                "mode": "s1_force",
                                "n_injected": state["n_injected"],
                                "n_stops": state["n_stops"],
                                "hit_cap": state["hit_cap"],
                                "finish_reason": state["finish"],
                            }
                        )
                        + "\n"
                    )
                    continue
                answer_prompts.append(closed)
                ready.append(state)
            if not answer_prompts:
                continue
            print(
                f"s1 shard{args.shard_id} answer generate n={len(answer_prompts)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
            answer_outs = llm.generate(answer_prompts, answer_params, use_tqdm=True)
            for state, out in zip(ready, answer_outs, strict=True):
                job = state["job"]
                gen = out.outputs[0]
                text = gen.text
                token_ids = list(gen.token_ids)
                thought = state["thought"]
                generated_text = f"{thought}\n</think>\n\n{text}"
                task_type = get_task_type(job["dataset"])
                answer = extract_answer(generated_text, task_type)
                try:
                    gold_ok = bool(check_is_correct(answer, job["gt"]))
                except Exception:
                    gold_ok = False
                n_think_tok = len(tokenizer.encode(thought, add_special_tokens=False)) if thought else 0
                n_left_tok = int(state.get("n_left_tok") or 0)
                rec = {
                    "status": "ok",
                    "mode": "s1_force",
                    "src": args.src,
                    "num_ignore": args.num_ignore,
                    "ignore_str": args.ignore_str,
                    "uid": job["uid"],
                    "dataset": job["dataset"],
                    "question_idx": job["question_idx"],
                    "kind": "s1_force",
                    "host_ok": job.get("host_ok", False),
                    "old_answer": job.get("old_answer", ""),
                    "original_tokens": job.get("original_tokens"),
                    "new_answer": answer,
                    "keep": bool(rg.same(job.get("old_answer", ""), answer)),
                    "new_gold_ok": gold_ok,
                    "n_injected": state["n_injected"],
                    "n_stops": state["n_stops"],
                    "n_left_tok": n_left_tok,
                    "n_cont_tok": max(0, n_think_tok - n_left_tok),
                    "n_think_tok": n_think_tok,
                    "n_ans_tok": len(token_ids),
                    "n_out_tok": n_think_tok + len(token_ids),
                    "finish_reason": state["finish"],
                    "hit_cap": state["hit_cap"],
                    "n_wait": len(WAIT_RE.findall(thought)),
                    "has_wait": bool(WAIT_RE.search(thought)),
                    "new_text": (thought[-80:] + " | " + text[:120])[:200],
                    "generated_text": generated_text,
                    "task_type": task_type,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            handle.flush()
            print(
                f"s1 shard{args.shard_id} wrote={wrote}/{len(pending)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    print(f"done s1-{args.num_ignore}x shard={args.shard_id} wrote={wrote}", flush=True)


if __name__ == "__main__":
    main()
