#!/usr/bin/env python3
"""Reproduce ASAG (arXiv:2606.15070, Algorithm 1) on R1-7B MATH.

Paper settings: greedy, λ=0.95, α=-0.1, s=1 jump, probe
\"\\n\\n Final Answer\\n\\n \\\\boxed\", Wait as ATP.
Modes: vanilla / deer (C>λ only) / asag (full) / asag_no_inject / asag_no_jump.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

PROBE = "\n\n Final Answer\n\n \\boxed"
JUMP = (
    "Wait, my previous reasoning is not correct. I should adopt a more concise "
    "and different approach to reexamine this problem.\n\n"
)
LAMBDA = 0.95
ALPHA = -0.1
MAX_NEW = 16000
PROBE_MAX = 64


def extract_boxed(text: str) -> str:
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    values: list[str] = []
    for start in starts:
        depth = 1
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[start:end].strip())
                    break
    return values[-1] if values else text.strip()


def load_math(path: Path) -> list[dict[str, Any]]:
    slim = AE / "results/asag_r1_7b/math-500/questions.jsonl"
    if slim.exists():
        return [json.loads(line) for line in slim.read_text().splitlines() if line.strip()]
    rows = json.loads(path.read_text())
    slim.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with slim.open("w") as handle:
        for index, row in enumerate(rows, start=1):
            item = {
                "question_idx": index,
                "question": row["question"],
                "gold": row.get("ground_truth_answer") or "",
            }
            handle.write(json.dumps(item) + "\n")
            out.append(item)
    return out


def shannon(probs) -> float:
    clipped = probs.clamp_min(1e-12)
    return float(-(clipped * clipped.log()).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("vanilla", "deer", "asag", "asag_no_inject", "asag_no_jump"), default="asag")
    parser.add_argument("--answers", type=Path, default=AE / "results/math500_official/puma_ds7b/answers.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda-th", type=float, default=LAMBDA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--max-jumps", type=int, default=1)
    args = parser.parse_args()
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LogitsProcessor,
        LogitsProcessorList,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    items = load_math(args.answers)[: args.max_questions]
    items = [row for i, row in enumerate(items) if i % args.num_shards == args.shard_id]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.open():
            if line.strip():
                done.add(int(json.loads(line)["question_idx"]))
    pending = [row for row in items if int(row["question_idx"]) not in done]
    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(SOLVER),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    wait_id = tokenizer.encode("Wait", add_special_tokens=False)
    wait_id = wait_id[0] if len(wait_id) == 1 else tokenizer.convert_tokens_to_ids("Wait")
    close_id = tokenizer.encode("</think>", add_special_tokens=False)
    close_id = close_id[0] if close_id else -1
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)

    class StopIds(StoppingCriteria):
        def __init__(self, ids: set[int], start: int):
            self.ids = ids
            self.start = start

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.shape[1] <= self.start:
                return False
            return int(input_ids[0, -1].item()) in self.ids

    def generate_until(ids: list[int], stop: set[int], limit: int, processor=None) -> list[int]:
        start = len(ids)
        tensor = torch.tensor([ids], device=model.device)
        extra = {}
        if processor is not None:
            extra["logits_processor"] = LogitsProcessorList(processor)
        with torch.inference_mode():
            out = model.generate(
                tensor,
                max_new_tokens=max(1, min(limit, MAX_NEW - max(0, start - prompt_len))),
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopIds(stop, start)]),
                pad_token_id=tokenizer.eos_token_id,
                **extra,
            )
        return out[0].tolist()

    def probe(ids: list[int]) -> tuple[str, float, float, list[float], torch.Tensor | None]:
        probe_ids = tokenizer.encode(PROBE, add_special_tokens=False)
        start = len(ids) + len(probe_ids)
        seq = generate_until(ids + probe_ids, {tokenizer.encode("}", add_special_tokens=False)[0], close_id}, PROBE_MAX)
        ans_ids = seq[start:]
        if not ans_ids:
            return "", float("nan"), float("nan"), [], None
        tensor = torch.tensor([seq], device=model.device)
        with torch.inference_mode():
            out = model(tensor, output_attentions=True, use_cache=False)
        logits = out.logits[0, start - 1 : start - 1 + len(ans_ids)].float()
        logp = torch.log_softmax(logits, dim=-1)
        gathered = logp.gather(-1, torch.tensor(ans_ids, device=model.device).unsqueeze(-1)).squeeze(-1)
        conf = float(gathered.exp().mean().item())
        answer_dist = torch.softmax(logits, dim=-1).mean(0).detach()
        win = len(probe_ids)
        thought_start = max(prompt_len, start - win - win)
        query_from = thought_start
        query_to = start  # T-tail + I, exclude A
        entropies = []
        masses = []
        for layer in range(n_layers - 4, n_layers):
            attn = out.attentions[layer][0].float()  # heads, q, k
            sl = attn[:, query_from:query_to, :]
            for head in range(n_heads):
                for q in range(sl.shape[1]):
                    entropies.append(shannon(sl[head, q]))
            masses.append(attn[:, query_from:query_to, :].mean(dim=(0, 1)))
        h = float(sum(entropies)) if entropies else float("nan")
        global_attn = torch.stack(masses).mean(0) if masses else None
        text = tokenizer.decode(ans_ids, skip_special_tokens=True)
        return extract_boxed("\\boxed{" + text if "\\boxed" not in text else text), conf, h, ans_ids, global_attn

    class MixAnswer(LogitsProcessor):
        def __init__(self, dist: torch.Tensor, mix: float = 0.05):
            self.dist = dist
            self.mix = mix

        def __call__(self, input_ids, scores):
            probs = torch.softmax(scores.float(), dim=-1)
            mixed = (1.0 - self.mix) * probs + self.mix * self.dist.to(probs.device).unsqueeze(0)
            return mixed.clamp_min(1e-12).log()

    print(f"asag mode={args.mode} shard={args.shard_id}/{args.num_shards} pending={len(pending)} wait_id={wait_id}", flush=True)
    started = time.perf_counter()
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(ids)
            decisions = []
            h1 = None
            jumps = 0
            injects = 0
            exited = False
            inject_proc = None
            remaining = MAX_NEW
            while remaining > 0 and len(ids) < prompt_len + MAX_NEW:
                before = len(ids)
                stop = {wait_id, close_id} if args.mode != "vanilla" else {close_id}
                ids = generate_until(ids, stop, remaining, processor=inject_proc)
                inject_proc = None
                remaining -= len(ids) - before
                if len(ids) == before:
                    break
                last = ids[-1]
                if last == close_id or args.mode == "vanilla":
                    if last != close_id and remaining > 0:
                        ids = generate_until(ids, set(), remaining)
                    break
                if last != wait_id:
                    continue
                trial, conf, h, ans_ids, global_attn = probe(ids)
                delta = None if h1 is None else ((h - h1) / h1 if h1 else float("nan"))
                if h1 is None:
                    h1 = h
                action = "continue"
                if args.mode == "deer":
                    if math.isfinite(conf) and conf > args.lambda_th:
                        action = "exit"
                else:
                    allow_inject = args.mode in {"asag", "asag_no_jump"}
                    allow_jump = args.mode in {"asag", "asag_no_inject"}
                    if (delta is None and conf > args.lambda_th) or (
                        delta is not None and delta < args.alpha and conf > args.lambda_th
                    ):
                        action = "exit"
                    elif delta is not None and delta < args.alpha and conf < args.lambda_th:
                        action = "inject" if allow_inject else "continue"
                    elif delta is not None and delta > args.alpha:
                        trap = False
                        if global_attn is not None:
                            # thought actions: split generated ids on Wait
                            wait_pos = [i for i, tok in enumerate(ids[prompt_len:], start=prompt_len) if tok == wait_id]
                            if len(wait_pos) >= 2:
                                cur_a, cur_b = wait_pos[-2] + 1, wait_pos[-1]
                                prev_a = wait_pos[-3] + 1 if len(wait_pos) >= 3 else prompt_len
                                prev_b = wait_pos[-2]
                                prev_m = float(global_attn[prev_a:prev_b].mean()) if prev_b > prev_a else 0.0
                                cur_m = float(global_attn[cur_a:cur_b].mean()) if cur_b > cur_a else 0.0
                                trap = prev_m > cur_m
                            elif len(wait_pos) == 1:
                                trap = False
                        if trap and allow_jump:
                            action = "jump" if jumps < args.max_jumps else "exit"
                        else:
                            action = "continue"
                decisions.append(
                    {
                        "conf": conf,
                        "entropy": h,
                        "delta_h": delta,
                        "action": action,
                        "trial": trial,
                    }
                )
                if action == "exit":
                    ids = ids + tokenizer.encode("</think>\n", add_special_tokens=False)
                    ids = generate_until(ids, set(), min(256, remaining))
                    exited = True
                    break
                if action == "inject" and ans_ids:
                    injects += 1
                    dist = torch.zeros(model.config.vocab_size, device=model.device)
                    for tok in ans_ids:
                        if 0 <= tok < dist.numel():
                            dist[tok] += 1.0
                    if float(dist.sum()) > 0:
                        dist = dist / dist.sum()
                    inject_proc = [MixAnswer(dist)]
                    if ids[-1] == wait_id:
                        ids = ids[:-1]
                    continue
                if action == "jump":
                    jumps += 1
                    if ids[-1] == wait_id:
                        ids = ids[:-1]
                    ids = ids + tokenizer.encode(JUMP, add_special_tokens=False)
                    continue
                # keep Wait and continue
            text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
            pred = extract_boxed(text)
            rec = {
                "question_idx": int(row["question_idx"]),
                "mode": args.mode,
                "status": "ok",
                "pred": pred,
                "gold": row["gold"],
                "correct": int(_fast_eq(pred, row["gold"])),
                "n_tokens": len(ids) - prompt_len,
                "exited": int(exited),
                "n_wait": sum(1 for tok in ids[prompt_len:] if tok == wait_id),
                "n_inject": injects,
                "n_jump": jumps,
                "n_exit_decisions": sum(1 for d in decisions if d["action"] == "exit"),
                "n_inject_decisions": sum(1 for d in decisions if d["action"] == "inject"),
                "n_jump_decisions": sum(1 for d in decisions if d["action"] == "jump"),
                "n_continue_decisions": sum(1 for d in decisions if d["action"] == "continue"),
                "decisions": decisions,
            }
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(pending)}] q={row['question_idx']} acc={rec['correct']} "
                f"tok={rec['n_tokens']} wait={rec['n_wait']} inj={injects} jump={jumps} "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
