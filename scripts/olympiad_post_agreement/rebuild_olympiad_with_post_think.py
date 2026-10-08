#!/usr/bin/env python3
"""Append post-</think> onto Olympiad series without disturbing the thinking curve.

Phase 1 = existing think-only freeze (identical to the old token series).
Phase 2 = after every question has finished thinking, advance through post-</think>
trial probes (freeze finished posts at their last value).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if "PLWS_ROOT" in __import__("os").environ:
    ROOT = Path(__import__("os").environ["PLWS_ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tmp" / "PUMA" / "puma"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer

from plot_count_bias_s42 import window_end_ok  # noqa: E402
from plot_count_bias_s42_tokens import (  # noqa: E402
    continuation_token_counts,
    count_bias_tokens,
    fullcot_tokens,
    mean_axis,
    trail,
)
from plws.grading import grade_many  # noqa: E402
from run_r1_7b_seed42 import continuation_ends, load_jsonl  # noqa: E402
from separate_steps import separate_steps  # noqa: E402

MODEL = "/ssd/share/models/DeepSeek-R1-Distill-Qwen-7B"
DATASET = "olympiadbench"
SEED = 42
WIN = 15
JOBS = (
    ROOT
    / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b"
    / DATASET
    / f"seed_{SEED}"
    / "jobs/firstwin.jsonl"
)
ACC_CACHE = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_series.json"
TOKEN_CACHE = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_token_series.json"
FULL_META = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b/fullcot_seed42_series.json"
THINK_PROBES = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b" / DATASET / f"seed_{SEED}"
POST_FULL = ROOT / "tmp/plws_step_probe/post_think/fullcot/r1_7b" / DATASET / f"seed_{SEED}"
POST_BIAS = (
    ROOT
    / "tmp/plws_step_probe/post_think/count_bias/rho_0p98/r1_7b"
    / DATASET
    / f"seed_{SEED}"
)
FULLCOT_ANSWERS = ROOT / "samples/r1_7b" / DATASET / f"seed_{SEED}" / "answers.json"
BIAS_SCORES = (
    ROOT
    / "results/runs/plws/count_bias/rho_0p98/r1_7b"
    / DATASET
    / f"seed_{SEED}"
)
OUT_TOKEN = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_token_series_post.json"


def load_probe_flags(directory: Path) -> dict[str, dict[int, bool]]:
    found: dict[str, dict[int, bool]] = defaultdict(dict)
    for path in sorted(directory.glob("trials*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                raise SystemExit(f"bad probe {path} {row.get('uid')}")
            found[row["uid"]][int(row["rel_step"])] = bool(row["gold_ok"])
    return found


def curve(seqs: list[list[bool]]) -> list[float]:
    n = len(seqs)
    tmax = max(len(row) for row in seqs)
    return [
        100.0
        * sum(row[step] if step < len(row) else row[-1] for row in seqs)
        / n
        for step in range(tmax)
    ]


def post_phase_mean(
    token_rows: list[list[int]],
    flag_rows: list[list[bool]],
    baseline: float,
) -> tuple[list[float], list[float]]:
    n = len(token_rows)
    width = max(len(row) for row in token_rows)
    xs: list[float] = []
    ys: list[float] = []
    for step in range(width):
        tok = 0.0
        ok = 0
        for tokens, flags in zip(token_rows, flag_rows):
            tok += tokens[step] if step < len(tokens) else tokens[-1]
            ok += int(flags[step] if step < len(flags) else flags[-1])
        xs.append(tok / n - baseline)
        ys.append(100.0 * ok / n)
    return xs, ys


def post_token_extras(tokenizer, text: str, think_end_tokens: int) -> list[int]:
    if "</think>" not in text:
        raise SystemExit("missing </think>")
    _think, post = text.split("</think>", 1)
    chunks = [chunk for chunk in separate_steps(post) if chunk.strip()]
    ends = continuation_ends(post, chunks)
    close_and_post = "</think>" + post
    abs_ends = [len("</think>") + end for end in ends]
    counts = continuation_token_counts(tokenizer, close_and_post, abs_ends)
    return [think_end_tokens + count for count in counts]


def main() -> None:
    print("load tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    jobs = load_jsonl(JOBS)
    meta = json.loads(FULL_META.read_text())["datasets"][DATASET]
    acc_cache = json.loads(ACC_CACHE.read_text())
    token_cache = json.loads(TOKEN_CACHE.read_text())
    old = token_cache[DATASET]

    think_bias = load_probe_flags(THINK_PROBES)
    post_full = load_probe_flags(POST_FULL)
    post_bias = load_probe_flags(POST_BIAS)
    for job in jobs:
        uid = job["uid"]
        for name, table in (
            ("think_bias", think_bias),
            ("post_full", post_full),
            ("post_bias", post_bias),
        ):
            if uid not in table:
                raise SystemExit(f"missing {name} {uid}")
            keys = sorted(table[uid])
            if keys != list(range(1, keys[-1] + 1)):
                raise SystemExit(f"gap {name} {uid}")

    t0 = window_end_ok(DATASET, jobs, meta["trials"])
    t0_rate = acc_cache[DATASET]["count_bias"][0]
    if abs(100.0 * sum(t0) / len(t0) - t0_rate) > 1e-6:
        raise SystemExit(f"t0 mismatch {100.0 * sum(t0) / len(t0)} vs {t0_rate}")

    bias_think_flags = []
    for job, base in zip(jobs, t0):
        steps = think_bias[job["uid"]]
        bias_think_flags.append(
            [base] + [steps[step] for step in range(1, max(steps) + 1)]
        )

    full_rows = fullcot_tokens(DATASET, jobs, meta["trials"])
    window = [row[0] for row in full_rows]
    baseline = sum(window) / len(window)
    bias_rows = count_bias_tokens(tokenizer, DATASET, jobs, window)

    by_question: dict[int, dict[int, str]] = defaultdict(dict)
    for row in json.loads((ROOT / meta["trials"]).read_text()):
        by_question[int(row["question_idx"])][int(row["stopped_len"])] = row.get(
            "final_answer"
        )
    pairs = []
    spans = []
    for job in jobs:
        index = int(job["question_idx"])
        left = int(job["left_step"])
        steps = by_question[index]
        post = sorted(step for step in steps if step >= left)
        spans.append(len(post))
        gold = job["gt"]
        for step in post:
            pairs.append((steps[step], gold))
    graded = grade_many(pairs, workers=8, chunksize=16)
    cursor = 0
    full_think_flags: list[list[bool]] = []
    for job, npost in zip(jobs, spans):
        chunk = graded[cursor : cursor + npost]
        row_flags = []
        for ok, err in chunk:
            if err:
                raise SystemExit(f"grade error {job['uid']}: {err}")
            row_flags.append(bool(ok))
        cursor += npost
        full_think_flags.append(row_flags)

    answers = json.loads(FULLCOT_ANSWERS.read_text())
    bias_texts = {
        row["uid"]: row["generated_text"]
        for path in sorted(BIAS_SCORES.glob("shard_*.jsonl"))
        for row in load_jsonl(path)
    }

    full_post_tok_rows: list[list[int]] = []
    bias_post_tok_rows: list[list[int]] = []
    full_post_flag_rows: list[list[bool]] = []
    bias_post_flag_rows: list[list[bool]] = []
    for job, full_think_tok, bias_think_tok in zip(jobs, full_rows, bias_rows):
        uid = job["uid"]
        q = int(job["question_idx"])
        full_text = answers[q - 1]["generated_text"]
        if answers[q - 1]["question"].strip() != job["question"].strip():
            raise SystemExit(f"question mismatch {uid}")
        bias_text = bias_texts[uid]
        full_post_tok = post_token_extras(tokenizer, full_text, full_think_tok[-1])
        bias_post_tok = post_token_extras(tokenizer, bias_text, bias_think_tok[-1])
        pf = post_full[uid]
        pb = post_bias[uid]
        full_post_ok = [pf[step] for step in range(1, max(pf) + 1)]
        bias_post_ok = [pb[step] for step in range(1, max(pb) + 1)]
        if len(full_post_tok) != len(full_post_ok) or len(bias_post_tok) != len(
            bias_post_ok
        ):
            raise SystemExit(
                f"len mismatch {uid} full {len(full_post_tok)}/{len(full_post_ok)} "
                f"bias {len(bias_post_tok)}/{len(bias_post_ok)}"
            )
        full_post_tok_rows.append(full_post_tok)
        bias_post_tok_rows.append(bias_post_tok)
        full_post_flag_rows.append(full_post_ok)
        bias_post_flag_rows.append(bias_post_ok)

    # Phase 1: thinking only (must match the old cached curve).
    full_x_think = mean_axis(full_rows, baseline)
    bias_x_think = mean_axis(bias_rows, baseline)
    full_raw_think = curve(full_think_flags)
    bias_raw_think = curve(bias_think_flags)
    if [
        abs(a - b) for a, b in zip(full_x_think, old["full_x"])
    ] and max(abs(a - b) for a, b in zip(full_x_think, old["full_x"])) > 1e-6:
        raise SystemExit("full think x drifted from old cache")
    if max(abs(a - b) for a, b in zip(bias_x_think, old["count_bias_x"])) > 1e-6:
        raise SystemExit("bias think x drifted from old cache")
    if max(abs(a - b) for a, b in zip(full_raw_think, acc_cache[DATASET]["full"])) > 1e-6:
        raise SystemExit("full think acc drifted from old cache")
    if max(abs(a - b) for a, b in zip(bias_raw_think, acc_cache[DATASET]["count_bias"])) > 1e-6:
        raise SystemExit("bias think acc drifted from old cache")

    # Phase 2: post-</think> after the thinking curve has ended.
    full_x_post, full_raw_post = post_phase_mean(
        full_post_tok_rows, full_post_flag_rows, baseline
    )
    bias_x_post, bias_raw_post = post_phase_mean(
        bias_post_tok_rows, bias_post_flag_rows, baseline
    )

    full_x = [*full_x_think, *full_x_post]
    bias_x = [*bias_x_think, *bias_x_post]
    full_raw = [*full_raw_think, *full_raw_post]
    bias_raw = [*bias_raw_think, *bias_raw_post]
    full_y = trail(full_raw, WIN)
    bias_y = trail(bias_raw, WIN)

    # Thinking-region trailing MA must stay identical.
    think_n = len(full_x_think)
    if max(abs(a - b) for a, b in zip(full_y[:think_n], old["full_y"])) > 1e-6:
        raise SystemExit("full think trailed y drifted")
    if max(abs(a - b) for a, b in zip(bias_y[: len(bias_x_think)], old["count_bias_y"])) > 1e-6:
        raise SystemExit("bias think trailed y drifted")

    out = dict(token_cache)
    out[DATASET] = {
        "n": len(jobs),
        "baseline_tokens": baseline,
        "full_x": full_x,
        "count_bias_x": bias_x,
        "full_y": full_y,
        "count_bias_y": bias_y,
        "final": float(meta["final_acc"]),
        "full_end_raw": full_raw[-1],
        "count_bias_end_raw": bias_raw[-1],
        "full_steps": len(full_x),
        "count_bias_steps": len(bias_x),
        "think_steps_full": len(full_x_think),
        "think_steps_bias": len(bias_x_think),
        "post_think": True,
        "align": "think_then_post",
    }
    OUT_TOKEN.write_text(json.dumps(out))
    print(
        OUT_TOKEN,
        f"full {full_y[0]:.2f}->{full_y[-1]:.2f} x={full_x[-1]:.1f} "
        f"(think_end x={full_x_think[-1]:.1f} y={full_y[think_n-1]:.2f})",
        f"bias {bias_y[0]:.2f}->{bias_y[-1]:.2f} x={bias_x[-1]:.1f} "
        f"(think_end x={bias_x_think[-1]:.1f} y={bias_y[len(bias_x_think)-1]:.2f})",
        f"raw_end full={full_raw[-1]:.2f} bias={bias_raw[-1]:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
