#!/usr/bin/env python3
"""Accuracy vs tokens after the same-answer window, R1-7B seed 42."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if "PLWS_ROOT" in __import__("os").environ:
    ROOT = Path(__import__("os").environ["PLWS_ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "tmp" / "PUMA" / "puma"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_r1_7b_seed42 import continuation_ends  # noqa: E402
from separate_steps import separate_steps  # noqa: E402

MODEL = "/ssd/share/models/DeepSeek-R1-Distill-Qwen-7B"
WIN = 15
CACHE = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_series.json"
SERIES = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b/fullcot_seed42_series.json"
PROBES = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b"
JOBS = ROOT / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b"
SCORES = ROOT / "results/runs/plws/count_bias/rho_0p98/r1_7b"
OUT = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_token_acc.png"
OUT_JSON = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_token_series.json"
PANELS = (("OlympiadBench", "olympiadbench"),)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trail(values: list[float], win: int) -> list[float]:
    out = []
    for index in range(len(values)):
        start = max(0, index - win + 1)
        chunk = values[start : index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def mean_axis(rows: list[list[int]], baseline: float) -> list[float]:
    width = max(len(row) for row in rows)
    xs = []
    n = len(rows)
    for step in range(width):
        total = 0
        for row in rows:
            total += row[step] if step < len(row) else row[-1]
        xs.append(total / n - baseline)
    return xs


def fullcot_tokens(dataset: str, jobs: list[dict], trials_path: str) -> list[list[int]]:
    wanted = {int(job["question_idx"]): int(job["left_step"]) for job in jobs}
    by_question: dict[int, dict[int, int]] = defaultdict(dict)
    print(f"[{dataset}] load full-cot tokens", flush=True)
    for row in json.loads((ROOT / trials_path).read_text()):
        index = int(row["question_idx"])
        left = wanted.get(index)
        if left is None:
            continue
        stopped = int(row["stopped_len"])
        if stopped >= left:
            by_question[index][stopped] = int(row["count_reasoning_tokens"])
    rows = []
    for job in jobs:
        index = int(job["question_idx"])
        left = int(job["left_step"])
        steps = by_question[index]
        post = sorted(step for step in steps if step >= left)
        if not post or post[0] != left or post != list(range(post[0], post[-1] + 1)):
            raise SystemExit(f"{dataset} token gap uid={job['uid']}")
        counts = [steps[step] for step in post]
        if any(later < earlier for earlier, later in zip(counts, counts[1:])):
            raise SystemExit(f"{dataset} token decreased uid={job['uid']}")
        rows.append(counts)
    return rows


def continuation_token_counts(tokenizer, text: str, ends: list[int]) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    counts = []
    cursor = 0
    size = len(offsets)
    for end in ends:
        while cursor < size and offsets[cursor][0] < end:
            cursor += 1
        counts.append(cursor)
    return counts


def count_bias_tokens(
    tokenizer,
    dataset: str,
    jobs: list[dict],
    window_tokens: list[int],
) -> list[list[int]]:
    scores = {}
    for path in sorted((SCORES / dataset / "seed_42").glob("shard_*.jsonl")):
        for row in load_jsonl(path):
            scores[row["uid"]] = row
    probes: dict[str, int] = defaultdict(int)
    for path in sorted((PROBES / dataset / "seed_42").glob("trials*.jsonl")):
        for row in load_jsonl(path):
            probes[row["uid"]] += 1
    rows = []
    checked = 0
    worst = 0
    for job, base in zip(jobs, window_tokens):
        rec = scores[job["uid"]]
        text = rec["generated_text"]
        thought = job["thought"]
        if not text.startswith(thought):
            raise SystemExit(f"{dataset} prefix mismatch {job['uid']}")
        cont = text.split("</think>", 1)[0][len(thought) :]
        chunks = [chunk for chunk in separate_steps(cont) if chunk.strip()]
        ends = continuation_ends(cont, chunks)
        if len(ends) != probes[job["uid"]]:
            raise SystemExit(
                f"{dataset} steps {len(ends)} != probes {probes[job['uid']]} {job['uid']}"
            )
        extra = continuation_token_counts(tokenizer, cont, ends) if ends else []
        if checked < 3 and ends:
            exact = [
                len(tokenizer.encode(cont[:end], add_special_tokens=False))
                for end in (ends[0], ends[len(ends) // 2], ends[-1])
            ]
            approx = [
                continuation_token_counts(tokenizer, cont, [end])[0]
                for end in (ends[0], ends[len(ends) // 2], ends[-1])
            ]
            worst = max(worst, max(abs(a - b) for a, b in zip(exact, approx)))
            checked += 1
        rows.append([base, *[base + count for count in extra]])
    print(f"[{dataset}] tokenizer vs prefix encode max abs diff on sample {worst}", flush=True)
    if worst > 1:
        raise SystemExit(f"{dataset} token locator off by {worst}")
    return rows


def main() -> None:
    print("load tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    acc = json.loads(CACHE.read_text())
    meta = json.loads(SERIES.read_text())["datasets"]
    saved = {}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.2 * len(PANELS), 4.4), squeeze=False)
    for ax, (title, dataset) in zip(axes[0], PANELS):
        jobs = load_jsonl(JOBS / dataset / "seed_42/jobs/firstwin.jsonl")
        full_rows = fullcot_tokens(dataset, jobs, meta[dataset]["trials"])
        window = [row[0] for row in full_rows]
        baseline = sum(window) / len(window)
        bias_rows = count_bias_tokens(tokenizer, dataset, jobs, window)
        full_x = mean_axis(full_rows, baseline)
        bias_x = mean_axis(bias_rows, baseline)
        full_y = trail(acc[dataset]["full"], WIN)
        bias_y = trail(acc[dataset]["count_bias"], WIN)
        if len(full_x) != len(full_y) or len(bias_x) != len(bias_y):
            raise SystemExit(
                f"{dataset} length full {len(full_x)}/{len(full_y)} "
                f"bias {len(bias_x)}/{len(bias_y)}"
            )
        if abs(full_x[0]) > 1e-6 or abs(bias_x[0]) > 1e-6:
            raise SystemExit(f"{dataset} x0 not zero {full_x[0]} {bias_x[0]}")
        final = acc[dataset]["final"]
        ax.plot(full_x, full_y, color="#1f4e79", lw=1.8, label="Full-CoT")
        ax.plot(bias_x, bias_y, color="#c45911", lw=1.8, label="Count-bias")
        ax.axhline(
            final,
            color="#1f4e79",
            ls="--",
            lw=1.2,
            label=f"Full-CoT final {final:.1f}%",
        )
        ax.set_title(title)
        ax.set_xlabel("Avg. reasoning tokens after window")
        ax.set_ylabel("Acc (%)")
        ax.set_xlim(left=0)
        lo = min(min(full_y), min(bias_y), final)
        hi = max(max(full_y), max(bias_y), final)
        pad = max(2.0, 0.08 * (hi - lo + 1))
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        saved[dataset] = {
            "n": acc[dataset]["n"],
            "baseline_tokens": baseline,
            "full_x": full_x,
            "count_bias_x": bias_x,
            "full_y": full_y,
            "count_bias_y": bias_y,
            "final": final,
        }
        print(
            f"{dataset} x0={baseline:.1f} "
            f"full x {full_x[-1]:.0f} y {full_y[0]:.1f}->{full_y[-1]:.1f} "
            f"cb x {bias_x[-1]:.0f} y {bias_y[0]:.1f}->{bias_y[-1]:.1f}",
            flush=True,
        )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    OUT_JSON.write_text(json.dumps(saved))
    print(OUT)


if __name__ == "__main__":
    main()
