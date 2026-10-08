#!/usr/bin/env python3
"""Full-CoT vs count-bias post-window accuracy, R1-7B seed 42, rho 0.98."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if "PLWS_ROOT" in __import__("os").environ:
    ROOT = Path(__import__("os").environ["PLWS_ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tmp" / "PUMA" / "puma"))

from plws.grading import grade_many  # noqa: E402

WIN = 15
SERIES = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b/fullcot_seed42_series.json"
PROBES = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b"
JOBS = ROOT / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b"
OUT = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_window_acc.png"
CACHE = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_series.json"
PANELS = (
    ("GPQA-Diamond", "gpqa-diamond"),
    ("OlympiadBench", "olympiadbench"),
    ("AIME25", "aime25"),
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trail(values: list[float], win: int) -> list[float]:
    out = []
    for index in range(len(values)):
        start = max(0, index - win + 1)
        chunk = values[start : index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def curve(seqs: list[list[bool]]) -> list[float]:
    n = len(seqs)
    tmax = max(len(row) for row in seqs)
    acc = []
    for step in range(tmax):
        correct = sum(row[step] if step < len(row) else row[-1] for row in seqs)
        acc.append(100.0 * correct / n)
    return acc


def probe_flags(dataset: str, jobs: list[dict]) -> dict[str, dict[int, bool]]:
    found: dict[str, dict[int, bool]] = defaultdict(dict)
    directory = PROBES / dataset / "seed_42"
    for path in sorted(directory.glob("trials*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok" or row.get("gold_error"):
                raise SystemExit(f"bad probe {dataset} {row.get('uid')}")
            found[row["uid"]][int(row["rel_step"])] = bool(row["gold_ok"])
    missing = [job["uid"] for job in jobs if job["uid"] not in found]
    if missing:
        raise SystemExit(f"{dataset} missing {len(missing)} questions")
    for job in jobs:
        keys = sorted(found[job["uid"]])
        if keys != list(range(1, keys[-1] + 1)):
            raise SystemExit(f"{dataset} gap {job['uid']}")
    return found


def window_end_ok(dataset: str, jobs: list[dict], trials_path: str) -> list[bool]:
    wanted = {int(job["question_idx"]): int(job["left_step"]) for job in jobs}
    answers: dict[int, str] = {}
    for row in json.loads((ROOT / trials_path).read_text()):
        index = int(row["question_idx"])
        if wanted.get(index) == int(row["stopped_len"]):
            answers[index] = row.get("final_answer")
    missing = [index for index in wanted if index not in answers]
    if missing:
        raise SystemExit(f"{dataset} missing window-end trials: {len(missing)}")
    pairs = [(answers[int(job["question_idx"])], job["gt"]) for job in jobs]
    graded = grade_many(pairs, workers=8, chunksize=16)
    flags = []
    for job, (ok, error) in zip(jobs, graded):
        if error:
            raise SystemExit(f"{dataset} t0 {job['uid']}: {error}")
        flags.append(bool(ok))
    return flags


def main() -> None:
    full = json.loads(SERIES.read_text())["datasets"]
    cache = {}
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))
    for ax, (title, dataset) in zip(axes, PANELS):
        jobs = load_jsonl(JOBS / dataset / "seed_42/jobs/firstwin.jsonl")
        probes = probe_flags(dataset, jobs)
        info = full[dataset]
        t0 = window_end_ok(dataset, jobs, info["trials"])
        if sum(t0) != info["t0_correct"]:
            raise SystemExit(
                f"{dataset} t0 {sum(t0)} != series {info['t0_correct']}"
            )
        seqs = []
        for job, base in zip(jobs, t0):
            steps = probes[job["uid"]]
            seqs.append([base] + [steps[step] for step in range(1, max(steps) + 1)])
        cb = curve(seqs)
        fy = trail(info["acc"], WIN)
        cy = trail(cb, WIN)
        final = info["final_acc"]
        ax.plot(range(len(fy)), fy, color="#1f4e79", lw=1.8, label="Full-CoT")
        ax.plot(range(len(cy)), cy, color="#c45911", lw=1.8, label="Count-bias")
        ax.axhline(
            final,
            color="#1f4e79",
            ls="--",
            lw=1.2,
            label=f"Full-CoT final {final:.1f}%",
        )
        ax.set_title(f"{title}  seed 42   n={info['n']}")
        ax.set_xlabel("steps after window end")
        ax.set_ylabel("Acc (%)")
        ax.set_xlim(-0.5, max(len(fy), len(cy)) - 0.5)
        lo = min(min(fy), min(cy), final)
        hi = max(max(fy), max(cy), final)
        pad = max(2.0, 0.08 * (hi - lo + 1))
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        cache[dataset] = {
            "n": info["n"],
            "final": final,
            "full": info["acc"],
            "count_bias": cb,
        }
        print(
            f"{dataset} n={info['n']} full {info['acc'][0]:.1f}->{info['acc'][-1]:.1f} "
            f"t={len(info['acc']) - 1} cb {cb[0]:.1f}->{cb[-1]:.1f} t={len(cb) - 1} "
            f"start {fy[0]:.2f} {cy[0]:.2f}",
            flush=True,
        )
    fig.suptitle(
        "R1-7B seed 42, count-bias ρ=0.98, 15-step trailing average, fixed denominator",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    CACHE.write_text(json.dumps(cache))
    print(OUT)


if __name__ == "__main__":
    main()
