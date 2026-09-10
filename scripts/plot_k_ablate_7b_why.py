#!/usr/bin/env python3
"""Two-panel figure: why the PLWS–Full-CoT gap is smallest at k=4."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import load_json  # noqa: E402
from report_k_ablate_7b import (  # noqa: E402
    DATASETS,
    KS,
    MODEL,
    SEEDS,
    load_k_cell_scores,
    load_k_jobs,
)

OUT_PNG = ROOT / "tables" / "firstwin_wait" / "k_ablate_7b_why.png"
OUT_JSON = ROOT / "results" / "reports" / "k_ablate_7b_why_figure.json"


def load_rows() -> list[dict]:
    paths = PLWSPaths.discover(ROOT)
    rows: list[dict] = []
    for dataset, _dszh, _n in DATASETS:
        for seed in SEEDS:
            official = {
                int(row["question_idx"]): row
                for row in load_json(paths.puma_statistics_path(MODEL, dataset, seed))
            }
            scores = {
                k: load_k_cell_scores(paths, dataset, seed, k) for k in KS
            }
            jobs = {k: load_k_jobs(paths, dataset, seed, k) for k in KS}
            for qid, info in official.items():
                item = {"full": bool(info.get("original_correct"))}
                for k in KS:
                    rec = scores[k].get((dataset, int(qid)))
                    job = jobs[k].get(int(qid))
                    if rec is None:
                        item[f"w{k}"] = False
                        continue
                    item[f"w{k}"] = True
                    item[f"ok{k}"] = bool(rec.get("new_gold_ok"))
                    item[f"lok{k}"] = bool(rec.get("left_ok"))
                    left = rec.get("left_step")
                    if left is None and job is not None:
                        left = job.get("left_step")
                    item[f"left{k}"] = int(left) if left is not None else None
                rows.append(item)
    return rows


def stats_for(group: list[dict], k: int) -> dict:
    c = Counter()
    steps = []
    for row in group:
        left, full, plws = row[f"lok{k}"], row["full"], row[f"ok{k}"]
        if row.get(f"left{k}") is not None:
            steps.append(row[f"left{k}"])
        if plws and not full:
            c["we_better"] += 1
        if (not plws) and full:
            c["full_better"] += 1
        if (not left) and (not full) and plws:
            c["we_only"] += 1
        if (not left) and full and (not plws):
            c["missed_full_recovery"] += 1
        if left and (not full) and plws:
            c["kept_good_lock"] += 1
        if left and full and (not plws):
            c["broke_good_lock"] += 1
        if left:
            c["lock_ok"] += 1
    n = len(group)
    return {
        "n": n,
        "acc": 100.0 * sum(1 for row in group if row[f"ok{k}"]) / n,
        "full_acc": 100.0 * sum(1 for row in group if row["full"]) / n,
        "we_better": c["we_better"],
        "full_better": c["full_better"],
        "delta": c["we_better"] - c["full_better"],
        "missed_full_recovery": c["missed_full_recovery"],
        "broke_good_lock": c["broke_good_lock"],
        "we_only": c["we_only"],
        "kept_good_lock": c["kept_good_lock"],
        "lock_ok_rate": 100.0 * c["lock_ok"] / n,
        "median_lock_step": median(steps) if steps else None,
    }


def draw(by_k: dict) -> None:
    ks = list(KS)
    we = [by_k[k]["we_better"] for k in ks]
    full = [by_k[k]["full_better"] for k in ks]
    missed = [by_k[k]["missed_full_recovery"] for k in ks]
    broke = [by_k[k]["broke_good_lock"] for k in ks]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.15), constrained_layout=True)
    x = np.arange(len(ks))

    ax = axes[0]
    ax.plot(x, we, "-o", color="#2A6F97", lw=1.8, ms=6, label="PLWS right, Full-CoT wrong")
    ax.plot(x, full, "-s", color="#C44536", lw=1.8, ms=6, label="PLWS wrong, Full-CoT right")
    for i, k in enumerate(ks):
        if k == 4:
            ax.axvline(i, color="#9AA0A6", ls="--", lw=0.8, zorder=0)
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_xlabel("First-window width $k$")
    ax.set_ylabel("Questions (same 4802)")
    ax.set_ylim(150, 300)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Disagreements with Full-CoT", fontsize=10, pad=6)

    ax = axes[1]
    w = 0.62
    ax.bar(x, missed, w, color="#E09F3E", label="Lock still wrong: Full-CoT later fixes it, we do not")
    ax.bar(
        x,
        broke,
        w,
        bottom=missed,
        color="#9B2226",
        label="Lock already right: we overwrite it, Full-CoT keeps it",
    )
    ax.plot(x, full, "o", color="#1D1D1D", ms=4, zorder=3)
    for i, k in enumerate(ks):
        if k == 4:
            ax.axvline(i, color="#9AA0A6", ls="--", lw=0.8, zorder=0)
        ax.text(
            i,
            full[i] + 8,
            str(full[i]),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1D1D1D",
        )
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_xlabel("First-window width $k$")
    ax.set_ylabel("PLWS wrong, Full-CoT right")
    ax.set_ylim(0, 320)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("Where those losses come from", fontsize=10, pad=6)

    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    inter = [row for row in rows if all(row.get(f"w{k}") for k in KS)]
    if len(inter) != 4802:
        raise RuntimeError(f"intersection n={len(inter)}, expected 4802")
    by_k = {k: stats_for(inter, k) for k in KS}
    draw(by_k)
    OUT_JSON.write_text(
        json.dumps(
            {
                "n": 4802,
                "full_acc": by_k[4]["full_acc"],
                "by_k": {str(k): by_k[k] for k in KS},
                "figure": str(OUT_PNG.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_JSON}")
    for k in KS:
        s = by_k[k]
        print(
            f"k={k} acc={s['acc']:.2f} we={s['we_better']} full={s['full_better']} "
            f"d={s['delta']:+d} miss={s['missed_full_recovery']} "
            f"broke={s['broke_good_lock']} lock_ok={s['lock_ok_rate']:.1f}% "
            f"med_step={s['median_lock_step']}"
        )


if __name__ == "__main__":
    main()
