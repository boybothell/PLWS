#!/usr/bin/env python3
"""Compare finished wrap-sample CORE+But/So/Therefore scores against CORE.

Only model × dataset cells whose wrap jobs are all scored. The sample is
flip-heavy selected questions, not the full official set.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from plws.matrix import reusable_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_fullcot_puma_plws import DATASETS  # noqa: E402

JOBS_DIR = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
WRAP_ROOT = (
    ROOT / "results" / "experiments" / "lexicon_ablation" / "wrap_sample"
)
OUT = ROOT / "results" / "reports" / "wrap_sample_vs_core.json"
TABLE = ROOT / "tables" / "firstwin_wait" / "wrap_sample_vs_core.md"
LEXICON = "core_plus_but_so_therefore"
MODELS = (
    ("qwen3_4b", "4B"),
    ("nemotron_8b", "Nemotron"),
)
DSZH = {name: zh for name, zh, _n in DATASETS}
TZ = ZoneInfo("Asia/Shanghai")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def wrap_scores(model: str) -> dict[str, dict[str, Any]]:
    folder = WRAP_ROOT / model / LEXICON
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for rec in load_jsonl(path):
            if not reusable_score(rec):
                continue
            uid = str(rec.get("uid") or "")
            if uid:
                out[uid] = rec
    return out


def delivery(think: Any, ans: Any) -> float | None:
    if think is None or ans is None:
        return None
    return float(think) + float(ans)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pct(part: int, whole: int) -> float | None:
    return 100.0 * part / whole if whole else None


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(pairs)
    full_ok = sum(1 for row in pairs if row["full_ok"])
    core_ok = sum(1 for row in pairs if row["core_ok"])
    wrap_ok = sum(1 for row in pairs if row["wrap_ok"])
    core_up = sum(1 for row in pairs if row["core_ok"] and not row["full_ok"])
    core_down = sum(1 for row in pairs if (not row["core_ok"]) and row["full_ok"])
    wrap_up = sum(1 for row in pairs if row["wrap_ok"] and not row["full_ok"])
    wrap_down = sum(1 for row in pairs if (not row["wrap_ok"]) and row["full_ok"])
    wrap_beat_core = sum(
        1 for row in pairs if row["wrap_ok"] and not row["core_ok"]
    )
    wrap_lose_core = sum(
        1 for row in pairs if (not row["wrap_ok"]) and row["core_ok"]
    )
    same_ok = sum(1 for row in pairs if row["wrap_ok"] == row["core_ok"])
    leftover_cut = sum(
        1
        for row in pairs
        if row["wrap_cont"] is not None
        and row["core_cont"] is not None
        and row["wrap_cont"] + 8 < row["core_cont"]
    )
    leftover_grew = sum(
        1
        for row in pairs
        if row["wrap_cont"] is not None
        and row["core_cont"] is not None
        and row["wrap_cont"] > row["core_cont"] + 8
    )
    return {
        "n": n,
        "n_questions": len({(row["dataset"], row["qid"]) for row in pairs}),
        "full_acc": pct(full_ok, n),
        "core_acc": pct(core_ok, n),
        "wrap_acc": pct(wrap_ok, n),
        "d_acc_wrap_minus_core": (
            None
            if n == 0
            else (wrap_ok - core_ok) * 100.0 / n
        ),
        "core_tok": mean([row["core_tok"] for row in pairs if row["core_tok"] is not None]),
        "wrap_tok": mean([row["wrap_tok"] for row in pairs if row["wrap_tok"] is not None]),
        "d_tok_wrap_minus_core": mean(
            [
                row["wrap_tok"] - row["core_tok"]
                for row in pairs
                if row["wrap_tok"] is not None and row["core_tok"] is not None
            ]
        ),
        "core_cont": mean(
            [row["core_cont"] for row in pairs if row["core_cont"] is not None]
        ),
        "wrap_cont": mean(
            [row["wrap_cont"] for row in pairs if row["wrap_cont"] is not None]
        ),
        "d_cont_wrap_minus_core": mean(
            [
                row["wrap_cont"] - row["core_cont"]
                for row in pairs
                if row["wrap_cont"] is not None and row["core_cont"] is not None
            ]
        ),
        "full_tok": mean(
            [row["full_tok"] for row in pairs if row["full_tok"] is not None]
        ),
        "same_ok": same_ok,
        "wrap_beat_core": wrap_beat_core,
        "wrap_lose_core": wrap_lose_core,
        "core_flip_up": core_up,
        "core_flip_down": core_down,
        "wrap_flip_up": wrap_up,
        "wrap_flip_down": wrap_down,
        "leftover_cut": leftover_cut,
        "leftover_grew": leftover_grew,
        "by_reason": {
            reason: summarize_leaf([row for row in pairs if row["reason"] == reason])
            for reason in (
                "flip",
                "keep_right_long",
                "keep_right_short",
                "keep_wrong",
            )
            if any(row["reason"] == reason for row in pairs)
        },
    }


def summarize_leaf(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    core_ok = sum(1 for row in pairs if row["core_ok"])
    wrap_ok = sum(1 for row in pairs if row["wrap_ok"])
    full_ok = sum(1 for row in pairs if row["full_ok"])
    return {
        "n": n,
        "n_questions": len({(row["dataset"], row["qid"]) for row in pairs}),
        "full_acc": pct(full_ok, n),
        "core_acc": pct(core_ok, n),
        "wrap_acc": pct(wrap_ok, n),
        "d_acc_wrap_minus_core": (wrap_ok - core_ok) * 100.0 / n,
        "core_tok": mean([row["core_tok"] for row in pairs if row["core_tok"] is not None]),
        "wrap_tok": mean([row["wrap_tok"] for row in pairs if row["wrap_tok"] is not None]),
        "d_tok_wrap_minus_core": mean(
            [
                row["wrap_tok"] - row["core_tok"]
                for row in pairs
                if row["wrap_tok"] is not None and row["core_tok"] is not None
            ]
        ),
        "core_cont": mean(
            [row["core_cont"] for row in pairs if row["core_cont"] is not None]
        ),
        "wrap_cont": mean(
            [row["wrap_cont"] for row in pairs if row["wrap_cont"] is not None]
        ),
        "wrap_beat_core": sum(
            1 for row in pairs if row["wrap_ok"] and not row["core_ok"]
        ),
        "wrap_lose_core": sum(
            1 for row in pairs if (not row["wrap_ok"]) and row["core_ok"]
        ),
    }


def fmt_acc(x: float | None) -> str:
    return "" if x is None else f"{x:.2f}%"


def fmt_tok(x: float | None) -> str:
    return "" if x is None else f"{x:.0f}"


def fmt_dacc(x: float | None) -> str:
    return "" if x is None else f"{x:+.2f}"


def fmt_dtok(x: float | None) -> str:
    return "" if x is None else f"{x:+.0f}"


def cell_line(model_zh: str, dszh: str, rec: dict[str, Any]) -> str:
    return (
        f"| {model_zh} | {dszh} | {rec['n']} | {rec['n_questions']} | "
        f"{fmt_acc(rec['full_acc'])} | {fmt_acc(rec['core_acc'])} | "
        f"{fmt_acc(rec['wrap_acc'])} | {fmt_dacc(rec['d_acc_wrap_minus_core'])} | "
        f"{fmt_tok(rec['core_tok'])} | {fmt_tok(rec['wrap_tok'])} | "
        f"{fmt_dtok(rec['d_tok_wrap_minus_core'])} | "
        f"{fmt_tok(rec['core_cont'])} | {fmt_tok(rec['wrap_cont'])} | "
        f"{fmt_dtok(rec['d_cont_wrap_minus_core'])} | "
        f"{rec['wrap_beat_core']} | {rec['wrap_lose_core']} |"
    )


def main() -> None:
    cells: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for model, zh in MODELS:
        jobs = load_jsonl(JOBS_DIR / f"wrap_sample_{model}.jsonl")
        scores = wrap_scores(model)
        by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for job in jobs:
            by_ds[str(job["dataset"])].append(job)
        for dataset, ds_jobs in sorted(by_ds.items()):
            pairs: list[dict[str, Any]] = []
            missing = 0
            for job in ds_jobs:
                rec = scores.get(str(job["uid"]))
                if rec is None:
                    missing += 1
                    continue
                pairs.append(
                    {
                        "dataset": dataset,
                        "qid": int(job["question_idx"]),
                        "seed": int(job["seed"]),
                        "reason": job.get("sample_reason") or "",
                        "full_ok": bool(job.get("orig_ok", job.get("host_ok"))),
                        "core_ok": bool(job.get("core_gold_ok")),
                        "wrap_ok": bool(rec.get("new_gold_ok")),
                        "core_tok": delivery(
                            job.get("core_n_think_tok"), job.get("core_n_ans_tok")
                        ),
                        "wrap_tok": delivery(
                            rec.get("n_think_tok"), rec.get("n_ans_tok")
                        ),
                        "core_cont": (
                            None
                            if job.get("core_n_cont_tok") is None
                            else float(job["core_n_cont_tok"])
                        ),
                        "wrap_cont": (
                            None
                            if rec.get("n_cont_tok") is None
                            else float(rec["n_cont_tok"])
                        ),
                        "full_tok": (
                            float(rec["original_tokens"])
                            if isinstance(rec.get("original_tokens"), (int, float))
                            else None
                        ),
                    }
                )
            progress = {
                "model": model,
                "model_zh": zh,
                "dataset": dataset,
                "dataset_zh": DSZH.get(dataset, dataset),
                "jobs": len(ds_jobs),
                "scored": len(pairs),
                "missing": missing,
                "complete": missing == 0 and len(pairs) == len(ds_jobs),
            }
            if not progress["complete"]:
                pending.append(progress)
                continue
            summary = summarize(pairs)
            summary.update(progress)
            cells.append(summary)

    generated = datetime.now(TZ).isoformat(timespec="seconds")
    payload = {
        "generated_at": generated,
        "script": "scripts/analyze_wrap_sample_vs_core.py",
        "lexicon": LEXICON,
        "unit": "wrap-sample jobs that already finished a whole dataset",
        "token": "delivery = n_think_tok + n_ans_tok; leftover = n_cont_tok; trial not added (same prefix)",
        "note": "Selected flip-heavy sample, not official full-set Acc/token.",
        "cells": cells,
        "pending": pending,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# wrap-sample：CORE vs CORE+But/So/Therefore",
        "",
        "只收已经整集跑完的 wrap-sample 题。样本是翻转题加对照，不是官方全量。",
        "Token 是窗后交付（思考 + 终答）；试答前缀相同，两边一样，未再加。",
        f"词表 `{LEXICON}`。生成 {generated}。",
        "",
        "| 模型 | 集 | jobs | 题 | Full-CoT Acc | CORE Acc | +But/So/Therefore Acc | ΔAcc | CORE tok | +wrap tok | Δtok | CORE leftover | +wrap leftover | Δleftover | wrap 纠 CORE | wrap 损 CORE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in cells:
        lines.append(cell_line(rec["model_zh"], rec["dataset_zh"], rec))
    if cells:
        lines.append("")
        lines.append("按抽样原因：")
        lines.append("")
        lines.append(
            "| 模型 | 集 | 原因 | jobs | CORE Acc | +wrap Acc | ΔAcc | CORE tok | +wrap tok | Δtok |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for rec in cells:
            for reason, leaf in rec["by_reason"].items():
                lines.append(
                    f"| {rec['model_zh']} | {rec['dataset_zh']} | {reason} | "
                    f"{leaf['n']} | {fmt_acc(leaf['core_acc'])} | "
                    f"{fmt_acc(leaf['wrap_acc'])} | "
                    f"{fmt_dacc(leaf['d_acc_wrap_minus_core'])} | "
                    f"{fmt_tok(leaf['core_tok'])} | {fmt_tok(leaf['wrap_tok'])} | "
                    f"{fmt_dtok(leaf['d_tok_wrap_minus_core'])} |"
                )
    if pending:
        lines.extend(
            [
                "",
                "未齐（不进表）：",
                "",
            ]
        )
        for rec in pending:
            lines.append(
                f"- {rec['model_zh']} {rec['dataset_zh']}: "
                f"{rec['scored']}/{rec['jobs']}"
            )
    lines.append("")
    TABLE.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    print(f"wrote {TABLE}")
    for rec in cells:
        print(
            f"{rec['model_zh']} {rec['dataset_zh']}: n={rec['n']} "
            f"acc {rec['core_acc']:.2f}->{rec['wrap_acc']:.2f} "
            f"tok {rec['core_tok']:.0f}->{rec['wrap_tok']:.0f}"
        )
    for rec in pending:
        print(f"pending {rec['model_zh']} {rec['dataset_zh']} {rec['scored']}/{rec['jobs']}")


if __name__ == "__main__":
    main()
