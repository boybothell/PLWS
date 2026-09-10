#!/usr/bin/env python3
"""AIME 多种子：压 Wait vs 官方 Full-CoT。按 2024 / 2025 分开，不按种子拆。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402

ROOT = AE / "results/leftover_suppress_toend"
JOBS = AE / "results/leftover_jump"
TABLE = AE / "tables/firstwin_wait/leftover_suppress_aime_seeds.md"
MODELS = (("r1_7b", "7B"), ("nemotron_8b", "8B"), ("r1_14b", "14B"))
SEEDS = (42, 0, 1, 123)
AIME = ("aime24", "aime25")


def load_jobs(tag: str, seed: int) -> dict[str, dict]:
    jobs: dict[str, dict] = {}
    root = JOBS / f"{tag}_s{seed}"
    for name, kind in (("jobs.jsonl", "low"), ("jobs_high.jsonl", "high"), ("jobs_mix.jsonl", "mix")):
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            job = json.loads(line)
            if job.get("dataset") not in AIME:
                continue
            job["kind"] = job.get("kind") or kind
            jobs[job["uid"]] = job
    return jobs


def load_scores(tag: str, seed: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for suf in ("", "_high", "_mix"):
        folder = ROOT / f"{tag}_s{seed}_suppress{suf}"
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("scores_shard*.jsonl")):
            for line in path.open():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") != "ok" or not rec.get("generated_text"):
                    continue
                out[rec["uid"]] = rec
    return out


def official(tag: str, ds: str, seed: int) -> dict[int, dict]:
    path = dd.puma_stat_path(tag, ds, seed)
    if not path.is_file():
        return {}
    return {int(row["question_idx"]): row for row in dd.load_json(path)}


def tot(rec: dict) -> float:
    return float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)


def agg(xs: list[dict]) -> dict | None:
    if not xs:
        return None
    n = len(xs)
    denom = sum(x["ot"] for x in xs)
    return {
        "n": n,
        "acc": 100.0 * sum(x["ok"] for x in xs) / n,
        "oacc": 100.0 * sum(x["orig"] for x in xs) / n,
        "tok": sum(x["tot"] for x in xs) / n,
        "ot": sum(x["ot"] for x in xs) / n,
        "r": (sum(x["tot"] for x in xs) / denom) if denom else float("nan"),
    }


def line(name: str, s: dict | None) -> str | None:
    if not s:
        return None
    return (
        f"| {name} | {s['n']} | {s['acc']:.1f}% | {s['oacc']:.1f}% | "
        f"{s['tok']:.0f} | {s['ot']:.0f} | {s['r']:.2f} |"
    )


def main() -> None:
    window: dict[tuple[str, str], list[dict]] = {(tag, ds): [] for tag, _ in MODELS for ds in AIME}
    full: dict[tuple[str, str], list[dict]] = {(tag, ds): [] for tag, _ in MODELS for ds in AIME}
    notes = []
    for tag, zh in MODELS:
        for seed in SEEDS:
            jobs = load_jobs(tag, seed)
            scores = load_scores(tag, seed)
            offs = {ds: official(tag, ds, seed) for ds in AIME}
            if not jobs:
                continue
            n_ok = sum(1 for uid in jobs if uid in scores)
            if n_ok < len(jobs):
                notes.append(f"{zh} s{seed} {n_ok}/{len(jobs)}")
            by: dict[tuple[str, int], dict] = {}
            for uid, job in jobs.items():
                rec = scores.get(uid)
                if not rec:
                    continue
                ds = job["dataset"]
                qi = int(job["question_idx"])
                info = offs.get(ds, {}).get(qi, {})
                orig = bool(
                    info.get("original_correct")
                    if "original_correct" in info
                    else job.get("orig_ok") or job.get("host_ok")
                )
                ot = float(info.get("original_tokens") or job.get("original_tokens") or 0)
                row = {"ok": bool(rec.get("new_gold_ok")), "orig": orig, "tot": tot(rec), "ot": ot}
                window[(tag, ds)].append(row)
                by[(ds, qi)] = row
            for ds in AIME:
                if not offs[ds]:
                    continue
                for qi, info in offs[ds].items():
                    orig = bool(info.get("original_correct"))
                    ot = float(info.get("original_tokens") or 0)
                    hit = by.get((ds, int(qi)))
                    if hit:
                        full[(tag, ds)].append({**hit, "orig": orig, "ot": ot})
                    else:
                        full[(tag, ds)].append({"ok": orig, "orig": orig, "tot": ot, "ot": ot})

    header = "| 切片 | n | 压 Wait Acc | 官方 Acc | 压 Wait 总 tok | 官方总 tok | 总/官 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    dszh = {"aime24": "A24", "aime25": "A25"}
    lines = [
        "# AIME 多种子：压 Wait vs 官方 Full-CoT",
        "",
        "种子 42/0/1/123 并起来，AIME 2024 / 2025 分开。不按种子拆。",
        "压 Wait = 第一扇四步同答窗后续写，禁 Wait / Alternatively / Hmm。",
        "有窗 = 有第一扇窗的题。整集 = 有窗用压 Wait，无窗用官方 CoT。总 tok = 思考 + 终答。",
        "",
    ]
    if notes:
        lines += ["未齐：" + "；".join(notes), ""]
    lines += ["## 有窗", "", header, sep]
    all_w: list[dict] = []
    for tag, zh in MODELS:
        for ds in AIME:
            xs = window[(tag, ds)]
            text = line(f"{zh} {dszh[ds]}", agg(xs))
            if text:
                lines.append(text)
                all_w.extend(xs)
    text = line("三模型", agg(all_w))
    if text:
        lines.append(text)
    lines += ["", "## 整集", "", header, sep]
    all_f: list[dict] = []
    for tag, zh in MODELS:
        for ds in AIME:
            xs = full[(tag, ds)]
            text = line(f"{zh} {dszh[ds]}", agg(xs))
            if text:
                lines.append(text)
                all_f.extend(xs)
    text = line("三模型", agg(all_f))
    if text:
        lines.append(text)
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
