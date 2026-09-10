#!/usr/bin/env python3
"""AIME seed-42：7B/8B/14B 扩词表 vs 核三词 vs 官方。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402

TABLE = AE / "tables/firstwin_wait/fw_lex_aime.md"
MODELS = (("r1_7b", "7B"), ("nemotron_8b", "8B"), ("r1_14b", "14B"))
AIME = ("aime24", "aime25")
ZH = {"aime24": "A24", "aime25": "A25"}
SEED = 42


def load_jobs(tag: str) -> dict[str, dict]:
    jobs: dict[str, dict] = {}
    root = AE / "results/leftover_jump" / f"{tag}_s{SEED}"
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


def load_scores(folder: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok" or not rec.get("generated_text"):
                continue
            out[rec["uid"]] = rec
    return out


def official(tag: str, ds: str) -> dict[int, dict]:
    path = dd.puma_stat_path(tag, ds, SEED)
    if not path.is_file():
        return {}
    return {int(row["question_idx"]): row for row in dd.load_json(path)}


def tot(rec: dict) -> float:
    return float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)


def rows_of(tag: str, scores: dict[str, dict], jobs: dict[str, dict]) -> list[dict]:
    offs = {ds: official(tag, ds) for ds in AIME}
    out = []
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
        out.append(
            {
                "ds": ds,
                "ok": bool(rec.get("new_gold_ok")),
                "orig": orig,
                "tot": tot(rec),
                "ot": ot,
            }
        )
    return out


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


def line(name: str, safe: dict | None, core: dict | None) -> str | None:
    if not safe or not core:
        return None
    return (
        f"| {name} | {safe['n']} | {safe['acc']:.1f}% | {core['acc']:.1f}% | {safe['oacc']:.1f}% | "
        f"{safe['acc'] - safe['oacc']:+.1f} | {core['acc'] - core['oacc']:+.1f} | "
        f"{safe['tok']:.0f} | {core['tok']:.0f} | {safe['ot']:.0f} | "
        f"{safe['r']:.2f} | {core['r']:.2f} |"
    )


def main() -> None:
    header = (
        "| 切片 | n | 扩词表 Acc | 核三词 Acc | 官方 Acc | 扩Δ | 核Δ | "
        "扩 tok | 核 tok | 官 tok | 扩/官 | 核/官 |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    notes = []
    lines = [
        "# AIME seed-42：扩词表 vs 核三词 vs 官方 Full-CoT",
        "",
        "第一扇四步同答窗后续写。核三词 = Wait / Alternatively / Hmm。",
        "扩词表 = 核三词 + However / Maybe / Perhaps / another way|approach|method / double-check / Hold on / 换一种。",
        "只含有窗 AIME。总 tok = 思考 + 终答。",
        "",
        header,
        sep,
    ]
    for tag, zh in MODELS:
        jobs = load_jobs(tag)
        if tag == "r1_7b":
            safe = load_scores(AE / "results/fw_lex_safe/r1_7b_s42_suppress")
        else:
            safe = load_scores(AE / "results/fw_lex_safe" / f"{tag}_s{SEED}_suppress")
        core: dict[str, dict] = {}
        for suf in ("", "_high", "_mix"):
            core.update(load_scores(AE / "results/leftover_suppress_toend" / f"{tag}_s{SEED}_suppress{suf}"))
        srows = rows_of(tag, safe, jobs)
        crows = rows_of(tag, core, jobs)
        if len(srows) < len(jobs):
            notes.append(f"{zh} 扩 {len(srows)}/{len(jobs)}")
        if len(crows) < len(jobs):
            notes.append(f"{zh} 核 {len(crows)}/{len(jobs)}")
        for ds in AIME:
            text = line(
                f"{zh} {ZH[ds]}",
                agg([x for x in srows if x["ds"] == ds]),
                agg([x for x in crows if x["ds"] == ds]),
            )
            if text:
                lines.append(text)
        text = line(f"{zh} AIME", agg(srows), agg(crows))
        if text:
            lines.append(text)
    if notes:
        lines[5:5] = ["未齐：" + "；".join(notes), ""]
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
