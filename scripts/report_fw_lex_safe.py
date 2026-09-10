#!/usr/bin/env python3
"""第一扇窗扩词表压词 vs 官方 Full-CoT。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402

ROOT = AE / "results/fw_lex_safe"
JOBS = AE / "results/leftover_jump"
TABLE = AE / "tables/firstwin_wait/fw_lex_safe.md"
MODELS = (("r1_7b", "7B"), ("nemotron_8b", "8B"), ("r1_14b", "14B"))
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
DS = (
    ("", "all"),
    ("math-500", "MATH"),
    ("olympiadbench", "oly"),
    ("gpqa-diamond", "GPQA"),
    ("aime24", "A24"),
    ("aime25", "A25"),
)
SEED = 42
KEEP = (
    "status",
    "uid",
    "dataset",
    "question_idx",
    "kind",
    "new_gold_ok",
    "n_think_tok",
    "n_ans_tok",
    "generated_text",
)


def load_jobs(tag: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    folder = JOBS / f"{tag}_s{SEED}"
    for name, kind in (("jobs.jsonl", "low"), ("jobs_high.jsonl", "high"), ("jobs_mix.jsonl", "mix")):
        path = folder / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            job = json.loads(line)
            job["kind"] = job.get("kind") or kind
            out[job["uid"]] = job
    return out


def load_scores(folder: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("status") != "ok" or not raw.get("generated_text"):
                    continue
                out[raw["uid"]] = {k: raw.get(k) for k in KEEP}
    return out


def official_map(tag: str, dataset: str) -> dict[int, dict]:
    path = dd.puma_stat_path(tag, dataset, SEED)
    if not path.is_file():
        return {}
    return {int(row["question_idx"]): row for row in dd.load_json(path)}


def tot(rec: dict) -> float:
    return float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)


def line(xs: list[dict], model: str, dszh: str) -> str | None:
    if not xs:
        return None
    n = len(xs)
    denom = sum(x["ot"] for x in xs)
    return (
        f"| {model} | {dszh} | {n} | "
        f"{100.0 * sum(x['ok'] for x in xs) / n:.1f}% | "
        f"{100.0 * sum(x['orig'] for x in xs) / n:.1f}% | "
        f"{sum(x['tot'] for x in xs) / n:.0f} | "
        f"{sum(x['ot'] for x in xs) / n:.0f} | "
        f"{(sum(x['tot'] for x in xs) / denom) if denom else float('nan'):.2f} |"
    )


def section(title: str, by_model: dict[str, list[dict]]) -> list[str]:
    out = [
        f"## {title}",
        "",
        "| 模型 | 集 | n | 扩词表 Acc | 官方 Acc | 扩词表总 tok | 官方总 tok | 总/官 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, zh in MODELS:
        rows = by_model.get(tag, [])
        if not rows:
            continue
        for ds, dszh in DS:
            xs = [x for x in rows if not ds or x["ds"] == ds]
            text = line(xs, zh, dszh)
            if text:
                out.append(text)
    out.append("")
    return out


def stitch(tag: str, window: list[dict]) -> list[dict]:
    official = {ds: official_map(tag, ds) for ds in DATASETS}
    by = {(x["ds"], x["qi"]): x for x in window}
    out = []
    for ds in DATASETS:
        for qi, info in official[ds].items():
            orig = bool(info.get("original_correct"))
            ot = float(info.get("original_tokens") or 0)
            hit = by.get((ds, int(qi)))
            if hit:
                out.append({**hit, "orig": orig, "ot": ot})
            else:
                out.append({"ds": ds, "qi": int(qi), "ok": orig, "orig": orig, "tot": ot, "ot": ot})
    return out


def main() -> None:
    window: dict[str, list[dict]] = {}
    full: dict[str, list[dict]] = {}
    notes = []
    for tag, zh in MODELS:
        jobs = load_jobs(tag)
        scores = load_scores(ROOT / f"{tag}_s{SEED}_suppress")
        official = {ds: official_map(tag, ds) for ds in DATASETS}
        rows = []
        for uid, job in jobs.items():
            rec = scores.get(uid)
            if not rec:
                continue
            info = official.get(job["dataset"], {}).get(int(job["question_idx"]), {})
            rows.append(
                {
                    "ds": job["dataset"],
                    "qi": int(job["question_idx"]),
                    "ok": bool(rec.get("new_gold_ok")),
                    "orig": bool(info.get("original_correct") if "original_correct" in info else job.get("orig_ok") or job.get("host_ok")),
                    "tot": tot(rec),
                    "ot": float(info.get("original_tokens") or job.get("original_tokens") or 0),
                }
            )
        if not rows:
            notes.append(f"{zh} 未齐")
            continue
        if len(rows) < len(jobs):
            notes.append(f"{zh} {len(rows)}/{len(jobs)}")
        window[tag] = rows
        full[tag] = stitch(tag, rows)
    lines = [
        "# 第一扇四步同答窗：扩词表 vs 官方 Full-CoT",
        "",
        "seed-42。第一扇四步同答窗后续写，思考阶段压扩词表。",
        "扩词表 = Wait / Alternatively / Hmm + However / Maybe / Perhaps / another way|approach|method / double-check / Hold on / 换一种。",
        "有窗 = 有第一扇窗的题。整集 = 有窗用扩词表，无窗用官方 CoT。总 tok = 思考 + 终答。",
        "",
    ]
    if notes:
        lines += ["未齐：" + "；".join(notes), ""]
    lines += section("有窗", window)
    lines += section("整集", full)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
