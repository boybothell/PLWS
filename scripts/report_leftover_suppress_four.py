#!/usr/bin/env python3
"""第一扇窗压 Wait vs 官方 Full-CoT。只出有窗合计和整集缝。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402

ROOT = AE / "results/leftover_suppress_toend"
JOBS = AE / "results/leftover_jump"
TABLE = AE / "tables/firstwin_wait/leftover_suppress_four.md"
MODELS = (("r1_7b", "7B"), ("nemotron_8b", "8B"), ("r1_14b", "14B"))
KINDS = ("low", "high", "mix")
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
STAT: dict[tuple[str, str], dict[int, dict]] = {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jobs_path(model: str, kind: str) -> Path:
    folder = JOBS / f"{model}_s{SEED}"
    return folder / ("jobs.jsonl" if kind == "low" else f"jobs_{kind}.jsonl")


def score_dir(model: str, kind: str) -> Path:
    tag = f"{model}_s{SEED}_suppress" if kind == "low" else f"{model}_s{SEED}_suppress_{kind}"
    return ROOT / tag


def official_map(model: str, dataset: str) -> dict[int, dict]:
    key = (model, dataset)
    if key not in STAT:
        path = dd.puma_stat_path(model, dataset, SEED)
        STAT[key] = (
            {int(row["question_idx"]): row for row in dd.load_json(path)}
            if path.is_file()
            else {}
        )
    return STAT[key]


def orig_tok(model: str, dataset: str, qi: int) -> int:
    return int(official_map(model, dataset).get(int(qi), {}).get("original_tokens") or 0)


def collect() -> list[dict]:
    rows: list[dict] = []
    for model, zh in MODELS:
        for kind in KINDS:
            jobs = {row["uid"]: row for row in load_jsonl(jobs_path(model, kind))}
            scores: list[dict] = []
            folder = score_dir(model, kind)
            if folder.is_dir():
                for path in sorted(folder.glob("scores_shard*.jsonl")):
                    scores.extend(load_jsonl(path))
            by = {
                row["uid"]: row
                for row in scores
                if row.get("status") == "ok"
                and row.get("generated_text")
                and row["uid"] in jobs
            }
            for uid, job in jobs.items():
                rec = by.get(uid)
                if not rec:
                    continue
                tot = float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)
                rows.append(
                    {
                        "model": zh,
                        "tag": model,
                        "kind": kind,
                        "ds": job["dataset"],
                        "qi": int(job["question_idx"]),
                        "ok": bool(rec.get("new_gold_ok")),
                        "orig": bool(job.get("orig_ok")),
                        "tot": tot,
                        "ot": orig_tok(model, job["dataset"], job["question_idx"]),
                    }
                )
    return rows


def collect_full(window: list[dict]) -> list[dict]:
    by = {(x["tag"], x["ds"], x["qi"]): x for x in window}
    rows: list[dict] = []
    for tag, zh in MODELS:
        for dataset in DATASETS:
            official = official_map(tag, dataset)
            for qi, info in official.items():
                hit = by.get((tag, dataset, int(qi)))
                orig = bool(info.get("original_correct"))
                ot = int(info.get("original_tokens") or 0)
                if hit:
                    rows.append(
                        {
                            "model": zh,
                            "kind": "full",
                            "ds": dataset,
                            "ok": hit["ok"],
                            "orig": orig,
                            "tot": hit["tot"],
                            "ot": ot,
                        }
                    )
                else:
                    rows.append(
                        {
                            "model": zh,
                            "kind": "full",
                            "ds": dataset,
                            "ok": orig,
                            "orig": orig,
                            "tot": float(ot),
                            "ot": ot,
                        }
                    )
    return rows


def line(xs: list[dict], model: str, dszh: str) -> str | None:
    if not xs:
        return None
    n = len(xs)
    acc = 100.0 * sum(x["ok"] for x in xs) / n
    orig = 100.0 * sum(x["orig"] for x in xs) / n
    tot = sum(x["tot"] for x in xs) / n
    ot = sum(x["ot"] for x in xs) / n
    denom = sum(x["ot"] for x in xs)
    ratio = sum(x["tot"] for x in xs) / denom if denom else float("nan")
    return (
        f"| {model} | {dszh} | {n} | {acc:.1f}% | {orig:.1f}% | "
        f"{tot:.0f} | {ot:.0f} | {ratio:.2f} |"
    )


def section(title: str, rows: list[dict], kind: str | None) -> list[str]:
    out = [
        f"## {title}",
        "",
        "| 模型 | 集 | n | 压 Wait Acc | 官方 Acc | 压 Wait 总 tok | 官方总 tok | 总/官 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, zh in MODELS:
        for ds, dszh in DS:
            xs = [
                x
                for x in rows
                if x["model"] == zh
                and (kind is None or x["kind"] == kind)
                and (not ds or x["ds"] == ds)
            ]
            text = line(xs, zh, dszh)
            if text:
                out.append(text)
    out.append("")
    return out


def main() -> None:
    rows = collect()
    lines = [
        "# 第一扇四步同答窗：压 Wait vs 官方 Full-CoT",
        "",
        "seed-42。第一扇四步同答窗后续写，思考阶段禁 Wait / Alternatively / Hmm。",
        "有窗 = 有第一扇窗的题。整集 = 有窗用压 Wait，无窗用官方 CoT。",
        "两边总 tok = 思考 + 终答。总/官 = 压 Wait 总 tok / 官方 `original_tokens`。",
        "32B / 30B 还没有分。",
        "",
    ]
    lines += section("有窗", rows, None)
    lines += section("整集", collect_full(rows), "full")
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
