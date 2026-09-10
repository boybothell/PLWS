#!/usr/bin/env python3
"""有窗题：第一扇窗 high/mix/low 上的窗后压 vs 官方。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402

TABLE = AE / "tables/firstwin_wait/window_hml.md"
MODELS = (("r1_7b", "7B"), ("nemotron_8b", "8B"), ("r1_14b", "14B"))
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
DSZH = {
    "math-500": "MATH",
    "olympiadbench": "oly",
    "gpqa-diamond": "GPQA",
    "aime24": "A24",
    "aime25": "A25",
}
KINDS = ("high", "mix", "low")
KINDZH = {"high": "High", "mix": "Mix", "low": "Low"}
SEEDS = (42, 0, 1, 123)


def pick_seeds(zh: str, ds: str, available: list[int]) -> list[int]:
    if not available:
        return []
    if zh == "7B":
        return [s for s in SEEDS if s in available]
    if ds.startswith("aime"):
        four = [s for s in SEEDS if s in available]
        return four if len(four) == 4 else ([42] if 42 in available else available[:1])
    if 42 in available:
        return [42]
    return available[:1]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def jobs_path(tag: str, seed: int, kind: str) -> Path:
    folder = AE / "results/leftover_jump" / f"{tag}_s{seed}"
    return folder / ("jobs.jsonl" if kind == "low" else f"jobs_{kind}.jsonl")


def score_map(tag: str, seed: int, kind: str) -> dict[str, dict]:
    suf = "" if kind == "low" else f"_{kind}"
    folder = AE / "results/leftover_suppress_toend" / f"{tag}_s{seed}_suppress{suf}"
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for rec in load_jsonl(path):
            if rec.get("status") not in ("ok", "too_long"):
                continue
            uid = rec.get("uid")
            if uid:
                out[str(uid)] = rec
    return out


OFF: dict[tuple[str, str, int], dict[int, dict]] = {}


def official_map(tag: str, ds: str, seed: int) -> dict[int, dict]:
    key = (tag, ds, seed)
    if key not in OFF:
        path = dd.puma_stat_path(tag, ds, seed)
        OFF[key] = (
            {int(row["question_idx"]): row for row in dd.load_json(path)} if path.is_file() else {}
        )
    return OFF[key]


def collect() -> dict[tuple[str, str, str], list[dict]]:
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for tag, zh in MODELS:
        for seed in SEEDS:
            for kind in KINDS:
                jobs = load_jsonl(jobs_path(tag, seed, kind))
                scores = score_map(tag, seed, kind)
                for job in jobs:
                    ds = job.get("dataset")
                    qi = job.get("question_idx")
                    rec = scores.get(job["uid"])
                    if ds is None or qi is None or rec is None:
                        continue
                    off = official_map(tag, ds, seed).get(int(qi), {})
                    tot = float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)
                    ot = float(off.get("original_tokens") or 0)
                    cells.setdefault((zh, ds, kind), []).append(
                        {
                            "seed": seed,
                            "ok": bool(rec.get("new_gold_ok")),
                            "orig": bool(off.get("original_correct"))
                            if "original_correct" in off
                            else bool(job.get("orig_ok") or job.get("host_ok")),
                            "tot": tot,
                            "ot": ot,
                        }
                    )
    picked: dict[tuple[str, str, str], list[dict]] = {}
    for zh, ds, kind in [(z, d, k) for _, z in MODELS for d in DATASETS for k in KINDS]:
        rows = cells.get((zh, ds, kind), [])
        avail = sorted({r["seed"] for r in rows})
        keep = set(pick_seeds(zh, ds, avail))
        picked[(zh, ds, kind)] = [r for r in rows if r["seed"] in keep]
    return picked


def agg(xs: list[dict]) -> dict | None:
    if not xs:
        return None
    n = len(xs)
    denom = sum(x["ot"] for x in xs)
    return {
        "n": n,
        "a": 100.0 * sum(x["ok"] for x in xs) / n,
        "o": 100.0 * sum(x["orig"] for x in xs) / n,
        "t": sum(x["tot"] for x in xs) / n,
        "ot": sum(x["ot"] for x in xs) / n,
        "r": (sum(x["tot"] for x in xs) / denom) if denom else float("nan"),
    }


def fmt(stats: dict) -> str:
    star = "*" if stats["n"] < 10 else ""
    return (
        f"{stats['n']}{star} | {stats['a']:.1f}%{star} | {stats['o']:.1f}%{star} | "
        f"{stats['t']:.0f} | {stats['ot']:.0f} | {stats['r']:.2f}"
    )


def main() -> None:
    cells = collect()
    lines = [
        "# 有窗题：High / Mix / Low 窗后压 vs 官方",
        "",
        "只统计有第一扇四步同答窗的题。档位只看这一扇窗里四次试答的 PUMA 把握，不是整道题后面的把握。",
        "窗后压 = 从该窗接着写，思考阶段禁 Wait / Alternatively / Hmm。对照是同一题官方 Full-CoT。",
        "总 tok = 思考 + 终答。`*` 表示 n<10，数字只看涨跌方向，不当精确值。",
        "",
        "## 档位含义",
        "",
        "第一扇窗：从步数 ≥10 起，第一次连续 4 次试答答案相同。",
        "",
        "- **High**：这四次把握满足 PUMA 高把握门——第一次 ≥ 0.995，后面三次都不低于「第一次 − 0.03」。模型在窗上已经很确信当前答案。",
        "- **Low**：这四次把握全部 < 0.995。答案锁住了，但密探并不确信。",
        "- **Mix**：有四次同答，但既不是全高、也不是全低——四次里有的过 0.995、有的没过。",
        "",
        "三档互不重叠，合起来就是全部有窗题。",
        "",
    ]
    for kind in KINDS:
        lines += [
            f"## {KINDZH[kind]}",
            "",
            "| 模型 | 集 | n | 窗后 Acc | 官方 Acc | tok | 官 tok | /官 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for zh in ("7B", "8B", "14B"):
            pooled: list[dict] = []
            for ds in DATASETS:
                rows = cells.get((zh, ds, kind), [])
                stats = agg(rows)
                if not stats:
                    continue
                pooled.extend(rows)
                lines.append(f"| {zh} | {DSZH[ds]} | {fmt(stats)} |")
            all_s = agg(pooled)
            if all_s:
                lines.append(f"| {zh} | 全部 | {fmt(all_s)} |")
        lines.append("")
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
