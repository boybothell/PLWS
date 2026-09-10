#!/usr/bin/env python3
"""开局压核三词 vs 第一扇窗后压 vs 官方。只出整体。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402

TABLE = AE / "tables/firstwin_wait/fromstart_vs_window.md"
MODELS = (("r1_7b", "7B"), ("nemotron_8b", "8B"), ("r1_14b", "14B"))
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
DSZH = {
    "math-500": "MATH",
    "olympiadbench": "oly",
    "gpqa-diamond": "GPQA",
    "aime24": "A24",
    "aime25": "A25",
}
SEEDS = (42, 0, 1, 123)


def load_scores(folder: Path, seed: int) -> dict[tuple[str, int, int], dict]:
    out: dict[tuple[str, int, int], dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") not in ("ok", "too_long"):
                continue
            ds = rec.get("dataset")
            qi = rec.get("question_idx")
            if ds is None or qi is None:
                parts = str(rec.get("uid") or "").split(":")
                if len(parts) >= 4:
                    ds = ds or parts[1]
                    qi = qi if qi is not None else parts[3]
            if ds is None or qi is None:
                continue
            out[(str(ds), int(seed), int(qi))] = rec
    return out


def load_window_scores(tag: str, seed: int) -> dict[tuple[str, int, int], dict]:
    out: dict[tuple[str, int, int], dict] = {}
    for suf in ("", "_high", "_mix"):
        folder = AE / "results/leftover_suppress_toend" / f"{tag}_s{seed}_suppress{suf}"
        out.update(load_scores(folder, seed))
    return out


def leftover_windowed(tag: str, seed: int) -> tuple[set[tuple[str, int]], set[str]]:
    """有第一扇窗的 (dataset, qi)，以及这份 leftover 已经导出过的数据集。"""
    keys: set[tuple[str, int]] = set()
    datasets: set[str] = set()
    folder = AE / "results/leftover_jump" / f"{tag}_s{seed}"
    for name in ("jobs.jsonl", "jobs_high.jsonl", "jobs_mix.jsonl"):
        path = folder / name
        if not path.is_file():
            continue
        for line in path.open():
            if not line.strip():
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError:
                continue
            ds = job.get("dataset")
            qi = job.get("question_idx")
            if ds is None or qi is None:
                continue
            keys.add((str(ds), int(qi)))
            datasets.add(str(ds))
    return keys, datasets


def tot(rec: dict) -> float:
    return float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)


def official_map(tag: str, ds: str, seed: int) -> dict[int, dict]:
    path = dd.puma_stat_path(tag, ds, seed)
    if not path.is_file():
        return {}
    return {int(row["question_idx"]): row for row in dd.load_json(path)}


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


def cell(s: dict | None, key: str, fmt: str) -> str:
    if not s:
        return "—"
    return fmt.format(s[key])


def line_vs(name: str, method: dict) -> str:
    return (
        f"| {name} | {method['n']} | {method['a']:.1f}% | {method['o']:.1f}% | "
        f"{method['t']:.0f} | {method['ot']:.0f} | {method['r']:.2f} |"
    )


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


def main() -> None:
    notes: list[str] = []
    start_cells: dict[tuple[str, str, int], list[dict]] = {}
    win_cells: dict[tuple[str, str, int], list[dict]] = {}
    for tag, zh in MODELS:
        for seed in SEEDS:
            start = load_scores(AE / "results/fromstart_core" / f"{tag}_s{seed}_suppress", seed)
            win = load_window_scores(tag, seed)
            windowed, exported = leftover_windowed(tag, seed)
            for ds in DATASETS:
                off = official_map(tag, ds, seed)
                n_off = len(off)
                if not n_off:
                    notes.append(f"{zh} {DSZH[ds]} s{seed} 无官方统计")
                    continue
                rows_s: list[dict] = []
                rows_w: list[dict] = []
                exported_ds = ds in exported or seed == 42 or ds.startswith("aime")
                miss_w = 0
                for qi, info in off.items():
                    orig = bool(info.get("original_correct"))
                    ot = float(info.get("original_tokens") or 0)
                    rec_s = start.get((ds, seed, int(qi)))
                    rec_w = win.get((ds, seed, int(qi)))
                    if rec_s:
                        rows_s.append({"ok": bool(rec_s.get("new_gold_ok")), "orig": orig, "tot": tot(rec_s), "ot": ot})
                    if rec_w:
                        rows_w.append({"ok": bool(rec_w.get("new_gold_ok")), "orig": orig, "tot": tot(rec_w), "ot": ot})
                    elif (ds, int(qi)) not in windowed and exported_ds:
                        rows_w.append({"ok": orig, "orig": orig, "tot": ot, "ot": ot})
                    else:
                        miss_w += 1
                if len(rows_s) == n_off:
                    start_cells[(zh, ds, seed)] = rows_s
                elif rows_s:
                    notes.append(f"{zh} {DSZH[ds]} s{seed} 开局 {len(rows_s)}/{n_off}")
                if miss_w == 0 and len(rows_w) == n_off:
                    win_cells[(zh, ds, seed)] = rows_w
                elif exported_ds and miss_w:
                    notes.append(f"{zh} {DSZH[ds]} s{seed} 窗后 {n_off - miss_w}/{n_off}")

    def seeds_of(store: dict, zh: str, ds: str) -> list[int]:
        return [s for (z, d, s) in store if z == zh and d == ds]

    def rows_of(store: dict, zh: str, ds: str) -> list[dict]:
        chosen = pick_seeds(zh, ds, seeds_of(store, zh, ds))
        return [r for s in chosen for r in store[(zh, ds, s)]]

    def table(title: str, store: dict) -> list[str]:
        out = [
            title,
            "",
            "| 模型 | 集 | n | Acc | 官方 Acc | tok | 官 tok | /官 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for zh in ("7B", "8B", "14B"):
            pooled: list[dict] = []
            for ds in DATASETS:
                rows = rows_of(store, zh, ds)
                stats = agg(rows)
                if not stats:
                    continue
                pooled.extend(rows)
                out.append(
                    f"| {zh} | {DSZH[ds]} | {stats['n']} | {stats['a']:.1f}% | {stats['o']:.1f}% | "
                    f"{stats['t']:.0f} | {stats['ot']:.0f} | {stats['r']:.2f} |"
                )
            all_s = agg(pooled)
            if all_s and len({ds for ds in DATASETS if rows_of(store, zh, ds)}) >= 4:
                out.append(
                    f"| {zh} | 全部 | {all_s['n']} | {all_s['a']:.1f}% | {all_s['o']:.1f}% | "
                    f"{all_s['t']:.0f} | {all_s['ot']:.0f} | {all_s['r']:.2f} |"
                )
        return out

    lines = [
        "# 窗后压 / 开局压 vs 官方",
        "",
        "核三词 = Wait / Alternatively / Hmm。不开扩词表。窗后压无窗贴官方。总 tok = 思考 + 终答。",
        "",
        *table("## 窗后压", win_cells),
        "",
        *table("## 开局压", start_cells),
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
