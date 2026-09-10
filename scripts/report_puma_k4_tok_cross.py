#!/usr/bin/env python3
"""Official PUMA stop reason × 密探k4 branch: where the token delta comes from."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_first_lock_room as room

TABLE = AE / "tables/puma_k4_tok_cross.md"
AIME_SEEDS = (42, 0, 1, 123)
CELLS = (
    ("7B MATH", "r1_7b", "math-500"),
    ("7B 奥赛", "r1_7b", "olympiadbench"),
    ("7B GPQA", "r1_7b", "gpqa-diamond"),
    ("7B AIME24", "r1_7b", "aime24"),
    ("7B AIME25", "r1_7b", "aime25"),
    ("8B MATH", "nemotron_8b", "math-500"),
    ("8B 奥赛", "nemotron_8b", "olympiadbench"),
    ("8B GPQA", "nemotron_8b", "gpqa-diamond"),
    ("8B AIME24", "nemotron_8b", "aime24"),
    ("8B AIME25", "nemotron_8b", "aime25"),
)
P_ORDER = (
    ("full_reasoning", "写完"),
    ("confidence_consecutive", "连答"),
    ("forced_stop_redundancy", "后路"),
)
K_ORDER = (("consec", "连答"), ("fs", "后路"), ("full", "写完"))
STORY = (
    ("full_reasoning", "consec", "官方写完 → k4连答"),
    ("full_reasoning", "full", "两边都写完"),
    ("full_reasoning", "fs", "官方写完 → k4后路"),
    ("confidence_consecutive", "consec", "两边都连答"),
    ("confidence_consecutive", "full", "官方连答 → k4写完"),
    ("confidence_consecutive", "fs", "官方连答 → k4后路"),
    ("forced_stop_redundancy", "consec", "官方后路 → k4连答"),
    ("forced_stop_redundancy", "fs", "两边都后路"),
    ("forced_stop_redundancy", "full", "官方后路 → k4写完"),
)


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    return xs[len(xs) // 2]


def fmt(x: float, nd: int = 0) -> str:
    return "—" if x != x else f"{x:+.{nd}f}" if nd else f"{x:+.0f}"


def absf(x: float, nd: int = 0) -> str:
    return "—" if x != x else f"{x:.{nd}f}"


def parts(info: dict[str, Any]) -> tuple[int, int, int]:
    prefix = int(info.get("compressed_tokens") or 0)
    trial = int(info.get("tokens_trial_answers") or 0)
    return prefix, trial, prefix + trial


def load_named(name: str, model: str, dataset: str) -> dict[str, Any] | None:
    seeds = AIME_SEEDS if dataset in ("aime24", "aime25") else (42,)
    rows = []
    for seed in seeds:
        pack = load_cell(name, model, dataset, seed)
        if pack:
            rows.extend(pack["rows"])
    if not rows:
        return None
    return {"name": name, "rows": rows}


def load_cell(name: str, model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    regen_path = dd.regen_stat_path(model, dataset, seed)
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not puma_path.is_file() or not regen_path.is_file() or not trials_path.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    rows_out = []
    for qi, info in official.items():
        host = regen.get(qi)
        trials = by.get(qi)
        if not host or not trials:
            continue
        orig_tok = int(info.get("original_tokens") or 0)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        p_pre, p_tr, p_tot = parts(info)
        k_pre, k_tr, k_tot = parts(host)
        rows_out.append(
            {
                "p_reason": str(info.get("stop_reason") or ""),
                "k_branch": sim["branch"],
                "p_pre": p_pre,
                "p_tr": p_tr,
                "p_tot": p_tot,
                "k_pre": k_pre,
                "k_tr": k_tr,
                "k_tot": k_tot,
                "d_tot": k_tot - p_tot,
                "d_pre": k_pre - p_pre,
                "d_tr": k_tr - p_tr,
                "p_step": int(info.get("stopped_len") or 0),
                "k_step": int(host.get("stopped_len") or sim["step"]),
                "p_n": int(info.get("generated_trial_answers") or 0),
                "k_n": int(host.get("generated_trial_answers") or sim.get("generated_trial_answers") or 0),
                "p_ok": bool(info.get("compressed_correct")),
                "k_ok": bool(host.get("compressed_correct")),
            }
        )
    return {"name": name, "rows": rows_out}


def summarize(xs: list[dict[str, Any]], n_all: int) -> dict[str, Any]:
    n = len(xs)
    return {
        "n": n,
        "share": n / n_all if n_all else float("nan"),
        "p_tot": mean([x["p_tot"] for x in xs]),
        "k_tot": mean([x["k_tot"] for x in xs]),
        "d_tot": mean([x["d_tot"] for x in xs]),
        "d_med": med([x["d_tot"] for x in xs]),
        "d_pre": mean([x["d_pre"] for x in xs]),
        "d_tr": mean([x["d_tr"] for x in xs]),
        "contrib": n * mean([x["d_tot"] for x in xs]) / n_all if n_all and xs else 0.0,
        "p_step": mean([x["p_step"] for x in xs]),
        "k_step": mean([x["k_step"] for x in xs]),
        "p_n": mean([x["p_n"] for x in xs]),
        "k_n": mean([x["k_n"] for x in xs]),
        "p_acc": sum(x["p_ok"] for x in xs) / n if n else float("nan"),
        "k_acc": sum(x["k_ok"] for x in xs) / n if n else float("nan"),
    }


def line_of(name: str, rec: dict[str, Any]) -> str:
    if rec["n"] == 0:
        return f"| {name} | 0 | — | — | — | — | — | — | — |"
    return (
        f"| {name} | {rec['n']}（{100.0 * rec['share']:.0f}%） "
        f"| {absf(rec['p_tot'])} → {absf(rec['k_tot'])} "
        f"| {fmt(rec['d_tot'])} / {fmt(rec['d_med'])} "
        f"| {fmt(rec['d_pre'])} / {fmt(rec['d_tr'])} "
        f"| {fmt(rec['contrib'])} "
        f"| {absf(rec['p_step'])} → {absf(rec['k_step'])} "
        f"| {absf(rec['p_n'], 1)} → {absf(rec['k_n'], 1)} "
        f"| {rg.fmt_pct(rec['p_acc'])} → {rg.fmt_pct(rec['k_acc'])} |"
    )


def main() -> None:
    packs = []
    for name, model, dataset in CELLS:
        pack = load_named(name, model, dataset)
        if pack:
            packs.append(pack)
            print(f"{name} n={len(pack['rows'])}", flush=True)
    header = (
        "| 桶 | 题数 | PUMA → k4 token | 每题差 均/中位 | 前缀差 / 试答差 | 均摊到全体 | 步数 | 试答次数 | Acc |"
    )
    sep = "|---|---:|---|---:|---|---:|---|---|---|"
    lines = [
        "# 官方 PUMA × 密探k4：token 差从哪来",
        "",
        "同一条 CoT。官方 PUMA 是稀探；密探k4 从第 7 步起每步都探，截断后再写终答。",
        "Token = 前缀（截断或写完）+ 实际探过的试答。官方写完复用原终答。",
        "均摊 = 该桶题数 × 每题均差 / 全集题数，加起来等于表上那一格的 token 差。",
        "",
    ]
    for pack in packs:
        xs = pack["rows"]
        n = len(xs)
        all_rec = summarize(xs, n)
        lines += [
            f"## {pack['name']}",
            "",
            f"全体 {n} 题：PUMA {absf(all_rec['p_tot'])} → k4 {absf(all_rec['k_tot'])}（{fmt(all_rec['d_tot'])}）。"
            f"前缀 {fmt(all_rec['d_pre'])}，试答 {fmt(all_rec['d_tr'])}。",
            "",
            header,
            sep,
            line_of("全体", all_rec),
        ]
        for p_key, k_key, zh in STORY:
            rec = summarize([x for x in xs if x["p_reason"] == p_key and x["k_branch"] == k_key], n)
            if rec["n"] == 0:
                continue
            lines.append(line_of(zh, rec))
        lines.append("")
        lines.append("官方停法 × k4 停法（题数 / 均摊 token）：")
        lines.append("")
        kh = " | ".join(f"k4{zh}" for _, zh in K_ORDER)
        lines.append(f"| 官方 \\ k4 | {kh} |")
        lines.append("|---|---:|---:|---:|")
        for p_key, p_zh in P_ORDER:
            cells = []
            for k_key, _ in K_ORDER:
                rec = summarize([x for x in xs if x["p_reason"] == p_key and x["k_branch"] == k_key], n)
                cells.append(f"{rec['n']} / {fmt(rec['contrib'])}" if rec["n"] else "—")
            lines.append(f"| 官方{p_zh} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## 读法",
        "",
        "MATH 上「官方写完 → k4连答」每题少几百，但均摊只有几十到一百；官方后路那一小撮 k4 没锁住、继续写，每题贵一两千，把平均吃回去。",
        "GPQA 上官方写完之后 k4 几乎也写完，早停那一格几乎是空的。",
        "AIME 四个 seed 并在一起（每集 120 题）。后路题更多，k4 去掉后路之后写完比例大约一半。",
        "试答差一律为正：密探每步都探，试答次数大约是官方的 2–6 倍。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
