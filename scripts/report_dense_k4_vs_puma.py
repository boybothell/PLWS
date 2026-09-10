#!/usr/bin/env python3
"""Write 密探k4 vs PUMA tables into tables/dense_k4/."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
OUT = AE / "tables/dense_k4"
SUMMARY = AE / "results/default_dense_gate/summary.json"

MODEL_ZH = {
    "r1_7b": "7B",
    "nemotron_8b": "8B",
    "r1_14b": "14B",
    "r1_32b": "32B",
    "qwen3_30b_a3b": "30B",
}
DS_ZH = {
    "math-500": "MATH",
    "olympiadbench": "奥赛",
    "gpqa-diamond": "GPQA",
    "aime24": "AIME24",
    "aime25": "AIME25",
}
MODELS = list(MODEL_ZH)
DATASETS = list(DS_ZH)


def pct(x: float | None) -> str:
    return "—" if x is None or x != x else f"{100.0 * x:.1f}%"


def tok(x: float | None) -> str:
    return "—" if x is None or x != x else f"{x:.0f}"


def pp(x: float | None) -> str:
    return "—" if x is None or x != x else f"{x:+.2f}"


def dtok(x: float | None) -> str:
    return "—" if x is None or x != x else f"{x:+.0f}"


def pair(acc: float | None, tokens: float | None) -> str:
    if acc is None or tokens is None or acc != acc or tokens != tokens:
        return "—"
    return f"{pct(acc)} / {tok(tokens)}"


def delta(acc_pp: float | None, tokens: float | None) -> str:
    if acc_pp is None or tokens is None or acc_pp != acc_pp or tokens != tokens:
        return "—"
    return f"{pp(acc_pp)} / {dtok(tokens)}"


def merge(cells: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(c["n"] for c in cells)
    ready = [c for c in cells if c.get("source") == "regen" and c.get("ours_acc") is not None]
    out: dict[str, Any] = {"n": n, "n_runs": len(cells)}
    for key in (
        "trial_acc",
        "puma_acc",
        "puma_tok",
        "full_acc",
        "full_tok",
        "d_trial_acc_pp",
        "frac_consec",
        "frac_fs",
        "frac_full",
    ):
        out[key] = sum(c[key] * c["n"] for c in cells) / n
    if ready and sum(c["n"] for c in ready) == n:
        out["ours_acc"] = sum(c["ours_acc"] * c["n"] for c in ready) / n
        out["ours_tok"] = sum(c["ours_tok"] * c["n"] for c in ready) / n
        out["d_acc_pp"] = sum(c["d_acc_pp"] * c["n"] for c in ready) / n
        out["d_tok"] = sum(c["d_tok"] * c["n"] for c in ready) / n
        out["source"] = "regen"
    else:
        out["ours_acc"] = out["ours_tok"] = out["d_acc_pp"] = out["d_tok"] = None
        out["source"] = "pending_regen"
    return out


def main() -> None:
    cells = json.loads(SUMMARY.read_text())["cells"]
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        if c.get("missing"):
            continue
        by[(c["model"], c["dataset"])].append(c)
    merged = {k: merge(v) for k, v in by.items()}

    lines = [
        "# 密探k4 vs 官方 PUMA",
        "",
        "这个目录只放点名要的表。本文件：密探k4 对官方 PUMA 的已存对比。",
        "",
        "密探k4：第 1 步必探，2–6 步不探，从第 7 步起每步都试答。",
        "连答停：同一答案连续 4 次，第一次 ≥ 0.995，后面不低于第一次 − 0.03，步数 ≥ 10。",
        "后路停：步数 ≥ 80，相邻步字数比 0.84–1.16 或相差 ≤ 35 连续 2 次，",
        "全程最高置信度 ≥ 0.85，最近 2 次试答相同，交同答里置信度最高且 ≥ 0.7。",
        "闸门只截断。从半截思路重写终答，Acc 看重写对金标。写完的题复用原终答。",
        "Token = 截断前缀 + 重写终答 + 实际探过的试答。",
        "",
        "7B / 8B 已重写。14B / 32B / 30B 只有门的停点，没有重写 Acc/Tok，显示 —。",
        "非 AIME seed 42。AIME 四个 seed（42 / 0 / 1 / 123），主表按题数加权，分 seed 在后面。",
        "数字来自 `results/default_dense_gate/summary.json`。",
        "",
        "## 相对 PUMA（Acc 百分点 / token）",
        "",
        "| 模型 | MATH | 奥赛 | GPQA | AIME24 | AIME25 |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        row = [MODEL_ZH[model]]
        for ds in DATASETS:
            rec = merged.get((model, ds))
            if not rec or rec["source"] != "regen":
                row.append("—")
            else:
                row.append(f"{pp(rec['d_acc_pp'])} / {dtok(rec['d_tok'])}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## 绝对数",
        "",
        "每格是 Acc / token。相对 PUMA 是百分点差 / token 差。",
        "",
        "| 模型 | 集 | n | 密探k4 | PUMA | 原始CoT | 相对 PUMA |",
        "|---|---|---:|---|---|---|---|",
    ]
    for model in MODELS:
        for ds in DATASETS:
            rec = merged.get((model, ds))
            if not rec:
                continue
            lines.append(
                f"| {MODEL_ZH[model]} | {DS_ZH[ds]} | {rec['n']} "
                f"| {pair(rec['ours_acc'], rec['ours_tok'])} "
                f"| {pair(rec['puma_acc'], rec['puma_tok'])} "
                f"| {pair(rec['full_acc'], rec['full_tok'])} "
                f"| {delta(rec['d_acc_pp'], rec['d_tok'])} |"
            )

    lines += [
        "",
        "## 停因",
        "",
        "| 模型 | 集 | 连答停 | 后路停 | 写完 |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for ds in DATASETS:
            rec = merged.get((model, ds))
            if not rec:
                continue
            lines.append(
                f"| {MODEL_ZH[model]} | {DS_ZH[ds]} "
                f"| {pct(rec['frac_consec'])} | {pct(rec['frac_fs'])} | {pct(rec['frac_full'])} |"
            )

    lines += [
        "",
        "## 只交试答（不重写）对 PUMA",
        "",
        "同一套密探k4 停点，交试答 boxed，不重写。7B / 8B 可对照重写列。",
        "",
        "| 模型 | 集 | 试答 Acc | 重写 Acc | PUMA Acc | 试答相对 PUMA |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        for ds in DATASETS:
            rec = merged.get((model, ds))
            if not rec:
                continue
            lines.append(
                f"| {MODEL_ZH[model]} | {DS_ZH[ds]} "
                f"| {pct(rec['trial_acc'])} | {pct(rec['ours_acc'])} | {pct(rec['puma_acc'])} "
                f"| {pp(rec['d_trial_acc_pp'])} |"
            )

    lines += [
        "",
        "## AIME 分 seed",
        "",
        "每格 30 题。",
        "",
        "| 模型 | 集 | seed | 密探k4 | PUMA | 原始CoT | 相对 PUMA | 连答 | 后路 | 写完 |",
        "|---|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for ds in ("aime24", "aime25"):
            for cell in sorted(by.get((model, ds), []), key=lambda x: x["seed"]):
                ours_acc = cell.get("ours_acc") if cell.get("source") == "regen" else None
                ours_tok = cell.get("ours_tok") if cell.get("source") == "regen" else None
                d_acc = cell.get("d_acc_pp") if cell.get("source") == "regen" else None
                d_tok = cell.get("d_tok") if cell.get("source") == "regen" else None
                lines.append(
                    f"| {MODEL_ZH[model]} | {DS_ZH[ds]} | {cell['seed']} "
                    f"| {pair(ours_acc, ours_tok)} "
                    f"| {pair(cell['puma_acc'], cell['puma_tok'])} "
                    f"| {pair(cell['full_acc'], cell.get('full_tok'))} "
                    f"| {delta(d_acc, d_tok)} "
                    f"| {pct(cell['frac_consec'])} | {pct(cell['frac_fs'])} | {pct(cell['frac_full'])} |"
                )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "vs_puma.md"
    path.write_text("\n".join(lines) + "\n")
    readme = OUT / "README.md"
    readme.write_text(
        "# dense_k4\n"
        "\n"
        "这个目录只有点名才往里放文件。\n"
        "\n"
        "方法名：密探k4。密探，连答 k=4、τ=0.995，后路强停，截断后重写终答。\n"
        "\n"
        "| 文件 | 内容 |\n"
        "|---|---|\n"
        "| [`vs_puma.md`](vs_puma.md) | 密探k4 对官方 PUMA 的已存对比（含 AIME 分 seed） |\n"
    )
    print(f"写成 {path}", flush=True)


if __name__ == "__main__":
    main()
