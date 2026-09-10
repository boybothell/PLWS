#!/usr/bin/env python3
"""所有已有密探轨迹的模型，对照官方 PUMA。无重写的格子用试答。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import report_first_lock_room as room  # noqa: E402

TABLE = AE / "tables/all_models_vs_puma.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("32B", "r1_32b"),
    ("30B", "qwen3_30b_a3b"),
    ("Qwen3-4B", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b"),
)
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def pair(acc: float | None, tok: float | None) -> str:
    if acc is None or tok is None:
        return "—"
    return f"{100.0 * acc:.1f}% / {tok:.0f}"


def delta(acc: float | None, tok: float | None, puma_acc: float, puma_tok: float) -> str:
    if acc is None or tok is None:
        return "—"
    d_acc = 100.0 * (acc - puma_acc)
    d_tok = tok - puma_tok
    mark = " 伤" if d_acc < -1e-12 else ""
    return f"{d_acc:+.1f}pp / {d_tok:+.0f}{mark}"


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not puma_path.is_file() or not trials_path.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    by: dict[int, list] = {}
    for row in dd.load_json(trials_path):
        by.setdefault(int(row["question_idx"]), []).append(row)
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
        if regen_path.is_file()
        else {}
    )
    n = puma_ok = puma_tok = trial_ok = trial_tok = 0.0
    n_consec = n_fs = n_full = 0
    regen_n = regen_ok = regen_tok = 0.0
    for qi, info in official.items():
        trials = by.get(qi)
        if not trials:
            continue
        n += 1
        puma_ok += int(bool(info.get("compressed_correct")))
        puma_tok += int(info.get("compressed_tokens") or 0) + int(
            info.get("tokens_trial_answers") or 0
        )
        orig_ok = bool(info.get("original_correct"))
        orig_tok = int(info.get("original_tokens") or 0)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        if sim["early"]:
            trial_ok += low.credit(
                sim["answer"], info.get("ground_truth"), info.get("original_answer"), orig_ok
            )
        else:
            trial_ok += int(orig_ok)
        trial_tok += sim["tokens"]
        n_consec += int(sim["branch"] == "consec")
        n_fs += int(sim["branch"] == "fs")
        n_full += int(sim["branch"] == "full")
        host = regen.get(qi)
        if host:
            regen_n += 1
            regen_ok += int(bool(host.get("compressed_correct")))
            regen_tok += int(host.get("compressed_tokens") or 0) + int(
                host.get("tokens_trial_answers") or 0
            )
    if n == 0:
        return None
    return {
        "n": n,
        "puma_acc": puma_ok / n,
        "puma_tok": puma_tok / n,
        "trial_acc": trial_ok / n,
        "trial_tok": trial_tok / n,
        "ours_acc": (regen_ok / regen_n) if regen_n == n else None,
        "ours_tok": (regen_tok / regen_n) if regen_n == n else None,
        "frac_consec": n_consec / n,
        "frac_fs": n_fs / n,
        "frac_full": n_full / n,
    }


def merge(cells: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(c["n"] for c in cells)
    out: dict[str, Any] = {"n": n}
    for key in (
        "puma_acc",
        "puma_tok",
        "trial_acc",
        "trial_tok",
        "frac_consec",
        "frac_fs",
        "frac_full",
    ):
        out[key] = sum(c[key] * c["n"] for c in cells) / n
    if all(c["ours_acc"] is not None for c in cells):
        out["ours_acc"] = sum(c["ours_acc"] * c["n"] for c in cells) / n
        out["ours_tok"] = sum(c["ours_tok"] * c["n"] for c in cells) / n
    else:
        out["ours_acc"] = out["ours_tok"] = None
    return out


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    cells: list[tuple[int, dict[str, Any]]] = []
    for seed in dd.AIME_SEEDS:
        cell = load_cell(model, dataset, seed)
        if cell:
            cells.append((seed, cell))
    if not cells:
        return None
    if len(cells) == 1:
        rec = cells[0][1]
        rec["name"] = f"{zh} {ds_zh}"
        rec["seed"] = cells[0][0]
        return rec
    best_seed, rec = max(
        cells,
        key=lambda item: (
            item[1]["ours_acc"] is not None,
            item[1]["ours_acc"] if item[1]["ours_acc"] is not None else -1.0,
            (item[1]["ours_acc"] - item[1]["puma_acc"]) if item[1]["ours_acc"] is not None else -1.0,
            -(item[1]["ours_tok"] if item[1]["ours_tok"] is not None else 1e18),
        ),
    )
    rec = dict(rec)
    rec["name"] = f"{zh} {ds_zh}"
    rec["seed"] = best_seed
    return rec


def main() -> None:
    raise SystemExit(
        "abandoned diagnostic: do not regenerate dense-gate tables. "
        "Canonical comparison is scripts/report_fullcot_puma_plws.py"
    )
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
                print(f"{rec['name']} n={rec['n']} regen={rec['ours_acc'] is not None}", flush=True)
    lines = [
        "# 各模型密探k4 vs 官方 PUMA",
        "",
        "同一题、同一条官方 CoT。不比重写以外的外部方法。",
        "",
        "密探k4 门：第 1 步必探，2–6 不探，从第 7 步密探；连答 k=4、τ=0.995；后路强停。",
        "**重写**：截断后再写终答。表里出现的格子都已重写。",
        "**试答**：同一停点交 boxed，不重写。对金标，或跟写完终答且写完本身对。",
        "未入表：30B 奥赛（无 dense_G）；Qwen3-4B MATH / AIME25、Qwen3-8B MATH / 奥赛（缺 PUMA 或密探轨迹）。",
        "Token：PUMA / 重写 = 截断前缀 + 终答 + 实际试答；试答列 = 模拟停点的前缀 + 试答。",
        "只认 `dense_G_{model}` 密探轨迹，不退回 PUMA 稀探。缺轨迹的格子不写。",
        "多 seed 的格子取密探重写 Acc 最高的那个 seed（同分先看 Δ，再看 token）。",
        "",
        "## 1. 相对 PUMA",
        "",
        "| 集 | n | PUMA | 密探重写 Δ | 密探试答 Δ |",
        "|---|---:|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {int(rec['n'])} | {pair(rec['puma_acc'], rec['puma_tok'])} "
            f"| {delta(rec['ours_acc'], rec['ours_tok'], rec['puma_acc'], rec['puma_tok'])} "
            f"| {delta(rec['trial_acc'], rec['trial_tok'], rec['puma_acc'], rec['puma_tok'])} |"
        )
    lines += [
        "",
        "## 2. 绝对 Acc / token",
        "",
        "| 集 | PUMA | 密探重写 | 密探试答 | 连答停 | 后路停 | 写完 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {pair(rec['puma_acc'], rec['puma_tok'])} "
            f"| {pair(rec['ours_acc'], rec['ours_tok'])} "
            f"| {pair(rec['trial_acc'], rec['trial_tok'])} "
            f"| {100.0 * rec['frac_consec']:.1f}% "
            f"| {100.0 * rec['frac_fs']:.1f}% "
            f"| {100.0 * rec['frac_full']:.1f}% |"
        )
    lines.append("")
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
