#!/usr/bin/env python3
"""全部四步同答窗：已有 4B 校准读数能不能分开对窗/错窗。零 GPU。"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_confcal_4b_variants as var
import analyze_confcal_v1 as v1
import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_second_lock as sl
import train_leftover_commit_probe as tp

TABLE = AE / "tables/samewin_4b_all.md"
VAR = AE / "results/confcal_judge/qwen4b_variants"
CELLS = (
    ("MATH", "math-500", (42,)),
    ("奥赛", "olympiadbench", (42,)),
    ("GPQA", "gpqa-diamond", (42,)),
    ("AIME24", "aime24", (42, 0, 1, 123)),
    ("AIME25", "aime25", (42, 0, 1, 123)),
)
SIGNALS = (
    ("geo", "试答把握（对照）"),
    ("verbal", "口头 0–100"),
    ("B_final", "Direct P(Yes)"),
    ("B_3way_correct", "三路：Correct/(Correct+Incorrect)"),
    ("B_3way_evidence", "三路：1−证据不足"),
    ("EAGLE_yn", "EAGLE Yes/No"),
    ("EAGLE_10bin", "EAGLE 十分箱"),
    ("EAGLE_10bin_final", "EAGLE 十分箱终档"),
    ("SteerConf_mean", "SteerConf 均值"),
    ("SteerConf_min", "SteerConf 最小"),
    ("SteerConf_stable", "SteerConf 均值×(1−散度)"),
    ("CMP_neg", "−跨模型惊讶"),
    ("CME_neg", "−跨模型熵"),
    ("h4_margin", "Yes−No 边距"),
    ("h4_coverage", "Yes+No 覆盖率"),
    ("h4_neg_entropy", "−Yes/No 熵"),
)


def finite(x: Any) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def load_var(dataset: str, seed: int) -> dict[tuple[int, int], dict[str, float]]:
    folder = VAR / f"{dataset}_s{seed}"
    out: dict[tuple[int, int], dict[str, float]] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            pack = var.signal_values(row)
            pack["geo"] = finite(row.get("geo_conf"))
            out[(int(row["question_idx"]), int(row["decision_step"]))] = pack
    return out


def load_v1_extra(dataset: str) -> dict[tuple[int, int], dict[str, float]]:
    if dataset not in ("math-500", "olympiadbench", "gpqa-diamond"):
        return {}
    out: dict[tuple[int, int], dict[str, float]] = {}
    for row in v1.assemble(dataset):
        out[(row["question_idx"], row["decision_step"])] = {
            "verbal": finite(row.get("verbal_h3")),
            "h4_margin": finite(row.get("h4_margin")),
            "h4_coverage": finite(row.get("h4_coverage")),
            "h4_neg_entropy": finite(row.get("h4_neg_entropy")),
        }
    return out


def load_wins(dataset: str, seed: int) -> list[dict[str, Any]]:
    scores = load_var(dataset, seed)
    extra = load_v1_extra(dataset) if seed == 42 else {}
    trials_path = room.dense_trial_path("r1_7b", dataset, seed)
    if not trials_path.is_file() or not scores:
        return []
    puma_path = dd.puma_stat_path("r1_7b", dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    gp = low.gpath("r1_7b", dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    out = []
    for qi, trials in by.items():
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        info = official.get(qi) or {}
        g = gmap.get(qi) or {}
        last = max(trials, key=lambda x: int(x["stopped_len"]))
        gt = info.get("ground_truth") or g.get("ground_truth")
        original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
        orig_ok = bool(info.get("original_correct")) if "original_correct" in info else bool(
            low.credit(original, gt, original, True)
        )
        for win in sl.same_windows(rows):
            pack = dict(scores.get((qi, win["step"])) or {})
            pack.update(extra.get((qi, win["step"])) or {})
            if "geo" not in pack:
                continue
            pack.update(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "kind": win["kind"],
                    "gold_ok": bool(low.credit(win["ans"], gt, original, orig_ok)),
                }
            )
            out.append(pack)
    return out


def auroc(xs: list[dict[str, Any]], key: str) -> float:
    pairs = [(int(bool(x["gold_ok"])), finite(x.get(key))) for x in xs]
    pairs = [(y, s) for y, s in pairs if s == s]
    if len(pairs) < 8:
        return float("nan")
    y = np.asarray([p[0] for p in pairs])
    s = np.asarray([p[1] for p in pairs])
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def line(name: str, xs: list[dict[str, Any]]) -> str:
    cells = [tp.fmt(auroc(xs, key)) for key, _ in SIGNALS]
    ok = sum(1 for x in xs if x["gold_ok"])
    return f"| {name} | {ok}/{len(xs)-ok} | " + " | ".join(cells) + " |"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rows: list[dict[str, Any]] = []
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for zh, dataset, seeds in CELLS:
        for seed in seeds:
            part = load_wins(dataset, seed)
            rows.extend(part)
            by_ds[zh].extend(part)
            print(f"{dataset} s{seed} n={len(part)}", flush=True)
    head = "| 切片 | 对/错 | " + " | ".join(name for _, name in SIGNALS) + " |"
    sep = "|---|---:|" + "|".join(["---:"] * len(SIGNALS)) + "|"
    lines = [
        "# 全部四步同答窗：4B 校准方案能不能分开对/错",
        "",
        "7B 轨迹。每一扇连续 4 步同一 boxed 都进，含高把握 / 混合 / 低把握。",
        "对窗 = 试答对金标。分数是已有 4B 抽取，不新开卡。",
        "数字是 AUROC，越大越像对窗。0.50 是猜。口头 0–100 只有 MATH/奥赛/GPQA。",
        "",
        "## 1. 全体窗",
        "",
        head,
        sep,
    ]
    for zh, _dataset, _seeds in CELLS:
        lines.append(line(zh, by_ds[zh]))
    lines.append(line("合计", rows))
    lines += ["", "## 2. 按窗档", "", head, sep]
    for kind, kind_zh in (("high", "高把握"), ("mix", "混合"), ("low", "低把握")):
        lines.append(line(f"合计·{kind_zh}", [x for x in rows if x["kind"] == kind]))
        for zh, _dataset, _seeds in CELLS:
            part = [x for x in by_ds[zh] if x["kind"] == kind]
            if part:
                lines.append(line(f"{zh}·{kind_zh}", part))
    lines += [
        "",
        f"接到分的窗 {len(rows)}。滑动窗：长平台会重复计。",
        "",
        "## 3. 读法",
        "",
        "先看合计，再看低把握。只在高把握上高、低把握掉到 0.6，还是锁在撑，不是新特征。",
        "没有一条明显高于试答把握，就不算分开了窗。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
