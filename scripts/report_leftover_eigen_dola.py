#!/usr/bin/env python3
"""剩窗 EigenScore / 谱熵 / DoLA JSD，正类 wait_helps。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
from report_leftover_waithelp import cell_name, fmt_auroc

OUT = AE / "tables/leftover_eigen_dola.md"
SCORE = AE / "results/leftover_eigen_dola"
SIGS = (
    ("confidence", "把握"),
    ("eigen_pre_k4", "思路末 eigen k4"),
    ("eigen_pre_k8", "思路末 eigen k8"),
    ("eigen_pre_k16", "思路末 eigen k16"),
    ("h_pre_k8", "思路末谱熵 k8"),
    ("eigen_box_k8", "boxed 末 eigen k8"),
    ("h_box_k8", "boxed 末谱熵 k8"),
    ("mid_eigen_pre_k8", "一半深 eigen k8"),
    ("dola_jsd", "末层−一半深 JSD"),
    ("dola_kl_lh", "KL 末‖半"),
    ("dola_kl_hl", "KL 半‖末"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n_ok = n_bad = 0
    for path in sorted(SCORE.glob("*_shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                n_bad += 1
                continue
            n_ok += 1
            by[cell_name(row)].append(row)
    lines = [
        "# 剩窗：EigenScore / 谱熵 / DoLA 层间 KL",
        "",
        "免训。正类 wait_helps = 交剩窗会错、k4 无后路再等是对的。",
        "EigenScore / 谱熵：思路末或 boxed 末近 K 个 last-layer hidden。",
        "DoLA：思路最后一个 token 上，末层 vs 一半深的下一词分布 JSD / KL。",
        f"打成 {n_ok}，失败 {n_bad}。",
        "",
        "| 集 | 必须等/可停 | 把握 | eigen k4 | eigen k8 | eigen k16 | 谱熵 k8 | boxed eigen | boxed 谱熵 | 一半深 eigen | JSD | KL末‖半 | KL半‖末 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(by):
        xs = by[name]
        pos = sum(1 for x in xs if x.get("wait_helps"))
        cells = [
            fmt_auroc(rg.auroc(
                [float(x[f]) for x in xs if x.get("wait_helps") and x.get(f) == x.get(f)],
                [float(x[f]) for x in xs if (not x.get("wait_helps")) and x.get(f) == x.get(f)],
            ))
            for f, _ in SIGS
        ]
        lines.append(f"| {name} | {pos}/{len(xs) - pos} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "读法：要通用，各集都得明显高于 0.50。贴 0.5 就停，不训探针。",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} cells={len(by)} ok={n_ok} bad={n_bad}", flush=True)


if __name__ == "__main__":
    main()
