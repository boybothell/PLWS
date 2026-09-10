#!/usr/bin/env python3
"""剩窗最后一搏：CoDE-Stop 退化分 / 早段把握，REFRAIN 反思词。免训，零 GPU。"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_first_lock_room as room

WAIT = AE / "results/leftover_waithelp"
TABLE = AE / "tables/leftover_paper_last.md"
DELTA = 0.55
MODEL_ZH = {
    "r1_7b": "7B",
    "nemotron_8b": "8B",
    "r1_14b": "14B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}
DS_ZH = {
    "math-500": "MATH",
    "olympiadbench": "奥赛",
    "gpqa-diamond": "GPQA",
    "aime24": "AIME24",
    "aime25": "AIME25",
}
# REFRAIN 四类触发词（论文 + 常见同义，不扩新类）
CUES = (
    r"\bwait\b",
    r"double[- ]?check",
    r"let me (check|verify|recheck)",
    r"\bverify\b",
    r"make sure",
    r"\balternatively\b",
    r"another (way|approach|method)",
    r"\binstead\b",
    r"let(?:'s| us) try",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"not sure",
    r"\bmight be\b",
    r"\bactually\b",
    r"\bhowever\b",
    r"i was wrong",
    r"wait,? no",
    r"不对",
    r"等一下",
    r"换一种",
)
CUE_RE = re.compile("|".join(CUES), re.I)
WAIT_RE = re.compile(r"\bwait\b|等一下", re.I)
ANS_CUE_RE = re.compile(r"answer (is|should be)|final answer|\\boxed", re.I)
FEATS = (
    ("code_D", "CoDE 退化分"),
    ("code_vfrac", "不稳步占比"),
    ("early_c", "前20%把握"),
    ("max_c", "迄今最高把握"),
    ("rel_c", "当前/最高"),
    ("gap_c", "最高−当前"),
    ("is_return", "锁回旧答"),
    ("early_same", "早段已出此答"),
    ("reflect_n", "近文反思词"),
    ("wait_n", "Wait 次数"),
    ("has_ans_cue", "已出现答句"),
    ("confidence", "窗末把握"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell_name(row: dict[str, Any]) -> str:
    return f"{MODEL_ZH[row['model']]} {DS_ZH[row['dataset']]}"


def fmt_auroc(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def degeneration(confs: list[float], times: list[float]) -> tuple[float, float]:
    if len(confs) < 2:
        return float("nan"), float("nan")
    tk = max(times[-1], 1.0)
    d = 0.0
    n_v = 0
    for i in range(1, len(confs)):
        v = 1.0 if (2.0 * confs[i] - confs[i - 1] < DELTA) else 0.0
        w = math.log(tk / max(times[i], 1.0)) + 1.0
        d += w * v
        n_v += v
    return d, n_v / (len(confs) - 1)


def feats(hist: list[dict[str, Any]], left: dict[str, Any]) -> dict[str, float]:
    confs = [float(r["confidence"]) for r in hist]
    times = [float(r.get("count_reasoning_tokens") or r["stopped_len"]) for r in hist]
    k = len(confs)
    early_n = max(1, k // 5)
    early = hist[:early_n]
    ans = left["answer"]
    appeared = [i for i, r in enumerate(hist) if rg.same(r.get("final_answer"), ans)]
    run_start = appeared[0] if appeared else 0
    for i in reversed(range(len(hist))):
        if not rg.same(hist[i].get("final_answer"), ans):
            run_start = i + 1
            break
    is_return = 1.0 if any(rg.same(r.get("final_answer"), ans) for r in hist[:run_start]) else 0.0
    early_same = 1.0 if any(rg.same(r.get("final_answer"), ans) for r in early) else 0.0
    d, vfrac = degeneration(confs, times)
    max_c = max(confs)
    cur = float(left["confidence"])
    prefix = str(left.get("reasoning_prefix") or "")
    tail = prefix[-1200:]
    head = prefix[:-200] if len(prefix) > 200 else ""
    return {
        "code_D": d,
        "code_vfrac": vfrac,
        "early_c": sum(float(r["confidence"]) for r in early) / len(early),
        "max_c": max_c,
        "rel_c": cur / max_c if max_c > 0 else float("nan"),
        "gap_c": max_c - cur,
        "is_return": is_return,
        "early_same": early_same,
        "reflect_n": float(len(CUE_RE.findall(tail))),
        "wait_n": float(len(WAIT_RE.findall(prefix))),
        "has_ans_cue": 1.0 if ANS_CUE_RE.search(head) else 0.0,
        "confidence": cur,
        "leftover_ok": bool(left["leftover_ok"]),
        "wait_helps": bool(left["wait_helps"]),
        "will_change": bool(left.get("high_later") and not left.get("same_as_high")),
        "kind": left["kind"],
        "model": left["model"],
        "dataset": left["dataset"],
    }


def collect() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(WAIT.glob("*.jsonl")):
        if path.stem not in MODEL_ZH:
            continue
        model = path.stem
        leftover = load_jsonl(path)
        by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in leftover:
            by_cell[(row["dataset"], int(row["seed"]))].append(row)
        for (dataset, seed), jobs in by_cell.items():
            trials_path = room.dense_trial_path(model, dataset, seed)
            if not trials_path.is_file():
                continue
            by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in dd.load_json(trials_path):
                by_q[int(row["question_idx"])].append(row)
            n = 0
            for job in jobs:
                rows = rg.usable_rows(by_q.get(int(job["question_idx"])) or [])
                hist = [r for r in rows if int(r["stopped_len"]) <= int(job["decision_step"])]
                if len(hist) < 2:
                    continue
                out.append(feats(hist, job))
                n += 1
            print(f"{model} {dataset} seed={seed} n={n}", flush=True)
    return out


def auroc_of(xs: list[dict[str, Any]], y: str, feat: str) -> float:
    return rg.auroc([x[feat] for x in xs if x[y]], [x[feat] for x in xs if not x[y]])


def auroc_row(name: str, xs: list[dict[str, Any]], y: str) -> str:
    npos = sum(1 for x in xs if x[y])
    nneg = len(xs) - npos
    if npos == 0 or nneg == 0:
        return f"| {name} | {npos}/{nneg} | " + " | ".join("—" for _ in FEATS) + " |"
    cells = [fmt_auroc(auroc_of(xs, y, f)) for f, _ in FEATS]
    return f"| {name} | {npos}/{nneg} | " + " | ".join(cells) + " |"


def main() -> None:
    rows = collect()
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[cell_name(row)].append(row)
    names = sorted(by)
    low = [x for x in rows if x["kind"] == "low"]
    lines = [
        "# 剩窗最后一搏：论文信号 × 现成轨迹",
        "",
        "只看第一扇剩窗。零 GPU。正类见各节。",
        "",
        "- **CoDE-Stop**（Hosseini et al. 2026）：`v_i=1(2c_i−c_{i-1}<0.55)`，早步权重大，`D=Σ w_i v_i`。另报前 20% 把握、当前/迄今最高。",
        "- **REFRAIN**（ACL 2026）：近 1200 字反思触发词次数、全文 Wait、此前是否已有答句。不做句向量。",
        "- 对照：窗末把握。试答集合熵已在 `asag_like_leftover` 死过，不重算。",
        "",
        "表头：退化分 / 不稳占比 / 前20%把握 / 迄今最高 / 当前÷最高 / 最高−当前 / 锁回旧答 / 早段已出此答 / 近文反思 / Wait次数 / 已有答句 / 窗末把握。",
        "",
        "## leftover_ok（这扇对不对）",
        "",
        "| 集 | 对/错 | 退化分 | 不稳占比 | 前20%把握 | 迄今最高 | 当前/最高 | 最高−当前 | 锁回 | 早段已出 | 反思词 | Wait | 已有答句 | 窗末把握 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        lines.append(auroc_row(name, by[name], "leftover_ok"))
    lines.append(auroc_row("全体", rows, "leftover_ok"))
    lines.append(auroc_row("全体低把握", low, "leftover_ok"))
    lines += [
        "",
        "## wait_helps（再等会更好 = 不该停）",
        "",
        "| 集 | 必须等/其余 | 退化分 | 不稳占比 | 前20%把握 | 迄今最高 | 当前/最高 | 最高−当前 | 锁回 | 早段已出 | 反思词 | Wait | 已有答句 | 窗末把握 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        lines.append(auroc_row(name, by[name], "wait_helps"))
    lines.append(auroc_row("全体", rows, "wait_helps"))
    lines.append(auroc_row("全体低把握", low, "wait_helps"))
    lines += [
        "",
        "## will_change（后面高把握换答）",
        "",
        "| 集 | 会换/其余 | 退化分 | 不稳占比 | 前20%把握 | 迄今最高 | 当前/最高 | 最高−当前 | 锁回 | 早段已出 | 反思词 | Wait | 已有答句 | 窗末把握 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        lines.append(auroc_row(name, by[name], "will_change"))
    lines.append(auroc_row("全体", rows, "will_change"))
    lines.append(auroc_row("全体低把握", low, "will_change"))
    lines += [
        "",
        "读法：新信号要在各集都明显高于 0.50 且压过窗末把握，才谈门。贴 0.5 或只在单集亮，停。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
