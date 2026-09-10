#!/usr/bin/env python3
"""不看 GPQA：密探每步探 vs DEER 只在 Wait 探，剩窗还在不在、把握能不能分。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_deer_official_plus as deer

TABLE = AE / "tables/wait_schedule_nongpqa.md"
WAIT = AE / "results/leftover_waithelp"
SKIP_DS = {"gpqa-diamond"}
DEER_JOBS = (
    ("7B MATH", "math500_ds7b", "math-500"),
    ("7B 奥赛", "olympiadbench_ds7b", "olympiadbench"),
    ("7B AIME24", "aime24_ds7b", "aime24"),
    ("7B AIME25", "aime25_ds7b", "aime25"),
)
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
    "aime24": "AIME24",
    "aime25": "AIME25",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell_name(row: dict[str, Any]) -> str:
    return f"{MODEL_ZH[row['model']]} {DS_ZH[row['dataset']]}"


def auroc_conf(pairs: list[tuple[float, bool]]) -> float:
    pos = [c for c, y in pairs if y]
    neg = [c for c, y in pairs if not y]
    return rg.auroc(pos, neg)


def wait_windows(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for end in range(3, len(checks)):
        window = checks[end - 3 : end + 1]
        ans = window[-1].get("trial_answer")
        if not ans or not all(rg.same(ans, x.get("trial_answer")) for x in window):
            continue
        confs = [rg.finite(x.get("conf")) for x in window]
        if any(c != c for c in confs):
            continue
        kind = "high" if rg.is_high(confs) else ("low" if rg.is_low(confs) else "mix")
        out.append({"end": end, "ans": ans, "c": confs[-1], "kind": kind, "confs": confs})
    return out


def gold_map(dataset: str) -> dict[int, Any]:
    path = deer.puma_stat(dataset)
    rows = json.loads(path.read_text()) if path.is_file() else []
    return {int(r["question_idx"]): r.get("ground_truth") for r in rows}


def deer_cell(name: str, tag: str, dataset: str) -> dict[str, Any]:
    raw = json.loads((deer.FIG / tag / "data/per_sample.json").read_text())
    gold = gold_map(dataset)
    for row in raw:
        row["ground_truth"] = gold.get(int(row["idx"]) + 1)
    hosts = [deer.host_stop(r) for r in raw]
    n = len(hosts)
    first_pairs = []
    all_pairs = []
    left_pairs = []
    n_left = n_left_ok = n_will = n_high_first = n_wait4 = 0
    for h in hosts:
        checks = h["checks"]
        gt = h.get("gt")
        for ch in checks:
            ok = bool(gt) and rg.same(ch.get("trial_answer"), gt)
            c = rg.finite(ch.get("conf"))
            if c == c:
                all_pairs.append((c, ok))
        if checks:
            c0 = rg.finite(checks[0].get("conf"))
            ok0 = bool(gt) and rg.same(checks[0].get("trial_answer"), gt)
            if c0 == c0:
                first_pairs.append((c0, ok0))
                if c0 > 0.95:
                    n_high_first += 1
        if len(checks) >= 4:
            n_wait4 += 1
        wins = wait_windows(checks)
        left = next((w for w in wins if w["kind"] != "high"), None)
        if left is None:
            continue
        n_left += 1
        ok = bool(gt) and rg.same(left["ans"], gt)
        n_left_ok += int(ok)
        left_pairs.append((left["c"], ok))
        later = next((w for w in wins if w["kind"] == "high" and w["end"] > left["end"]), None)
        if later is not None and not rg.same(left["ans"], later["ans"]):
            n_will += 1
    return {
        "name": name,
        "n": n,
        "early": sum(int(h["early"]) for h in hosts),
        "n_high_first": n_high_first,
        "n_wait4": n_wait4,
        "n_left": n_left,
        "n_left_ok": n_left_ok,
        "n_will": n_will,
        "auroc_all": auroc_conf(all_pairs),
        "auroc_first": auroc_conf(first_pairs),
        "auroc_left": auroc_conf(left_pairs),
        "n_all": len(all_pairs),
        "ok_all": sum(1 for _, y in all_pairs if y),
        "n_first": len(first_pairs),
        "ok_first": sum(1 for _, y in first_pairs if y),
    }


def dense_rows() -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(WAIT.glob("*.jsonl")):
        if path.stem not in MODEL_ZH:
            continue
        for row in load_jsonl(path):
            if row["dataset"] in SKIP_DS:
                continue
            by[cell_name(row)].append(row)
    return by


def main() -> None:
    deer_rows = [deer_cell(name, tag, ds) for name, tag, ds in DEER_JOBS]
    dense = dense_rows()
    names = sorted(dense)
    lines = [
        "# 不看 GPQA：密探每步探 vs 只在 Wait 探",
        "",
        "7B 的 Wait 日程来自官方 DEER 转储（最多 10 次 Wait 试答）。密探剩窗来自已导出 jsonl，去掉 GPQA。",
        "Wait 剩窗 = Wait 探点上连续 4 次同答、且还不是高把握锁。",
        "",
        "## 1. 7B：只在 Wait 探",
        "",
        "| 集 | 题 | 第一次 Wait>0.95 | 能凑齐 4 次 Wait | Wait 剩窗 | 其中试答对 | 后面 Wait 高把握换答 | 全部 Wait 把握 AUROC | 第一次 Wait AUROC | Wait 剩窗把握 AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in deer_rows:
        lines.append(
            f"| {r['name']} | {r['n']} | {r['n_high_first']}/{r['n']} | {r['n_wait4']}/{r['n']} | "
            f"{r['n_left']} | {r['n_left_ok']}（{100.0 * r['n_left_ok'] / r['n_left']:.0f}%） | {r['n_will']} | "
            f"{r['auroc_all']:.2f}（{r['ok_all']}/{r['n_all']}） | "
            f"{r['auroc_first']:.2f}（{r['ok_first']}/{r['n_first']}） | "
            f"{r['auroc_left']:.2f} |"
        )
    lines += [
        "",
        "## 2. 密探每步探：剩窗（无 GPQA）",
        "",
        "| 集 | 剩窗 | 试答对 | 再等会更好 | 后面高把握换答 | 把握 AUROC（试答对） | 把握 AUROC（再等会更好） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_d: list[dict[str, Any]] = []
    for name in names:
        xs = dense[name]
        all_d.extend(xs)
        ok = [x for x in xs if x["leftover_ok"]]
        helps = [x for x in xs if x["wait_helps"]]
        will = [x for x in xs if x.get("high_later") and not x.get("same_as_high")]
        a_ok = rg.auroc(
            [float(x["confidence"]) for x in ok],
            [float(x["confidence"]) for x in xs if not x["leftover_ok"]],
        )
        a_w = rg.auroc(
            [float(x["confidence"]) for x in helps],
            [float(x["confidence"]) for x in xs if not x["wait_helps"]],
        )
        lines.append(
            f"| {name} | {len(xs)} | {len(ok)}（{100.0 * len(ok) / len(xs):.0f}%） | "
            f"{len(helps)}（{100.0 * len(helps) / len(xs):.0f}%） | {len(will)} | "
            f"{a_ok:.2f} | {a_w:.2f} |"
        )
    ok = [x for x in all_d if x["leftover_ok"]]
    helps = [x for x in all_d if x["wait_helps"]]
    will = [x for x in all_d if x.get("high_later") and not x.get("same_as_high")]
    a_ok = rg.auroc(
        [float(x["confidence"]) for x in ok],
        [float(x["confidence"]) for x in all_d if not x["leftover_ok"]],
    )
    a_w = rg.auroc(
        [float(x["confidence"]) for x in helps],
        [float(x["confidence"]) for x in all_d if not x["wait_helps"]],
    )
    lines.append(
        f"| 全体无 GPQA | {len(all_d)} | {len(ok)}（{100.0 * len(ok) / len(all_d):.0f}%） | "
        f"{len(helps)}（{100.0 * len(helps) / len(all_d):.0f}%） | {len(will)} | "
        f"{a_ok:.2f} | {a_w:.2f} |"
    )
    lines += [
        "",
        "读法：若少探能洗干净，Wait 剩窗应接近 0，或剩窗把握 AUROC 明显高于密探。",
        "MATH 第一次 Wait 大多已经 >0.95，剩窗少。奥赛 / AIME 仍能凑出 Wait 剩窗，把握区分度不会比密探更好。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)
    for r in deer_rows:
        print(
            f"{r['name']} wait_left={r['n_left']} ok={r['n_left_ok']} will={r['n_will']} "
            f"auroc_all={r['auroc_all']:.2f} first={r['auroc_first']:.2f} left={r['auroc_left']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    rg.K = 4
    rg.TAU = 0.995
    main()
