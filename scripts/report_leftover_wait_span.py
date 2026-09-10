#!/usr/bin/env python3
"""锁后到换答之间有没有先写 Wait。方向1能不能压 Wait 的离线闸门。零 GPU。"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_first_lock_room as room
import report_k4_hyps as hy
import report_k4_second_lock as sl
import report_leftover_after as after

TABLE = AE / "tables/leftover_wait_span.md"
CONTROL_STEPS = 2
DS_ZH = {**after.DS_ZH, "math-500": "MATH"}

# 压 Wait 时会动的词。不把句中 But / However 算进去，数学推导里太常见。
WAIT_RE = re.compile(r"\bwait\b|等一下", re.I)
ALT_RE = re.compile(
    r"\balternatively\b|another (way|approach|method)|换一种|换个(?:思路|方法)",
    re.I,
)
HMM_RE = re.compile(r"\bhmm+\b|\bhuh\b", re.I)
START_RE = re.compile(
    r"^(?:wait|alternatively|hmm+|huh|等一下|换一种)\b",
    re.I,
)
PARA_START_RE = re.compile(
    r"(?:^|(?<=\n))[ \t]*(?:wait|alternatively|hmm+|huh|等一下|换一种)\b",
    re.I,
)


def prefix_of(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("reasoning_prefix") or "")


def span_of(old: str, new: str) -> str | None:
    if not old or not new:
        return None
    if new.startswith(old):
        return new[len(old) :]
    head = old[:80]
    if not head or not new.startswith(head):
        return None
    i = 0
    for a, b in zip(old, new):
        if a != b:
            break
        i += 1
    return new[i:]


def hits(text: str, cre: re.Pattern[str]) -> int:
    return len(cre.findall(text)) if text else 0


def pack_span(text: str | None) -> dict[str, Any] | None:
    if text is None:
        return None
    stripped = text.lstrip(" \n\t-*#")
    n_wait = hits(text, WAIT_RE)
    n_alt = hits(text, ALT_RE)
    n_hmm = hits(text, HMM_RE)
    n_any = n_wait + n_alt + n_hmm
    last = text[-80:] if text else ""
    return {
        "n_char": len(text),
        "n_wait": n_wait,
        "n_alt": n_alt,
        "n_hmm": n_hmm,
        "n_any": n_any,
        "has_wait": n_wait > 0,
        "has_any": n_any > 0,
        "starts": bool(START_RE.match(stripped)),
        "para": bool(PARA_START_RE.search(text)),
        "tail_wait": hits(last, WAIT_RE) > 0,
    }


def row_at(after_rows: list[dict[str, Any]], pred) -> dict[str, Any] | None:
    for row in after_rows:
        if pred(row):
            return row
    return None


def same_after(after_rows: list[dict[str, Any]], ans: Any, n: int) -> dict[str, Any] | None:
    seen = 0
    last = None
    for row in after_rows:
        if not rg.same(row.get("final_answer"), ans):
            return None
        seen += 1
        last = row
        if seen >= n:
            return last
    return None


def attach_spans(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    xs = after.load_left(model, dataset, seed)
    if not xs:
        return []
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not trials_path.is_file():
        return []
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    out = []
    for rec in xs:
        trials = by.get(int(rec["question_idx"]))
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        if left is None:
            continue
        left_row = rows[left["end"]]
        leftover_prefix = prefix_of(left_row)
        after_rows = rows[left["end"] + 1 :]
        if not leftover_prefix or not after_rows:
            rec = dict(rec)
            rec["align"] = False
            out.append(rec)
            continue
        nxt = after_rows[0]
        brk = row_at(
            after_rows,
            lambda row, ans=left["ans"]: not rg.same(row.get("final_answer"), ans),
        )
        ctrl = same_after(after_rows, left["ans"], CONTROL_STEPS)
        next_span = span_of(leftover_prefix, prefix_of(nxt))
        break_span = span_of(leftover_prefix, prefix_of(brk)) if brk is not None else None
        ctrl_span = span_of(leftover_prefix, prefix_of(ctrl)) if ctrl is not None else None
        packed = {
            **rec,
            "align": next_span is not None,
            "next": pack_span(next_span),
            "brk": pack_span(break_span),
            "ctrl": pack_span(ctrl_span),
        }
        out.append(packed)
    return out


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "—"
    return f"{100.0 * n / d:.0f}%"


def line(name: str, xs: list[dict[str, Any]], key: str) -> str:
    got = [x[key] for x in xs if x.get(key)]
    n = len(got)
    if not n:
        return f"| {name} | {len(xs)} | 0 | — | — | — | — | — | — |"
    return (
        f"| {name} | {len(xs)} | {n} | "
        f"{pct(sum(int(s['starts']) for s in got), n)} | "
        f"{pct(sum(int(s['para']) for s in got), n)} | "
        f"{pct(sum(int(s['has_wait']) for s in got), n)} | "
        f"{pct(sum(int(s['has_any']) for s in got), n)} | "
        f"{mean([s['n_wait'] for s in got]):.2f} | "
        f"{mean([s['n_char'] for s in got]):.0f} |"
    )


GROUPS = (
    ("误杀，后面换答", lambda x: x["wait_helps"] and x["will_change"]),
    ("误杀，写完才对", lambda x: x["wait_helps"] and x["never_high"]),
    ("已对，后面同答锁", lambda x: x["left_ok"] and x["same_as_high"]),
    ("已对，写完才停", lambda x: x["left_ok"] and x["never_high"]),
    ("两边都错，后面换答", lambda x: (not x["left_ok"]) and (not x["wait_helps"]) and x["will_change"]),
    ("全体对齐", lambda x: bool(x.get("align"))),
)


def rate(xs: list[dict[str, Any]], key: str, field: str) -> tuple[int, int]:
    got = [x[key] for x in xs if x.get(key)]
    return sum(int(s[field]) for s in got), len(got)


def verdict(rows: list[dict[str, Any]]) -> list[str]:
    chg = [x for x in rows if x["wait_helps"] and x["will_change"] and x.get("brk")]
    ok = [x for x in rows if x["left_ok"] and x["same_as_high"] and x.get("ctrl")]
    chg_start, n_chg_next = rate(
        [x for x in rows if x["wait_helps"] and x["will_change"]], "next", "starts"
    )
    ok_start, n_ok_next = rate(
        [x for x in rows if x["left_ok"] and x["same_as_high"]], "next", "starts"
    )
    chg_wait, n_chg = rate(chg, "brk", "has_wait")
    chg_tail, _ = rate(chg, "brk", "tail_wait")
    ok_wait, n_ok = rate(ok, "ctrl", "has_wait")
    lines = [
        "## 4. 读法",
        "",
        f"误杀换答、锁后下一步以 Wait / Alternatively 开头：{pct(chg_start, n_chg_next)}（{chg_start}/{n_chg_next}）。",
        f"已对同答锁、锁后下一步同样开头：{pct(ok_start, n_ok_next)}（{ok_start}/{n_ok_next}）。",
        f"误杀换答、锁到第一次换答正文里出现过 Wait：{pct(chg_wait, n_chg)}（{chg_wait}/{n_chg}）；"
        f"换答前 80 字里有 Wait：{pct(chg_tail, n_chg)}（{chg_tail}/{n_chg}）。",
        f"已对同答锁、锁后再走 {CONTROL_STEPS} 步仍同答的等长正文里出现过 Wait："
        f"{pct(ok_wait, n_ok)}（{ok_wait}/{n_ok}）。",
        "",
    ]
    start_chg = (chg_start / n_chg_next) if n_chg_next else 0.0
    start_ok = (ok_start / n_ok_next) if n_ok_next else 0.0
    wait_chg = (chg_wait / n_chg) if n_chg else 0.0
    wait_ok = (ok_wait / n_ok) if n_ok else 0.0
    tail_chg = (chg_tail / n_chg) if n_chg else 0.0
    if n_chg < 30 or n_chg_next < 30:
        lines.append("对齐样本不够，不当闸门。")
    elif start_chg >= start_ok + 0.10 and tail_chg >= 0.25:
        lines.append(
            "方向 1 关。误杀换答更常先写 Wait，换答当步前也常有 Wait，压 Wait 会挡住改口通道。"
        )
    elif start_chg <= start_ok + 0.03 and tail_chg <= 0.15:
        lines.append(
            "insight 站住：Wait 更常出现在已对验算上，换答当步前几乎没有 Wait。 "
            "可以做在线压 Wait，但不能靠这张表宣布 Acc 安全——"
            f"仍有 {pct(chg_wait, n_chg)} 的误杀换答路径上中途出现过 Wait。"
        )
    else:
        lines.append(
            "方向 1 先不要灌全量。两摊 Wait 差不够大，压了会不会挡改口，离线表分不清。"
        )
    lines.append("")
    return lines


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rows: list[dict[str, Any]] = []
    ds = (("MATH", "math-500"),) + after.DS
    for _zh, model in after.MODELS:
        for _ds_zh, dataset in ds:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                part = attach_spans(model, dataset, seed)
                rows.extend(part)
                n_al = sum(1 for x in part if x.get("align"))
                print(f"{model} {dataset} s{seed} leftover={len(part)} align={n_al}", flush=True)
    aligned = [x for x in rows if x.get("align")]
    lines = [
        "# 锁后到换答：中间有没有先写 Wait",
        "",
        "第一扇非高把握连答窗之后，看新写下的思路正文，不训、不灌 logit。",
        "Wait 只数 `Wait` / `等一下`。反思词再加 `Alternatively` / `Hmm` / `换一种`。句中 But 不算。",
        "下一步：锁后到下一次试答之间新写下的字。",
        "锁到换答：锁后到第一次不同试答之间新写下的字。",
        f"等长对照：锁后再走 {CONTROL_STEPS} 步仍是同一答时新写下的字（换答误杀中位间隔大约 2 步）。",
        "以 Wait 开头：这段去掉空白后第一个词就是 Wait / Alternatively / Hmm。",
        "段首是 Wait：这段里任意一行以这些词起头。",
        "对不上：后一步推理正文接不上锁点前缀，这类题不进比例。",
        "",
        "## 1. 锁后下一步新思路",
        "",
        "| 切片 | 题 | 对上 | 下一步以 Wait 开头 | 段首是 Wait | 这段有 Wait | 这段有反思词 | 平均 Wait 次数 | 新写字数 |",
        "|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for name, pred in GROUPS:
        lines.append(line(name, [x for x in rows if pred(x)], "next"))

    lines += [
        "",
        "## 2. 锁到第一次换答（只看后来改口的题）",
        "",
        "| 切片 | 题 | 对上 | 整段以 Wait 开头 | 段首是 Wait | 这段有 Wait | 这段有反思词 | 平均 Wait 次数 | 新写字数 |",
        "|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for name, pred in GROUPS:
        if name in ("已对，后面同答锁", "已对，写完才停", "全体对齐"):
            continue
        lines.append(line(name, [x for x in rows if pred(x)], "brk"))

    lines += [
        "",
        f"## 3. 锁后再走 {CONTROL_STEPS} 步仍同答（等长对照）",
        "",
        "| 切片 | 题 | 对上 | 整段以 Wait 开头 | 段首是 Wait | 这段有 Wait | 这段有反思词 | 平均 Wait 次数 | 新写字数 |",
        "|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for name, pred in GROUPS:
        lines.append(line(name, [x for x in rows if pred(x)], "ctrl"))

    lines += [
        "",
        "## 3b. 按数据集：误杀换答 vs 已对同答锁",
        "",
        "| 集 | 误杀换答 下一步开头 | 误杀换答 锁到换答有 Wait | 已对同答锁 下一步开头 | 已对同答锁 后2步有 Wait |",
        "|---|---|---|---|---|",
    ]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in rows:
        by_cell[f"{x['model']}\t{x['dataset']}"].append(x)
    for key in sorted(by_cell):
        model, ds_name = key.split("\t")
        xs = by_cell[key]
        chg = [x for x in xs if x["wait_helps"] and x["will_change"]]
        ok = [x for x in xs if x["left_ok"] and x["same_as_high"]]
        a, na = rate(chg, "next", "starts")
        b, nb = rate([x for x in chg if x.get("brk")], "brk", "has_wait")
        c, nc = rate(ok, "next", "starts")
        d, nd = rate([x for x in ok if x.get("ctrl")], "ctrl", "has_wait")
        lines.append(
            f"| {after.MODEL_ZH.get(model, model)} {DS_ZH.get(ds_name, ds_name)} | "
            f"{pct(a, na)}（{a}/{na}） | {pct(b, nb)}（{b}/{nb}） | "
            f"{pct(c, nc)}（{c}/{nc}） | {pct(d, nd)}（{d}/{nd}） |"
        )

    lines += [""] + verdict(rows)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE} leftover={len(rows)} align={len(aligned)}", flush=True)


if __name__ == "__main__":
    main()
