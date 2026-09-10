#!/usr/bin/env python3
"""Feature snapshot of G/R/W and low-conf consecutive-wrong (L)."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
_WS = re.compile(r"\s+")
_LATEX = re.compile(r"\\(left|right|cdot|times|dfrac|tfrac|frac|mathrm|text)")
K = 4
TAU = 0.98


def norm(text):
    s = _WS.sub("", str(text or "").strip().lower())
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = _LATEX.sub("", s)
    return s.replace("\\", "")


def same(a, b):
    x, y = norm(a), norm(b)
    return bool(x) and x == y


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def p50(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[len(xs) // 2]


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def auroc(pos, neg):
    pos = [x for x in pos if x == x]
    neg = [x for x in neg if x == x]
    if not pos or not neg:
        return float("nan")
    better = tie = 0
    for a in pos:
        for b in neg:
            if a > b:
                better += 1
            elif a == b:
                tie += 1
    return (better + 0.5 * tie) / (len(pos) * len(neg))


def sem_type(text):
    t = (text or "").lower()
    if any(p in t for p in ("therefore", "so the answer", "final answer", "in conclusion", "thus", "hence")):
        return "conclude"
    if any(p in t for p in ("let me check", "double-check", "verify", "confirm", "seems correct")):
        return "verify"
    if any(p in t for p in ("wait", "however", "let me reconsider", "or maybe", "i'm not sure")):
        return "doubt"
    if any(p in t for p in ("calculate", "compute", " = ", "multiply", "total", "sum")):
        return "calculate"
    return "other"


def load_internal(dataset: str) -> dict:
    folder = AE / f"results/confcal_judge/v2/dense_internal/{dataset}"
    out = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") and row.get("status") != "ok":
                continue
            key = (int(row["question_idx"]), int(row.get("decision_step") or row.get("stopped_len") or 0))
            out[key] = row
    return out


def load_cell(name, trial, stat, gpath, fs):
    trials = json.loads(Path(trial).read_text())
    official = {int(r["question_idx"]): r for r in json.loads(Path(stat).read_text())}
    gmap = {int(r["question_idx"]): r for r in json.loads(Path(gpath).read_text())}
    by = defaultdict(list)
    for row in trials:
        by[int(row["question_idx"])].append(row)
    for qi in by:
        by[qi].sort(key=lambda x: int(x["stopped_len"]))
    steps_by = {}
    for i, row in enumerate(json.loads(Path(fs).read_text()), 1):
        steps_by[i] = row.get("reasoning_steps") or []
    return name, by, official, gmap, steps_by


CELLS = [
    (
        "MATH",
        AE / "results/dense_G_r1_7b/math-500/dense_puma/trial_answers.json",
        AE / "results/math500_official/puma_ds7b/statistics.json",
        AE / "results/dense_G_r1_7b/math-500/per_sample.json",
        AE / "results/math500_official/puma_ds7b/filtered_steps.json",
        "math-500",
    ),
    (
        "OLY",
        AE / "results/dense_G_r1_7b/olympiadbench/dense_puma/trial_answers.json",
        AE / "results/puma_offline_r1_7b/olympiadbench/statistics.json",
        AE / "results/dense_G_r1_7b/olympiadbench/per_sample.json",
        AE / "results/puma_offline_r1_7b/olympiadbench/filtered_steps.json",
        None,
    ),
    (
        "GPQA",
        AE / "results/dense_G_r1_7b/gpqa-diamond/dense_puma/trial_answers.json",
        AE / "results/puma_offline_r1_7b/gpqa-diamond/statistics.json",
        AE / "results/dense_G_r1_7b/gpqa-diamond/per_sample.json",
        AE / "results/puma_offline_r1_7b/gpqa-diamond/filtered_steps.json",
        None,
    ),
]


def main() -> None:
    for name, trial, stat, gpath, fs, internal_ds in CELLS:
        by, official, gmap, steps_by = load_cell(name, trial, stat, gpath, fs)[1:]
        internal = load_internal(internal_ds) if internal_ds else {}
        bags = defaultdict(lambda: defaultdict(list))
        first_win = defaultdict(list)
        q_r_only = q_r_then_g = q_r = 0
        nq = 0
        for qi, trials in by.items():
            info = official.get(qi)
            gg = gmap.get(qi)
            if not info or not gg:
                continue
            nq += 1
            af = gg.get("A_final") or info.get("original_answer")
            gt = info.get("ground_truth")
            steps = steps_by.get(qi) or []
            nst = int(trials[-1]["stopped_len"])
            prev_text = ""
            rows = []
            for row in trials:
                st = int(row["stopped_len"])
                ans = str(row.get("final_answer") or "")
                conf = finite(row.get("confidence"))
                inc = steps[st - 1] if 0 < st <= len(steps) else ""
                ok = bool(ans) and (same(ans, af) or same(ans, gt))
                typ = sem_type(inc)
                in_inc = bool(norm(ans)) and (norm(ans) in norm(inc) or ans in inc)
                in_prev = bool(norm(ans)) and norm(ans) in norm(prev_text)
                wait = bool(re.search(r"\bWait\b", inc))
                prev_len = len(steps[st - 2]) if st >= 2 and st - 1 <= len(steps) else float("nan")
                ratio = (len(inc) / prev_len) if prev_len and prev_len == prev_len and prev_len > 0 else float("nan")
                inn = internal.get((qi, st), {})
                rows.append(
                    {
                        "st": st,
                        "ans": ans,
                        "conf": conf,
                        "ok": ok,
                        "inc_len": len(inc),
                        "ratio": ratio,
                        "in_inc": float(in_inc),
                        "in_prev": float(in_prev),
                        "wait": float(wait),
                        "conclude": float(typ == "conclude"),
                        "verify": float(typ == "verify"),
                        "doubt": float(typ == "doubt"),
                        "ans_len": len(ans),
                        "tok": float(row.get("count_reasoning_tokens") or float("nan")),
                        "frac": st / max(nst, 1),
                        "left": nst - st,
                        "stop_margin": finite(inn.get("stop_margin")),
                        "dola_rise": finite(inn.get("dola_mean_rise")),
                        "lookback": finite(inn.get("lookback_ratio")),
                    }
                )
                prev_text += "\n" + inc

            labels = ["O"] * len(rows)
            seen_ans = []
            for i, row in enumerate(rows):
                seen_ans.append(norm(row["ans"]))
                if i + 1 < K:
                    continue
                win = rows[i + 1 - K : i + 1]
                a0 = win[0]["ans"]
                if not a0 or not all(same(a0, x["ans"]) for x in win):
                    continue
                confs = [x["conf"] for x in win]
                if not all(math.isfinite(c) for c in confs):
                    continue
                mn, mx = min(confs), max(confs)
                all_ok = all(x["ok"] for x in win)
                all_bad = not any(x["ok"] for x in win)
                high = mn >= TAU
                low = mx < TAU
                if all_ok and high:
                    tag = "G"
                elif all_ok and low:
                    tag = "R"
                elif all_bad and high:
                    tag = "W"
                elif all_bad and low:
                    tag = "L"
                else:
                    continue
                for j in range(i + 1 - K, i + 1):
                    if tag == "G":
                        labels[j] = "G"
                    elif tag == "W" and labels[j] != "G":
                        labels[j] = "W"
                    elif tag == "R" and labels[j] not in ("G", "W"):
                        labels[j] = "R"
                    elif tag == "L" and labels[j] == "O":
                        labels[j] = "L"
                # first time this window type closes, take the last step of the window
                feat = dict(win[-1])
                feat["min_conf"] = mn
                feat["n_ans"] = len({x for x in seen_ans if x})
                first_win[tag].append(feat)

            if "R" in labels:
                q_r += 1
                if "G" in labels:
                    q_r_then_g += 1
                else:
                    q_r_only += 1
            for row, lab in zip(rows, labels):
                if lab in ("G", "R", "W", "L"):
                    for key, value in row.items():
                        if key in ("ans", "ok"):
                            continue
                        if isinstance(value, (int, float)):
                            bags[lab][key].append(float(value))

        print(f"\n======== {name} nq={nq}  R题={q_r} 其中后来仍有G={q_r_then_g} 只有R没有G={q_r_only} ========")
        order = ["G", "R", "L", "W"]
        keys = [
            "frac",
            "left",
            "conf",
            "inc_len",
            "ratio",
            "in_inc",
            "in_prev",
            "wait",
            "conclude",
            "verify",
            "doubt",
            "ans_len",
            "n_ans",
            "stop_margin",
            "dola_rise",
            "lookback",
        ]
        print("窗口刚满4次时（每窗一次）：")
        for key in keys:
            parts = []
            for tag in order:
                xs = [r[key] for r in first_win[tag] if key in r]
                parts.append(f"{tag} p50={p50(xs):.3f} n={len([x for x in xs if x==x])}")
            print(f"  {key:12} " + " | ".join(parts))
        print("OR 捞 R 的对照 = R vs L（低把握连错）。越大越像该交：")
        for key, higher_better in [
            ("frac", False),
            ("left", True),
            ("in_inc", True),
            ("in_prev", True),
            ("conclude", True),
            ("verify", True),
            ("wait", False),
            ("doubt", False),
            ("inc_len", False),
            ("ans_len", False),
            ("n_ans", False),
            ("stop_margin", True),
            ("dola_rise", True),
            ("lookback", True),
        ]:
            pos = [r[key] for r in first_win["R"]]
            neg = [r[key] for r in first_win["L"]]
            if not higher_better:
                pos = [-x if x == x else x for x in pos]
                neg = [-x if x == x else x for x in neg]
            score = auroc(pos, neg)
            print(f"  R对L {key:12} AUROC={score:.3f}  R窗={len(first_win['R'])} L窗={len(first_win['L'])}")
        print("拿掉 W 的对照 = G vs W（高把握连对 vs 高把握连错）：")
        for key, higher_better in [
            ("frac", False),
            ("in_inc", True),
            ("conclude", True),
            ("stop_margin", True),
            ("dola_rise", True),
            ("lookback", True),
            ("n_ans", False),
        ]:
            pos = [r[key] for r in first_win["G"]]
            neg = [r[key] for r in first_win["W"]]
            if not higher_better:
                pos = [-x if x == x else x for x in pos]
                neg = [-x if x == x else x for x in neg]
            print(f"  G对W {key:12} AUROC={auroc(pos, neg):.3f}")


if __name__ == "__main__":
    main()
