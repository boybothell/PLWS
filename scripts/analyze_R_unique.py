#!/usr/bin/env python3
"""What is unique to R versus G / L / W / unlocked-correct / other steps."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
K = 4
TAU = 0.98


def norm(text):
    s = "".join(str(text or "").split()).strip().lower()
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    for tok in ("left", "right", "cdot", "times", "frac", "mathrm", "text"):
        s = s.replace("\\" + tok, "")
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


def auroc(pos, neg):
    pos = [x for x in pos if x == x]
    neg = [x for x in neg if x == x]
    if len(pos) < 8 or len(neg) < 8:
        return float("nan")
    xs = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    xs.sort()
    n1, n0 = len(pos), len(neg)
    rank_sum = 0.0
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rank_sum += avg * sum(xs[k][1] for k in range(i, j))
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load_folder(folder: Path) -> dict:
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
            out.setdefault(key, {}).update(row)
    return out


def load_scores(dataset: str) -> dict:
    out = {}
    for folder in (
        AE / f"results/confcal_judge/v2/dense_internal/{dataset}",
        AE / f"results/confcal_judge/v2/dense_lens/{dataset}",
        AE / f"results/confcal_judge/v2/dense_solver_probes/{dataset}",
    ):
        for key, row in load_folder(folder).items():
            out.setdefault(key, {}).update(row)
    return out


CELLS = [
    (
        "MATH",
        "math-500",
        AE / "results/dense_G_r1_7b/math-500/dense_puma/trial_answers.json",
        AE / "results/math500_official/puma_ds7b/statistics.json",
        AE / "results/dense_G_r1_7b/math-500/per_sample.json",
    ),
    (
        "GPQA",
        "gpqa-diamond",
        AE / "results/dense_G_r1_7b/gpqa-diamond/dense_puma/trial_answers.json",
        AE / "results/puma_offline_r1_7b/gpqa-diamond/statistics.json",
        AE / "results/dense_G_r1_7b/gpqa-diamond/per_sample.json",
    ),
]

FEATS = [
    ("conf", True, "置信度"),
    ("stop_margin", True, "写完后收口比 Wait 高多少"),
    ("stop_vs_cont", True, "收口比 Wait/Alternatively 高多少"),
    ("dola_mean_rise", True, "浅层到深层，试答越来越好写"),
    ("dola_logp_l20", True, "第 20 层对试答第一个词"),
    ("dola_logp_l27", True, "第 27 层对试答第一个词"),
    ("dola_top1_l20", True, "第 20 层是否已经把试答词排第一"),
    ("dola_top1_l27", True, "第 27 层是否已经把试答词排第一"),
    ("lens_rise", True, "最后一层比第 14 层好写多少"),
    ("neg_ans_entropy", True, "试答用词有多集中"),
    ("reasoning_pmi", True, "有草稿比只看题更顺多少"),
    ("margin", True, "当前试答比历史上别的试答高多少"),
    ("lookback_ratio", True, "写完试答时还在看题的比例"),
    ("lb_q", True, "注意力落在题目上多少"),
    ("frac", True, "已经走完的思考比例"),
    ("left", False, "后面还剩多少步"),
    ("n_ans", False, "到这一步换过多少种答案"),
    ("run", True, "同一答案已经连了几次"),
    ("late_rise", True, "第 20 层到第 27 层，试答词抬了多少"),
    ("gap", True, "同置信度段里，收口分偏高多少"),
]


def window_tag(window):
    confs = [x["conf"] for x in window]
    if not all(c == c for c in confs):
        return None
    a0 = window[0]["ans"]
    if not a0 or not all(same(a0, x["ans"]) for x in window):
        return None
    all_ok = all(x["ok"] for x in window)
    all_bad = not any(x["ok"] for x in window)
    high = min(confs) >= TAU
    low = max(confs) < TAU
    if all_ok and high:
        return "G"
    if all_ok and low:
        return "R"
    if all_bad and high:
        return "W"
    if all_bad and low:
        return "L"
    return None


def build_rows(trials, official, gmap, scores):
    out = []
    first = defaultdict(list)
    for qi, seq in trials.items():
        info = official.get(qi)
        gg = gmap.get(qi)
        if not info or not gg:
            continue
        af = gg.get("A_final") or info.get("original_answer")
        gt = info.get("ground_truth")
        nst = int(seq[-1]["stopped_len"])
        rows = []
        seen = set()
        run = 0
        prev = ""
        for row in seq:
            st = int(row["stopped_len"])
            ans = str(row.get("final_answer") or "")
            conf = finite(row.get("confidence"))
            ok = bool(ans) and (same(ans, af) or same(ans, gt))
            if ans and same(ans, prev):
                run += 1
            else:
                run = 1 if ans else 0
            prev = ans
            if ans:
                seen.add(norm(ans))
            inn = scores.get((qi, st), {})
            item = {
                "qi": qi,
                "st": st,
                "ans": ans,
                "conf": conf,
                "ok": ok,
                "run": float(run),
                "frac": st / max(nst, 1),
                "left": float(nst - st),
                "n_ans": float(len(seen)),
                "nst": nst,
                "stop_margin": finite(inn.get("stop_margin")),
                "stop_vs_cont": finite(inn.get("stop_vs_cont")),
                "dola_mean_rise": finite(inn.get("dola_mean_rise")),
                "dola_logp_l20": finite(inn.get("dola_logp_l20")),
                "dola_logp_l27": finite(inn.get("dola_logp_l27")),
                "dola_top1_l20": finite(inn.get("dola_top1_l20")),
                "dola_top1_l27": finite(inn.get("dola_top1_l27")),
                "lens_rise": finite(inn.get("lens_rise")),
                "neg_ans_entropy": finite(inn.get("neg_ans_entropy")),
                "reasoning_pmi": finite(inn.get("reasoning_pmi")),
                "margin": finite(inn.get("margin")),
                "lookback_ratio": finite(inn.get("lookback_ratio")),
                "lb_q": finite(inn.get("lb_q")),
                "late_rise": (
                    finite(inn.get("dola_logp_l27")) - finite(inn.get("dola_logp_l20"))
                    if math.isfinite(finite(inn.get("dola_logp_l27")))
                    and math.isfinite(finite(inn.get("dola_logp_l20")))
                    else float("nan")
                ),
            }
            rows.append(item)
        labels = ["O"] * len(rows)
        for i, row in enumerate(rows):
            if i + 1 < K:
                if row["ok"] and row["conf"] == row["conf"] and row["conf"] < TAU:
                    labels[i] = "U"
                continue
            win = rows[i + 1 - K : i + 1]
            steps = [x["st"] for x in win]
            if steps != list(range(steps[0], steps[0] + K)):
                if row["ok"] and row["conf"] == row["conf"] and row["conf"] < TAU:
                    labels[i] = "U"
                continue
            tag = window_tag(win)
            if tag:
                for j in range(i + 1 - K, i + 1):
                    if tag == "G":
                        labels[j] = "G"
                    elif tag == "W" and labels[j] != "G":
                        labels[j] = "W"
                    elif tag == "R" and labels[j] not in ("G", "W"):
                        labels[j] = "R"
                    elif tag == "L" and labels[j] == "O":
                        labels[j] = "L"
                if tag not in first or first[tag][-1]["qi"] != qi:
                    feat = dict(win[-1])
                    feat["tag"] = tag
                    first[tag].append(feat)
            elif row["ok"] and row["conf"] == row["conf"] and row["conf"] < TAU and labels[i] == "O":
                labels[i] = "U"
        for row, lab in zip(rows, labels):
            row["tag"] = lab
            out.append(row)
        # first unlocked-correct (U) of the question
        us = [r for r, lab in zip(rows, labels) if lab == "U"]
        if us and ("U" not in first or first["U"][-1]["qi"] != qi):
            first["U"].append(dict(us[0], tag="U"))
    return out, first


def residual(rows, key):
    # bin by conf, subtract bin median so leftover is not just 置信度
    bins = defaultdict(list)
    for row in rows:
        c, v = row["conf"], row[key]
        if c == c and v == v:
            bins[round(c, 2)].append(v)
    med = {b: p50(vs) for b, vs in bins.items()}
    out = []
    for row in rows:
        c, v = row["conf"], row[key]
        if c != c or v != v:
            out.append(float("nan"))
            continue
            out.append(v - med.get(round(c, 2), float("nan")))
    return out


def main():
    lines = [
        "# R 相对其他步有没有独有特征",
        "",
        "R = 连续 4 次同一试答、每步置信度都 < 0.98、且都对。",
        "对照：G 高置信度连对；L 低置信度连错；W 高置信度连错；",
        "U = 也对、置信度也 < 0.98，但还没连满 4 次；O = 其余。",
        "数字是窗口刚满时（每题每种锁一次），避免长锁把中位数拉偏。",
        "",
    ]
    dump = {}
    for name, dataset, trial, stat, gpath in CELLS:
        print(f"load {name}", flush=True)
        trials = json.loads(trial.read_text())
        print(f"  trials={len(trials)}", flush=True)
        official = {int(r["question_idx"]): r for r in json.loads(stat.read_text())}
        gmap = {int(r["question_idx"]): r for r in json.loads(gpath.read_text())}
        by = defaultdict(list)
        for row in trials:
            by[int(row["question_idx"])].append(row)
        for qi in by:
            by[qi].sort(key=lambda x: int(x["stopped_len"]))
        scores = load_scores(dataset)
        print(f"  scores={len(scores)}", flush=True)
        steps, first = build_rows(by, official, gmap, scores)
        print(f"  first R/G/L/W/U = {len(first['R'])}/{len(first['G'])}/{len(first['L'])}/{len(first['W'])}/{len(first['U'])}", flush=True)
        # attach residual stop_margin / lens_rise / pmi
        for key in ("stop_margin", "lens_rise", "dola_mean_rise", "reasoning_pmi", "neg_ans_entropy"):
            res = residual(steps, key)
            for row, value in zip(steps, res):
                row[f"gap_{key}"] = value
            for tag in first:
                for row in first[tag]:
                    row["gap"] = float("nan")
            # map residual onto first windows
            by_key = {(r["qi"], r["st"]): r.get(f"gap_{key}") for r in steps}
            if key == "stop_margin":
                for tag in first:
                    for row in first[tag]:
                        row["gap"] = finite(by_key.get((row["qi"], row["st"])))

        lines.append(f"## {name}")
        lines.append("")
        counts = {tag: len(first[tag]) for tag in ("G", "R", "L", "W", "U")}
        q_r = len({r["qi"] for r in first["R"]})
        q_u = len({r["qi"] for r in first["U"]})
        lines.append(
            f"窗口数：G {counts['G']}，R {counts['R']}，L {counts['L']}，W {counts['W']}，"
            f"还没锁住的低置信度对步（U）{counts['U']}。有 R 的题 {q_r}，有 U 的题 {q_u}。"
            f"内部读数覆盖 {len(scores)} 步。"
        )
        lines.append("")
        order = ["R", "G", "L", "W", "U", "O"]
        # first-window bags; O uses a sample of other first-of-question other steps
        bags = {tag: first[tag] for tag in ("G", "R", "L", "W", "U")}
        other = []
        seen_q = set()
        for row in steps:
            if row["tag"] == "O" and row["qi"] not in seen_q:
                other.append(row)
                seen_q.add(row["qi"])
        bags["O"] = other

        lines.append("各锁刚出现时的中位数（空 = 这套读数还没有）：")
        lines.append("")
        lines.append("| 读数 | R | G 高置信度连对 | L 低置信度连错 | W 高置信度连错 | U 对了但还没锁 | 其他步 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key, _higher, zh in FEATS:
            parts = []
            for tag in order:
                xs = [finite(r.get(key)) for r in bags[tag]]
                n = sum(x == x for x in xs)
                parts.append("—" if n < 5 else f"{p50(xs):.3f}")
            lines.append(f"| {zh} | " + " | ".join(parts) + " |")
        lines.append("")
        lines.append("R 对每一类的分开程度（越大越像 R；0.5 是猜）。要独有，必须对每一类都不是 0.5：")
        lines.append("")
        lines.append("| 读数 | 对 G | 对 L | 对 W | 对 U（对了还没锁） | 对其余 | 对「所有非 R」 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        r_rows = bags["R"]
        uniq = []
        for key, higher, zh in FEATS:
            pos = [finite(r.get(key)) for r in r_rows]
            if not higher:
                pos = [-x if x == x else x for x in pos]
            cells = []
            vs_all = []
            ok_all = True
            for tag in ("G", "L", "W", "U", "O"):
                neg = [finite(r.get(key)) for r in bags[tag]]
                if not higher:
                    neg = [-x if x == x else x for x in neg]
                score = auroc(pos, neg)
                cells.append("—" if score != score else f"{score:.2f}")
                if tag != "O":
                    if score != score or not (score >= 0.62 or score <= 0.38):
                        ok_all = False
                vs_all.extend(neg if tag != "O" else [])
            # all non-R first windows
            neg_all = []
            for tag in ("G", "L", "W", "U"):
                xs = [finite(r.get(key)) for r in bags[tag]]
                if not higher:
                    xs = [-x if x == x else x for x in xs]
                neg_all.extend(xs)
            all_score = auroc(pos, neg_all)
            cells.append("—" if all_score != all_score else f"{all_score:.2f}")
            if ok_all and all_score == all_score and (all_score >= 0.62 or all_score <= 0.38):
                uniq.append((zh, all_score, cells))
            lines.append(f"| {zh} | " + " | ".join(cells) + " |")
        lines.append("")
        if uniq:
            lines.append("按上面的粗标准，没有哪条对 G/L/W/U 同时拉开的，就不在这里标「独有」。")
            lines.append("下面这些至少对四类里多数拉开了：")
            for zh, score, _ in uniq:
                lines.append(f"- {zh}（对非 R 合计 {score:.2f}）")
        else:
            lines.append(
                "没有一条读数能同时把 R 从 G、L、W、U 里都拉开。"
                "R 更像「已经对了、只是置信度还没上去」：对 G 像对的锁，对 L 像低置信度。"
            )
        lines.append("")

        # mismatch: high internal, low conf
        if any(finite(r.get("stop_margin")) == finite(r.get("stop_margin")) for r in r_rows):
            lines.append("错位：置信度低、但内部已经像该交。分数 = 同置信度段里收口分偏高多少。")
            pos = [finite(r.get("gap")) for r in r_rows]
            for tag, zh in (("G", "G"), ("L", "L"), ("U", "U"), ("W", "W")):
                neg = [finite(r.get("gap")) for r in bags[tag]]
                lines.append(
                    f"- 对 {zh}：AUROC {auroc(pos, neg):.2f}；R 中位 {p50(pos):.2f}，{zh} 中位 {p50(neg):.2f}"
                )
            lines.append("")

        # later becomes G?
        r_then_g = 0
        r_only = 0
        q_labels = defaultdict(set)
        for row in steps:
            q_labels[row["qi"]].add(row["tag"])
        for qi, tags in q_labels.items():
            if "R" in tags:
                if "G" in tags:
                    r_then_g += 1
                else:
                    r_only += 1
        lines.append(f"有 R 的题里，后来仍出现 G 的 {r_then_g} 题，一直没有 G 的 {r_only} 题。")
        # compare those two R first windows
        r_g = []
        r_n = []
        for row in first["R"]:
            if "G" in q_labels[row["qi"]]:
                r_g.append(row)
            else:
                r_n.append(row)
        lines.append("只比较「后来会有 G 的 R」和「只有 R 的 R」，看后来会不会涨置信度，现在能不能看出来：")
        for key, higher, zh in FEATS[:12]:
            a = [finite(r.get(key)) for r in r_n]
            b = [finite(r.get(key)) for r in r_g]
            if not higher:
                a = [-x if x == x else x for x in a]
                b = [-x if x == x else x for x in b]
            score = auroc(a, b)
            if score == score:
                lines.append(
                    f"- {zh}：只有 R 对 后来有 G 的分开程度 {score:.2f}；"
                    f"中位 {p50([finite(r.get(key)) for r in r_n]):.3f} vs {p50([finite(r.get(key)) for r in r_g]):.3f}"
                )
        lines.append("")
        dump[name] = {
            "counts": counts,
            "r_then_g": r_then_g,
            "r_only": r_only,
            "n_scores": len(scores),
        }

    out = AE / "tables/R_unique_features.md"
    out.write_text("\n".join(lines) + "\n")
    (AE / "results/rescue_R_gate/unique_preview.json").parent.mkdir(parents=True, exist_ok=True)
    (AE / "results/rescue_R_gate/unique_preview.json").write_text(json.dumps(dump, indent=2) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
