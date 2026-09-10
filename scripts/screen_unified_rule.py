#!/usr/bin/env python3
"""Threshold-free / one-number interpretable rules on PUMA-base, 7B+8B."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import screen_puma_layers as sc
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_unified_rule.md"


def last(vals: list[float]) -> float:
    return vals[-1] if vals and vals[-1] == vals[-1] else float("nan")


def finite_all(vals: list[float]) -> bool:
    return bool(vals) and all(v == v for v in vals)


def eval_rule(pack: dict[str, Any], fire: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = puma_acc = puma_tok = 0.0
    n_fire = 0
    for q in pack["questions"]:
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        hit = None
        hist: list[float] = []
        for ev in q["_act"]:
            ev = dict(ev)
            ev["_hist"] = list(hist)
            dola = last(ev["vals"].get(pack["_dola"]) or [])
            if dola == dola:
                hist.append(dola)
            if fire(ev):
                hit = ev
                break
        if hit is None:
            acc += int(q["puma_ok"])
            tok += q["puma_tok"]
            continue
        n_fire += 1
        sim = rg.pack(
            q["trials"], q["rows"], hit["end"], "rescue",
            original_tokens=q["orig_tok"], label=hit["tag"],
        )
        ok = rg.same(sim["answer"], q["gt"])
        acc += int(ok)
        tok += sim["tokens"]
    return {
        "acc": acc / n,
        "tok": tok / n,
        "d_acc": 100.0 * (acc / n - puma_acc / n),
        "d_tok": tok / n - puma_tok / n,
        "n_fire": n_fire,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
    }


def make_rules(pack: dict[str, Any]) -> list[tuple[str, Callable]]:
    half = pack.get("_half_layer")
    last_id = max(pack["_ids"])
    dola = f"dola_mean_l{last_id}"
    rise = "exit_minus_half"
    lens_h = f"lens_l{half}" if half is not None else None
    pack["_dola"] = dola

    def v(ev, name):
        return ev["vals"].get(name) or []

    rules = []

    def r_last_max(ev):
        xs = v(ev, dola)
        return finite_all(xs) and xs[-1] == max(xs)

    def r_exit_gt_half(ev):
        xs = v(ev, rise)
        return bool(xs) and xs[-1] == xs[-1] and xs[-1] > 0

    def r_and(ev):
        return r_last_max(ev) and r_exit_gt_half(ev)

    def r_last_gt_prev(ev):
        xs = v(ev, dola)
        return finite_all(xs) and xs[-1] > max(xs[:-1])

    def r_conf_last_max(ev):
        end = ev["end"]
        rows = ev["_rows"]
        confs = [rg.finite(rows[end + 1 - 4 + i].get("confidence")) for i in range(4)]
        return all(c == c for c in confs) and confs[-1] == max(confs)

    def r_conf_ge(thr):
        def fn(ev):
            end = ev["end"]
            c = rg.finite(ev["_rows"][end].get("confidence"))
            return c == c and c >= thr
        return fn

    def r_qtile(p):
        def fn(ev):
            xs = v(ev, dola)
            if not xs or xs[-1] != xs[-1]:
                return False
            hist = ev.get("_hist") or []
            pool = hist + [xs[-1]]
            if len(pool) < 2:
                return False
            s = sorted(pool)
            rank = sum(1 for x in s if x <= xs[-1]) / len(s)
            return rank >= p
        return fn

    def r_odds(k):
        def fn(ev):
            xs = v(ev, rise)
            return bool(xs) and xs[-1] == xs[-1] and math.exp(xs[-1]) >= k
        return fn

    rules += [
        ("窗内最后一步最顺（最后一层好写）", r_last_max),
        ("最新一步出口比一半深更亮", r_exit_gt_half),
        ("最顺 且 出口亮过一半深", r_and),
        ("最后一步好写高于前三步", r_last_gt_prev),
        ("窗内最后一步把握最高", r_conf_last_max),
        ("最后一步把握≥0.90", r_conf_ge(0.90)),
        ("最后一步把握≥0.85", r_conf_ge(0.85)),
        ("最后一步把握≥0.80", r_conf_ge(0.80)),
        ("本题历史里好写≥70%分位", r_qtile(0.70)),
        ("本题历史里好写≥50%分位", r_qtile(0.50)),
        ("出口比一半深至少亮 e^2 倍", r_odds(math.e ** 2)),
        ("出口比一半深至少亮 10 倍", r_odds(10)),
        ("出口比一半深至少亮 100 倍", r_odds(100)),
        ("出口比一半深至少亮 1000 倍", r_odds(1000)),
    ]
    return rules


def main() -> None:
    packs = []
    for name, pack in sc.load_ready():
        for q in pack["questions"]:
            for ev in q["_act"]:
                ev["_rows"] = q["rows"]
        packs.append((name, pack))

    names = [n for n, _ in packs]
    seven = [n for n in names if n.startswith("7B ")]
    eight = [n for n in names if n.startswith("8B ")]
    rule_names = [zh for zh, _ in make_rules(packs[0][1])]
    grid: dict[str, dict[str, dict[str, Any]]] = {zh: {} for zh in rule_names}

    for name, pack in packs:
        print(f"eval {name}", flush=True)
        for zh, fn in make_rules(pack):
            grid[zh][name] = eval_rule(pack, fn)

    def summarize(zh: str) -> dict[str, Any]:
        recs = grid[zh]
        d = {n: recs[n]["d_acc"] for n in names}
        return {
            "zh": zh,
            "mean": sum(d.values()) / len(d),
            "m7": sum(d[n] for n in seven) / len(seven),
            "m8": sum(d[n] for n in eight) / len(eight),
            "n_pos": sum(int(v > 1e-9) for v in d.values()),
            "n_neg": sum(int(v < -1e-9) for v in d.values()),
            "worst": min(d.values()),
            "d": d,
            "recs": recs,
        }

    ranked = sorted((summarize(zh) for zh in rule_names), key=lambda r: (-r["m8"], -r["mean"], r["n_neg"]))
    lines = [
        "# 跨模型跨集：一条规则、不调每集门槛",
        "",
        "官方 PUMA 当底。只在停点前的低置信连答窗上判断。下面每条都是同一句话，7B/8B 十集共用。正确率只对金标。",
        "对照：一半深 / 最后一层好写 的本集自选门槛（乐观）。",
        "",
        "| 规则 | 全体 | 7B | 8B | 赢几集 | 伤几集 | 最差一集 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    print(lines[-2], flush=True)
    for r in ranked:
        row = (
            f"| {r['zh']} | {r['mean']:+.1f} | {r['m7']:+.1f} | {r['m8']:+.1f} | "
            f"{r['n_pos']}/10 | {r['n_neg']}/10 | {r['worst']:+.1f} |"
        )
        print(row, flush=True)
        lines.append(row)

    best = ranked[0]
    lines += ["", f"## 8B 平均最好的一条：{best['zh']}", "", "| 集 | vs PUMA | token | 放进几题 |", "|---|---|---:|---:|"]
    for n, _ in packs:
        rec = best["recs"][n]
        lines.append(
            f"| {n} | {rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])} | "
            f"{rec['tok']:.0f} | {rec['n_fire']} |"
        )
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
