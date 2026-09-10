#!/usr/bin/env python3
"""Half-depth + Wait: AND / OR / z-sum vs each alone. Acc-first among both."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as old
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import report_8b_extra_vs_puma as extra8

HALF = "exit_minus_half"
WAIT = "stop_margin"
TABLE = AE / "tables/half_wait_combo.md"


def win_min(ev: dict[str, Any], sig: str) -> float:
    vals = ev["vals"].get(sig) or []
    if not vals or any(v != v for v in vals):
        return float("nan")
    return min(vals)


def decide_combo(q, *, mode: str, thr_a: float, thr_b: float, need: str = "all"):
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials = q["trials"]
    rows = q["rows"]
    lens = old.step_char_lens(trials)
    red_run = 0
    best_conf = float("-inf")
    probed: list[dict[str, Any]] = []
    walked = 0
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if tstep >= rg.FS_MIN_STEP:
                red_run = red_run + 1 if old.is_red(tstep, lens) else 0
            walked += 1
        conf = rg.finite(row.get("confidence"))
        answer = str(row.get("final_answer") or "")
        if answer and conf == conf:
            best_conf = max(best_conf, conf)
            probed.append({"step": step, "answer": answer, "conf": conf})
        ev = events_by_end.get(end)
        if ev is not None and step >= rg.MSS:
            if ev["high"]:
                return rg.pack(
                    trials, rows, end, "conf", original_tokens=q["orig_tok"], label=ev["tag"]
                )
            if not ev["mixed"]:
                pa = layers.passed(ev["vals"].get(HALF) or [], thr_a, need)
                pb = layers.passed(ev["vals"].get(WAIT) or [], thr_b, need)
                fire = (pa and pb) if mode == "and" else (pa or pb)
                if fire:
                    return rg.pack(
                        trials,
                        rows,
                        end,
                        "rescue",
                        original_tokens=q["orig_tok"],
                        label=ev["tag"],
                    )
        pick = rg.fs_ready(step, red_run, best_conf, probed)
        if pick is not None:
            tag = ev["tag"] if ev is not None else "O"
            return rg.pack_fs(trials, rows, end, pick, q["orig_tok"], label=tag)
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def decide_sum(q, *, thr: float, mid_a: float, spr_a: float, mid_b: float, spr_b: float):
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials = q["trials"]
    rows = q["rows"]
    lens = old.step_char_lens(trials)
    red_run = 0
    best_conf = float("-inf")
    probed: list[dict[str, Any]] = []
    walked = 0
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if tstep >= rg.FS_MIN_STEP:
                red_run = red_run + 1 if old.is_red(tstep, lens) else 0
            walked += 1
        conf = rg.finite(row.get("confidence"))
        answer = str(row.get("final_answer") or "")
        if answer and conf == conf:
            best_conf = max(best_conf, conf)
            probed.append({"step": step, "answer": answer, "conf": conf})
        ev = events_by_end.get(end)
        if ev is not None and step >= rg.MSS:
            if ev["high"]:
                return rg.pack(
                    trials, rows, end, "conf", original_tokens=q["orig_tok"], label=ev["tag"]
                )
            if not ev["mixed"]:
                za = cons_z(win_min(ev, HALF), mid_a, spr_a)
                zb = cons_z(win_min(ev, WAIT), mid_b, spr_b)
                if za == za and zb == zb and za + zb >= thr:
                    return rg.pack(
                        trials,
                        rows,
                        end,
                        "rescue",
                        original_tokens=q["orig_tok"],
                        score=za + zb,
                        label=ev["tag"],
                    )
        pick = rg.fs_ready(step, red_run, best_conf, probed)
        if pick is not None:
            tag = ev["tag"] if ev is not None else "O"
            return rg.pack_fs(trials, rows, end, pick, q["orig_tok"], label=tag)
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def cons_z(x: float, mid: float, spr: float) -> float:
    if x != x or not math.isfinite(mid) or not math.isfinite(spr) or abs(spr) < 1e-6:
        return float("nan")
    return (x - mid) / spr


def eval_rows(pack, fs_rows, fs_sum, ours):
    rec = layers.contrast(fs_rows, ours, fs_sum)
    rec["d_acc_pp_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
    rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
    return rec


def run_combo(pack, mode, thr_a, thr_b, need="all"):
    return [
        {
            **decide_combo(q, mode=mode, thr_a=thr_a, thr_b=thr_b, need=need),
            "ok": rg.hit(
                decide_combo(q, mode=mode, thr_a=thr_a, thr_b=thr_b, need=need)["answer"],
                q["gt"],
                q["a_final"],
                q["orig_ok"],
            )
            if decide_combo(q, mode=mode, thr_a=thr_a, thr_b=thr_b, need=need)["early"]
            else q["orig_ok"],
            "has_r": "R" in q["labels"],
            "r_only": "R" in q["labels"] and "G" not in q["labels"],
        }
        for q in pack["questions"]
    ]


def run_combo_once(pack, mode, thr_a, thr_b):
    out = []
    for q in pack["questions"]:
        sim = decide_combo(q, mode=mode, thr_a=thr_a, thr_b=thr_b)
        out.append(
            {
                **sim,
                "ok": rg.hit(sim["answer"], q["gt"], q["a_final"], q["orig_ok"])
                if sim["early"]
                else q["orig_ok"],
                "has_r": "R" in q["labels"],
                "r_only": "R" in q["labels"] and "G" not in q["labels"],
            }
        )
    return out


def run_sum_once(pack, thr, mid_a, spr_a, mid_b, spr_b):
    out = []
    for q in pack["questions"]:
        sim = decide_sum(q, thr=thr, mid_a=mid_a, spr_a=spr_a, mid_b=mid_b, spr_b=spr_b)
        out.append(
            {
                **sim,
                "ok": rg.hit(sim["answer"], q["gt"], q["a_final"], q["orig_ok"])
                if sim["early"]
                else q["orig_ok"],
                "has_r": "R" in q["labels"],
                "r_only": "R" in q["labels"] and "G" not in q["labels"],
            }
        )
    return out


def fmt(rec) -> str:
    if rec is None:
        return "—"
    extra = rec.get("note") or ""
    return (
        f"{rg.fmt_pct(rec['acc'])}/{rec['tok']:.0f} "
        f"（{rg.fmt_pp(rec['d_acc_pp_puma'])}/{rg.fmt_tok(rec['d_tok_puma'])}；"
        f"{rec['rescue_R']}/{rec['rescue_L']}{extra}）"
    )


def load_named():
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    named = []
    seen = set()
    for cell in list(rg.CELLS) + [more.EXTRA] + list(extra8.CELLS):
        if cell["name"] in seen:
            continue
        pack = cmp.load_ready(cell)
        if pack is None:
            continue
        seen.add(cell["name"])
        print(f"load {cell['name']}", flush=True)
        named.append((cell["name"], pack))
    for ds, zh in (("aime24", "7B AIME24"), ("aime25", "7B AIME25")):
        parts = []
        for seed in accfirst.SEEDS:
            pack = cmp.load_ready(accfirst.aime_cell(ds, seed))
            if pack is None:
                continue
            pack["_cov"] = 1.0
            parts.append(pack)
        if len(parts) == 4:
            print(f"load {zh}", flush=True)
            named.append((zh, accfirst.merge_packs(parts, zh)))
    return named


def iqr_stats(pack, sig):
    xs = cmp.low_xs(pack, sig)
    xs = sorted(x for x in xs if x == x)
    if len(xs) < 8:
        return float("nan"), float("nan")
    lo = xs[int(0.25 * (len(xs) - 1))]
    hi = xs[int(0.75 * (len(xs) - 1))]
    return xs[len(xs) // 2], hi - lo


def main() -> None:
    named = load_named()
    lines = [
        "# 一半深 + Wait 叠加",
        "",
        "高置信 + 后路强停照旧。滞后门：两扇都过才停 / 一扇过就停 / 两分数标准化后相加。",
        "选门槛：相对高置信+FS，正确率不降且 token 不多时先取正确率最高，再少 token。",
        "括号：相对 PUMA；放进对/错锁。",
        "",
        "| 集 | 只一半深 | 只 Wait | 两扇都过 | 一扇过就停 | 两分数相加 |",
        "|---|---|---|---|---|---|",
    ]
    print(lines[-2])
    print(lines[-1], flush=True)
    for name, pack in named:
        print(f"sweep {name}", flush=True)
        if "scores" in pack:
            layers.precompute_events(pack, [HALF, WAIT])
        fs_rows = layers.run_pack(pack, None, float("inf"), "all")
        fs_sum = rg.summarize(fs_rows)
        xa = [float("inf"), *rg.quantiles(cmp.low_xs(pack, HALF), n=13)]
        xb = [float("inf"), *rg.quantiles(cmp.low_xs(pack, WAIT), n=13)]
        half = cmp.sweep_sig(pack, fs_rows, fs_sum, HALF)
        wait = cmp.sweep_sig(pack, fs_rows, fs_sum, WAIT)
        and_pts, or_pts = [], []
        for ta in xa:
            for tb in xb:
                if ta == float("inf") and tb == float("inf"):
                    rec = eval_rows(pack, fs_rows, fs_sum, fs_rows)
                    rec["rescue_R"] = rec["rescue_L"] = 0
                    rec["threshold"] = float("inf")
                    and_pts.append(rec)
                    or_pts.append(dict(rec))
                    continue
                if ta == float("inf"):
                    and_rec = None
                else:
                    and_rec = eval_rows(
                        pack, fs_rows, fs_sum, run_combo_once(pack, "and", ta, tb if tb == tb else float("inf"))
                    )
                    and_rec["threshold"] = ta
                    and_pts.append(and_rec)
                or_rec = eval_rows(pack, fs_rows, fs_sum, run_combo_once(pack, "or", ta, tb))
                or_rec["threshold"] = ta
                or_pts.append(or_rec)
        and_best = accfirst.pick_acc_first_both(and_pts, fs_sum["acc"], fs_sum["tok"]) if and_pts else None
        or_best = accfirst.pick_acc_first_both(or_pts, fs_sum["acc"], fs_sum["tok"])
        mid_a, spr_a = iqr_stats(pack, HALF)
        mid_b, spr_b = iqr_stats(pack, WAIT)
        sum_pts = []
        if spr_a == spr_a and spr_b == spr_b:
            zs = []
            for q in pack["questions"]:
                for ev in q["events"]:
                    if ev["high"] or ev["mixed"]:
                        continue
                    za = cons_z(win_min(ev, HALF), mid_a, spr_a)
                    zb = cons_z(win_min(ev, WAIT), mid_b, spr_b)
                    if za == za and zb == zb:
                        zs.append(za + zb)
            for thr in [float("inf"), *rg.quantiles(zs, n=17)]:
                ours = run_sum_once(pack, thr, mid_a, spr_a, mid_b, spr_b)
                rec = eval_rows(pack, fs_rows, fs_sum, ours)
                rec["threshold"] = thr
                sum_pts.append(rec)
        sum_best = accfirst.pick_acc_first_both(sum_pts, fs_sum["acc"], fs_sum["tok"]) if sum_pts else None
        row = (
            f"| {name} | {fmt(half)} | {fmt(wait)} | {fmt(and_best)} | {fmt(or_best)} | {fmt(sum_best)} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
