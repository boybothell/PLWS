#!/usr/bin/env python3
"""Acc-first: residual conf / stop_logp / layer_agree as extra high-conf floor."""
from __future__ import annotations

import math
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as old
import replay_rescue_R_gate as rg
import report_14b_vs_puma as r14
import report_8b_extra_vs_puma as extra8
import screen_gw_veto as gw

SIGS = ("stop_logp", "layer_agree", "prefix_ans_cos")


def score_of(ev, rows, end, kind):
    window = rows[end + 1 - rg.K : end + 1]
    if kind == "first_conf":
        return rg.finite(window[0].get("confidence"))
    if kind == "min_conf":
        xs = [rg.finite(x.get("confidence")) for x in window]
        xs = [x for x in xs if x == x]
        return min(xs) if xs else float("nan")
    xs = [v for v in (ev["vals"].get(kind) or []) if v == v]
    if kind.endswith("_min"):
        raw = kind[: -len("_min")]
        xs = [v for v in (ev["vals"].get(raw) or []) if v == v]
        return min(xs) if xs else float("nan")
    return xs[-1] if xs else float("nan")


def decide(q, *, kind, thr):
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials, rows = q["trials"], q["rows"]
    lens = old.step_char_lens(trials)
    red_run = 0
    best_conf = float("-inf")
    probed = []
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
        if ev is not None and step >= rg.MSS and ev["high"]:
            extra = score_of(ev, rows, end, kind)
            if extra == extra and extra >= thr:
                return rg.pack(
                    trials, rows, end, "conf", original_tokens=q["orig_tok"], label=ev["tag"]
                )
        pick = rg.fs_ready(step, red_run, best_conf, probed)
        if pick is not None:
            tag = ev["tag"] if ev is not None else "O"
            return rg.pack_fs(trials, rows, end, pick, q["orig_tok"], label=tag)
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def eval_rows(pack, rows):
    out = []
    for q, sim in zip(pack["questions"], rows):
        out.append(
            {
                **sim,
                "ok": rg.hit(sim["answer"], q["gt"], q["a_final"], q["orig_ok"])
                if sim["early"]
                else q["orig_ok"],
            }
        )
    return rg.summarize(out)


def pick_thr(pack, kind, cands, base):
    keep = []
    any_acc = []
    for thr in cands:
        got = eval_rows(pack, [decide(q, kind=kind, thr=thr) for q in pack["questions"]])
        rec = (got["acc"], -got["tok"], thr, got)
        any_acc.append(rec)
        if got["acc"] + 1e-12 >= base["acc"] and got["tok"] <= base["tok"] + 1e-6:
            keep.append(rec)
    keep.sort(reverse=True)
    any_acc.sort(key=lambda x: (-abs(x[0] - base["acc"]) if x[0] < base["acc"] else 0, -x[0], x[2]))
    # among Acc not down (tokens may rise), best Acc then min tokens
    acc_ok = [x for x in any_acc if x[0] + 1e-12 >= base["acc"]]
    acc_ok.sort(reverse=True)
    return keep[0] if keep else None, acc_ok[0] if acc_ok else None


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = [rg.CELLS[0], rg.CELLS[2], extra8.CELLS[0], extra8.CELLS[1], r14.CELLS[0]]
    kinds = (
        ("first_conf", [0.98, 0.99, 0.995, 0.998, 0.999]),
        ("min_conf", [0.98, 0.99, 0.995, 0.998, 0.999]),
        ("stop_logp", None),
        ("layer_agree_min", None),
        ("prefix_ans_cos", None),
    )
    print(
        "正确率优先。高置信门多一道地板：不够就不在这扇停，后面还能强停/写完。"
        "赢=正确率不降且字数不升。",
        flush=True,
    )
    for cell in cells:
        pack = gw.load(cell)
        if pack is None:
            continue
        layers.precompute_events(pack, list(SIGS))
        base = eval_rows(
            pack,
            [layers.decide(q, signal=None, threshold=float("inf"), need="all") for q in pack["questions"]],
        )
        print(
            f"\n=== {cell['name']}  现在 {rg.fmt_pct(base['acc'])} / {base['tok']:.0f}  "
            f"PUMA {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f} ===",
            flush=True,
        )
        for kind, fixed in kinds:
            if fixed is None:
                vals = []
                for q in pack["questions"]:
                    for ev in q["events"]:
                        if ev["high"] and int(q["rows"][ev["end"]]["stopped_len"]) >= rg.MSS:
                            v = score_of(ev, q["rows"], ev["end"], kind)
                            if v == v:
                                vals.append(v)
                            break
                if len(vals) < 20:
                    print(f"  {kind}: 票不够", flush=True)
                    continue
                xs = sorted(vals)
                cands = sorted({xs[int(i * (len(xs) - 1) / 8)] for i in range(9)})
            else:
                cands = fixed
            win, acc_ok = pick_thr(pack, kind, cands, base)
            if win is None:
                if acc_ok is None:
                    print(f"  {kind}: 正确率不降的点也没有", flush=True)
                else:
                    _, _, thr, got = acc_ok
                    print(
                        f"  {kind}: 正确率+字数都守住=空。"
                        f"正确率不降最好 {rg.fmt_pct(got['acc'])} / {got['tok']:.0f} "
                        f"（{rg.fmt_pp(100*(got['acc']-base['acc']))} / {rg.fmt_tok(got['tok']-base['tok'])}）"
                        f"  地板 {thr:.4g}",
                        flush=True,
                    )
            else:
                _, _, thr, got = win
                print(
                    f"  {kind}: 守住  {rg.fmt_pct(got['acc'])} / {got['tok']:.0f} "
                    f"（{rg.fmt_pp(100*(got['acc']-base['acc']))} / {rg.fmt_tok(got['tok']-base['tok'])}）"
                    f"  地板 {thr:.4g}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
