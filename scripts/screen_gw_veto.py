#!/usr/bin/env python3
"""Can half/Wait/exit scores tell high-conf G from high-conf W?

Oracle = never stop on a W lock (perfect W detector). Then sweep real
signals as a veto on the high-conf door. Acc first vs current high-conf+FS.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_14b_vs_puma as r14
import report_8b_extra_vs_puma as extra8
import report_conf_fs_stop_margin as cmp

SIGS = (
    "exit_minus_half",
    "stop_margin",
    "stop_margin_alt",
    "last_mean_logp",
    "ans_entropy",
    "neg_ans_entropy",
    "lookback_ratio",
    "lb_q",
    "dola_mean_rise",
    "margin",
    "reasoning_pmi",
)


def win_min(ev: dict[str, Any], sig: str) -> float:
    vals = ev["vals"].get(sig) or []
    if not vals or any(v != v for v in vals):
        return float("nan")
    return min(vals)


def decide_veto(q, *, sig: str | None, thr: float, flip: bool, oracle: bool):
    import replay_default_dense_gate as old

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
        if ev is not None and step >= rg.MSS and ev["high"]:
            fire = True
            if oracle and ev.get("tag") == "W":
                fire = False
            elif sig:
                score = win_min(ev, sig)
                if score != score:
                    fire = True
                elif flip:
                    fire = score <= thr
                else:
                    fire = score >= thr
            if fire:
                return rg.pack(
                    trials, rows, end, "conf", original_tokens=q["orig_tok"], label=ev["tag"]
                )
        if ev is not None and step >= rg.MSS:
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


def auroc(pos, neg):
    return rg.auroc(pos, neg)


def first_high(pack):
    g, w = [], []
    for q in pack["questions"]:
        for ev in q["events"]:
            if not ev["high"] or int(q["rows"][ev["end"]]["stopped_len"]) < rg.MSS:
                continue
            rec = {sig: win_min(ev, sig) for sig in SIGS}
            rec["tag"] = ev["tag"]
            rec["end"] = ev["end"]
            if ev["tag"] == "G":
                g.append(rec)
            elif ev["tag"] == "W":
                w.append(rec)
            break
    return g, w


def load(cell):
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
        return None
    pack = layers.load_pack(cell)
    cons.attach_shared(pack)
    layers.precompute_events(pack, list(SIGS))
    return pack


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = list(rg.CELLS) + list(extra8.CELLS) + [r14.CELLS[0]]
    print(
        "高置信门上 G 对 W。Oracle = 看见 W 就不停（上界）。"
        "真信号：4 步最差过门槛才允许高置信停，否则接着走。",
        flush=True,
    )
    for cell in cells:
        pack = load(cell)
        if pack is None:
            print(f"skip {cell['name']}", flush=True)
            continue
        g, w = first_high(pack)
        print(f"\n=== {cell['name']}  half={pack.get('_half_layer')}  第一扇高置信 G={len(g)} W={len(w)} ===", flush=True)
        if len(w) < 3:
            print("  W 太少，这集高置信几乎不连错", flush=True)
        base_rows = [layers.decide(q, signal=None, threshold=float("inf"), need="all") for q in pack["questions"]]
        base = eval_rows(pack, base_rows)
        ora_rows = [decide_veto(q, sig=None, thr=0, flip=False, oracle=True) for q in pack["questions"]]
        ora = eval_rows(pack, ora_rows)
        print(
            f"  现在高置信+FS {rg.fmt_pct(base['acc'])} / {base['tok']:.0f}   "
            f"PUMA {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f}   "
            f"完美拿掉 W {rg.fmt_pct(ora['acc'])} / {ora['tok']:.0f} "
            f"（{rg.fmt_pp(100*(ora['acc']-base['acc']))} / {rg.fmt_tok(ora['tok']-base['tok'])}）",
            flush=True,
        )
        if len(g) < 8 or len(w) < 3:
            continue
        print("  第一扇 4 步最差  G中位 / W中位  AUROC(G>W)", flush=True)
        ranked = []
        for sig in SIGS:
            gp = [r[sig] for r in g]
            wp = [r[sig] for r in w]
            if sum(x == x for x in gp) < 8 or sum(x == x for x in wp) < 3:
                continue
            roc = auroc(gp, wp)
            roc_f = auroc([-x if x == x else x for x in gp], [-x if x == x else x for x in wp])
            flip = roc_f > roc + 0.02
            use = roc_f if flip else roc
            gm = statistics.median([x for x in gp if x == x])
            wm = statistics.median([x for x in wp if x == x])
            ranked.append((use, flip, sig, gm, wm, roc, roc_f))
            print(
                f"    {sig:18} G {gm:8.3f}  W {wm:8.3f}  AUROC {roc:.3f}"
                f"{'  反过来 '+format(roc_f,'.3f') if flip else ''}",
                flush=True,
            )
        ranked.sort(reverse=True)
        for use, flip, sig, *_ in ranked[:4]:
            xs = [win_min(ev, sig) for q in pack["questions"] for ev in q["events"] if ev["high"]]
            xs = [x for x in xs if x == x]
            thrs = rg.quantiles(xs, n=21) if xs else []
            best = None
            for thr in thrs:
                rows = [decide_veto(q, sig=sig, thr=thr, flip=flip, oracle=False) for q in pack["questions"]]
                rec = eval_rows(pack, rows)
                rec["thr"] = thr
                rec["sig"] = sig
                rec["flip"] = flip
                if best is None or (rec["acc"], -rec["tok"]) > (best["acc"], -best["tok"]):
                    best = rec
            if best:
                print(
                    f"  真门槛 {sig}{'↓' if flip else '↑'} {best['thr']:.3f} → "
                    f"{rg.fmt_pct(best['acc'])} / {best['tok']:.0f}  "
                    f"相对现在 {rg.fmt_pp(100*(best['acc']-base['acc']))} / {rg.fmt_tok(best['tok']-base['tok'])}  "
                    f"相对PUMA {rg.fmt_pp(100*(best['acc']-pack['puma_acc']))}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
