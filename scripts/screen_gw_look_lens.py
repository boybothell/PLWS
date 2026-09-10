#!/usr/bin/env python3
"""G vs W on the first high-conf lock: lookback, layer lens, Wait.

Online only (no answer-change count). Acc-first-among-both vs high-conf+FS:
Acc not down and tokens not up; then max Acc, then min tokens.
"""
from __future__ import annotations

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
import screen_gw_veto as gw

LOOK = (
    "lookback_ratio",
    "lb_q",
    "lb_cot",
    "lb_ans",
    "lb_rev",
    "lb_recency5",
    "pre_lookback_ratio",
    "pre_lb_q",
)
WAIT = (
    "stop_margin",
    "stop_margin_alt",
    "stop_vs_cont",
    "stop_logp",
    "wait_logp",
    "alt_logp",
)


def rel_layers(ids: list[int]) -> dict[str, int]:
    if not ids:
        return {}
    last = max(ids)
    n = last + 1
    out = {}
    for name, t in (("¼", n // 4), ("½", n // 2), ("¾", (3 * n) // 4), ("末", last)):
        out[name] = min(ids, key=lambda layer: (abs(layer - t), layer))
    return out


def win_score(ev: dict[str, Any], sig: str, how: str) -> float:
    vals = ev["vals"].get(sig) or []
    good = [v for v in vals if v == v]
    if not good:
        return float("nan")
    if how == "last":
        return good[-1]
    return min(good)


def first_high_rows(pack, names: list[str]) -> tuple[list[dict], list[dict]]:
    g, w = [], []
    for q in pack["questions"]:
        for ev in q["events"]:
            if not ev["high"] or int(q["rows"][ev["end"]]["stopped_len"]) < rg.MSS:
                continue
            rec = {"tag": ev["tag"], "ev": ev}
            for sig in names:
                rec[f"{sig}|all"] = win_score(ev, sig, "all")
                rec[f"{sig}|last"] = win_score(ev, sig, "last")
            if ev["tag"] == "G":
                g.append(rec)
            elif ev["tag"] == "W":
                w.append(rec)
            break
    return g, w


def roc_pair(g, w, key: str) -> tuple[float, float, float, float]:
    gp = [r[key] for r in g]
    wp = [r[key] for r in w]
    if sum(x == x for x in gp) < 8 or sum(x == x for x in wp) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    roc = rg.auroc(gp, wp)
    rocf = rg.auroc([-x if x == x else x for x in gp], [-x if x == x else x for x in wp])
    gm = statistics.median([x for x in gp if x == x])
    wm = statistics.median([x for x in wp if x == x])
    return roc, rocf, gm, wm


def sweep_veto(pack, base, sig: str, how: str, flip: bool):
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"]:
                xs.append(win_score(ev, sig, how))
    xs = [x for x in xs if x == x]
    if len(xs) < 8:
        return None
    best_both = None
    best_acc = None
    for thr in rg.quantiles(xs, n=21):
        rows = [
            gw.decide_veto(q, sig=sig, thr=thr, flip=flip, oracle=False)
            if how == "all"
            else decide_last(q, sig, thr, flip)
            for q in pack["questions"]
        ]
        rec = gw.eval_rows(pack, rows)
        rec.update(thr=thr, sig=sig, how=how, flip=flip)
        if best_acc is None or (rec["acc"], -rec["tok"]) > (best_acc["acc"], -best_acc["tok"]):
            best_acc = rec
        if rec["acc"] + 1e-12 >= base["acc"] and rec["tok"] <= base["tok"] + 1e-6:
            if best_both is None or (rec["acc"], -rec["tok"]) > (best_both["acc"], -best_both["tok"]):
                best_both = rec
    return best_both, best_acc


def decide_last(q, sig, thr, flip):
    # last-step veto: reuse decide_veto but score = last not min
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
            score = win_score(ev, sig, "last")
            fire = True
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
                return rg.pack_fs(
                    trials, rows, end, pick, q["orig_tok"], label=ev["tag"] if ev else "O"
                )
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def fmt(rec):
    if rec is None:
        return "没有（一开正确率就掉或 token 涨）"
    arrow = "↓" if rec["flip"] else "↑"
    return (
        f"{rec['sig']}{arrow} {rec['how']} {rec['thr']:.3f} → "
        f"{rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}"
    )


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = [rg.CELLS[0], rg.CELLS[2], extra8.CELLS[0], extra8.CELLS[1], r14.CELLS[0]]
    print(
        "高置信第一扇：回看题目 / 各层 lens / Wait。换答次数不用。"
        "有用 = 相对高置信+FS，正确率不降且 token 不涨。",
        flush=True,
    )
    for cell in cells:
        pack = gw.load(cell)
        if pack is None:
            print(f"skip {cell['name']}", flush=True)
            continue
        ids = more.discover_layers(pack["scores"])
        rel = rel_layers(ids)
        lens_sigs = [f"lens_l{L}" for L in ids] + [f"exit_minus_l{L}" for L in ids]
        lens_sigs += [f"dola_mean_l{L}" for L in ids]
        if rel:
            pairs = list(rel.items())
            for i, (na, la) in enumerate(pairs):
                for nb, lb in pairs[i + 1 :]:
                    lens_sigs.append(f"dola_mean_{la}_{lb}")
                    lens_sigs.append(f"lens_{la}_{lb}")
        names = list(dict.fromkeys([*LOOK, *WAIT, "last_mean_logp", "ans_entropy", *lens_sigs]))
        layers.precompute_events(pack, names)
        g, w = first_high_rows(pack, names)
        base_rows = [layers.decide(q, signal=None, threshold=float("inf"), need="all") for q in pack["questions"]]
        base = gw.eval_rows(pack, base_rows)
        print(
            f"\n=== {cell['name']} 层={rel}  G={len(g)} W={len(w)}  "
            f"现在 {rg.fmt_pct(base['acc'])}/{base['tok']:.0f}  "
            f"PUMA {rg.fmt_pct(pack['puma_acc'])}/{pack['puma_tok']:.0f} ===",
            flush=True,
        )
        if len(w) < 5:
            print("  W 太少", flush=True)
            continue

        def show_family(title: str, keys: list[str]) -> list[tuple]:
            print(f"  {title}", flush=True)
            ranked = []
            for sig in keys:
                for how in ("all", "last"):
                    key = f"{sig}|{how}"
                    roc, rocf, gm, wm = roc_pair(g, w, key)
                    if roc != roc:
                        continue
                    flip = rocf > roc + 0.02
                    use = rocf if flip else roc
                    ranked.append((use, flip, sig, how, gm, wm, roc, rocf))
            ranked.sort(reverse=True)
            for use, flip, sig, how, gm, wm, roc, rocf in ranked[:8]:
                extra = f"  反{rocf:.3f}" if flip else ""
                print(
                    f"    {sig:22} {how:4} G {gm:8.3f} W {wm:8.3f}  AUROC {roc:.3f}{extra}",
                    flush=True,
                )
            return ranked

        look_r = show_family("回看题目", [s for s in LOOK if s in names])
        wait_r = show_family("Wait / 收口", list(WAIT) + ["last_mean_logp", "ans_entropy"])
        # compact lens: relative depths + last-minus
        lens_show = ["last_mean_logp"]
        for name, L in rel.items():
            lens_show += [f"lens_l{L}", f"exit_minus_l{L}"]
        if rel:
            la, lb = rel["½"], rel["末"]
            lens_show += [f"exit_minus_l{rel['½']}", f"dola_mean_{la}_{lb}", f"lens_{la}_{lb}"]
            if "¼" in rel:
                lens_show.append(f"exit_minus_l{rel['¼']}")
                lens_show.append(f"exit_minus_l{rel['¾']}")
        lens_r = show_family("中间层 / 出口减该层", list(dict.fromkeys(lens_show)))

        # Acc sweep: lookback + wait + top-3 lens by AUROC
        cands = []
        for ranked in (look_r, wait_r, lens_r):
            cands.extend(ranked[:3])
        seen = set()
        print("  正确率优先（不降且 token 不涨） / 只追正确率", flush=True)
        for use, flip, sig, how, *_ in cands:
            tag = (sig, how, flip)
            if tag in seen or use < 0.55:
                continue
            seen.add(tag)
            both, accm = sweep_veto(pack, base, sig, how, flip)
            print(
                f"    {sig} {'↓' if flip else '↑'} {how} AUROC {use:.3f} | "
                f"两边都好: {fmt(both)} | 只追Acc: {fmt(accm)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
