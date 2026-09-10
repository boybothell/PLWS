#!/usr/bin/env python3
"""R vs L AUROC of Wait aggregations already in the 4-step window."""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_brightest_stop_auroc as br
import report_8b_extra_vs_puma as extra8

SIGS = ("stop_margin", "wait_logp", "stop_logp", "stop_vs_cont")


def xs(ev, sig):
    vals = [v for v in (ev["vals"].get(sig) or []) if v == v]
    return vals


def agg(ev, kind):
    m = xs(ev, "stop_margin")
    w = xs(ev, "wait_logp")
    s = xs(ev, "stop_logp")
    if kind.startswith("m:") and len(m) < 2 and kind not in {"m:min", "m:last", "m:mean", "m:max"}:
        if len(m) < 1:
            return float("nan")
    if kind == "m:min":
        return min(m) if m else float("nan")
    if kind == "m:mean":
        return sum(m) / len(m) if m else float("nan")
    if kind == "m:last":
        return m[-1] if m else float("nan")
    if kind == "m:max":
        return max(m) if m else float("nan")
    if kind == "m:med":
        return statistics.median(m) if m else float("nan")
    if kind == "m:d":
        return m[-1] - m[0] if len(m) >= 2 else float("nan")
    if kind == "m:std":
        return -statistics.pstdev(m) if len(m) >= 2 else float("nan")
    if kind == "m:sharpe":
        if len(m) < 2:
            return float("nan")
        sd = statistics.pstdev(m)
        return (sum(m) / len(m)) - sd
    if kind == "m:prob":
        if not m:
            return float("nan")
        ps = [1.0 / (1.0 + math.exp(-x)) for x in m]
        return min(ps)
    if kind == "w:min":
        return -min(w) if w else float("nan")
    if kind == "w:mean":
        return -(sum(w) / len(w)) if w else float("nan")
    if kind == "w:last":
        return -w[-1] if w else float("nan")
    if kind == "w:d":
        return w[0] - w[-1] if len(w) >= 2 else float("nan")
    if kind == "s:min":
        return min(s) if s else float("nan")
    if kind == "pair:stop_up_wait_down":
        if len(s) < 2 or len(w) < 2:
            return float("nan")
        return (s[-1] - s[0]) + (w[0] - w[-1])
    return float("nan")


def vote(q, kind):
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials, rows = q["trials"], q["rows"]
    import replay_default_dense_gate as old

    lens = old.step_char_lens(trials) if rg.USE_FS else {}
    red_run = 0
    best_conf = float("-inf")
    probed = []
    walked = 0
    best_score = None
    best_tag = None
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if rg.USE_FS and tstep >= rg.FS_MIN_STEP:
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
                break
            if (not ev["mixed"]) and ev["tag"] in {"R", "L"}:
                score = agg(ev, kind)
                if score == score and (best_score is None or score > best_score):
                    best_score, best_tag = score, ev["tag"]
        if rg.USE_FS:
            pick = rg.fs_ready(step, red_run, best_conf, probed)
            if pick is not None:
                break
    if best_score is None:
        return None
    return best_score, best_tag


def collect(pack, kind):
    pos, neg = [], []
    for q in pack["questions"]:
        got = vote(q, kind)
        if got is None:
            continue
        score, tag = got
        (pos if tag == "R" else neg).append(score)
    return pos, neg


KINDS = (
    ("m:min", "相减4步最差(现在)"),
    ("m:mean", "相减4步平均"),
    ("m:last", "相减只看最后一步"),
    ("m:max", "相减4步最好"),
    ("m:med", "相减中位"),
    ("m:d", "相减末−首(趋势)"),
    ("m:std", "相减更稳(−标准差)"),
    ("m:sharpe", "平均−标准差"),
    ("m:prob", "先变概率再取最差"),
    ("w:min", "最不想Wait(最差步)"),
    ("w:mean", "平均不想Wait"),
    ("w:last", "最后一步不想Wait"),
    ("w:d", "Wait在降"),
    ("s:min", "收口最差步"),
    ("pair:stop_up_wait_down", "收口升且Wait降"),
)


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = [rg.CELLS[0], rg.CELLS[2], rg.CELLS[1], extra8.CELLS[0], extra8.CELLS[1], extra8.CELLS[2]]
    print(
        f"TAU={rg.TAU}  一题一票最亮扇。只改 4 步怎么聚，不新抽。",
        flush=True,
    )
    header = f"{'聚合':<22}" + "".join(f"{c['name']:>10}" for c in cells)
    print(header, flush=True)
    packs = []
    for cell in cells:
        pack = layers.load_pack(cell)
        cons.attach_shared(pack)
        layers.precompute_events(pack, list(SIGS))
        packs.append(pack)
    rows = {k: [] for k, _ in KINDS}
    for pack in packs:
        for kind, _zh in KINDS:
            pos, neg = collect(pack, kind)
            roc = rg.auroc(pos, neg) if pos and neg else float("nan")
            rows[kind].append(roc)
    for kind, zh in KINDS:
        bits = "".join(f"{x:10.3f}" if x == x else f"{'—':>10}" for x in rows[kind])
        print(f"{zh:<22}{bits}", flush=True)
    print("\n相对现在(最差步)的平均差", flush=True)
    base = rows["m:min"]
    ranked = []
    for kind, zh in KINDS:
        xs = [a - b for a, b in zip(rows[kind], base) if a == a and b == b]
        if not xs:
            continue
        ranked.append((sum(xs) / len(xs), zh))
    for mean_d, zh in sorted(ranked, reverse=True):
        print(f"  {zh:<22} {mean_d:+.3f}", flush=True)


if __name__ == "__main__":
    main()
