#!/usr/bin/env python3
"""PUMA-base add-on: other stored layers + window aggregations vs last-minus-half."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_puma_plus_r_oracle as base
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_puma_layers.md"
AGGS = ("min", "last", "mean", "k3", "rise")
AGG_ZH = {
    "min": "4步都过",
    "last": "只看最后一步",
    "mean": "4步平均",
    "k3": "4步里3步过",
    "rise": "最后减第一步",
}
BASE_KEY = ("exit_minus_half", "min", False)


def rel_layer(ids: list[int], frac: float) -> int:
    n = max(ids) + 1
    target = n - 1 if frac >= 1 else int(round((n - 1) * frac))
    return min(ids, key=lambda layer: (abs(layer - target), layer))


def attach_extra(pack: dict[str, Any]) -> None:
    ids = more.discover_layers(pack["scores"])
    half = pack.get("_half_layer")
    for row in pack["scores"].values():
        last = rg.finite(row.get("last_mean_logp"))
        lens = {L: rg.finite(row.get(f"lens_l{L}")) for L in ids}
        shallow = [lens[L] for L in ids if half is not None and L <= half and lens[L] == lens[L]]
        deep = [lens[L] for L in ids if half is not None and L > half and lens[L] == lens[L]]
        if last == last and shallow:
            row["exit_minus_min_shallow"] = last - min(shallow)
            row["exit_minus_mean_shallow"] = last - (sum(shallow) / len(shallow))
        if last == last and deep:
            row["exit_minus_mean_deep"] = last - (sum(deep) / len(deep))


def candidate_sigs(pack: dict[str, Any]) -> list[str]:
    ids = more.discover_layers(pack["scores"])
    names = [
        "last_mean_logp",
        "last_min_logp",
        "exit_minus_half",
        "exit_minus_min_shallow",
        "exit_minus_mean_shallow",
        "exit_minus_mean_deep",
        "neg_ans_entropy",
        "dola_mean_rise",
        "dola_score",
        "dola_jsd_max",
        "emerge_layer",
        "lens_best",
    ]
    for L in ids:
        names += [f"exit_minus_l{L}", f"lens_l{L}", f"dola_mean_l{L}", f"dola_logp_l{L}"]
        if f"dola_jsd_l{L}" in next(iter(pack["scores"].values()), {}):
            names.append(f"dola_jsd_l{L}")
    if len(ids) >= 2:
        picked = []
        for frac in (0.25, 0.50, 0.75, 1.0):
            picked.append(rel_layer(ids, frac))
        picked = list(dict.fromkeys(picked))
        for i, lo in enumerate(picked):
            for hi in picked[i + 1 :]:
                names += [f"lens_{lo}_{hi}", f"dola_mean_{lo}_{hi}", f"dola_logp_{lo}_{hi}"]
        if len(ids) <= 6:
            for i, lo in enumerate(ids):
                for hi in ids[i + 1 :]:
                    names += [f"lens_{lo}_{hi}", f"dola_mean_{lo}_{hi}"]
    return [n for n in dict.fromkeys(names) if any(n in row for row in pack["scores"].values())]


def zh_sig(name: str, pack: dict[str, Any]) -> str:
    half = pack.get("_half_layer")
    if name == "exit_minus_half":
        return f"末层减一半深（第{half}层）"
    if name == "exit_minus_min_shallow":
        return "末层减前半最浅"
    if name == "exit_minus_mean_shallow":
        return "末层减前半平均"
    if name == "exit_minus_mean_deep":
        return "末层减后半平均"
    if name == "last_mean_logp":
        return "末层整段好写"
    if name == "last_min_logp":
        return "末层最差一个词"
    if name.startswith("exit_minus_l"):
        return f"末层减第{name[len('exit_minus_l'):]}层"
    if name.startswith("lens_l") and name.count("_") == 1:
        return f"第{name[len('lens_l'):]}层好写"
    if name.startswith("dola_mean_l"):
        return f"第{name[len('dola_mean_l'):]}层 DoLA 均值"
    if name.startswith("dola_logp_l"):
        return f"第{name[len('dola_logp_l'):]}层首词"
    if name.startswith("lens_"):
        return f"lens {name[5:].replace('_', '→')}"
    if name.startswith("dola_mean_"):
        return f"DoLA均值 {name[10:].replace('_', '→')}"
    if name.startswith("dola_logp_"):
        return f"首词 {name[10:].replace('_', '→')}"
    return name


def window_score(vals: list[float], agg: str, flip: bool) -> float:
    if not vals:
        return float("nan")
    usable = [v for v in vals if v == v]
    if agg == "min":
        if len(usable) < 4:
            return float("nan")
        score = min(usable)
    elif agg == "last":
        score = vals[-1] if vals[-1] == vals[-1] else float("nan")
    elif agg == "mean":
        if not usable:
            return float("nan")
        score = sum(usable) / len(usable)
    elif agg == "max":
        if not usable:
            return float("nan")
        score = max(usable)
    elif agg == "k3":
        if len(usable) < 4:
            return float("nan")
        score = sorted(usable)[-3]
    elif agg == "rise":
        if vals[0] != vals[0] or vals[-1] != vals[-1]:
            return float("nan")
        score = vals[-1] - vals[0]
    else:
        raise ValueError(agg)
    return -score if flip else score


def actionable(q: dict[str, Any]) -> list[dict[str, Any]]:
    puma_step = int(q.get("puma_step") or 10**9)
    out = []
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed") or ev.get("tag") not in {"R", "L"}:
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= puma_step:
            continue
        out.append(ev)
    return out


def first_scores(pack: dict[str, Any], sig: str, agg: str, flip: bool) -> tuple[list[float], list[float]]:
    pos, neg = [], []
    for q in pack["questions"]:
        evs = q["_act"]
        if not evs:
            continue
        score = window_score(evs[0]["vals"].get(sig) or [], agg, flip)
        if score != score:
            continue
        (pos if evs[0]["tag"] == "R" else neg).append(score)
    return pos, neg


def choose_flip(pack: dict[str, Any], sig: str, agg: str) -> bool:
    pos, neg = first_scores(pack, sig, agg, False)
    if not pos or not neg:
        return False
    return rg.p50(pos) < rg.p50(neg)


def eval_cfg(pack: dict[str, Any], sig: str, agg: str, flip: bool, thr: float) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = 0
    tok = 0.0
    puma_acc = 0
    puma_tok = 0.0
    n_r = n_l = n_gain = n_hurt = 0
    for q in pack["questions"]:
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        hit = None
        for ev in q["_act"]:
            score = window_score(ev["vals"].get(sig) or [], agg, flip)
            if score == score and score >= thr:
                hit = ev
                break
        if hit is None:
            acc += int(q["puma_ok"])
            tok += q["puma_tok"]
            continue
        sim = rg.pack(
            q["trials"],
            q["rows"],
            hit["end"],
            "rescue",
            original_tokens=q["orig_tok"],
            label=hit["tag"],
        )
        ok = base.trial_ok(q, sim["answer"])
        acc += int(ok)
        tok += sim["tokens"]
        if hit["tag"] == "R":
            n_r += 1
        else:
            n_l += 1
        if ok and not q["puma_ok"]:
            n_gain += 1
        if (not ok) and q["puma_ok"]:
            n_hurt += 1
    return {
        "sig": sig,
        "agg": agg,
        "flip": flip,
        "threshold": thr,
        "acc": acc / max(n, 1),
        "tok": tok / max(n, 1),
        "d_acc": 100.0 * (acc / max(n, 1) - puma_acc / max(n, 1)),
        "d_tok": tok / max(n, 1) - puma_tok / max(n, 1),
        "rescue_R": n_r,
        "rescue_L": n_l,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "auroc": rg.auroc(*first_scores(pack, sig, agg, flip)),
    }


def sweep_cfg(pack: dict[str, Any], sig: str, agg: str, flip: bool) -> dict[str, Any] | None:
    xs = []
    for q in pack["questions"]:
        for ev in q["_act"]:
            score = window_score(ev["vals"].get(sig) or [], agg, flip)
            if score == score:
                xs.append(score)
    if len(xs) < 8:
        return None
    thrs = [float("inf"), *rg.quantiles(xs, n=41)]
    points = [eval_cfg(pack, sig, agg, flip, thr) for thr in thrs]
    n = max(len(pack["questions"]), 1)
    puma_acc = sum(int(q["puma_ok"]) for q in pack["questions"]) / n
    puma_tok = sum(q["puma_tok"] for q in pack["questions"]) / n
    chosen = accfirst.pick_acc_first_both(points, puma_acc, puma_tok)
    if chosen is None:
        return None
    chosen["puma_acc"] = puma_acc
    chosen["puma_tok"] = puma_tok
    return chosen


def load_ready() -> list[tuple[str, dict[str, Any]]]:
    layers.enrich = more.enrich_all
    out: list[tuple[str, dict[str, Any]]] = []
    for name, cells in sw.all_jobs():
        if not (name.startswith("7B ") or name.startswith("8B ")):
            continue
        packs = []
        names: list[str] | None = None
        for cell in cells:
            cell = dict(cell)
            if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
                continue
            if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
                continue
            pack = layers.load_pack(cell)
            base.attach_puma_step(pack, Path(cell["stat"]))
            cons.attach_shared(pack)
            attach_extra(pack)
            if names is None:
                names = candidate_sigs(pack)
            layers.precompute_events(pack, names or [])
            pack["_half_cov"] = 1.0
            pack["_cov"] = 1.0
            packs.append(pack)
        if not packs:
            continue
        if len(cells) > 1 and len(packs) != len(cells):
            print(f"skip {name} seeds={len(packs)}/{len(cells)}", flush=True)
            continue
        pack = accfirst.merge_packs(packs, name) if len(packs) > 1 else packs[0]
        pack["_half_layer"] = packs[0].get("_half_layer")
        pack["_ids"] = more.discover_layers(packs[0]["scores"])
        pack["_names"] = names or []
        for q in pack["questions"]:
            q["_act"] = actionable(q)
        out.append((name, pack))
        print(
            f"load {name} n={len(pack['questions'])} layers={pack['_ids']} sigs={len(pack['_names'])}",
            flush=True,
        )
    return out


def fmt_cfg(rec: dict[str, Any] | None, pack: dict[str, Any]) -> str:
    if rec is None or rec["threshold"] == float("inf"):
        return "不开"
    flip = "，取反" if rec["flip"] else ""
    return (
        f"{sw.fmt_pair(rec['acc'], rec['tok'])} "
        f"（{zh_sig(rec['sig'], pack)}，{AGG_ZH[rec['agg']]}{flip}）"
    )


def main() -> None:
    lines = [
        "# PUMA 底：其他层 + 4 步聚法",
        "",
        "官方 PUMA 当底。只在停点之前的低置信窗上开门。正确率只对金标。",
        "对照 = 末层减一半深、4 步都过。其余是现成分数里的层对 / 单层 / 浅层聚合，再换 4 步怎么收。",
        "门槛仍是本集自选：正确率不低于 PUMA，再尽量高、再尽量短。",
        "取反 = 这集上对窗分数更低，门槛按越大越该停来翻。",
        "",
        "| 集 | 层 | PUMA | 一半深 | 本集最好 | 最好比一半深 |",
        "|---|---|---|---|---|---|",
    ]
    print(lines[-2], flush=True)
    for name, pack in load_ready():
        jobs: list[tuple[str, str, bool]] = []
        for sig in pack["_names"]:
            for agg in AGGS:
                flip = choose_flip(pack, sig, agg)
                jobs.append((sig, agg, flip))
        # always include the written half-depth, no flip, 4-step all
        if BASE_KEY not in jobs:
            jobs.insert(0, BASE_KEY)
        recs = []
        half = None
        print(f"sweep {name} jobs={len(jobs)}", flush=True)
        for sig, agg, flip in jobs:
            rec = sweep_cfg(pack, sig, agg, flip)
            if rec is None:
                continue
            recs.append(rec)
            if sig == "exit_minus_half" and agg == "min" and not flip:
                half = rec
        if not recs:
            continue
        best = sorted(recs, key=lambda p: (-p["acc"], p["tok"]))[0]
        half = half or next((p for p in recs if p["sig"] == "exit_minus_half" and p["agg"] == "min"), recs[0])
        beat = [p for p in recs if p["acc"] > half["acc"] + 1e-12]
        print(
            f"  half {sw.fmt_pair(half['acc'], half['tok'])} {rg.fmt_pp(half['d_acc'])} "
            f"best {zh_sig(best['sig'], pack)} {AGG_ZH[best['agg']]} "
            f"{sw.fmt_pair(best['acc'], best['tok'])} {rg.fmt_pp(best['d_acc'])} "
            f"beat_half={len(beat)}",
            flush=True,
        )
        d_acc = best["d_acc"] - half["d_acc"]
        d_tok = best["tok"] - half["tok"]
        row = (
            f"| {name} | {pack['_ids']} | {sw.fmt_pair(pack['puma_acc'], pack['puma_tok'])} | "
            f"{fmt_cfg(half, pack)} | {fmt_cfg(best, pack)} | "
            f"{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)} |"
        )
        print(row, flush=True)
        lines.append(row)
        pack["_best"] = best
        pack["_half"] = half
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
