#!/usr/bin/env python3
"""Stack existing low-conf doors on 密探k4. Gold Acc. No regen on fire."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_brightest_stop_auroc as jobs
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import report_dense_regen_default_door as door
import report_puma_plus_r_oracle as base

TABLE = AE / "tables/dense_k4/lag_doors.md"
READY = 0.85
NAMES = ("exit_minus_half", "last_mean_logp", "stop_margin")
PEEK = (
    ("half", "一半深", "exit_minus_half", "all", False),
    ("last", "最后一层好写", "last_mean_logp", "last", False),
    ("wait", "收口比 Wait", "stop_margin", "last", False),
    ("ratio", "对比率", "exit_minus_half", "last", True),
)


def load_one(zh: str, tag: str, gdir: str, puma: str, ds_zh: str, dataset: str, seed: int | None):
    cell = jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset, seed)
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
        return None
    pack = layers.load_pack(cell)
    ids = more.discover_layers(pack["scores"])
    if len(ids) < 2:
        return None
    cons.attach_shared(pack)
    layers.precompute_events(pack, list(NAMES))
    pack["_half_cov"] = cmp.cov(pack, "exit_minus_half")
    pack["_last_cov"] = cmp.cov(pack, "last_mean_logp")
    pack["_wait_cov"] = cmp.cov(pack, "stop_margin")
    pack["_cov"] = pack["_half_cov"]
    use_seed = 42 if seed is None else seed
    if not door.attach_host(pack, tag, dataset, use_seed):
        return None
    return pack


def load_named(zh: str, tag: str, gdir: str, puma: str, ds_zh: str, dataset: str, is_aime: bool):
    if not is_aime:
        pack = load_one(zh, tag, gdir, puma, ds_zh, dataset, None)
        if pack is None:
            return None
        print(f"load {zh} {ds_zh} n={len(pack['questions'])} half={pack['_half_cov']:.2f} wait={pack['_wait_cov']:.2f}", flush=True)
        return pack
    packs = []
    for seed in jobs.SEEDS:
        pack = load_one(zh, tag, gdir, puma, ds_zh, dataset, seed)
        if pack is None:
            continue
        print(f"load {zh} {ds_zh}-s{seed} n={len(pack['questions'])} half={pack['_half_cov']:.2f} wait={pack['_wait_cov']:.2f}", flush=True)
        packs.append(pack)
    if len(packs) != 4:
        print(f"skip {zh} {ds_zh} seeds={len(packs)}/4", flush=True)
        return None
    pack = accfirst.merge_packs(packs, f"{zh} {ds_zh}")
    pack["_half_cov"] = min(p["_half_cov"] for p in packs)
    pack["_last_cov"] = min(p["_last_cov"] for p in packs)
    pack["_wait_cov"] = min(p["_wait_cov"] for p in packs)
    return pack


def fire(q: dict[str, Any], kind: str, thr: float, need: str = "last") -> dict[str, Any] | None:
    host_step = int(q.get("host_step") or 10**9)
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= host_step:
            continue
        if kind == "ratio":
            rise = base.last_val(ev["vals"].get("exit_minus_half"))
            exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
            if rise != rise or exit_s != exit_s:
                continue
            if rise / (abs(exit_s) + 1e-6) >= thr:
                return ev
            continue
        if kind == "conf":
            conf = rg.finite(q["rows"][ev["end"]].get("confidence"))
            if conf == conf and conf >= thr:
                return ev
            continue
        if kind == "oracle":
            ans = q["rows"][ev["end"]].get("final_answer")
            if rg.same(ans, q["gt"]) or rg.same(ans, q["a_final"]):
                return ev
            continue
        vals = ev["vals"].get(kind) or []
        if layers.passed(vals, thr, need):
            return ev
    return None


def eval_kind(pack: dict[str, Any], kind: str, thr: float, need: str = "last") -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = host_acc = host_tok = puma_acc = puma_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        host_ok = bool(q["host_ok"])
        host_t = float(q["host_tok"])
        host_acc += int(host_ok)
        host_tok += host_t
        puma_acc += int(bool(q["puma_ok"]))
        puma_tok += float(q["puma_tok"])
        ev = fire(q, kind, thr, need)
        if ev is None:
            acc += int(host_ok)
            tok += host_t
            continue
        sim = rg.pack(q["trials"], q["rows"], ev["end"], "rescue", original_tokens=q["orig_tok"])
        if kind == "oracle":
            ok = rg.same(sim["answer"], q["gt"]) or (
                bool(q["orig_ok"]) and rg.same(sim["answer"], q["a_final"])
            )
        else:
            ok = rg.same(sim["answer"], q["gt"])
        acc += int(ok)
        tok += sim["tokens"]
        n_fire += 1
        n_gain += int(ok and not host_ok)
        n_hurt += int((not ok) and host_ok)
    return {
        "n": n,
        "threshold": thr,
        "acc": acc / n,
        "tok": tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "d_host_acc": 100.0 * (acc / n - host_acc / n),
        "d_host_tok": tok / n - host_tok / n,
        "d_puma_acc": 100.0 * (acc / n - puma_acc / n),
        "d_puma_tok": tok / n - puma_tok / n,
    }


def score_xs(pack: dict[str, Any], kind: str, ratio: bool) -> list[float]:
    xs = []
    for q in pack["questions"]:
        host_step = int(q.get("host_step") or 10**9)
        for ev in q["events"]:
            if ev.get("high") or ev.get("mixed"):
                continue
            step = int(q["rows"][ev["end"]]["stopped_len"])
            if step < rg.MSS or step >= host_step:
                continue
            if ratio:
                rise = base.last_val(ev["vals"].get("exit_minus_half"))
                exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
                if rise == rise and exit_s == exit_s:
                    xs.append(rise / (abs(exit_s) + 1e-6))
            elif kind == "conf":
                conf = rg.finite(q["rows"][ev["end"]].get("confidence"))
                if conf == conf:
                    xs.append(conf)
            else:
                val = base.last_val(ev["vals"].get(kind))
                if val == val:
                    xs.append(val)
    return xs


def sweep(pack: dict[str, Any], kind: str, need: str, ratio: bool, cov_key: str) -> dict[str, Any] | None:
    if float(pack.get(cov_key) or 0.0) < READY:
        return None
    host = eval_kind(pack, kind, float("inf"), need)
    xs = score_xs(pack, kind, ratio)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = [eval_kind(pack, kind, thr, need) for thr in thrs]
    pick = accfirst.pick_acc_first_both(points, host["host_acc"], host["host_tok"])
    if pick is None:
        pick = host
    return pick


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def cell(rec: dict[str, Any] | None, *, show_thr: bool = False) -> str:
    if rec is None:
        return "未齐"
    if rec["threshold"] == float("inf") and rec["n_fire"] == 0:
        return f"{pair(rec['acc'], rec['tok'])}（不开）"
    extra = f"；门 {rec['threshold']:.3f}" if show_thr and rec["threshold"] == rec["threshold"] else ""
    return (
        f"{pair(rec['acc'], rec['tok'])}"
        f"（{rg.fmt_pp(rec['d_host_acc'])} / {rg.fmt_tok(rec['d_host_tok'])}；"
        f"{rec['n_fire']}火/{rec['n_gain']}救/{rec['n_hurt']}伤{extra}）"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    frozen_h = "| 集 | PUMA | 密探k4 | 对比率≥2.85 | 把握≥0.90 | 上限（金标或CoT） |"
    peek_h = "| 集 | 一半深 | 最后一层好写 | 收口比 Wait | 对比率 |"
    lines = [
        "# 密探k4 + 现有低置信门",
        "",
        "宿主是已存密探k4。附加门只看停点前的低置信连答窗，开火交试答、不重写；没开留密探k4 重写。",
        "Acc 只对金标。括号相对密探k4：百分点 / token；开火 / 救回 / 伤。",
        "冻住的门十集同一条。同集门偷看本集对错：相对密探k4 Acc 不降，再少 token。换集会塌。",
        "AIME 四个 seed。中间层覆盖不到 85% 的不报。",
        "",
        "## 冻住的门",
        "",
        frozen_h,
        "|---|---|---|---|---|---|",
    ]
    peek_lines = ["", "## 同集偷看（相对密探k4 Acc 不降）", "", peek_h, "|---|---|---|---|---|"]
    print(frozen_h, flush=True)
    for zh, tag, gdir, puma in door.MODELS:
        for ds_zh, dataset, is_aime in door.DS:
            pack = load_named(zh, tag, gdir, puma, ds_zh, dataset, is_aime)
            if pack is None:
                continue
            name = f"{zh} {ds_zh}"
            frozen = eval_kind(pack, "ratio", rg.REL_RISE_THR)
            conf = eval_kind(pack, "conf", 0.90)
            oracle = eval_kind(pack, "oracle", 0.0)
            host = frozen
            row = (
                f"| {name} | {pair(host['puma_acc'], host['puma_tok'])} "
                f"| {pair(host['host_acc'], host['host_tok'])} "
                f"| {cell(frozen)} | {cell(conf)} | {cell(oracle)} |"
            )
            print(row, flush=True)
            lines.append(row)
            peek_cells = []
            for key, _zh, sig, need, ratio in PEEK:
                cov = {
                    "half": "_half_cov",
                    "last": "_last_cov",
                    "wait": "_wait_cov",
                    "ratio": "_half_cov",
                }[key]
                rec = sweep(pack, "ratio" if ratio else sig, need, ratio, cov)
                peek_cells.append(cell(rec, show_thr=True))
                print(f"  {name} {key} {peek_cells[-1]}", flush=True)
            peek_lines.append(f"| {name} | " + " | ".join(peek_cells) + " |")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines + peek_lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
