#!/usr/bin/env python3
"""Previous lag doors, same-set: Acc ≥ 写完, then min tokens."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_brightest_stop_auroc as jobs
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import report_puma_plus_r_oracle as base

TABLE = AE / "tables/prev_sameset_tok.md"
READY = 0.85
SIGS = (
    ("half", "一半深", "exit_minus_half", "all", False),
    ("last", "最后一层好写", "last_mean_logp", "last", False),
    ("wait", "收口比 Wait", "stop_margin", "last", False),
    ("ratio", "对比率", "exit_minus_half", "last", True),
)


def credit(q: dict[str, Any], ans: Any) -> int:
    if rg.same(ans, q["gt"]):
        return 1
    if q["orig_ok"] and rg.same(ans, q["a_final"]):
        return 1
    return 0


def full_step(q: dict[str, Any]) -> int:
    if q["rows"]:
        return int(q["rows"][-1]["stopped_len"]) + 1
    return 10**9


def fire(q: dict[str, Any], sig: str, need: str, thr: float, ratio: bool) -> dict[str, Any] | None:
    host = full_step(q)
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= host:
            continue
        if ratio:
            rise = base.last_val(ev["vals"].get("exit_minus_half"))
            exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
            if rise != rise or exit_s != exit_s:
                continue
            if rise / (abs(exit_s) + 1e-6) >= thr:
                return ev
            continue
        vals = ev["vals"].get(sig) or []
        if layers.passed(vals, thr, need):
            return ev
    return None


def score_xs(pack: dict[str, Any], sig: str, ratio: bool) -> list[float]:
    xs = []
    for q in pack["questions"]:
        host = full_step(q)
        for ev in q["events"]:
            if ev.get("high") or ev.get("mixed"):
                continue
            step = int(q["rows"][ev["end"]]["stopped_len"])
            if step < rg.MSS or step >= host:
                continue
            if ratio:
                rise = base.last_val(ev["vals"].get("exit_minus_half"))
                exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
                if rise == rise and exit_s == exit_s:
                    xs.append(rise / (abs(exit_s) + 1e-6))
            else:
                val = base.last_val(ev["vals"].get(sig))
                if val == val:
                    xs.append(val)
    return xs


def eval_thr(
    pack: dict[str, Any], sig: str, need: str, thr: float, ratio: bool
) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = orig_acc = orig_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        orig_acc += int(q["orig_ok"])
        orig_tok += q["orig_tok"]
        ev = fire(q, sig, need, thr, ratio)
        if ev is None:
            acc += int(q["orig_ok"])
            tok += q["orig_tok"]
            continue
        sim = rg.pack(
            q["trials"],
            q["rows"],
            ev["end"],
            "rescue",
            original_tokens=q["orig_tok"],
        )
        ok = credit(q, sim["answer"])
        acc += ok
        tok += sim["tokens"]
        n_fire += 1
        n_gain += int(ok and not q["orig_ok"])
        n_hurt += int((not ok) and q["orig_ok"])
    return {
        "threshold": thr,
        "acc": acc / n,
        "tok": tok / n,
        "orig_acc": orig_acc / n,
        "orig_tok": orig_tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "d_acc": 100.0 * (acc / n - orig_acc / n),
        "d_tok": tok / n - orig_tok / n,
    }


def pick_tok_first(points: list[dict[str, Any]], base_acc: float) -> dict[str, Any]:
    ok = [p for p in points if p["acc"] + 1e-12 >= base_acc]
    if not ok:
        return sorted(points, key=lambda p: (-p["acc"], p["tok"]))[0]
    return sorted(ok, key=lambda p: (p["tok"], -p["acc"]))[0]


def sweep(pack: dict[str, Any], sig: str, need: str, ratio: bool) -> dict[str, Any] | None:
    cov = cmp.cov(pack, "exit_minus_half" if ratio else sig)
    if cov < READY:
        return None
    xs = score_xs(pack, sig, ratio)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = [eval_thr(pack, sig, need, thr, ratio) for thr in thrs]
    return pick_tok_first(points, points[0]["orig_acc"])


def load_78() -> list[tuple[str, dict[str, Any]]]:
    layers.enrich = more.enrich_all
    out = []
    for zh, tag, gdir, puma in jobs.MODELS[:2]:
        for ds_zh, dataset in jobs.BIG + jobs.AIME:
            is_aime = dataset in ("aime24", "aime25")
            cells = (
                [jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset, seed) for seed in jobs.SEEDS]
                if is_aime
                else [jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset)]
            )
            packs = []
            for cell in cells:
                cell = dict(cell)
                if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
                    continue
                if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
                    continue
                pack = layers.load_pack(cell)
                official = {int(r["question_idx"]): r for r in rg.load_json(cell["stat"])}
                for q in pack["questions"]:
                    q["puma_step"] = int((official.get(int(q["qi"])) or {}).get("stopped_len") or 10**9)
                cons.attach_shared(pack)
                names = ["exit_minus_half", "last_mean_logp", "stop_margin"]
                layers.precompute_events(pack, names)
                pack["_half_cov"] = cmp.cov(pack, "exit_minus_half")
                pack["_last_cov"] = cmp.cov(pack, "last_mean_logp")
                pack["_wait_cov"] = cmp.cov(pack, "stop_margin")
                pack["_cov"] = pack["_half_cov"]
                packs.append(pack)
                print(
                    f"load {zh} {ds_zh}"
                    + (f"-s{cell.get('seed')}" if is_aime else "")
                    + f" n={len(pack['questions'])} half={pack['_half_cov']:.2f} "
                    f"wait={pack['_wait_cov']:.2f}",
                    flush=True,
                )
            if is_aime and len(packs) != 4:
                print(f"skip {zh} {ds_zh} seeds={len(packs)}/4", flush=True)
                continue
            if not packs:
                continue
            pack = accfirst.merge_packs(packs, f"{zh} {ds_zh}") if len(packs) > 1 else packs[0]
            pack["_half_cov"] = min(p["_half_cov"] for p in packs)
            pack["_last_cov"] = min(p["_last_cov"] for p in packs)
            pack["_wait_cov"] = min(p["_wait_cov"] for p in packs)
            out.append((f"{zh} {ds_zh}", pack))
    return out


def fmt_cell(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "未齐"
    if rec["threshold"] == float("inf"):
        return f"{rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}（不开）"
    return (
        f"{rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}"
        f"（{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])}；"
        f"{rec['n_fire']}火/{rec['n_hurt']}伤）"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    header = "| 集 | 写完全程 | 一半深 | 最后一层好写 | 收口比 Wait | 对比率 |"
    sep = "|---|---|---|---|---|---|"
    lines = [
        "# 对比率之前的门：同集 Acc 不降、最少 token",
        "",
        "写完底，交试答。只看低置信连答窗。同集偷看门槛：Acc ≥ 写完时取 token 最少。",
        "一半深 = 末层好写 − 一半深，4 步都过。最后一层好写 / 收口比 Wait = 只看窗最后一步。",
        "对比率列是同一协议下的对照。换集会塌。AIME 四个 seed。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, pack in load_78():
        orig = None
        cells = []
        for key, _zh, sig, need, ratio in SIGS:
            rec = sweep(pack, sig, need, ratio)
            if rec is not None and orig is None:
                orig = rec
            cells.append(fmt_cell(rec))
            print(f"  {name} {key} {cells[-1]}", flush=True)
        if orig is None:
            row = f"| {name} | 未齐 | 未齐 | 未齐 | 未齐 | 未齐 |"
        else:
            row = (
                f"| {name} | {rg.fmt_pct(orig['orig_acc'])} / {orig['orig_tok']:.0f} "
                f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"
            )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
