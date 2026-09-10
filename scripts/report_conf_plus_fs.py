#!/usr/bin/env python3
"""Add the old FS force-stop onto the current high-conf door. Trial-as-final.

Current door: k=4, first>=0.98, later>=first-0.03, no stop before step 10.
FS (unchanged): t>=80, length-red run>=2, traj best_conf>=0.85,
                last 2 probes same answer, peak-same conf>=0.7.
Length red: adjacent step char-len ratio in [0.84, 1.16] or |Δlen|<=35.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_acctok as acc
import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as old
import replay_rescue_R_gate as rg
import report_conf_vs_puma_all as grid
import report_unified_same_set as uni

SIG = "exit_minus_half"

FS_MIN_STEP = old.FS_MIN_STEP
FS_RUN = old.FS_RUN
FS_BEST = old.FS_BEST
FS_DEC = old.FS_DEC
FS_SAME_K = old.FS_SAME_K


def pick_peak_same(probed: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not probed:
        return None
    last = probed[-1]["answer"]
    same_ans = [x for x in probed if rg.same(x["answer"], last)]
    if not same_ans:
        return None
    return max(same_ans, key=lambda x: x["conf"])


def pack_fs(
    trials: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    end: int,
    pick: dict[str, Any],
    original_tokens: int,
) -> dict[str, Any]:
    rec = rg.pack(trials, rows, end, "fs", original_tokens=original_tokens)
    rec["answer"] = pick["answer"]
    rec["conf"] = pick["conf"]
    rec["fs_pick_step"] = pick["step"]
    return rec


def simulate(
    q: dict[str, Any],
    *,
    scores: dict[tuple[int, int], dict[str, Any]] | None = None,
    signal: str | None = None,
    threshold: float = float("inf"),
    use_fs: bool = True,
) -> dict[str, Any]:
    trials = sorted(q["trials"], key=lambda x: int(x["stopped_len"]))
    rows = q.get("rows") or rg.usable_rows(trials)
    lens = old.step_char_lens(trials)
    red_run = 0
    best_conf = float("-inf")
    probed: list[dict[str, Any]] = []
    walked = 0
    scores = scores or {}

    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if tstep >= FS_MIN_STEP:
                red_run = red_run + 1 if old.is_red(tstep, lens) else 0
            walked += 1
        conf = rg.finite(row.get("confidence"))
        answer = str(row.get("final_answer") or "")
        if answer and conf == conf:
            best_conf = max(best_conf, conf)
            probed.append({"step": step, "answer": answer, "conf": conf})
        if not rg.can_stop_step(row):
            continue
        if rg.window_ok(rows, end):
            window = rows[end + 1 - rg.K : end + 1]
            confs = rg.confs_of(window)
            if rg.is_high(confs):
                return rg.pack(trials, rows, end, "conf", original_tokens=q["orig_tok"])
            if (
                signal
                and (not rg.is_high(confs))
                and rg.is_low(confs)
            ):
                vals = [
                    rg.finite(scores.get((q["qi"], int(x["stopped_len"])), {}).get(signal))
                    for x in window
                ]
                if layers.passed(vals, threshold, "all"):
                    usable = [v for v in vals if v == v]
                    return rg.pack(
                        trials,
                        rows,
                        end,
                        "rescue",
                        original_tokens=q["orig_tok"],
                        score=min(usable) if usable else float("nan"),
                    )
        if use_fs and (
            step >= FS_MIN_STEP
            and red_run >= FS_RUN
            and best_conf >= FS_BEST
            and len(probed) >= FS_SAME_K
            and all(rg.same(x["answer"], probed[-1]["answer"]) for x in probed[-FS_SAME_K:])
        ):
            pick = pick_peak_same(probed)
            if pick is not None and pick["conf"] >= FS_DEC:
                return pack_fs(trials, rows, end, pick, q["orig_tok"])
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def run_combo(
    pack: dict[str, Any],
    *,
    signal: str | None = None,
    threshold: float = float("inf"),
    use_fs: bool = True,
) -> list[dict[str, Any]]:
    out = []
    for q in pack["questions"]:
        sim = simulate(
            q, scores=pack.get("scores"), signal=signal, threshold=threshold, use_fs=use_fs
        )
        out.append(
            {
                **sim,
                "ok": rg.hit(sim["answer"], q["gt"], q["a_final"], q["orig_ok"])
                if sim["early"]
                else q["orig_ok"],
                "qi": q["qi"],
                "puma_tok": q["puma_tok"],
                "puma_ok": q["puma_ok"],
                "orig_tok": q["orig_tok"],
                "orig_ok": q["orig_ok"],
                "has_r": "R" in q.get("labels", []),
                "r_only": "R" in q.get("labels", []) and "G" not in q.get("labels", []),
            }
        )
    return out


def run_fs(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return run_combo(pack, use_fs=True)


def lag_values(pack: dict[str, Any]) -> list[float]:
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            vals = ev["vals"].get(SIG) or []
            if vals and vals[-1] == vals[-1]:
                xs.append(vals[-1])
    return xs


def sweep_combo(pack: dict[str, Any], base_rows: list[dict[str, Any]], base_acc: float):
    points = []
    for thr in rg.quantiles(lag_values(pack), n=17):
        rows = run_combo(pack, signal=SIG, threshold=thr, use_fs=True)
        rec = layers.contrast(base_rows, rows, {"acc": base_acc, "tok": 0.0})
        rec["threshold"] = thr
        rec["rows"] = rows
        rec.update(summarize(rows))
        rec["d_acc_pp_conf"] = 100.0 * (rec["acc"] - base_acc)
        rec["d_tok_conf"] = rec["tok"] - sum(r["tokens"] for r in base_rows) / max(len(base_rows), 1)
        points.append(rec)
    return acc.pick_acc(points), uni.pick_flat(points, base_acc)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = rg.summarize(rows)
    s["n_fs"] = sum(int(r["branch"] == "fs") for r in rows)
    s["frac_fs"] = s["n_fs"] / max(s["n"], 1)
    fs = [r for r in rows if r["branch"] == "fs"]
    s["fs_ok"] = sum(int(r["ok"]) for r in fs)
    s["fs_would_full_ok"] = sum(int(r["orig_ok"]) for r in fs)
    return s


def bucket_line(rows: list[dict[str, Any]]) -> None:
    b: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        we = "我们提前" if r["early"] else "我们写完"
        puma = "PUMA提前" if int(r["puma_tok"]) < int(r["orig_tok"]) - 1 else "PUMA写完"
        b[(we, puma)].append(r)
    print(f"    {'两边怎么停':<22} 题数  我们token  PUMA  差/题  我们对  PUMA对")
    for key in sorted(b):
        rs = b[key]
        wt = sum(x["tokens"] for x in rs) / len(rs)
        pt = sum(x["puma_tok"] for x in rs) / len(rs)
        print(
            f"    {key[0]}/{key[1]:<10} {len(rs):4d}  {wt:8.0f}  {pt:7.0f}  "
            f"{wt - pt:+6.0f}  {sum(int(x['ok']) for x in rs):4d}  "
            f"{sum(int(x['puma_ok']) for x in rs):4d}"
        )


def line(name: str, acc: float, tok: float, pack: dict[str, Any], extra: str = "") -> None:
    print(
        f"  {name:<28} {rg.fmt_pct(acc)} / {tok:.0f}  "
        f"vs PUMA {rg.fmt_pp(100 * (acc - pack['puma_acc']))} / {rg.fmt_tok(tok - pack['puma_tok'])}"
        f"{extra}",
        flush=True,
    )


def has_scores(cell: dict[str, Any]) -> bool:
    return any(Path(p).is_dir() for p in cell.get("scores") or ())


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    print(
        f"高置信度门：k={rg.K} 第一次≥{rg.TAU} 后面≥第一次−{rg.EPS} 前{rg.MSS}步不许停。试答即终答。",
        flush=True,
    )
    print(
        f"FS：步数≥{FS_MIN_STEP}，长度红连续≥{FS_RUN}，轨迹最高置信度≥{FS_BEST}，"
        f"最近{FS_SAME_K}次同答，peak_same 置信度≥{FS_DEC}。",
        flush=True,
    )
    print(
        "滞后：最后一层减一半深。选门槛按题：正确率不降且 token 不多时先取正确率最高；"
        "不要求每题第一扇低置信窗就停，理想第一扇、后面停也算。",
        flush=True,
    )

    cells = [c for c in list(rg.CELLS) + [more.EXTRA] if has_scores(c)]
    for cell in cells:
        pack = layers.load_pack(cell)
        cons.attach_shared(pack)
        layers.precompute_events(pack, [SIG])
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf = rg.summarize(conf_rows)
        lag_best, lag_flat = uni.sweep(pack, conf_rows, conf)
        fs_rows = run_fs(pack)
        fs = summarize(fs_rows)
        combo_best, combo_flat = sweep_combo(pack, fs_rows, fs["acc"])
        combo_vs_conf = uni.pick_flat(
            [p for p in ([combo_flat, combo_best] if combo_flat or combo_best else []) if p],
            conf["acc"],
        )
        # stricter: lag+FS holding conf-only Acc
        points_vs_conf = []
        for thr in rg.quantiles(lag_values(pack), n=17):
            rows = run_combo(pack, signal=SIG, threshold=thr, use_fs=True)
            rec = summarize(rows)
            rec["threshold"] = thr
            rec["rows"] = rows
            rec["d_acc_pp_conf"] = 100.0 * (rec["acc"] - conf["acc"])
            rec["d_tok_conf"] = rec["tok"] - conf["tok"]
            points_vs_conf.append(rec)
        combo_flat_vs_conf = uni.pick_flat(points_vs_conf, conf["acc"])

        print(f"\n======== {cell['name']}  {len(pack['questions'])}题 ========", flush=True)
        line("官方 PUMA", pack["puma_acc"], pack["puma_tok"], pack)
        line("只看置信度", conf["acc"], conf["tok"], pack, f"  写完{conf['n_full']}")
        if lag_flat:
            line(
                "置信度+滞后 · 正确率不降",
                lag_flat["acc"],
                lag_flat["tok"],
                pack,
                f"  门槛{lag_flat['threshold']:.3f}  vs置信度 {rg.fmt_pp(lag_flat['d_acc_pp_conf'])} / {rg.fmt_tok(lag_flat['d_tok_conf'])}  "
                f"放进{lag_flat['rescue_R']}/{lag_flat['rescue_L']}",
            )
        if lag_best and (not lag_flat or abs(lag_best["acc"] - lag_flat["acc"]) > 1e-12 or abs(lag_best["tok"] - lag_flat["tok"]) > 0.5):
            line(
                "置信度+滞后 · 正确率优先(同集)",
                lag_best["acc"],
                lag_best["tok"],
                pack,
                f"  门槛{lag_best['threshold']:.3f}  vs置信度 {rg.fmt_pp(lag_best['d_acc_pp_conf'])} / {rg.fmt_tok(lag_best['d_tok_conf'])}",
            )
        line(
            "置信度+FS",
            fs["acc"],
            fs["tok"],
            pack,
            f"  vs置信度 {rg.fmt_pp(100*(fs['acc']-conf['acc']))} / {rg.fmt_tok(fs['tok']-conf['tok'])}  "
            f"FS{fs['n_fs']} 写完{fs['n_full']}",
        )
        if combo_flat:
            line(
                "置信度+FS+滞后 · 不降相对FS",
                combo_flat["acc"],
                combo_flat["tok"],
                pack,
                f"  门槛{combo_flat['threshold']:.3f}  vs FS {rg.fmt_pp(combo_flat['d_acc_pp_conf'])} / {rg.fmt_tok(combo_flat['d_tok_conf'])}  "
                f"滞后{combo_flat['n_rescue']} FS{combo_flat['n_fs']}",
            )
        if combo_flat_vs_conf:
            line(
                "置信度+FS+滞后 · 不降相对置信度",
                combo_flat_vs_conf["acc"],
                combo_flat_vs_conf["tok"],
                pack,
                f"  门槛{combo_flat_vs_conf['threshold']:.3f}  vs置信度 {rg.fmt_pp(combo_flat_vs_conf['d_acc_pp_conf'])} / {rg.fmt_tok(combo_flat_vs_conf['d_tok_conf'])}  "
                f"滞后{combo_flat_vs_conf['n_rescue']} FS{combo_flat_vs_conf['n_fs']}",
            )
        elif points_vs_conf:
            print("  置信度+FS+滞后无法在正确率不低于只看置信度时打开", flush=True)

        if cell["name"] == "奥赛":
            print("  分桶 只看置信度 / 置信度+滞后不降 / 置信度+FS / 三者都开（相对FS不降）", flush=True)
            print("  只看置信度：")
            bucket_line(
                [
                    {**r, **{k: q[k] for k in ("puma_tok", "puma_ok", "orig_tok", "orig_ok")}}
                    for r, q in zip(conf_rows, pack["questions"])
                ]
            )
            if lag_flat:
                print("  置信度+滞后 · 正确率不降：")
                bucket_line(
                    [
                        {**r, **{k: q[k] for k in ("puma_tok", "puma_ok", "orig_tok", "orig_ok")}}
                        for r, q in zip(
                            layers.run_pack(pack, SIG, lag_flat["threshold"], "all"),
                            pack["questions"],
                        )
                    ]
                )
            print("  置信度+FS：")
            bucket_line(fs_rows)
            if combo_flat:
                print("  置信度+FS+滞后 · 不降相对FS：")
                bucket_line(combo_flat["rows"])


if __name__ == "__main__":
    main()
