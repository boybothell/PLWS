#!/usr/bin/env python3
"""On leftover low-conf windows: does keep/revise (forecast or future peek) separate stoppable?"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_cand_collapse as cc

TABLE = AE / "tables/leftover_stability.md"
FORECAST = AE / "results/confcal_judge/v2/dense_forecast/math-500"
MODELS = (("7B", "r1_7b"), ("8B", "nemotron_8b"))
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def load_forecast() -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    if not FORECAST.is_dir():
        return out
    for path in sorted(FORECAST.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in (None, "ok"):
                continue
            change = rg.finite(row.get("forecast_p_change"))
            if change != change:
                continue
            out[(int(row["question_idx"]), int(row["decision_step"]))] = 1.0 - change
    return out


def future_frac(rows: list[dict[str, Any]], end: int, k: int) -> float:
    ans = rows[end].get("final_answer")
    fut = rows[end + 1 : end + 1 + k]
    if not fut:
        return float("nan")
    return sum(rg.same(x.get("final_answer"), ans) for x in fut) / len(fut)


def stays_all(rows: list[dict[str, Any]], end: int) -> float:
    ans = rows[end].get("final_answer")
    fut = rows[end + 1 :]
    if not fut:
        return float("nan")
    return float(all(rg.same(x.get("final_answer"), ans) for x in fut))


def load_cell(model: str, dataset: str, seed: int, forecast: dict[tuple[int, int], float]) -> dict[str, Any] | None:
    regen_path = dd.regen_stat_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    gp = cc.gpath(model, dataset, seed)
    if not regen_path.is_file() or not puma_path.is_file() or not trials_path.is_file():
        return None
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    questions = []
    for qi, info in sorted(official.items()):
        host = regen.get(qi)
        trials = by.get(qi)
        if not host or not trials:
            continue
        rows = rg.usable_rows(trials)
        host_step = int(host.get("stopped_len") or 10**9)
        windows = []
        for end in cc.leftover_windows(rows, host_step):
            row = rows[end]
            ans = row.get("final_answer")
            step = int(row["stopped_len"])
            keep = forecast.get((qi, step), float("nan"))
            windows.append(
                {
                    "end": end,
                    "step": step,
                    "ans": ans,
                    "pos": cc.stoppable(
                        ans,
                        info.get("ground_truth"),
                        info.get("original_answer"),
                        (gmap.get(qi) or {}).get("A_final") or info.get("original_answer"),
                    ),
                    "geo": rg.finite(row.get("confidence")),
                    "forecast_keep": keep,
                    "future4": future_frac(rows, end, 4),
                    "future8": future_frac(rows, end, 8),
                    "stays": stays_all(rows, end),
                }
            )
        questions.append(
            {
                "qi": qi,
                "rows": rows,
                "trials": trials,
                "gt": info.get("ground_truth"),
                "original": info.get("original_answer"),
                "orig_ok": bool(info.get("original_correct")),
                "orig_tok": int(info.get("original_tokens") or 0),
                "host_ok": bool(host.get("compressed_correct")),
                "host_tok": int(host.get("compressed_tokens") or 0) + int(host.get("tokens_trial_answers") or 0),
                "windows": windows,
            }
        )
    return {"questions": questions}


def merge_packs(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs = []
    for pack in packs:
        qs.extend(pack["questions"])
    return {"questions": qs}


def fire_end(q: dict[str, Any], field: str, thr: float) -> int | None:
    for w in q["windows"]:
        val = float(w[field])
        if val == val and val >= thr:
            return int(w["end"])
    return None


def eval_door(pack: dict[str, Any], field: str, thr: float) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = host_acc = host_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        host_acc += int(q["host_ok"])
        host_tok += q["host_tok"]
        end = fire_end(q, field, thr)
        if end is None:
            acc += int(q["host_ok"])
            tok += q["host_tok"]
            continue
        sim = rg.pack(q["trials"], q["rows"], end, "rescue", original_tokens=q["orig_tok"])
        ok = cc.gold_ok(sim["answer"], q["gt"], q["original"], q["orig_ok"])
        acc += int(ok)
        tok += sim["tokens"]
        n_fire += 1
        n_gain += int(ok and not q["host_ok"])
        n_hurt += int((not ok) and q["host_ok"])
    return {
        "n": n,
        "threshold": thr,
        "acc": acc / n,
        "tok": tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "d_host_acc": 100.0 * (acc / n - host_acc / n),
        "d_host_tok": tok / n - host_tok / n,
    }


def peek(pack: dict[str, Any], field: str) -> dict[str, Any]:
    host = eval_door(pack, field, float("inf"))
    xs = [float(w[field]) for q in pack["questions"] for w in q["windows"] if w[field] == w[field]]
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=21))
    points = [eval_door(pack, field, thr) for thr in thrs]
    ok = [p for p in points if p["acc"] + 1e-12 >= host["host_acc"]]
    return min(ok, key=lambda p: (p["tok"], -p["acc"])) if ok else host


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def cell(rec: dict[str, Any]) -> str:
    if rec["n_fire"] == 0:
        return f"{pair(rec['acc'], rec['tok'])}（不开）"
    return (
        f"{pair(rec['acc'], rec['tok'])}"
        f"（{rg.fmt_pp(rec['d_host_acc'])} / {rg.fmt_tok(rec['d_host_tok'])}；"
        f"{rec['n_fire']}火/{rec['n_gain']}救/{rec['n_hurt']}伤）"
    )


def fmt_auc(v: float) -> str:
    return "—" if v != v else f"{v:.3f}"


def fmt_rate(v: float) -> str:
    return "—" if v != v else f"{100.0 * v:.0f}%"


def summarize(zh: str, model: str, ds_zh: str, dataset: str, forecast: dict[tuple[int, int], float]) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = [load_cell(model, dataset, seed, {}) for seed in dd.AIME_SEEDS]
        packs = [p for p in packs if p]
        if len(packs) != 4:
            print(f"skip {zh} {ds_zh} seeds={len(packs)}/4", flush=True)
            return None
        pack = merge_packs(packs)
    else:
        use_fc = forecast if (model == "r1_7b" and dataset == "math-500") else {}
        pack = load_cell(model, dataset, 42, use_fc)
        if pack is None:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
    wins = [w for q in pack["questions"] for w in q["windows"]]
    firsts = [q["windows"][0] for q in pack["questions"] if q["windows"]]
    rec: dict[str, Any] = {
        "name": f"{zh} {ds_zh}",
        "n_left": sum(1 for q in pack["questions"] if q["windows"]),
        "n_win": len(wins),
        "pack": pack,
    }
    for field in ("forecast_keep", "geo", "future4", "future8", "stays"):
        rec[f"auc_{field}"] = cc.auroc_of(wins, field)[0]
        rec[f"fauc_{field}"] = cc.auroc_of(firsts, field)[0]
        rec[f"{field}_pos"] = cc.rate(wins, field, True)
        rec[f"{field}_neg"] = cc.rate(wins, field, False)
    rec["n_fc"] = sum(1 for w in wins if w["forecast_keep"] == w["forecast_keep"])
    rec["peek_fc"] = peek(pack, "forecast_keep") if rec["n_fc"] else None
    rec["peek_f4"] = peek(pack, "future4")
    rec["peek_geo"] = peek(pack, "geo")
    rec["door_fc"] = eval_door(pack, "forecast_keep", 0.7) if rec["n_fc"] else None
    return rec


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    forecast = load_forecast()
    print(f"forecast keys={len(forecast)}", flush=True)
    lines = [
        "# 低把握窗上，稳不稳能不能分开可停",
        "",
        "只看密探k4 停点前、连续 4 步同一试答、把握都 < 0.995 的窗。正类 = 试答已等于金标或写完终答。",
        "4B 会不会改答：已有分数只在 7B MATH。后 4/8 步还同答、一路不改是偷看，只当诊断，不能当在线门。",
        "对照是同一窗上的几何置信度。门开火交试答、不重写。",
        "",
        "## 分得开吗",
        "",
        "| 集 | 剩窗 | 4B保持 AUROC/对上均值/没对上 | 把握 AUROC | 后4步同答 AUROC/对上/没对上 | 第一扇 4B/后4步 |",
        "|---|---:|---|---|---|---|",
    ]
    door_lines = [
        "",
        "## 当门（相对密探k4；后4步是偷看上限）",
        "",
        "| 集 | 密探k4 | 4B保持≥0.7 | 同集偷看4B | 同集偷看把握 | 偷看后4步同答 |",
        "|---|---|---|---|---|---|",
    ]
    print(lines[-2], flush=True)
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = summarize(zh, model, ds_zh, dataset, forecast)
            if rec is None:
                continue
            fc = (
                f"{fmt_auc(rec['auc_forecast_keep'])} / {fmt_rate(rec['forecast_keep_pos'])} / {fmt_rate(rec['forecast_keep_neg'])}"
                if rec["n_fc"]
                else "未抽"
            )
            row = (
                f"| {rec['name']} | {rec['n_left']} / {rec['n_win']} "
                f"| {fc} "
                f"| {fmt_auc(rec['auc_geo'])} "
                f"| {fmt_auc(rec['auc_future4'])} / {fmt_rate(rec['future4_pos'])} / {fmt_rate(rec['future4_neg'])} "
                f"| {fmt_auc(rec['fauc_forecast_keep'])} / {fmt_auc(rec['fauc_future4'])} |"
            )
            host = rec["peek_f4"]
            drow = (
                f"| {rec['name']} | {pair(host['host_acc'], host['host_tok'])} "
                f"| {cell(rec['door_fc']) if rec['door_fc'] else '未抽'} "
                f"| {cell(rec['peek_fc']) if rec['peek_fc'] else '未抽'} "
                f"| {cell(rec['peek_geo'])} "
                f"| {cell(rec['peek_f4'])} |"
            )
            print(row, flush=True)
            print(" ", drow, flush=True)
            if rec["n_fc"]:
                mid = [w for q in rec["pack"]["questions"] for w in q["windows"] if 0.5 <= float(w["geo"]) < 0.995]
                print(
                    f"  leftover mid-geo forecast {fmt_auc(cc.auroc_of(mid, 'forecast_keep')[0])} "
                    f"vs geo {fmt_auc(cc.auroc_of(mid, 'geo')[0])} n={len(mid)}",
                    flush=True,
                )
            lines.append(row)
            door_lines.append(drow)
    TABLE.write_text("\n".join(lines + door_lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
