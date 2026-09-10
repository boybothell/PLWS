#!/usr/bin/env python3
"""密探k4 多角度：停因、第一次可停、重写还是试答、全窗新尺子。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_cand_collapse as cc
import report_dense_k4_lowconf_ceiling as low

TABLE = AE / "tables/k4_angles.md"
MODELS = (("7B", "r1_7b"), ("8B", "nemotron_8b"))
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def window_kind(confs: list[float]) -> str:
    if rg.is_high(confs):
        return "high"
    if rg.is_low(confs):
        return "low"
    return "mix"


def distinct(rows: list[dict[str, Any]]) -> int:
    seen = []
    for row in rows:
        ans = row.get("final_answer")
        if not any(rg.same(ans, x) for x in seen):
            seen.append(ans)
    return len(seen)


def first_window(
    rows: list[dict[str, Any]],
    *,
    want: str | None,
    stoppable: bool | None,
    gt: Any,
    original: Any,
    a_final: Any,
) -> dict[str, Any] | None:
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        window = rows[end + 1 - rg.K : end + 1]
        confs = rg.confs_of(window)
        if not all(c == c for c in confs):
            continue
        kind = window_kind(confs)
        if want and kind != want:
            continue
        ans = row.get("final_answer")
        hit = cc.stoppable(ans, gt, original, a_final)
        if stoppable is True and not hit:
            continue
        if stoppable is False and hit:
            continue
        return {
            "end": end,
            "step": int(row["stopped_len"]),
            "ans": ans,
            "kind": kind,
            "conf": confs[-1],
            "conf_min": min(confs),
            "conf_rise": confs[-1] - confs[0],
            "hit": hit,
            "n_dist": distinct(rows[: end + 1]),
            "churn8": distinct(rows[max(0, end + 1 - 8) : end + 1]),
            "rel": int(row["stopped_len"]) / max(int(rows[-1]["stopped_len"]), 1),
        }
    return None


def all_windows(rows: list[dict[str, Any]], gt: Any, original: Any, a_final: Any) -> list[dict[str, Any]]:
    out = []
    n_last = max(int(rows[-1]["stopped_len"]), 1)
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        window = rows[end + 1 - rg.K : end + 1]
        confs = rg.confs_of(window)
        if not all(c == c for c in confs):
            continue
        ans = row.get("final_answer")
        out.append(
            {
                "end": end,
                "step": int(row["stopped_len"]),
                "kind": window_kind(confs),
                "pos": cc.stoppable(ans, gt, original, a_final),
                "geo": confs[-1],
                "geo_min": min(confs),
                "geo_rise": confs[-1] - confs[0],
                "neg_dist": -float(distinct(rows[: end + 1])),
                "neg_churn": -float(distinct(rows[max(0, end + 1 - 8) : end + 1])),
                "rel": int(row["stopped_len"]) / n_last,
            }
        )
    return out


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    regen_path = dd.regen_stat_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    gp = low.gpath(model, dataset, seed)
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
        if not rows:
            continue
        gt = info.get("ground_truth")
        original = info.get("original_answer")
        a_final = (gmap.get(qi) or {}).get("A_final") or original
        orig_ok = bool(info.get("original_correct"))
        orig_tok = int(info.get("original_tokens") or 0)
        host_step = int(host.get("stopped_len") or 10**9)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        trial_at_host = next((r for r in rows if int(r["stopped_len"]) == host_step), rows[-1])
        questions.append(
            {
                "qi": qi,
                "rows": rows,
                "trials": trials,
                "gt": gt,
                "original": original,
                "a_final": a_final,
                "orig_ok": orig_ok,
                "orig_tok": orig_tok,
                "host_ok": bool(host.get("compressed_correct")),
                "host_tok": int(host.get("compressed_tokens") or 0) + int(host.get("tokens_trial_answers") or 0),
                "puma_ok": bool(info.get("compressed_correct")),
                "puma_tok": int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0),
                "host_step": host_step,
                "host_reason": str(host.get("stop_reason") or sim["branch"]),
                "host_branch": sim["branch"],
                "trial_host_ok": low.credit(trial_at_host.get("final_answer"), gt, original, orig_ok),
                "first_ok": first_window(rows, want=None, stoppable=True, gt=gt, original=original, a_final=a_final),
                "first_w": first_window(rows, want="high", stoppable=False, gt=gt, original=original, a_final=a_final),
                "windows": all_windows(rows, gt, original, a_final),
            }
        )
    return {"questions": questions}


def merge(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs = []
    for pack in packs:
        qs.extend(pack["questions"])
    return {"questions": qs}


def pct(n: int, d: int) -> str:
    return "—" if d <= 0 else f"{100.0 * n / d:.0f}%"


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(v: float, nd: int = 3) -> str:
    return "—" if v != v else f"{v:.{nd}f}"


def anatomy(pack: dict[str, Any]) -> dict[str, Any]:
    qs = pack["questions"]
    n = len(qs)
    rec: dict[str, Any] = {"n": n}
    rec["host_acc"] = sum(q["host_ok"] for q in qs) / n
    rec["host_tok"] = sum(q["host_tok"] for q in qs) / n
    rec["puma_acc"] = sum(q["puma_ok"] for q in qs) / n
    rec["puma_tok"] = sum(q["puma_tok"] for q in qs) / n
    rec["trial_acc"] = sum(q["trial_host_ok"] for q in qs) / n
    rec["n_hit"] = sum(1 for q in qs if q["first_ok"])
    rec["n_high"] = sum(1 for q in qs if q["first_ok"] and q["first_ok"]["kind"] == "high")
    rec["n_low"] = sum(1 for q in qs if q["first_ok"] and q["first_ok"]["kind"] == "low")
    rec["n_mix"] = sum(1 for q in qs if q["first_ok"] and q["first_ok"]["kind"] == "mix")
    rec["n_none"] = sum(1 for q in qs if not q["first_ok"])
    rec["n_w"] = sum(1 for q in qs if q["first_w"])
    rec["late"] = []
    rec["early"] = []
    rec["on_time"] = 0
    for q in qs:
        if not q["first_ok"]:
            continue
        lag = q["host_step"] - q["first_ok"]["step"]
        if lag > 1:
            rec["late"].append(lag)
        elif lag < 0:
            rec["early"].append(-lag)
        else:
            rec["on_time"] += 1
    rec["n_late"] = len(rec["late"])
    rec["n_early"] = len(rec["early"])
    rec["lag_p50"] = rg.p50(rec["late"]) if rec["late"] else float("nan")
    rec["n_consec"] = sum(1 for q in qs if q["host_branch"] == "consec")
    rec["n_fs"] = sum(1 for q in qs if q["host_branch"] == "fs")
    rec["n_full"] = sum(1 for q in qs if q["host_branch"] == "full")
    rec["tok_consec"] = mean([q["host_tok"] - q["puma_tok"] for q in qs if q["host_branch"] == "consec"])
    rec["tok_fs"] = mean([q["host_tok"] - q["puma_tok"] for q in qs if q["host_branch"] == "fs"])
    rec["tok_full"] = mean([q["host_tok"] - q["puma_tok"] for q in qs if q["host_branch"] == "full"])
    rec["gain"] = sum(1 for q in qs if q["host_ok"] and not q["puma_ok"])
    rec["hurt"] = sum(1 for q in qs if (not q["host_ok"]) and q["puma_ok"])
    rec["regen_save"] = sum(1 for q in qs if q["host_ok"] and not q["trial_host_ok"])
    rec["regen_hurt"] = sum(1 for q in qs if (not q["host_ok"]) and q["trial_host_ok"])
    rec["w_then_ok"] = sum(1 for q in qs if q["first_w"] and q["host_ok"])
    rec["w_then_bad"] = sum(1 for q in qs if q["first_w"] and not q["host_ok"])
    return rec


def signal_table(pack: dict[str, Any]) -> dict[str, Any]:
    wins = [w for q in pack["questions"] for w in q["windows"]]
    out: dict[str, Any] = {"n": len(wins), "n_pos": sum(1 for w in wins if w["pos"])}
    for field in ("geo", "geo_min", "geo_rise", "neg_dist", "neg_churn", "rel"):
        out[f"all_{field}"] = cc.auroc_of(wins, field)[0]
        for kind in ("high", "low", "mix"):
            sub = [w for w in wins if w["kind"] == kind]
            out[f"{kind}_{field}"] = cc.auroc_of(sub, field)[0]
        mid = [w for w in wins if 0.5 <= float(w["geo"]) < 0.995]
        out[f"mid_{field}"] = cc.auroc_of(mid, field)[0]
    out["n_high"] = sum(1 for w in wins if w["kind"] == "high")
    out["n_low"] = sum(1 for w in wins if w["kind"] == "low")
    out["n_mix"] = sum(1 for w in wins if w["kind"] == "mix")
    out["n_mid"] = sum(1 for w in wins if 0.5 <= float(w["geo"]) < 0.995)
    return out


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = [load_cell(model, dataset, seed) for seed in dd.AIME_SEEDS]
        packs = [p for p in packs if p]
        if len(packs) != 4:
            print(f"skip {zh} {ds_zh} seeds={len(packs)}/4", flush=True)
            return None
        pack = merge(packs)
    else:
        pack = load_cell(model, dataset, 42)
        if pack is None:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
    rec = anatomy(pack)
    rec["sig"] = signal_table(pack)
    rec["name"] = f"{zh} {ds_zh}"
    print(
        f"{rec['name']} n={rec['n']} first_ok={rec['n_hit']} high/low/mix="
        f"{rec['n_high']}/{rec['n_low']}/{rec['n_mix']} late={rec['n_late']}",
        flush=True,
    )
    return rec


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    lines = [
        "# 密探k4 多角度",
        "",
        "不再只盯低把握窗。看三件事：第一次可停（试答已等于金标或写完终答）在哪；",
        "密探k4 相对 PUMA 的对错和长短从哪来；全窗上几条便宜尺子还分不分得开。",
        "可停 = 连续 4 步同一试答，且等于金标或写完终答。高把握 = 窗里第一次 ≥ 0.995。",
        "重写 = 密探k4 现在的交卷；试答 = 停点上那次 boxed。",
        "",
        "## 1. 第一次可停是哪种窗",
        "",
        "| 集 | 题数 | 有可停 | 高把握先到 | 低把握先到 | 混合先到 | 从没可停 | 高把握但不可停（假锁） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n']} | {rec['n_hit']}（{pct(rec['n_hit'], rec['n'])}） "
            f"| {rec['n_high']} | {rec['n_low']} | {rec['n_mix']} | {rec['n_none']} | {rec['n_w']} |"
        )
    lines += [
        "",
        "## 2. 密探k4 相对第一次可停是早了还是晚了",
        "",
        "晚了 = 已经有可停锁，门还在往后写。早了 = 门先停，可停锁在停点之后才出现。",
        "",
        "| 集 | 对准 | 晚了（中位晚几步） | 早了 | 连答停 / 后路 / 写完 |",
        "|---|---:|---|---:|---|",
    ]
    for rec in recs:
        late = f"{rec['n_late']}（中位 {rec['lag_p50']:.0f} 步）" if rec["n_late"] else "0"
        lines.append(
            f"| {rec['name']} | {rec['on_time']} | {late} | {rec['n_early']} "
            f"| {rec['n_consec']} / {rec['n_fs']} / {rec['n_full']} |"
        )
    lines += [
        "",
        "## 3. 对 PUMA：谁在涨正确率、谁在多花钱",
        "",
        "| 集 | 密探k4 | 相对PUMA | 救回 / 伤 | 停点交试答Acc | 重写救回 / 重写写坏 | 连答题token差 | 后路 | 写完 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f} "
            f"| {rg.fmt_pp(100*(rec['host_acc']-rec['puma_acc']))} / {rg.fmt_tok(rec['host_tok']-rec['puma_tok'])} "
            f"| {rec['gain']} / {rec['hurt']} "
            f"| {rg.fmt_pct(rec['trial_acc'])} "
            f"| {rec['regen_save']} / {rec['regen_hurt']} "
            f"| {fmt(rec['tok_consec'], 0)} | {fmt(rec['tok_fs'], 0)} | {fmt(rec['tok_full'], 0)} |"
        )
    lines += [
        "",
        "## 4. 全窗新尺子（AUROC，越大越像可停）",
        "",
        "不只低把握。把握 = 窗最后一步；最差把握 = 四步里最低；把握升降 = 最后减第一；",
        "越少换答 = 到这步为止不同试答个数取负；近 8 步少换；相对步数 = 当前步 / 全长。",
        "中间带 = 把握在 0.50–0.995，看是不是只剩把握自己。",
        "",
        "| 集 | 窗数 | 把握 全/高/低/中间带 | 最差把握 | 把握升降 | 越少换答 | 近8步少换 | 相对步数 |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for rec in recs:
        s = rec["sig"]
        lines.append(
            f"| {rec['name']} | {s['n']} "
            f"| {fmt(s['all_geo'])} / {fmt(s['high_geo'])} / {fmt(s['low_geo'])} / {fmt(s['mid_geo'])} "
            f"| {fmt(s['all_geo_min'])} | {fmt(s['all_geo_rise'])} "
            f"| {fmt(s['all_neg_dist'])} | {fmt(s['all_neg_churn'])} | {fmt(s['all_rel'])} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "高把握先到的题，密探k4 已经该停；再找标量和把握抢功。",
        "低把握先到、或写完才可停，才是上限还空着的地方。假锁（高把握但不是终答/金标）要靠重写托 Acc。",
        "新尺子如果全窗接近把握、中间带掉到 0.5，就不是新信息。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
