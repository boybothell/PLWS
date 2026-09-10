#!/usr/bin/env python3
"""Scale-free relative layer scores on PUMA-base: where it forms, how much exit gained."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import screen_overall_layer as ov
import screen_puma_layers as sc
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_relative_layer.md"


def last(vals: list[float] | None) -> float:
    if not vals:
        return float("nan")
    return vals[-1] if vals[-1] == vals[-1] else float("nan")


def rel_best(ev: dict[str, Any], ids: list[int], prefix: str) -> float:
    top = max(ids)
    best_l, best_v = None, float("-inf")
    for layer in ids:
        val = last(ev["vals"].get(f"{prefix}{layer}"))
        if val == val and val > best_v:
            best_v, best_l = val, layer
    if best_l is None:
        return float("nan")
    return best_l / top


def rise_over_exit(ev: dict[str, Any]) -> float:
    rise = last(ev["vals"].get("exit_minus_half"))
    exit_s = last(ev["vals"].get("last_mean_logp"))
    if rise != rise or exit_s != exit_s:
        return float("nan")
    return rise / (abs(exit_s) + 1e-6)


def rise_over_sum(ev: dict[str, Any]) -> float:
    rise = last(ev["vals"].get("exit_minus_half"))
    exit_s = last(ev["vals"].get("last_mean_logp"))
    if rise != rise or exit_s != exit_s:
        return float("nan")
    half = exit_s - rise
    return rise / (abs(exit_s) + abs(half) + 1e-6)


def rel_emerge(ev: dict[str, Any], top: int) -> float:
    val = last(ev["vals"].get("emerge_layer"))
    if val != val or top <= 0:
        return float("nan")
    return val / top


def attach_rel(pack: dict[str, Any]) -> None:
    ids = pack["_ids"]
    top = max(ids) if ids else 1
    for q in pack["questions"]:
        for ev in q["_act"]:
            ev["rel"] = {
                "最好写层在哪一截（lens）": rel_best(ev, ids, "lens_l"),
                "最好写层在哪一截（DoLA）": rel_best(ev, ids, "dola_mean_l"),
                "答案浮出层在哪一截": rel_emerge(ev, top),
                "出口抬升 / 出口自身": rise_over_exit(ev),
                "出口抬升 / （出口+一半深）": rise_over_sum(ev),
                "出口减一半深（生值）": last(ev["vals"].get("exit_minus_half")),
            }


def first_auroc(pack: dict[str, Any], name: str) -> tuple[float, int, int]:
    pos, neg = [], []
    for q in pack["questions"]:
        if not q["_act"]:
            continue
        score = q["_act"][0]["rel"].get(name)
        if score != score:
            continue
        (pos if q["_act"][0]["tag"] == "R" else neg).append(score)
    return rg.auroc(pos, neg), len(pos), len(neg)


def cache(pack: dict[str, Any], name: str) -> list[list[tuple[float, dict[str, Any]]]]:
    out = []
    for q in pack["questions"]:
        rows = []
        for ev in q["_act"]:
            score = ev["rel"].get(name)
            if score == score:
                rows.append((score, ev))
        out.append(rows)
    return out


def main() -> None:
    packs = []
    for name, pack in sc.load_ready():
        attach_rel(pack)
        packs.append((name, pack))
    sigs = []
    for _, pack in packs:
        for q in pack["questions"]:
            if q["_act"]:
                sigs = list(q["_act"][0]["rel"])
                break
        if sigs:
            break
    names = [n for n, _ in packs]
    seven = [n for n in names if n.startswith("7B ")]
    eight = [n for n in names if n.startswith("8B ")]

    local: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in sigs}
    cached: dict[str, dict[str, Any]] = {s: {} for s in sigs}
    print("| 集 | 分数 | 先窗AUROC | 本集最好 vs PUMA | 门槛 |", flush=True)
    for name, pack in packs:
        for sig in sigs:
            auc, nr, nl = first_auroc(pack, sig)
            c = cache(pack, sig)
            cached[sig][name] = c
            rec = ov.sweep_cached(pack, c)
            local[sig][name] = rec
            if rec is None:
                print(f"| {name} | {sig} | {auc:.3f}（{nr}/{nl}） | 不开 | — |", flush=True)
                continue
            print(
                f"| {name} | {sig} | {auc:.3f}（{nr}/{nl}） | "
                f"{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])} | "
                f"{rec['threshold']:.3f} |",
                flush=True,
            )

    def freeze(sig: str, group: list[str]) -> tuple[float, dict[str, dict[str, Any]], float]:
        pool = []
        for n in group:
            for rows in cached[sig][n]:
                pool.extend(s for s, _ in rows)
        best_thr, best_recs, best_mean = 0.0, {}, -1e9
        for thr in rg.quantiles(pool, n=41):
            recs = {}
            for n, pack in packs:
                if n not in group:
                    continue
                recs[n] = ov.eval_cfg_from_cache if False else None
            recs = {n: ov.sweep_cached.__wrapped__ if False else None for n in group}
            recs = {}
            pack_map = dict(packs)
            for n in group:
                recs[n] = eval_thr(pack_map[n], cached[sig][n], thr)
            mean = sum(r["d_acc"] for r in recs.values()) / len(recs)
            if mean > best_mean:
                best_thr, best_recs, best_mean = thr, recs, mean
        return best_thr, best_recs, best_mean

    def eval_thr(pack, c, thr):
        # local copy of ov.sweep one thr
        n = len(pack["questions"])
        acc = tok = puma_acc = puma_tok = 0.0
        for q, rows in zip(pack["questions"], c):
            puma_acc += int(q["puma_ok"])
            puma_tok += q["puma_tok"]
            hit = next((ev for score, ev in rows if score >= thr), None)
            if hit is None:
                acc += int(q["puma_ok"])
                tok += q["puma_tok"]
                continue
            sim = rg.pack(
                q["trials"], q["rows"], hit["end"], "rescue",
                original_tokens=q["orig_tok"], label=hit["tag"],
            )
            ok = rg.same(sim["answer"], q["gt"])
            acc += int(ok)
            tok += sim["tokens"]
        return {
            "acc": acc / n,
            "tok": tok / n,
            "puma_acc": puma_acc / n,
            "puma_tok": puma_tok / n,
            "d_acc": 100.0 * (acc / n - puma_acc / n),
            "d_tok": tok / n - puma_tok / n,
        }


    def fmt_vs(rec: dict[str, Any]) -> str:
        return f"{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])}"


    def fmt_mean(recs: dict[str, dict[str, Any]]) -> str:
        mean = sum(r["d_acc"] for r in recs.values()) / len(recs)
        mean_tok = sum(r["d_tok"] for r in recs.values()) / len(recs)
        return f"{mean:+.1f} / {rg.fmt_tok(mean_tok)}"


    def fmt_bits(recs: dict[str, dict[str, Any]], group: list[str]) -> str:
        return ", ".join(f"{n} {fmt_vs(recs[n])}" for n in group)

    def freeze_sig(sig: str, group: list[str]):
        pool = [s for n in group for rows in cached[sig][n] for s, _ in rows]
        if len(pool) < 8:
            return None
        best = None
        pack_map = dict(packs)
        for thr in rg.quantiles(pool, n=41):
            recs = {n: eval_thr(pack_map[n], cached[sig][n], thr) for n in group}
            mean = sum(r["d_acc"] for r in recs.values()) / len(recs)
            n_pos = sum(int(r["d_acc"] > 1e-9) for r in recs.values())
            n_neg = sum(int(r["d_acc"] < -1e-9) for r in recs.values())
            cand = (mean, n_pos, -n_neg, thr, recs)
            if best is None or cand[:3] > best[:3]:
                best = cand
        return best

    fixed = [
        ("最好写层在哪一截（lens）", 1.0, "最好写就在最后一层"),
        ("最好写层在哪一截（lens）", 0.90, "最好写在最后 10% 深"),
        ("最好写层在哪一截（lens）", 0.75, "最好写在后四分之一"),
        ("最好写层在哪一截（DoLA）", 1.0, "DoLA 最好写在最后一层"),
        ("最好写层在哪一截（DoLA）", 0.90, "DoLA 最好写在最后 10% 深"),
        ("出口抬升 / 出口自身", 0.0, "出口比一半深更亮"),
        ("出口抬升 / 出口自身", 1.0, "抬升至少等于出口自身尺度"),
        ("出口抬升 / （出口+一半深）", 0.0, "对比率为正"),
        ("出口抬升 / （出口+一半深）", 0.50, "抬升至少占两层尺度一半"),
        ("出口减一半深（生值）", 0.0, "生值：出口亮过一半深"),
    ]

    lines = [
        "# 相对量：答案在哪一截成形、出口抬了自身多少",
        "",
        "官方 PUMA 当底。只看停点前的低置信窗最后一步。正确率只对金标。",
        "最好写层在哪一截 = argmax 层号 / 最深层号，0 是最浅，1 是最后一层。跨模型同一句话。",
        "出口抬升 / 出口自身 = (末层 − 一半深) / |末层|。生 logp 差除掉出口自己的尺度。",
        "本集最好仍自选门槛。冻 = 十集或单模型共用一个数。",
        "",
        "## 冻同一条（解释性门槛或扫出来的共用数）",
        "",
        "| 规则 | 范围 | 平均 vs PUMA | 赢 | 伤 | 最差 | 各集 vs PUMA |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    print(lines[-2], flush=True)

    pack_map = dict(packs)
    rows_out = []
    for sig, thr, zh in fixed:
        for title, group in (("7B+8B", names), ("只 8B", eight), ("只 7B", seven)):
            recs = {n: eval_thr(pack_map[n], cached[sig][n], thr) for n in group}
            mean = sum(r["d_acc"] for r in recs.values()) / len(recs)
            n_pos = sum(int(r["d_acc"] > 1e-9) for r in recs.values())
            n_neg = sum(int(r["d_acc"] < -1e-9) for r in recs.values())
            worst = min(r["d_acc"] for r in recs.values())
            row = (
                f"| {zh}（≥{thr:g}） | {title} | {fmt_mean(recs)} | "
                f"{n_pos}/{len(group)} | {n_neg}/{len(group)} | {worst:+.1f} | {fmt_bits(recs, group)} |"
            )
            print(row, flush=True)
            lines.append(row)
            rows_out.append((mean, n_neg, zh, title, recs))

    lines += [
        "",
        "## 扫一个共用门槛（仍是相对量，不是每集一只）",
        "",
        "| 分数 | 范围 | 冻的门槛 | 平均 vs PUMA | 赢 | 伤 | 各集 |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    swept: dict[tuple[str, str], tuple[float, dict[str, dict[str, Any]]]] = {}
    for sig in sigs:
        for title, group in (("7B+8B", names), ("只 8B", eight), ("只 7B", seven)):
            got = freeze_sig(sig, group)
            if got is None:
                continue
            mean, n_pos, n_neg, thr, recs = got
            n_neg = -n_neg
            swept[(sig, title)] = (thr, recs)
            row = (
                f"| {sig} | {title} | {thr:.3f} | {fmt_mean(recs)} | "
                f"{n_pos}/{len(group)} | {n_neg}/{len(group)} | {fmt_bits(recs, group)} |"
            )
            print(row, flush=True)
            lines.append(row)

    last_recs = {n: eval_thr(pack_map[n], cached["最好写层在哪一截（lens）"][n], 1.0) for n in names}
    rise_thr, rise_recs = swept.get(("出口抬升 / 出口自身", "7B+8B"), (2.848, {}))
    if not rise_recs:
        rise_recs = {n: eval_thr(pack_map[n], cached["出口抬升 / 出口自身"][n], rise_thr) for n in names}

    def cell(rec: dict[str, Any]) -> str:
        return f"{sw.fmt_pair(rec['acc'], rec['tok'])}（{fmt_vs(rec)}）"

    lines += [
        "",
        f"## 按集 Acc / token（冻：最好写=最后一层；抬升/自身 ≥ {rise_thr:.2f}）",
        "",
        "| 集 | PUMA | 最好写就在最后一层 | 抬升/自身（十集冻） | 本集最好抬升/自身 |",
        "|---|---|---|---|---|",
    ]
    for name, pack in packs:
        puma = sw.fmt_pair(pack["puma_acc"], pack["puma_tok"])
        local_rise = local["出口抬升 / 出口自身"].get(name)
        if local_rise is None or local_rise["threshold"] == float("inf"):
            local_s = "不开"
        else:
            local_s = (
                f"{sw.fmt_pair(local_rise['acc'], local_rise['tok'])}"
                f"（{rg.fmt_pp(local_rise['d_acc'])} / {rg.fmt_tok(local_rise['d_tok'])}，门 {local_rise['threshold']:.2f}）"
            )
        lines.append(
            f"| {name} | {puma} | {cell(last_recs[name])} | {cell(rise_recs[name])} | {local_s} |"
        )

    lines += ["", "## 本集自选（对照：相对量还有没有头）", "", "| 集 | 最好写层(lens) | 出口抬升/自身 | 生一半深 |", "|---|---|---|---|"]
    for name, _ in packs:
        def fmt(sig):
            rec = local[sig].get(name)
            if rec is None or rec["threshold"] == float("inf"):
                return "不开"
            return (
                f"{sw.fmt_pair(rec['acc'], rec['tok'])}"
                f"（{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])}，门 {rec['threshold']:.2f}）"
            )
        lines.append(
            f"| {name} | {fmt('最好写层在哪一截（lens）')} | "
            f"{fmt('出口抬升 / 出口自身')} | {fmt('出口减一半深（生值）')} |"
        )

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
