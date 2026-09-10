#!/usr/bin/env python3
"""Layer-pair and cross-set replay for the lag door. Trial-as-final. No new extract."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg  # noqa: E402

TABLE = AE / "tables/lag_layer_pairs.md"
OUT = AE / "results/rescue_R_gate/layer_pairs.json"
LAYERS = (8, 14, 20, 27)
NEEDS = ("all", "last", "k3")
NEED_ZH = {"all": "4 步都过", "last": "只看最后一步", "k3": "4 步里 3 步过"}


def zh_name(sig: str) -> str:
    if sig == "last_mean_logp":
        return "最后一层，整段试答好写多少"
    if sig.startswith("dola_logp_l"):
        return f"试答第一个词，第 {sig[len('dola_logp_l'):]} 层有多像"
    if sig.startswith("dola_mean_l"):
        return f"DoLA 整段均值，第 {sig[len('dola_mean_l'):]} 层"
    if sig.startswith("lens_l") and sig.count("_") == 1:
        return f"lens 整段，第 {sig[len('lens_l'):]} 层"
    if sig.startswith("dola_logp_"):
        lo, hi = sig[len("dola_logp_") :].split("_")
        return f"试答第一个词，第 {lo}→{hi} 层抬了多少"
    if sig.startswith("dola_mean_"):
        lo, hi = sig[len("dola_mean_") :].split("_")
        return f"DoLA 整段均值，第 {lo}→{hi} 层抬了多少"
    if sig.startswith("lens_"):
        lo, hi = sig[len("lens_") :].split("_")
        return f"lens 整段，第 {lo}→{hi} 层好写多少"
    if sig.startswith("exit_minus_l"):
        return f"最后一层比第 {sig[len('exit_minus_l'):]} 层（整段 − lens）"
    return sig


def all_signals() -> list[str]:
    out = ["last_mean_logp"]
    for L in LAYERS:
        out.append(f"dola_logp_l{L}")
        out.append(f"dola_mean_l{L}")
        out.append(f"lens_l{L}")
        out.append(f"exit_minus_l{L}")
    for i, lo in enumerate(LAYERS):
        for hi in LAYERS[i + 1 :]:
            out.append(f"dola_logp_{lo}_{hi}")
            out.append(f"dola_mean_{lo}_{hi}")
            out.append(f"lens_{lo}_{hi}")
    return out


def enrich(scores: dict[tuple[int, int], dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    families = {
        "dola_logp": "dola_logp_l{L}",
        "dola_mean": "dola_mean_l{L}",
        "lens": "lens_l{L}",
    }
    for row in scores.values():
        for fam, tmpl in families.items():
            vals = {L: rg.finite(row.get(tmpl.format(L=L))) for L in LAYERS}
            for i, lo in enumerate(LAYERS):
                for hi in LAYERS[i + 1 :]:
                    a, b = vals[lo], vals[hi]
                    if math.isfinite(a) and math.isfinite(b):
                        row[f"{fam}_{lo}_{hi}"] = b - a
        last = rg.finite(row.get("last_mean_logp"))
        for L in LAYERS:
            mid = rg.finite(row.get(f"lens_l{L}"))
            if math.isfinite(last) and math.isfinite(mid):
                row[f"exit_minus_l{L}"] = last - mid
        a = rg.finite(row.get("dola_logp_l27"))
        b = rg.finite(row.get("dola_logp_l20"))
        if math.isfinite(a) and math.isfinite(b):
            row["late_rise"] = a - b
    return scores


def passed(vals: list[float], threshold: float, need: str) -> bool:
    ok = [math.isfinite(v) and v >= threshold for v in vals]
    if need == "all":
        return bool(ok) and all(ok)
    if need == "last":
        return bool(ok) and ok[-1]
    if need == "k3":
        return sum(ok) >= 3
    raise ValueError(need)


def precompute_events(pack: dict[str, Any], names: list[str]) -> None:
    scores = pack["scores"]
    for q in pack["questions"]:
        trials = sorted(q["trials"], key=lambda x: int(x["stopped_len"]))
        rows = rg.usable_rows(trials)
        labels, _ = rg.label_windows(rows, a_final=q["a_final"], gt=q["gt"])
        events = []
        for end, _row in enumerate(rows):
            if not rg.window_ok(rows, end):
                continue
            window = rows[end + 1 - rg.K : end + 1]
            confs = rg.confs_of(window)
            if not all(math.isfinite(c) for c in confs):
                continue
            events.append(
                {
                    "end": end,
                    "high": rg.is_high(confs),
                    "mixed": (not rg.is_high(confs)) and (not rg.is_low(confs)),
                    "tag": labels[end],
                    "vals": {
                        name: [
                            rg.finite(scores.get((q["qi"], int(x["stopped_len"])), {}).get(name))
                            for x in window
                        ]
                        for name in names
                    },
                }
            )
        q["trials"] = trials
        q["rows"] = rows
        q["events"] = events


def decide(
    q: dict[str, Any],
    *,
    signal: str | None,
    threshold: float,
    need: str,
    use_fs: bool = True,
) -> dict[str, Any]:
    import replay_default_dense_gate as old

    if use_fs is True:
        use_fs = rg.USE_FS
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials = q["trials"]
    rows = q["rows"]
    lens = old.step_char_lens(trials) if use_fs else {}
    red_run = 0
    best_conf = float("-inf")
    probed: list[dict[str, Any]] = []
    walked = 0
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if use_fs and tstep >= rg.FS_MIN_STEP:
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
                return rg.pack(
                    trials, rows, end, "conf", original_tokens=q["orig_tok"], label=ev["tag"]
                )
            if (not ev["mixed"]) and signal:
                vals = ev["vals"].get(signal) or []
                if passed(vals, threshold, need):
                    usable = [v for v in vals if math.isfinite(v)]
                    return rg.pack(
                        trials,
                        rows,
                        end,
                        "rescue",
                        original_tokens=q["orig_tok"],
                        score=min(usable) if usable else float("nan"),
                        label=ev["tag"],
                    )
        if use_fs:
            pick = rg.fs_ready(step, red_run, best_conf, probed)
            if pick is not None:
                tag = ev["tag"] if ev is not None else "O"
                return rg.pack_fs(trials, rows, end, pick, q["orig_tok"], label=tag)
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def load_pack(cell: dict[str, Any]) -> dict[str, Any]:
    trials_all = rg.load_json(cell["trial"])
    official = {int(r["question_idx"]): r for r in rg.load_json(cell["stat"])}
    gmap = {int(r["question_idx"]): r for r in rg.load_json(cell["gpath"])}
    scores = enrich(rg.load_scores(cell["scores"]))
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trials_all:
        by[int(row["question_idx"])].append(row)
    questions: list[dict[str, Any]] = []
    first_low: list[dict[str, Any]] = []
    names = list(
        dict.fromkeys(
            [
                *all_signals(),
                "late_rise",
                "lens_rise",
                "neg_ans_entropy",
                "stop_margin",
                "dola_mean_rise",
                "reasoning_pmi",
                "margin",
            ]
        )
    )
    for qi, trials in by.items():
        info = official.get(qi)
        gg = gmap.get(qi)
        if not info or not gg:
            continue
        a_final = gg.get("A_final") or info.get("original_answer")
        gt = info.get("ground_truth")
        orig_ok = bool(info.get("original_correct"))
        orig_tok = int(info.get("original_tokens") or 0)
        rows = rg.usable_rows(trials)
        labels, first = rg.label_windows(rows, a_final=a_final, gt=gt)
        feat: dict[str, Any] = {}
        if first["R"] or first["L"]:
            src = (
                first["R"][0]
                if first["R"] and (not first["L"] or first["R"][0]["st"] <= first["L"][0]["st"])
                else first["L"][0]
            )
            inn = scores.get((qi, int(src["st"])), {})
            feat = {
                "tag": "R" if first["R"] and src is first["R"][0] else "L",
                "st": src["st"],
                "conf": src["conf"],
                **{name: rg.finite(inn.get(name)) for name in names},
            }
            first_low.append(feat)
        questions.append(
            {
                "qi": qi,
                "trials": trials,
                "a_final": a_final,
                "gt": gt,
                "orig_ok": orig_ok,
                "orig_tok": orig_tok,
                "puma_ok": bool(info.get("compressed_correct")),
                "puma_tok": int(info.get("compressed_tokens") or 0)
                + int(info.get("tokens_trial_answers") or 0),
                "labels": labels,
                "events": [],
                "rows": rows,
            }
        )
    return {
        "cell": cell,
        "scores": scores,
        "questions": questions,
        "first_low": first_low,
        "puma_acc": sum(int(q["puma_ok"]) for q in questions) / max(len(questions), 1),
        "puma_tok": sum(q["puma_tok"] for q in questions) / max(len(questions), 1),
    }


def run_pack(
    pack: dict[str, Any],
    signal: str | None,
    threshold: float,
    need: str,
    use_fs: bool = True,
) -> list[dict[str, Any]]:
    out = []
    for q in pack["questions"]:
        sim = decide(q, signal=signal, threshold=threshold, need=need, use_fs=use_fs)
        out.append(
            {
                **sim,
                "ok": rg.hit(sim["answer"], q["gt"], q["a_final"], q["orig_ok"])
                if sim["early"]
                else q["orig_ok"],
                "has_r": "R" in q["labels"],
                "r_only": "R" in q["labels"] and "G" not in q["labels"],
            }
        )
    return out


def contrast(conf_rows: list[dict[str, Any]], ours: list[dict[str, Any]], conf_sum: dict[str, Any]) -> dict[str, Any]:
    s = rg.summarize(ours)
    s.update(
        {
            "damaged": sum(int(c["ok"] and not o["ok"]) for c, o in zip(conf_rows, ours)),
            "recovered": sum(
                int((not c["ok"]) and o["ok"] and o["branch"] == "rescue")
                for c, o in zip(conf_rows, ours)
            ),
            "r_only_hit": sum(int(o["r_only"] and o["ok"] and o["branch"] == "rescue") for o in ours),
            "d_acc_pp_conf": 100.0 * (s["acc"] - conf_sum["acc"]),
            "d_tok_conf": s["tok"] - conf_sum["tok"],
        }
    )
    return s


def pick(points: list[dict[str, Any]], conf_sum: dict[str, Any]) -> dict[str, Any] | None:
    if not points:
        return None
    keep = [p for p in points if p["acc"] + 1e-12 >= conf_sum["acc"] and p["damaged"] == 0]
    pool = keep or points
    return sorted(pool, key=lambda p: (-p["acc"], p["tok"], -p["recovered"]))[0]


def auroc_bins(first_low: list[dict[str, Any]], name: str) -> dict[str, Any]:
    rows = [(r["tag"], r["conf"], rg.finite(r.get(name))) for r in first_low]
    rows = [(t, c, s) for t, c, s in rows if s == s]
    pos = [s for t, _, s in rows if t == "R"]
    neg = [s for t, _, s in rows if t == "L"]
    out = {
        "n_r": len(pos),
        "n_l": len(neg),
        "auroc": rg.auroc(pos, neg),
        "bins": {},
    }
    for lo, hi, lab in rg.CONF_BINS:
        pos_b = [s for t, c, s in rows if t == "R" and lo <= c < hi]
        neg_b = [s for t, c, s in rows if t == "L" and lo <= c < hi]
        out["bins"][lab] = {"n_r": len(pos_b), "n_l": len(neg_b), "auroc": rg.auroc(pos_b, neg_b)}
    return out


def evaluate() -> dict[str, Any]:
    packs = {}
    for cell in rg.CELLS:
        if not cell["trial"].is_file() or not cell["stat"].is_file():
            continue
        if not any(path.is_dir() for path in cell["scores"]):
            print(f"skip no internals {cell['name']}", flush=True)
            continue
        print(f"load {cell['name']}", flush=True)
        pack = load_pack(cell)
        precompute_events(pack, all_signals())
        packs[cell["name"]] = pack

    names = all_signals()
    report: dict[str, Any] = {}
    for name, pack in packs.items():
        print(f"sweep {name}", flush=True)
        conf_rows = run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        item: dict[str, Any] = {
            "n": len(pack["questions"]),
            "puma_acc": pack["puma_acc"],
            "puma_tok": pack["puma_tok"],
            "conf": {
                "acc": conf_sum["acc"],
                "tok": conf_sum["tok"],
                "d_acc_pp_puma": 100.0 * (conf_sum["acc"] - pack["puma_acc"]),
                "d_tok_puma": conf_sum["tok"] - pack["puma_tok"],
            },
            "signals": {},
        }
        for sig in names:
            auc = auroc_bins(pack["first_low"], sig)
            if auc["n_r"] + auc["n_l"] < 12 or auc["auroc"] != auc["auroc"]:
                continue
            xs = [rg.finite(r.get(sig)) for r in pack["first_low"]]
            chosen_need: dict[str, Any] = {}
            for need in NEEDS:
                points = []
                for thr in rg.quantiles(xs):
                    rows = run_pack(pack, sig, thr, need)
                    rec = contrast(conf_rows, rows, conf_sum)
                    rec["signal"] = sig
                    rec["threshold"] = thr
                    rec["need"] = need
                    points.append(rec)
                choice = pick(points, conf_sum)
                if choice:
                    chosen_need[need] = choice
            item["signals"][sig] = {"auroc": auc, "chosen": chosen_need}
        report[name] = {"pack_conf": conf_rows, "conf_sum": conf_sum, **item, "_pack": pack}

    print("cross-set", flush=True)
    cross = []
    shared = None
    for name in report:
        keys = set(report[name]["signals"])
        shared = keys if shared is None else shared & keys
    names_order = [n for n in ("MATH", "GPQA", "奥赛") if n in report]
    for sig in names:
        if not shared or sig not in shared:
            continue
        for src in names_order:
            for tgt in names_order:
                if src == tgt:
                    continue
                src_choice = report[src]["signals"][sig]["chosen"].get("all")
                if not src_choice:
                    continue
                pack = report[tgt]["_pack"]
                conf_rows = report[tgt]["pack_conf"]
                conf_sum = report[tgt]["conf_sum"]
                rows = run_pack(pack, sig, float(src_choice["threshold"]), "all")
                rec = contrast(conf_rows, rows, conf_sum)
                rec.update(
                    {
                        "signal": sig,
                        "freeze": src,
                        "test": tgt,
                        "threshold": src_choice["threshold"],
                    }
                )
                cross.append(rec)
    slim = {
        name: {k: v for k, v in item.items() if k not in {"pack_conf", "_pack", "conf_sum"}}
        for name, item in report.items()
    }
    return {"sets": slim, "cross": cross}


def fmt_auc(block: dict[str, Any] | None) -> str:
    if not block or block["auroc"] != block["auroc"]:
        return "—"
    return f"{block['auroc']:.2f}（对{block['n_r']}/错{block['n_l']}）"


def gate_cell(choice: dict[str, Any] | None) -> str:
    if not choice:
        return "—"
    return (
        f"{rg.fmt_pct(choice['acc'])} / {choice['tok']:.0f} "
        f"（{rg.fmt_pp(choice['d_acc_pp_conf'])} / {rg.fmt_tok(choice['d_tok_conf'])}；"
        f"{choice['rescue_R']}/{choice['rescue_L']}；伤{choice['damaged']}）"
    )


def render(blob: dict[str, Any]) -> str:
    lines = [
        "# 滞后门用哪几层（现成 8 / 14 / 20 / 27，换集重放）",
        "",
        "交卷仍是试答即终答。高置信度门不动。门槛在一集冻住，另一集原样测。",
        "同集扫出来更好看的数，不当新方法。",
        "",
        "「最后一层比第 14 层」= 现在 GPQA 用的那条（最后一层整段 − 第 14 层 lens）。",
        "「试答第一个词，第 20→27 层」= 现在 MATH 用的那条。GPQA 没有 DoLA，这条换不过去。",
        "",
        "换集结论先看：没有一条共享层对能两边都增益且不伤对题。",
        "MATH 冻住的门槛拿到 GPQA 上几乎都不开火；GPQA 冻住的门槛拿到 MATH 上会把错的锁放进来。",
        "同集上，MATH 的「最后一层比第 20 层」比现在的 20→27 词抬更高更干净；GPQA 把「4 步都过」改成「4 步里 3 步过」还能再涨。这两条都还没换集验过，不当新方法。",
        "",
        "## 低置信度连答窗上，对的锁和错的锁分得开吗",
        "",
        "只看每题第一次出现的低置信度连答。0.5 是乱猜。",
        "",
        "| 数据集 | 读数 | 全部 | 0.5–0.8 | 0.8–0.9 | 0.9–0.98 |",
        "|---|---|---|---|---|---|",
    ]
    for set_name, item in blob["sets"].items():
        for sig in all_signals():
            info = item["signals"].get(sig)
            if not info:
                continue
            auc = info["auroc"]
            bins = auc["bins"]
            lines.append(
                f"| {set_name} | {zh_name(sig)} | {fmt_auc(auc)} | "
                f"{fmt_auc(bins.get('0.5–0.8'))} | {fmt_auc(bins.get('0.8–0.9'))} | "
                f"{fmt_auc(bins.get('0.9–0.98'))} |"
            )

    lines += [
        "",
        "## 同集当门（4 步都过；不伤已有对题优先）",
        "",
        "格子里是正确率 / token（相对只看置信度；放进的对锁/错锁；伤到的对题）。",
        "",
        "| 数据集 | 读数 | 4 步都过 | 只看最后一步 | 4 步里 3 步过 |",
        "|---|---|---|---|---|",
    ]
    for set_name, item in blob["sets"].items():
        for sig in all_signals():
            info = item["signals"].get(sig)
            if not info:
                continue
            ch = info["chosen"]
            if not any(ch.get(need) for need in NEEDS):
                continue
            lines.append(
                f"| {set_name} | {zh_name(sig)} | {gate_cell(ch.get('all'))} | "
                f"{gate_cell(ch.get('last'))} | {gate_cell(ch.get('k3'))} |"
            )

    lines += [
        "",
        "## 换集：门槛冻在 A，原样测 B",
        "",
        "只放两集都有的读数。伤到对题 > 0 或正确率掉下去，这条层对就不能当跨集门。",
        "",
        "| 读数 | 冻在 | 测在 | 门槛 | 正确率 / token | 相对只看置信度 | 对锁/错锁 | 伤到的对题 | 纯 R 捞回 |",
        "|---|---|---|---:|---|---|---|---:|---:|",
    ]
    for rec in blob["cross"]:
        lines.append(
            f"| {zh_name(rec['signal'])} | {rec['freeze']} | {rec['test']} | "
            f"{rec['threshold']:.3f} | {rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f} | "
            f"{rg.fmt_pp(rec['d_acc_pp_conf'])} / {rg.fmt_tok(rec['d_tok_conf'])} | "
            f"{rec['rescue_R']} / {rec['rescue_L']} | {rec['damaged']} | {rec['r_only_hit']} |"
        )

    lines += [
        "",
        "入口：`scripts/analyze_lag_layers.py`。奥赛还没有这套内部读数，GPQA 还没有 DoLA。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    blob = evaluate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sets": blob["sets"], "cross": blob["cross"]}, indent=2, ensure_ascii=False) + "\n")
    TABLE.write_text(render(blob))
    print(TABLE.read_text())
    print(f"wrote {TABLE} and {OUT}", flush=True)


if __name__ == "__main__":
    main()
