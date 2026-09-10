#!/usr/bin/env python3
"""Re-rank lag-door signals by full-set Acc/Tok. No zero-damage filter."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402

TABLE = AE / "tables/lag_acctok_signals.md"
OUT = AE / "results/rescue_R_gate/acctok_signals.json"

# High AUROC / previously discarded families. Layer pairs only where scores exist.
FOCUS = (
    "late_rise",
    "dola_logp_l27",
    "dola_logp_20_27",
    "dola_mean_rise",
    "neg_ans_entropy",
    "lens_rise",
    "exit_minus_l14",
    "exit_minus_l20",
    "stop_margin",
    "reasoning_pmi",
    "margin",
    "last_mean_logp",
)

ZH = {
    "late_rise": "试答第一个词，20→27 抬了多少",
    "dola_logp_l27": "第 27 层对试答第一个词有多像",
    "dola_logp_20_27": "试答第一个词，20→27 抬了多少",
    "dola_mean_rise": "浅层到深层，整段试答抬了多少",
    "neg_ans_entropy": "试答用词有多集中",
    "lens_rise": "最后一层比第 14 层（现用 GPQA）",
    "exit_minus_l14": "最后一层比第 14 层（整段 − lens）",
    "exit_minus_l20": "最后一层比第 20 层",
    "stop_margin": "写完试答后，收口比 Wait 高多少",
    "reasoning_pmi": "有草稿比只看题，试答多顺多少",
    "margin": "当前试答比历史上别的试答高多少",
    "last_mean_logp": "最后一层，整段试答好写多少",
}


def pick_acc(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not points:
        return None
    return sorted(points, key=lambda p: (-p["acc"], p["tok"], -p.get("recovered", 0)))[0]


def pick_safe(points: list[dict[str, Any]], conf_acc: float) -> dict[str, Any] | None:
    keep = [p for p in points if p["acc"] + 1e-12 >= conf_acc and p["damaged"] == 0]
    if not keep:
        return None
    return sorted(keep, key=lambda p: (-p["acc"], p["tok"], -p.get("recovered", 0)))[0]


def evaluate() -> dict[str, Any]:
    names = list(dict.fromkeys([*FOCUS, *layers.all_signals()]))
    packs = {}
    for cell in rg.CELLS:
        if not cell["trial"].is_file() or not cell["stat"].is_file():
            continue
        if not any(path.is_dir() for path in cell["scores"]):
            print(f"skip no internals {cell['name']}", flush=True)
            continue
        print(f"load {cell['name']}", flush=True)
        pack = layers.load_pack(cell)
        layers.precompute_events(pack, names)
        packs[cell["name"]] = pack

    report: dict[str, Any] = {}
    for set_name, pack in packs.items():
        print(f"sweep {set_name}", flush=True)
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        item: dict[str, Any] = {
            "n": len(pack["questions"]),
            "puma_acc": pack["puma_acc"],
            "puma_tok": pack["puma_tok"],
            "conf": {"acc": conf_sum["acc"], "tok": conf_sum["tok"]},
            "signals": {},
        }
        for sig in FOCUS:
            auc = layers.auroc_bins(pack["first_low"], sig)
            if auc["n_r"] + auc["n_l"] < 12 or auc["auroc"] != auc["auroc"]:
                continue
            xs = [rg.finite(r.get(sig)) for r in pack["first_low"]]
            points = []
            for thr in rg.quantiles(xs):
                rows = layers.run_pack(pack, sig, thr, "all")
                rec = layers.contrast(conf_rows, rows, conf_sum)
                rec["signal"] = sig
                rec["threshold"] = thr
                rec["d_acc_pp_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
                rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
                points.append(rec)
            item["signals"][sig] = {
                "auroc": auc,
                "acc_best": pick_acc(points),
                "safe": pick_safe(points, conf_sum["acc"]),
            }
        report[set_name] = {**item, "_pack": pack, "pack_conf": conf_rows, "conf_sum": conf_sum}

    print("cross-set", flush=True)
    cross = []
    order = [n for n in ("MATH", "GPQA", "奥赛") if n in report]
    shared = None
    for name in order:
        keys = set(report[name]["signals"])
        shared = keys if shared is None else shared & keys
    for sig in FOCUS:
        if not shared or sig not in shared:
            continue
        for src in order:
            choice = report[src]["signals"][sig].get("acc_best")
            if not choice:
                continue
            for tgt in order:
                if src == tgt:
                    continue
                pack = report[tgt]["_pack"]
                rec = layers.contrast(
                    report[tgt]["pack_conf"],
                    layers.run_pack(pack, sig, float(choice["threshold"]), "all"),
                    report[tgt]["conf_sum"],
                )
                rec.update(
                    {
                        "signal": sig,
                        "freeze": src,
                        "test": tgt,
                        "threshold": choice["threshold"],
                        "d_acc_pp_puma": 100.0 * (rec["acc"] - pack["puma_acc"]),
                        "d_tok_puma": rec["tok"] - pack["puma_tok"],
                    }
                )
                cross.append(rec)
    slim = {
        name: {k: v for k, v in item.items() if k not in {"_pack", "pack_conf", "conf_sum"}}
        for name, item in report.items()
    }
    return {"sets": slim, "cross": cross}


def cell_txt(p: dict[str, Any] | None) -> str:
    if not p:
        return "—"
    return (
        f"{rg.fmt_pct(p['acc'])} / {p['tok']:.0f} "
        f"（相对只看置信度 {rg.fmt_pp(p['d_acc_pp_conf'])} / {rg.fmt_tok(p['d_tok_conf'])}；"
        f"相对 PUMA {rg.fmt_pp(p.get('d_acc_pp_puma', float('nan')))} / {rg.fmt_tok(p.get('d_tok_puma', float('nan')))}；"
        f"放进 {p['rescue_R']}/{p['rescue_L']}；伤 {p['damaged']}）"
    )


def render(blob: dict[str, Any]) -> str:
    lines = [
        "# 滞后门按全集 Acc / token 重选（不再要求伤 0）",
        "",
        "交卷仍是试答即终答。选点：正确率尽量高，其次 token 少。伤到几道已经对的题只记录，不刷掉。",
        "「旧规则」= 先要求伤 0，再比正确率。那条会把 Acc 更好的点扔掉。",
        "",
        "## 同集：旧规则 vs 全集 Acc",
        "",
        "| 数据集 | 读数 | 对/错分开 | 旧规则（伤 0） | 全集 Acc 最好 |",
        "|---|---|---|---|---|",
    ]
    for set_name, item in blob["sets"].items():
        for sig in FOCUS:
            info = item["signals"].get(sig)
            if not info:
                continue
            auc = info["auroc"]
            lines.append(
                f"| {set_name} | {ZH.get(sig, sig)} | {auc['auroc']:.2f}（对{auc['n_r']}/错{auc['n_l']}） | "
                f"{cell_txt(info.get('safe'))} | {cell_txt(info.get('acc_best'))} |"
            )
    lines += [
        "",
        "## 换集：用 A 集 Acc 最好的门槛，原样测 B",
        "",
        "| 读数 | 冻在 | 测在 | 门槛 | 正确率 / token | 相对只看置信度 | 相对 PUMA | 放进对/错 | 伤到的对题 |",
        "|---|---|---|---:|---|---|---|---|---:|",
    ]
    for rec in blob["cross"]:
        lines.append(
            f"| {ZH.get(rec['signal'], rec['signal'])} | {rec['freeze']} | {rec['test']} | "
            f"{rec['threshold']:.3f} | {rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f} | "
            f"{rg.fmt_pp(rec['d_acc_pp_conf'])} / {rg.fmt_tok(rec['d_tok_conf'])} | "
            f"{rg.fmt_pp(rec['d_acc_pp_puma'])} / {rg.fmt_tok(rec['d_tok_puma'])} | "
            f"{rec['rescue_R']} / {rec['rescue_L']} | {rec['damaged']} |"
        )
    lines += [
        "",
        "入口：`scripts/analyze_lag_acctok.py`。8B / 14B / 32B / 30B 还没有这套内部读数，跨模型还不能测。奥赛内部还在抽。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    blob = evaluate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sets": blob["sets"], "cross": blob["cross"]}, indent=2, ensure_ascii=False) + "\n")
    TABLE.write_text(render(blob))
    print(TABLE.read_text())
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
