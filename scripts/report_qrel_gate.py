#!/usr/bin/env python3
"""High-conf + FS + within-question relative lag. No global threshold."""
from __future__ import annotations

import math
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as old
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp

SIGS = ("stop_margin", "exit_minus_half")
RULES = ("new_high", "above_mean", "rise_last")


def cur_score(ev, sig) -> float:
    vals = ev["vals"].get(sig) or []
    if not vals or any(v != v for v in vals):
        return float("nan")
    return min(vals)


def rel_ok(rule: str, cur: float, past: list[float]) -> bool:
    past = [x for x in past if x == x]
    if cur != cur or not past:
        return False
    if rule == "new_high":
        return cur > max(past)
    if rule == "above_mean":
        return cur > sum(past) / len(past)
    if rule == "rise_last":
        return cur > past[-1]
    raise ValueError(rule)


def decide_rel(q, sig: str, rule: str) -> dict:
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials = q["trials"]
    rows = q["rows"]
    lens = old.step_char_lens(trials)
    red_run = 0
    best_conf = float("-inf")
    probed = []
    walked = 0
    past: list[float] = []
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if tstep >= rg.FS_MIN_STEP:
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
                return rg.pack(trials, rows, end, "conf", original_tokens=q["orig_tok"], label=ev["tag"])
            if not ev["mixed"]:
                cur = cur_score(ev, sig)
                if rel_ok(rule, cur, past):
                    return rg.pack(
                        trials,
                        rows,
                        end,
                        "rescue",
                        original_tokens=q["orig_tok"],
                        score=cur,
                        label=ev["tag"],
                    )
                if cur == cur:
                    past.append(cur)
        if rg.USE_FS:
            pick = rg.fs_ready(step, red_run, best_conf, probed)
            if pick is not None:
                tag = ev["tag"] if ev is not None else "O"
                return rg.pack_fs(trials, rows, end, pick, q["orig_tok"], label=tag)
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def run_rel(pack, sig, rule):
    out = []
    for q in pack["questions"]:
        sim = decide_rel(q, sig, rule)
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


def load_all():
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    named = []
    for cell in list(rg.CELLS) + [more.EXTRA]:
        pack = cmp.load_ready(cell)
        if pack is None:
            print(f"skip {cell['name']}", flush=True)
            continue
        print(f"load {cell['name']}", flush=True)
        named.append((cell["name"], pack))
    for ds, zh in (("aime24", "7B AIME24"), ("aime25", "7B AIME25")):
        parts = []
        for seed in accfirst.SEEDS:
            pack = cmp.load_ready(accfirst.aime_cell(ds, seed))
            if pack is None:
                continue
            pack["_cov"] = 1.0
            parts.append(pack)
            print(f"load {ds}-s{seed}", flush=True)
        if len(parts) == 4:
            named.append((zh, accfirst.merge_packs(parts, zh)))
    return named


def main() -> None:
    named = load_all()
    zh_rule = {
        "new_high": "比这道题以前的锁都亮",
        "above_mean": "比这道题以前的锁均值亮",
        "rise_last": "比上一扇锁亮",
    }
    zh_sig = {"stop_margin": "Wait", "exit_minus_half": "一半深"}
    print(
        "\n高置信 + FS + 题内相对。第一扇低置信锁不看滞后（没历史）。"
        "不用全集门槛。",
        flush=True,
    )
    header = "| 集 | 高置信+FS | 信号 / 规则 | 正确率 / token | 相对FS | 放进对/错 |"
    print(header)
    print("|---|---|---|---|---|---|")
    for name, pack in named:
        fs = layers.run_pack(pack, None, float("inf"), "all")
        fs_sum = rg.summarize(fs)
        print(
            f"| {name} | {rg.fmt_pct(fs_sum['acc'])} / {fs_sum['tok']:.0f} | — | "
            f"{rg.fmt_pct(fs_sum['acc'])} / {fs_sum['tok']:.0f} | — | — |",
            flush=True,
        )
        for sig in SIGS:
            for rule in RULES:
                ours = run_rel(pack, sig, rule)
                rec = layers.contrast(fs, ours, fs_sum)
                print(
                    f"| {name} |  | {zh_sig[sig]} / {zh_rule[rule]} | "
                    f"{rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f} | "
                    f"{rg.fmt_pp(rec['d_acc_pp_conf'])} / {rg.fmt_tok(rec['d_tok_conf'])} | "
                    f"{rec['rescue_R']}/{rec['rescue_L']} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
