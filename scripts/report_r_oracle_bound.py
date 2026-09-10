#!/usr/bin/env python3
"""Old host: 0.995 trial-as-final. Oracle = first gold-correct low-conf lock."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_puma_plus_r_oracle as base
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/r_oracle_bound.md"


def first_high(q: dict[str, Any]) -> dict[str, Any] | None:
    for ev in q["events"]:
        if not ev.get("high"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step >= rg.MSS:
            return ev
    return None


def first_gold_low(q: dict[str, Any]) -> dict[str, Any] | None:
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS:
            continue
        if rg.same(q["rows"][ev["end"]].get("final_answer"), q["gt"]):
            return ev
    return None


def eval_pack(pack: dict[str, Any]) -> dict[str, Any]:
    n = len(pack["questions"])
    has_r = rescued = acc = puma_acc = 0
    tok = puma_tok = acc995 = tok995 = 0.0
    for q in pack["questions"]:
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        high = first_high(q)
        gold = first_gold_low(q)
        high_step = int(q["rows"][high["end"]]["stopped_len"]) if high else 10**9
        gold_step = int(q["rows"][gold["end"]]["stopped_len"]) if gold else 10**9
        # 0.995 only
        if high is not None:
            sim = rg.pack(q["trials"], q["rows"], high["end"], "conf", original_tokens=q["orig_tok"], label="H")
            acc995 += int(base.trial_ok(q, sim["answer"]))
            tok995 += sim["tokens"]
        else:
            sim = rg.pack(q["trials"], q["rows"], max(len(q["rows"]) - 1, 0), "full", original_tokens=q["orig_tok"])
            acc995 += int(base.trial_ok(q, sim["answer"]))
            tok995 += sim["tokens"]
        # oracle: gold low-conf wins if it appears at or before high-conf
        if gold is not None:
            has_r += 1
        if gold is not None and gold_step <= high_step:
            rescued += 1
            sim = rg.pack(q["trials"], q["rows"], gold["end"], "rescue", original_tokens=q["orig_tok"], label="R")
            acc += 1
            tok += sim["tokens"]
        elif high is not None:
            sim = rg.pack(q["trials"], q["rows"], high["end"], "conf", original_tokens=q["orig_tok"], label="H")
            acc += int(base.trial_ok(q, sim["answer"]))
            tok += sim["tokens"]
        else:
            sim = rg.pack(q["trials"], q["rows"], max(len(q["rows"]) - 1, 0), "full", original_tokens=q["orig_tok"])
            acc += int(base.trial_ok(q, sim["answer"]))
            tok += sim["tokens"]
    return {
        "n": n,
        "has_r": has_r,
        "rescued": rescued,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "acc995": acc995 / n,
        "tok995": tok995 / n,
        "acc": acc / n,
        "tok": tok / n,
        "d_acc": 100.0 * (acc / n - puma_acc / n),
        "d_tok": tok / n - puma_tok / n,
    }


def main() -> None:
    lines = [
        "# 旧协议：0.995 试答即终答 + 低置信连对上界",
        "",
        "不要当主上界。主上界是挂在官方 PUMA 上的 [`puma_plus_r_oracle.md`](puma_plus_r_oracle.md)。",
        "",
        "0.995 高置信仍开，交试答。低置信连答且试答对金标，一扇就停、交试答。低置信错窗不停。不开强停。",
        "正确率只对金标。高置信假停仍会把 Acc 压到 PUMA 下面。AIME 四 seed。缺 seed 的不报。",
        "",
        "| 集 | 题 | 有R | 救回 | PUMA | 0.995 | R上限 | vs PUMA |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    print(lines[-2], flush=True)
    for name, cells in sw.all_jobs():
        packs = []
        for cell in cells:
            cell = dict(cell)
            cell["scores"] = ()
            if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
                continue
            pack = layers.load_pack(cell)
            base.attach_puma_step(pack, Path(cell["stat"]))
            pack["_cov"] = 1.0
            layers.precompute_events(pack, [])
            packs.append(pack)
        if not packs:
            continue
        if len(cells) > 1 and len(packs) != len(cells):
            if len(packs) < 3:
                print(f"skip {name} seeds={len(packs)}/{len(cells)}", flush=True)
                continue
            name = f"{name}（{len(packs)} seed）"
        pack = accfirst.merge_packs(packs, name) if len(packs) > 1 else packs[0]
        rec = eval_pack(pack)
        row = (
            f"| {name} | {rec['n']} | {rec['has_r']} | {rec['rescued']} | "
            f"{sw.fmt_pair(rec['puma_acc'], rec['puma_tok'])} | "
            f"{sw.fmt_pair(rec['acc995'], rec['tok995'])} | "
            f"{sw.fmt_pair(rec['acc'], rec['tok'])} | "
            f"{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
