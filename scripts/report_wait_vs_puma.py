#!/usr/bin/env python3
"""Same-set stop_margin (收口比 Wait) vs official PUMA: Acc and tokens."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg
import report_puma_plus_r_oracle as base
import report_prev_sameset_tok as prev

TABLE = AE / "tables/wait_vs_puma.md"


def attach_puma(pack: dict[str, Any], stat: Path) -> None:
    official = {int(r["question_idx"]): r for r in json.loads(stat.read_text())}
    for q in pack["questions"]:
        info = official.get(int(q["qi"])) or {}
        q["puma_step"] = int(info.get("stopped_len") or 10**9)


def fire_wait(q: dict[str, Any], thr: float, host_step: int) -> dict[str, Any] | None:
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= host_step:
            continue
        vals = ev["vals"].get("stop_margin") or []
        if layers.passed(vals, thr, "last"):
            return ev
    return None


def wait_xs(pack: dict[str, Any], host: str) -> list[float]:
    xs = []
    for q in pack["questions"]:
        limit = int(q.get("puma_step") or 10**9) if host == "puma" else prev.full_step(q)
        for ev in q["events"]:
            if ev.get("high") or ev.get("mixed"):
                continue
            step = int(q["rows"][ev["end"]]["stopped_len"])
            if step < rg.MSS or step >= limit:
                continue
            val = base.last_val(ev["vals"].get("stop_margin"))
            if val == val:
                xs.append(val)
    return xs


def eval_wait(pack: dict[str, Any], thr: float, host: str) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = puma_acc = puma_tok = orig_acc = orig_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        orig_acc += int(q["orig_ok"])
        orig_tok += q["orig_tok"]
        limit = int(q.get("puma_step") or 10**9) if host == "puma" else prev.full_step(q)
        ev = fire_wait(q, thr, limit)
        if ev is None:
            if host == "puma":
                acc += int(q["puma_ok"])
                tok += q["puma_tok"]
            else:
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
        ok = prev.credit(q, sim["answer"])
        acc += ok
        tok += sim["tokens"]
        n_fire += 1
        base_ok = q["puma_ok"] if host == "puma" else q["orig_ok"]
        n_gain += int(ok and not base_ok)
        n_hurt += int((not ok) and base_ok)
    return {
        "threshold": thr,
        "acc": acc / n,
        "tok": tok / n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "orig_acc": orig_acc / n,
        "orig_tok": orig_tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "d_puma_acc": 100.0 * (acc / n - puma_acc / n),
        "d_puma_tok": tok / n - puma_tok / n,
        "d_orig_acc": 100.0 * (acc / n - orig_acc / n),
        "d_orig_tok": tok / n - orig_tok / n,
    }


def sweep(pack: dict[str, Any], host: str) -> dict[str, Any]:
    floor = pack["questions"]
    n = max(len(floor), 1)
    base_acc = (
        sum(int(q["puma_ok"]) for q in floor) / n
        if host == "puma"
        else sum(int(q["orig_ok"]) for q in floor) / n
    )
    xs = wait_xs(pack, host)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = [eval_wait(pack, thr, host) for thr in thrs]
    return prev.pick_tok_first(points, base_acc)


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    header = (
        "| 集 | PUMA Acc | PUMA token | Wait（写完底）Acc | Wait token | 比 PUMA | "
        "Wait（挂 PUMA 上）Acc | Wait token | 比 PUMA |"
    )
    sep = "|---|---:|---:|---:|---:|---|---:|---:|---|"
    lines = [
        "# 收口比 Wait 对官方 PUMA：正确率和 token",
        "",
        "Wait = 写完 boxed 后，收口比继续 Wait 高多少（`stop_margin`），只看低置信连答窗最后一步。",
        "同集偷看门槛：相对底线 Acc 不降，再取 token 最少。换集会塌。",
        "写完底：没开 Wait 就写完。挂 PUMA 上：没开 Wait 就留官方 PUMA（含重写终答）。",
        "AIME 四个 seed。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, pack in prev.load_78():
        # puma_step: merge already has per-question puma_tok/ok; attach step from first cell if missing
        for q in pack["questions"]:
            q.setdefault("puma_step", 10**9)
        full = sweep(pack, "full")
        on_puma = sweep(pack, "puma")
        row = (
            f"| {name} | {rg.fmt_pct(full['puma_acc'])} | {full['puma_tok']:.0f} "
            f"| {rg.fmt_pct(full['acc'])} | {full['tok']:.0f} "
            f"| {rg.fmt_pp(full['d_puma_acc'])} / {rg.fmt_tok(full['d_puma_tok'])} "
            f"| {rg.fmt_pct(on_puma['acc'])} | {on_puma['tok']:.0f} "
            f"| {rg.fmt_pp(on_puma['d_puma_acc'])} / {rg.fmt_tok(on_puma['d_puma_tok'])} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
