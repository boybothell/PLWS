#!/usr/bin/env python3
"""Same-set best of the current rel-rise door: Acc ≥ 写完, then min tokens."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_puma_plus_r_oracle as base

TABLE = AE / "tables/method_sameset_tok.md"


def credit(q: dict[str, Any], ans: Any) -> int:
    if rg.same(ans, q["gt"]):
        return 1
    if q["orig_ok"] and rg.same(ans, q["a_final"]):
        return 1
    return 0


def fire_ratio_before(q: dict[str, Any], thr: float, host_step: int) -> dict[str, Any] | None:
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= host_step:
            continue
        rise = base.last_val(ev["vals"].get(base.SIG))
        exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
        if rise != rise or exit_s != exit_s:
            continue
        if rise / (abs(exit_s) + 1e-6) >= thr:
            return ev
    return None


def full_step(q: dict[str, Any]) -> int:
    if q["rows"]:
        return int(q["rows"][-1]["stopped_len"]) + 1
    return 10**9


def eval_thr(pack: dict[str, Any], thr: float) -> dict[str, Any] | None:
    if float(pack.get("_half_cov") or 0.0) < base.HALF_READY:
        return None
    n = len(pack["questions"])
    acc = tok = orig_acc = orig_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        orig_acc += int(q["orig_ok"])
        orig_tok += q["orig_tok"]
        ev = fire_ratio_before(q, thr, full_step(q))
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
            label=ev.get("tag") or "O",
        )
        ok = credit(q, sim["answer"])
        acc += ok
        tok += sim["tokens"]
        n_fire += 1
        n_gain += int(ok and not q["orig_ok"])
        n_hurt += int((not ok) and q["orig_ok"])
    return {
        "threshold": thr,
        "n": n,
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


def ratio_xs(pack: dict[str, Any]) -> list[float]:
    xs = []
    for q in pack["questions"]:
        host = full_step(q)
        for ev in q["events"]:
            if ev.get("high") or ev.get("mixed"):
                continue
            step = int(q["rows"][ev["end"]]["stopped_len"])
            if step < rg.MSS or step >= host:
                continue
            rise = base.last_val(ev["vals"].get(base.SIG))
            exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
            if rise != rise or exit_s != exit_s:
                continue
            xs.append(rise / (abs(exit_s) + 1e-6))
    return xs


def pick_tok_first(points: list[dict[str, Any]], base_acc: float) -> dict[str, Any]:
    ok = [p for p in points if p["acc"] + 1e-12 >= base_acc]
    if not ok:
        return sorted(points, key=lambda p: (-p["acc"], p["tok"]))[0]
    return sorted(ok, key=lambda p: (p["tok"], -p["acc"]))[0]


def eval_low_oracle(pack: dict[str, Any]) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = orig_acc = orig_tok = 0.0
    n_fire = n_gain = 0
    for q in pack["questions"]:
        orig_acc += int(q["orig_ok"])
        orig_tok += q["orig_tok"]
        ev = None
        host = full_step(q)
        for cand in q["events"]:
            if cand.get("high") or cand.get("mixed"):
                continue
            step = int(q["rows"][cand["end"]]["stopped_len"])
            if step < rg.MSS or step >= host:
                continue
            ans = q["rows"][cand["end"]].get("final_answer")
            if credit(q, ans):
                ev = cand
                break
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
    return {
        "n": n,
        "acc": acc / n,
        "tok": tok / n,
        "orig_acc": orig_acc / n,
        "orig_tok": orig_tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "d_acc": 100.0 * (acc / n - orig_acc / n),
        "d_tok": tok / n - orig_tok / n,
    }


def fmt(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    header = "| 集 | 写完全程 | 这一套同集（Acc不降、最少token） | 相对写完 | 开火 / 抬 / 伤 | 本门先知 |"
    sep = "|---|---|---|---|---:|---|"
    lines = [
        "# 现行对比率门：同集最好能不降 Acc 省多少 token",
        "",
        "方法还是这一套：低置信连答窗（k=4、四步都 < 0.995），最后一步对比率",
        "（末层好写 − 一半深好写）/ |末层好写| ≥ 本集自选门槛，交试答；没开就写完。",
        "不含高置信 0.995：那一扇相对写完已经降 Acc，同集也补不回来。",
        "同集门槛偷看本集对错：在 Acc ≥ 写完的点里取 token 最少。换集会塌。",
        "交跟写完同一串时，对错跟官方写完走。本门先知 = 第一扇低置信窗已经可停（写完或金标）。",
        "AIME 四个 seed。中间层覆盖不到 85% 的不报。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, pack in base.load_jobs():
        closed = eval_thr(pack, float("inf"))
        if closed is None:
            print(f"| {name} | 未齐 |", flush=True)
            lines.append(f"| {name} | 未齐 |")
            continue
        xs = ratio_xs(pack)
        thrs = [float("inf")]
        if xs:
            thrs.extend(rg.quantiles(xs, n=41))
        points = []
        for thr in thrs:
            rec = eval_thr(pack, thr)
            if rec is not None:
                points.append(rec)
        pick = pick_tok_first(points, closed["orig_acc"])
        ora = eval_low_oracle(pack)
        if pick["threshold"] == float("inf"):
            door = f"{fmt(pick['acc'], pick['tok'])}（不开）"
        else:
            door = f"{fmt(pick['acc'], pick['tok'])}（门 {pick['threshold']:.3f}）"
        row = (
            f"| {name} | {fmt(closed['orig_acc'], closed['orig_tok'])} "
            f"| {door} "
            f"| {rg.fmt_pp(pick['d_acc'])} / {rg.fmt_tok(pick['d_tok'])} "
            f"| {pick['n_fire']} / {pick['n_gain']} / {pick['n_hurt']} "
            f"| {fmt(ora['acc'], ora['tok'])}（{rg.fmt_pp(ora['d_acc'])} / {rg.fmt_tok(ora['d_tok'])}；{ora['n_fire']}） |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
