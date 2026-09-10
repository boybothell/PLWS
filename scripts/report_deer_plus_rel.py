#!/usr/bin/env python3
"""Hang the default relative door on a DEER-style host (same CoT, trial-as-final).

Official DEER: Wait-ATP, first probe conf > 0.95, ≤10 Wait ends.
No deer_offline_wait dumps yet, so Wait cuts are recovered from reasoning_prefix.
Every-step first conf > 0.95 is an earlier-stop ablation, not paper DEER.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import screen_puma_layers as sc
import screen_relative_layer as rel
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/deer_plus_rel.md"
WAIT_RE = re.compile(r"\bWait\b")
DEER_LAMB = 0.95
MAX_WAIT = 10


def wait_probe_rows(q: dict[str, Any]) -> list[dict[str, Any]]:
    prev = 0
    out = []
    by_step = {int(r["stopped_len"]): r for r in q["rows"]}
    for trial in sorted(q["trials"], key=lambda x: int(x["stopped_len"])):
        text = str(trial.get("reasoning_prefix") or "")
        n = len(WAIT_RE.findall(text))
        if n <= prev:
            continue
        prev = n
        row = by_step.get(int(trial["stopped_len"]))
        if row is not None:
            out.append(row)
        if len(out) >= MAX_WAIT:
            break
    return out


def first_over(rows: list[dict[str, Any]]) -> tuple[int | None, int]:
    last_step = int(rows[-1]["stopped_len"]) if rows else 10**9
    for end, row in enumerate(rows):
        conf = rg.finite(row.get("confidence"))
        if conf == conf and conf > DEER_LAMB:
            return end, int(row["stopped_len"])
    return None, last_step


def host_outcome(q: dict[str, Any], end: int | None) -> dict[str, Any]:
    if end is None:
        sim = rg.pack(
            q["trials"], q["rows"], max(len(q["rows"]) - 1, 0),
            "full", original_tokens=q["orig_tok"],
        )
    else:
        sim = rg.pack(
            q["trials"], q["rows"], end, "rescue",
            original_tokens=q["orig_tok"],
        )
    ok = rg.same(sim["answer"], q["gt"])
    return {"ok": ok, "tok": sim["tokens"], "step": sim["step"], "early": end is not None}


def before(q: dict[str, Any], host_step: int) -> list[dict[str, Any]]:
    out = []
    for ev in q.get("_act_all") or q["_act"]:
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if rg.MSS <= step < host_step:
            out.append(ev)
    return out


def apply_door(q: dict[str, Any], host_step: int, host: dict[str, Any], thr: float) -> dict[str, Any]:
    hit = None
    for ev in before(q, host_step):
        score = ev.get("rel", {}).get("出口抬升 / 出口自身")
        if score == score and score >= thr:
            hit = ev
            break
    if hit is None:
        return {**host, "fired": False, "oracle": False}
    sim = rg.pack(
        q["trials"], q["rows"], hit["end"], "rescue",
        original_tokens=q["orig_tok"], label=hit["tag"],
    )
    ok = rg.same(sim["answer"], q["gt"])
    return {"ok": ok, "tok": sim["tokens"], "step": sim["step"], "early": True, "fired": True, "oracle": False}


def apply_oracle(q: dict[str, Any], host_step: int, host: dict[str, Any]) -> dict[str, Any]:
    hit = next((ev for ev in before(q, host_step) if ev.get("tag") == "R"), None)
    if hit is None:
        return {**host, "fired": False, "oracle": False}
    sim = rg.pack(
        q["trials"], q["rows"], hit["end"], "rescue",
        original_tokens=q["orig_tok"], label="R",
    )
    ok = rg.same(sim["answer"], q["gt"])
    return {"ok": ok, "tok": sim["tokens"], "step": sim["step"], "early": True, "fired": True, "oracle": True}


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(rows), 1)
    return {
        "acc": sum(int(r["ok"]) for r in rows) / n,
        "tok": sum(r["tok"] for r in rows) / n,
        "fire": sum(int(r.get("fired")) for r in rows) / n,
        "early": sum(int(r.get("early")) for r in rows) / n,
    }


def fmt_vs(a: dict[str, float], b: dict[str, float]) -> str:
    return f"{rg.fmt_pp(100.0 * (a['acc'] - b['acc']))} / {rg.fmt_tok(a['tok'] - b['tok'])}"


def main() -> None:
    packs = []
    for name, pack in sc.load_ready():
        # windows before any host; filter later
        for q in pack["questions"]:
            q["_act_all"] = []
            for ev in q["events"]:
                if ev.get("high") or ev.get("mixed") or ev.get("tag") not in {"R", "L"}:
                    continue
                step = int(q["rows"][ev["end"]]["stopped_len"])
                if step >= rg.MSS:
                    q["_act_all"].append(ev)
        rel.attach_rel(pack)
        # attach_rel writes ev["rel"] on q["_act"]; copy onto _act_all via same objects if shared
        for q in pack["questions"]:
            for ev in q["_act_all"]:
                if "rel" not in ev:
                    ev["rel"] = {
                        "出口抬升 / 出口自身": rel.rise_over_exit(ev),
                    }
        packs.append((name, pack))

    lines = [
        "# DEER 底 + 默认对比率附加门",
        "",
        "同一条 Full-CoT。DEER 交卷是试答即终答，闸门是第一次把握 > 0.95。",
        "Wait-ATP = 冻 CoT 上前 10 个 Wait 切点（从 reasoning_prefix 数 \\bWait\\b），不是每步探。",
        "每步 >0.95 = 消融，比论文 DEER 更早停，附加门空间更小。",
        f"附加门仍是默认：低置信 k=4 窗最后一步，(末层−一半深)/|末层| ≥ {rg.REL_RISE_THR:.2f}，必须在 DEER 停点之前。",
        "官方 `deer_offline_wait_*` 还没有；这张表是现成密探重放，数据齐了要用 Wait-ATP 试答再核一次。",
        "",
        "## 有没有空间（停点前能不能开低置信窗）",
        "",
        "| 集 | 题 | Wait切点题 | DEER-Wait早停 | 窗在Wait停前 | 其中连对 | DEER-每步早停 | 窗在每步停前 | PUMA停前有窗 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    print(lines[-2], flush=True)

    cells = []
    for name, pack in packs:
        n = len(pack["questions"])
        wait_has = deer_w = room_w = r_w = deer_a = room_a = room_p = 0
        recs = []
        for q in pack["questions"]:
            puma_step = int(q.get("puma_step") or 10**9)
            waits = wait_probe_rows(q)
            if waits:
                wait_has += 1
            w_end, w_step = first_over(waits) if waits else (None, int(q["rows"][-1]["stopped_len"]) if q["rows"] else 10**9)
            a_end, a_step = first_over(q["rows"])
            if w_end is not None:
                deer_w += 1
            if a_end is not None:
                deer_a += 1
            bw = before(q, w_step)
            ba = before(q, a_step)
            if bw:
                room_w += 1
            if any(ev.get("tag") == "R" for ev in bw):
                r_w += 1
            if ba:
                room_a += 1
            if before(q, puma_step):
                room_p += 1
            host_w = host_outcome(q, w_end if waits else None)
            if not waits:
                host_w = host_outcome(q, None)
            # if there are wait probes, host uses the wait schedule's first over, else full
            if waits:
                host_w = host_outcome(q, w_end)
            host_a = host_outcome(q, a_end)
            recs.append(
                {
                    "host_w": host_w,
                    "door_w": apply_door(q, w_step, host_w, rg.REL_RISE_THR),
                    "ora_w": apply_oracle(q, w_step, host_w),
                    "host_a": host_a,
                    "door_a": apply_door(q, a_step, host_a, rg.REL_RISE_THR),
                    "ora_a": apply_oracle(q, a_step, host_a),
                    "puma": {"ok": q["puma_ok"], "tok": q["puma_tok"]},
                }
            )
        row = (
            f"| {name} | {n} | {wait_has} | {deer_w} | {room_w} | {r_w} | "
            f"{deer_a} | {room_a} | {room_p} |"
        )
        print(row, flush=True)
        lines.append(row)
        cells.append((name, pack, recs))

    lines += [
        "",
        "## Wait-ATP 底（更接近论文 DEER）",
        "",
        "| 集 | DEER-Wait | +默认门 | 门比DEER | 上限 | 上限比DEER | 开门题 |",
        "|---|---|---|---|---|---|---:|",
    ]
    print(lines[-2], flush=True)
    for name, pack, recs in cells:
        hw = summarize([r["host_w"] for r in recs])
        dw = summarize([r["door_w"] for r in recs])
        ow = summarize([r["ora_w"] for r in recs])
        row = (
            f"| {name} | {sw.fmt_pair(hw['acc'], hw['tok'])} | "
            f"{sw.fmt_pair(dw['acc'], dw['tok'])} | {fmt_vs(dw, hw)} | "
            f"{sw.fmt_pair(ow['acc'], ow['tok'])} | {fmt_vs(ow, hw)} | "
            f"{sum(int(r['door_w'].get('fired')) for r in recs)}/{len(recs)} |"
        )
        print(row, flush=True)
        lines.append(row)

    lines += [
        "",
        "## 每步 >0.95（消融，停得更早）",
        "",
        "| 集 | DEER-每步 | +默认门 | 门比DEER | 上限 | 上限比DEER | 开门题 |",
        "|---|---|---|---|---|---|---:|",
    ]
    print(lines[-2], flush=True)
    for name, pack, recs in cells:
        ha = summarize([r["host_a"] for r in recs])
        da = summarize([r["door_a"] for r in recs])
        oa = summarize([r["ora_a"] for r in recs])
        row = (
            f"| {name} | {sw.fmt_pair(ha['acc'], ha['tok'])} | "
            f"{sw.fmt_pair(da['acc'], da['tok'])} | {fmt_vs(da, ha)} | "
            f"{sw.fmt_pair(oa['acc'], oa['tok'])} | {fmt_vs(oa, ha)} | "
            f"{sum(int(r['door_a'].get('fired')) for r in recs)}/{len(recs)} |"
        )
        print(row, flush=True)
        lines.append(row)

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
