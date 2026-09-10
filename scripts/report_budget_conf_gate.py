#!/usr/bin/env python3
"""第一次连答后再走 B 步：仍同答才考虑停，用解题模型把握当不能停。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import report_first_lock_room as room  # noqa: E402
import report_k4_second_lock as sl  # noqa: E402

TABLE = AE / "tables/budget_conf_gate.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("32B", "r1_32b"),
    ("Qwen3-4B", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b"),
)
DS = (
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)
RULES: list[tuple[str, int, Callable[[float, float], bool]]] = [
    ("B2 仍同答就交", 2, lambda _c, _d: True),
    ("B4 仍同答就交", 4, lambda _c, _d: True),
    ("B8 仍同答就交", 8, lambda _c, _d: True),
    ("B4 且末步把握≥0.90", 4, lambda c, _d: c >= 0.90),
    ("B4 且末步把握≥0.95", 4, lambda c, _d: c >= 0.95),
    ("B4 且末步把握≥0.98", 4, lambda c, _d: c >= 0.98),
    ("B4 且把握升≥0.02", 4, lambda _c, d: d >= 0.02),
    ("B4 且把握升≥0.05", 4, lambda _c, d: d >= 0.05),
    ("B4 且(≥0.90 或升≥0.02)", 4, lambda c, d: c >= 0.90 or d >= 0.02),
    ("B4 且≥0.90 且不降", 4, lambda c, d: c >= 0.90 and d >= 0.0),
    ("B4 且≥0.95 且升≥0.02", 4, lambda c, d: c >= 0.95 and d >= 0.02),
    ("B2 且≥0.95 且升≥0.02", 2, lambda c, d: c >= 0.95 and d >= 0.02),
    ("B8 且≥0.95 且升≥0.02", 8, lambda c, d: c >= 0.95 and d >= 0.02),
]


def first_same(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for win in sl.same_windows(rows):
        return win
    return None


def persist(
    rows: list[dict[str, Any]],
    first: dict[str, Any],
    budget: int,
) -> dict[str, Any] | None:
    end = int(first["end"]) + budget
    if end >= len(rows):
        return None
    start_step = int(first["step"])
    tail = rows[int(first["end"]) + 1 : end + 1]
    if len(tail) != budget:
        return None
    steps = [start_step] + [int(x["stopped_len"]) for x in tail]
    if steps != list(range(steps[0], steps[0] + budget + 1)):
        return None
    ans = first["ans"]
    if not all(rg.same(ans, x.get("final_answer")) for x in tail):
        return None
    confs = [rg.finite(x.get("confidence")) for x in tail]
    if any(c != c for c in confs):
        return None
    last4 = rows[end + 1 - rg.K : end + 1]
    last_confs = rg.confs_of(last4)
    if rg.is_high(last_confs):
        return None
    c0 = float(first["c"])
    c1 = float(confs[-1])
    return {
        "end": end,
        "step": int(rows[end]["stopped_len"]),
        "ans": ans,
        "c0": c0,
        "c1": c1,
        "rise": c1 - c0,
        "kind": first["kind"],
    }


def apply_door(qs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    acc = tok = fire = gain = hurt = 0
    for q in qs:
        hit = q["cands"].get(key)
        if hit is not None:
            ok, t = hit["ok"], hit["tok"]
            fire += 1
            gain += int(ok and not q["host_ok"])
            hurt += int((not ok) and q["host_ok"])
        else:
            ok, t = q["host_ok"], q["host_tok"]
        acc += int(ok)
        tok += t
    n = len(qs)
    return {
        "acc": acc / n if n else 0.0,
        "tok": tok / n if n else 0.0,
        "fire": fire,
        "gain": gain,
        "hurt": hurt,
        "n": n,
    }


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not puma_path.is_file() or not trials_path.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
        if regen_path.is_file()
        else {}
    )
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    questions = []
    n_regen = 0
    for qi, info in sorted(official.items()):
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        gt = info.get("ground_truth")
        original = info.get("original_answer")
        orig_ok = bool(info.get("original_correct"))
        orig_tok = int(info.get("original_tokens") or 0)
        host = regen.get(qi)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        if host:
            n_regen += 1
            host_ok = bool(host.get("compressed_correct"))
            host_tok = int(host.get("compressed_tokens") or 0) + int(
                host.get("tokens_trial_answers") or 0
            )
            host_step = int(host.get("stopped_len") or sim["step"])
            host_name = "密探k4"
        else:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
            host_step = int(sim["step"])
            host_name = "PUMA"
        first = first_same(rows)
        cands: dict[str, dict[str, Any] | None] = {}
        for name, budget, pred in RULES:
            hit = None
            if first is not None and first["kind"] != "high":
                got = persist(rows, first, budget)
                if (
                    got is not None
                    and got["step"] < host_step
                    and pred(got["c1"], got["rise"])
                ):
                    packed = rg.pack(
                        trials, rows, got["end"], "rescue", original_tokens=orig_tok
                    )
                    hit = {
                        "ok": low.credit(packed["answer"], gt, original, orig_ok),
                        "tok": packed["tokens"],
                        "c1": got["c1"],
                        "rise": got["rise"],
                    }
            cands[name] = hit
        questions.append(
            {
                "host_ok": host_ok,
                "host_tok": host_tok,
                "host_name": host_name,
                "cands": cands,
            }
        )
    if not questions:
        return None
    return {"questions": questions, "n_regen": n_regen}


def merge(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs: list[dict[str, Any]] = []
    n_regen = 0
    for pack in packs:
        qs.extend(pack["questions"])
        n_regen += pack["n_regen"]
    return {"questions": qs, "n_regen": n_regen}


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = []
        seeds = []
        for seed in dd.AIME_SEEDS:
            pack = load_cell(model, dataset, seed)
            if pack:
                packs.append(pack)
                seeds.append(seed)
        if not packs:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        pack = merge(packs)
        name = f"{zh} {ds_zh}" if len(seeds) == 4 else f"{zh} {ds_zh}（{len(seeds)} seed）"
    else:
        pack = load_cell(model, dataset, 42)
        if pack is None:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        name = f"{zh} {ds_zh}"
    qs = pack["questions"]
    n = len(qs)
    rec: dict[str, Any] = {
        "name": name,
        "n": n,
        "host_name": "密探k4" if pack["n_regen"] == n else "PUMA",
        "host_acc": sum(q["host_ok"] for q in qs) / n,
        "host_tok": sum(q["host_tok"] for q in qs) / n,
    }
    for name, _budget, _pred in RULES:
        rec[name] = apply_door(qs, name)
    print(f"{name} n={n} host={rec['host_name']}", flush=True)
    return rec


def cell(rec: dict[str, Any], key: str) -> str:
    door = rec[key]
    d_acc = 100.0 * (door["acc"] - rec["host_acc"])
    d_tok = door["tok"] - rec["host_tok"]
    mark = " 伤" if d_acc < -1e-12 else ""
    return (
        f"{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)} "
        f"（{door['fire']}/{rec['n']}，伤 {door['hurt']}）{mark}"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    focus = [
        "B4 仍同答就交",
        "B4 且末步把握≥0.90",
        "B4 且末步把握≥0.95",
        "B4 且把握升≥0.02",
        "B4 且≥0.90 且不降",
        "B4 且≥0.95 且升≥0.02",
    ]
    extra = [
        "B2 仍同答就交",
        "B8 仍同答就交",
        "B4 且末步把握≥0.98",
        "B4 且把握升≥0.05",
        "B4 且(≥0.90 或升≥0.02)",
        "B2 且≥0.95 且升≥0.02",
        "B8 且≥0.95 且升≥0.02",
    ]
    lines = [
        "# 连答短预算 + 把握不能停",
        "",
        "第一次连续 4 步同一试答、且还不是高把握锁，再密探 B 步。",
        "这 B 步必须步号相连、仍是同一答；中途改口或已经变成高把握锁 → 不能停，留宿主。",
        "到点仍同答，再用解题模型自己的把握决定交不交。交的是当时试答，不重写。",
        "宿主：有密探k4 重写就留它，否则留 PUMA。正确率只对金标。AIME 四个 seed。",
        "",
        "括号：相对宿主的正确率差 / token 差（开火题数/总题，伤了几题）。",
        "伤 = 这题宿主对、提前交则错。",
        "",
        "## 1. B=4 主设定",
        "",
        "| 集 | 宿主 | "
        + " | ".join(focus)
        + " |",
        "|---|---|" + "|".join(["---"] * len(focus)) + "|",
    ]
    for rec in recs:
        lines.append(
            "| "
            + " | ".join(
                [
                    rec["name"],
                    f"{rec['host_name']} {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f}",
                    *[cell(rec, name) for name in focus],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## 2. 其它 B / 门槛",
        "",
        "| 集 | " + " | ".join(extra) + " |",
        "|---|" + "|".join(["---"] * len(extra)) + "|",
    ]
    for rec in recs:
        lines.append("| " + " | ".join([rec["name"], *[cell(rec, name) for name in extra]]) + " |")
    safe: list[str] = []
    for name, _b, _p in RULES:
        hurt_cells = [
            rec["name"]
            for rec in recs
            if rec[name]["acc"] + 1e-12 < rec["host_acc"]
        ]
        fires = sum(rec[name]["fire"] for rec in recs)
        d_tok = sum(
            (rec[name]["tok"] - rec["host_tok"]) * rec["n"] for rec in recs
        ) / max(1, sum(rec["n"] for rec in recs))
        if not hurt_cells and fires:
            safe.append(f"{name}：开火 {fires}，均 token {d_tok:.0f}")
        elif not hurt_cells:
            safe.append(f"{name}：不伤，但不开火")
    lines += [
        "",
        "## 3. 难集每一格 Acc 都不低于宿主的设定",
        "",
    ]
    if safe:
        lines.extend(f"- {x}" for x in safe)
    else:
        lines.append("没有。所有会开火的设定都至少伤一格。")
    lines.append("")
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
