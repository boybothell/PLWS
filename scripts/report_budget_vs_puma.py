#!/usr/bin/env python3
"""密探k4 / 连答短预算，一律对照官方 PUMA。不用论文 ASAG。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_budget_conf_gate as bg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import report_first_lock_room as room  # noqa: E402

TABLE = AE / "tables/budget_vs_puma.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("32B", "r1_32b"),
    ("Qwen3-4B", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b"),
)
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)
RULES: list[tuple[str, int, Callable[[float, float], bool]]] = [
    ("B4 仍同答就交", 4, lambda _c, _d: True),
    ("B4 把握≥0.90", 4, lambda c, _d: c >= 0.90),
    ("B4 把握≥0.95", 4, lambda c, _d: c >= 0.95),
    ("B4 把握≥0.90 且不降", 4, lambda c, d: c >= 0.90 and d >= 0.0),
]


def puma_tok_of(info: dict[str, Any]) -> int:
    return int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)


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
        puma_ok = bool(info.get("compressed_correct"))
        puma_t = puma_tok_of(info)
        puma_step = int(info.get("stopped_len") or 10**9)
        host = regen.get(qi)
        if host:
            n_regen += 1
            k4_ok = bool(host.get("compressed_correct"))
            k4_t = int(host.get("compressed_tokens") or 0) + int(host.get("tokens_trial_answers") or 0)
            k4_step = int(host.get("stopped_len") or puma_step)
        else:
            k4_ok = k4_t = k4_step = None
        first = bg.first_same(rows)
        cands: dict[str, dict[str, Any] | None] = {}
        for name, budget, pred in RULES:
            hit = None
            if first is not None and first["kind"] != "high":
                got = bg.persist(rows, first, budget)
                if (
                    got is not None
                    and got["step"] < puma_step
                    and pred(got["c1"], got["rise"])
                ):
                    packed = rg.pack(
                        trials, rows, got["end"], "rescue", original_tokens=orig_tok
                    )
                    hit = {
                        "ok": low.credit(packed["answer"], gt, original, orig_ok),
                        "tok": packed["tokens"],
                    }
            cands[name] = hit
        questions.append(
            {
                "puma_ok": puma_ok,
                "puma_tok": puma_t,
                "k4_ok": k4_ok,
                "k4_tok": k4_t,
                "k4_step": k4_step,
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


def apply_mix(qs: list[dict[str, Any]], key: str | None) -> dict[str, Any]:
    acc = tok = fire = gain = hurt = 0
    for q in qs:
        hit = q["cands"].get(key) if key else None
        if hit is not None:
            ok, t = hit["ok"], hit["tok"]
            fire += 1
            gain += int(ok and not q["puma_ok"])
            hurt += int((not ok) and q["puma_ok"])
        elif q["k4_ok"] is not None:
            ok, t = q["k4_ok"], q["k4_tok"]
        else:
            ok, t = q["puma_ok"], q["puma_tok"]
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
        "has_k4": pack["n_regen"] == n,
        "puma_acc": sum(q["puma_ok"] for q in qs) / n,
        "puma_tok": sum(q["puma_tok"] for q in qs) / n,
        "密探k4": apply_mix(qs, None),
    }
    if not rec["has_k4"]:
        rec["密探k4"] = None
    for rule_name, _b, _p in RULES:
        rec[rule_name] = apply_mix(qs, rule_name)
    print(f"{name} n={n} k4={'yes' if rec['has_k4'] else 'no'}", flush=True)
    return rec


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def delta(door: dict[str, Any], rec: dict[str, Any], *, leftover: bool) -> str:
    d_acc = 100.0 * (door["acc"] - rec["puma_acc"])
    d_tok = door["tok"] - rec["puma_tok"]
    mark = " 伤" if d_acc < -1e-12 else ""
    extra = ""
    if leftover and door["fire"]:
        extra = f"（{door['fire']}/{rec['n']}，伤 {door['hurt']}）"
    return f"{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)} {extra}{mark}".rstrip()


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    methods = ["密探k4", *[name for name, _b, _p in RULES]]
    lines = [
        "# 对照官方 PUMA",
        "",
        "不比论文 ASAG。同一题、同一条官方 CoT。",
        "",
        "密探k4：高把握连答 k=4 停，后路强停，截断后重写终答。只有 7B / 8B 有重写。",
        "连答短预算：第一次非高把握连答后再走 4 步，仍同答才考虑提前交当时试答（不重写）；",
        "不开火时，有密探k4 就留密探，否则留 PUMA。",
        "Token：PUMA / 密探 = 截断前缀 + 重写终答 + 实际试答；预算开火 = 当时试答轨迹。",
        "AIME 四个 seed 按题加权。正确率只对金标。",
        "",
        "括号：预算开火题数 / 总题，伤 = PUMA 对、提前交则错。",
        "",
        "## 1. 相对 PUMA（Acc 百分点 / token）",
        "",
        "| 集 | PUMA | " + " | ".join(methods) + " |",
        "|---|---|" + "|".join(["---"] * len(methods)) + "|",
    ]
    for rec in recs:
        cells = [rec["name"], pair(rec["puma_acc"], rec["puma_tok"])]
        for name in methods:
            door = rec.get(name)
            cells.append("—" if door is None else delta(door, rec, leftover=name != "密探k4"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 2. 绝对 Acc / token",
        "",
        "| 集 | PUMA | " + " | ".join(methods) + " |",
        "|---|---|" + "|".join(["---"] * len(methods)) + "|",
    ]
    for rec in recs:
        cells = [rec["name"], pair(rec["puma_acc"], rec["puma_tok"])]
        for name in methods:
            door = rec.get(name)
            cells.append("—" if door is None else pair(door["acc"], door["tok"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 3. 难集里 Acc 不低于 PUMA 的设定",
        "",
    ]
    hard = [rec for rec in recs if "MATH" not in rec["name"]]
    found = False
    for name in methods:
        hurt = [rec["name"] for rec in hard if rec.get(name) and rec[name]["acc"] + 1e-12 < rec["puma_acc"]]
        ready = [rec for rec in hard if rec.get(name)]
        if not ready:
            continue
        d_tok = sum((rec[name]["tok"] - rec["puma_tok"]) * rec["n"] for rec in ready) / max(
            1, sum(rec["n"] for rec in ready)
        )
        d_acc = 100.0 * sum((rec[name]["acc"] - rec["puma_acc"]) * rec["n"] for rec in ready) / max(
            1, sum(rec["n"] for rec in ready)
        )
        if not hurt:
            found = True
            lines.append(
                f"- {name}：难集 Acc 都不低于 PUMA，均 {rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)}"
            )
    if not found:
        lines.append("没有。难集上每个会改答案的设定都至少有一格低于 PUMA。")
    lines.append("")
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
