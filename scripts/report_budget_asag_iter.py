#!/usr/bin/env python3
"""连答短预算：扫把握 / 层间 JS / DoLA，对照密探和论文 ASAG。"""
from __future__ import annotations

import json
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

TABLE = AE / "tables/budget_asag_iter.md"
INT = AE / "results/confcal_judge/v2/dense_internal"
EIGEN = AE / "results/leftover_eigen_dola"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("Qwen3-4B", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b"),
)
DS = (
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)
ASAG = {
    "7B 奥赛": (0.523, 5325),
    "7B GPQA": (0.318, 5584),
    "7B AIME24": (0.467, 9154),
    "7B AIME25": (0.367, 7547),
    "Qwen3-4B 奥赛": (0.646, 7536),
    "Qwen3-4B GPQA": (0.480, 3563),
    "Qwen3-4B AIME24": (0.700, 8768),
    "Qwen3-8B GPQA": (0.561, 5714),
    "Qwen3-8B AIME24": (0.667, 8683),
    "Qwen3-8B AIME25": (0.600, 11826),
}
FEAT_KEYS = (
    "dola_jsd_max",
    "dola_mean_rise",
    "lookback_ratio",
    "stop_margin",
    "wait_logp",
    "eigen_k4",
    "geo_conf",
)


def feat(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if x == x else float("nan")


def internal_dirs(model: str, dataset: str, seed: int) -> list[Path]:
    if dataset in ("aime24", "aime25"):
        return [INT / model / f"{dataset}_s{seed}", INT / f"{dataset}_s{seed}"]
    if model == "r1_7b":
        return [INT / dataset, INT / model / dataset]
    return [INT / model / dataset, INT / dataset]


def load_internal(model: str, dataset: str, seed: int) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    for root in internal_dirs(model, dataset, seed):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("scores_shard*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                key = (int(row["question_idx"]), int(row["decision_step"]))
                out[key] = {name: feat(row.get(name)) for name in FEAT_KEYS}
    return out


def load_eigen() -> dict[tuple[str, str, int, int], dict[str, float]]:
    out: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for path in sorted(EIGEN.glob("*_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            key = (
                str(row["model"]),
                str(row["dataset"]),
                int(row.get("seed") or 42),
                int(row["question_idx"]),
            )
            out[key] = {
                "eigen_jsd": feat(row.get("dola_jsd")),
                "eigen_h": feat(row.get("h_pre_k8")),
                "eigen_pre": feat(row.get("eigen_pre_k8")),
                "eigen_step": feat(row.get("decision_step")),
            }
    return out


def pack_hit(
    trials: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    got: dict[str, Any],
    *,
    orig_tok: int,
    gt: Any,
    original: Any,
    orig_ok: bool,
    internals: dict[tuple[int, int], dict[str, float]],
    eigen: dict[str, float] | None,
    first_step: int,
) -> dict[str, Any]:
    packed = rg.pack(trials, rows, got["end"], "rescue", original_tokens=orig_tok)
    rec = {
        "ok": low.credit(packed["answer"], gt, original, orig_ok),
        "tok": packed["tokens"],
        "c1": got["c1"],
        "rise": got["rise"],
        "step": got["step"],
    }
    rec.update(internals.get((int(rows[0]["question_idx"]), int(got["step"]))) or {})
    t0 = internals.get((int(rows[0]["question_idx"]), int(first_step))) or {}
    rec["jsd0"] = t0.get("dola_jsd_max", float("nan"))
    rec["rise0"] = t0.get("dola_mean_rise", float("nan"))
    rec["lb0"] = t0.get("lookback_ratio", float("nan"))
    if eigen and int(eigen.get("eigen_step") or -1) == int(first_step):
        rec["eigen_jsd"] = eigen.get("eigen_jsd", float("nan"))
        rec["eigen_h"] = eigen.get("eigen_h", float("nan"))
    else:
        rec["eigen_jsd"] = rec["eigen_h"] = float("nan")
    return rec


def load_cell(
    model: str,
    dataset: str,
    seed: int,
    eigen_map: dict[tuple[str, str, int, int], dict[str, float]],
) -> dict[str, Any] | None:
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
    internals = load_internal(model, dataset, seed)
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
        else:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
            host_step = int(sim["step"])
        first = bg.first_same(rows)
        hits: dict[int, dict[str, Any] | None] = {}
        for budget in (2, 4, 8):
            hit = None
            if first is not None and first["kind"] != "high":
                got = bg.persist(rows, first, budget)
                if got is not None and got["step"] < host_step:
                    hit = pack_hit(
                        trials,
                        rows,
                        got,
                        orig_tok=orig_tok,
                        gt=gt,
                        original=original,
                        orig_ok=orig_ok,
                        internals=internals,
                        eigen=eigen_map.get((model, dataset, seed if dataset.startswith("aime") else 42, qi)),
                        first_step=int(first["step"]),
                    )
            hits[budget] = hit
        questions.append(
            {
                "host_ok": host_ok,
                "host_tok": host_tok,
                "hits": hits,
            }
        )
    if not questions:
        return None
    return {"questions": questions, "n_regen": n_regen}


def apply(
    qs: list[dict[str, Any]],
    budget: int,
    pred: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    acc = tok = fire = gain = hurt = 0
    for q in qs:
        hit = q["hits"].get(budget)
        use = hit is not None and pred(hit)
        if use:
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


def ge(name: str, tau: float) -> Callable[[dict[str, Any]], bool]:
    return lambda h: feat(h.get(name)) >= tau


def le(name: str, tau: float) -> Callable[[dict[str, Any]], bool]:
    return lambda h: feat(h.get(name)) <= tau


def both(
    left: Callable[[dict[str, Any]], bool],
    right: Callable[[dict[str, Any]], bool],
) -> Callable[[dict[str, Any]], bool]:
    return lambda h: left(h) and right(h)


def always(_hit: dict[str, Any]) -> bool:
    return True


RULES: list[tuple[str, int, Callable[[dict[str, Any]], bool]]] = [
    ("B2 仍同答", 2, always),
    ("B4 仍同答", 4, always),
    ("B8 仍同答", 8, always),
    ("B4 把握≥0.90", 4, ge("c1", 0.90)),
    ("B4 把握≥0.95", 4, ge("c1", 0.95)),
    ("B2 把握≥0.90", 2, ge("c1", 0.90)),
    ("B4 JS≤0.50", 4, le("dola_jsd_max", 0.50)),
    ("B4 JS≤0.60", 4, le("dola_jsd_max", 0.60)),
    ("B4 JS≤0.70", 4, le("dola_jsd_max", 0.70)),
    ("B4 JS≥0.65", 4, ge("dola_jsd_max", 0.65)),
    ("B4 窗时 JS≤0.60", 4, le("jsd0", 0.60)),
    ("B4 窗时 JS≤0.50", 4, le("jsd0", 0.50)),
    ("B4 DoLA升≥1", 4, ge("dola_mean_rise", 1.0)),
    ("B4 DoLA升≥2", 4, ge("dola_mean_rise", 2.0)),
    ("B4 回看题面≥0.40", 4, ge("lookback_ratio", 0.40)),
    ("B4 回看题面≥0.50", 4, ge("lookback_ratio", 0.50)),
    ("B4 停边距≥0", 4, ge("stop_margin", 0.0)),
    ("B4 剩窗JS≤0.60", 4, le("eigen_jsd", 0.60)),
    ("B4 把握≥0.90 且 JS≤0.60", 4, both(ge("c1", 0.90), le("dola_jsd_max", 0.60))),
    ("B4 把握≥0.90 且 JS≤0.70", 4, both(ge("c1", 0.90), le("dola_jsd_max", 0.70))),
    ("B4 把握≥0.85 且 JS≤0.60", 4, both(ge("c1", 0.85), le("dola_jsd_max", 0.60))),
    ("B4 把握≥0.90 或 JS≤0.50", 4, lambda h: feat(h.get("c1")) >= 0.90 or feat(h.get("dola_jsd_max")) <= 0.50),
    ("B4 把握≥0.90 且 DoLA升≥1", 4, both(ge("c1", 0.90), ge("dola_mean_rise", 1.0))),
    ("B4 把握≥0.90 且回看≥0.40", 4, both(ge("c1", 0.90), ge("lookback_ratio", 0.40))),
    ("B2 把握≥0.90 且 JS≤0.60", 2, both(ge("c1", 0.90), le("dola_jsd_max", 0.60))),
    ("B8 把握≥0.90", 8, ge("c1", 0.90)),
    ("B8 把握≥0.90 且 JS≤0.60", 8, both(ge("c1", 0.90), le("dola_jsd_max", 0.60))),
]


def merge(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs: list[dict[str, Any]] = []
    n_regen = 0
    for pack in packs:
        qs.extend(pack["questions"])
        n_regen += pack["n_regen"]
    return {"questions": qs, "n_regen": n_regen}


def score_cell(name: str, pack: dict[str, Any]) -> dict[str, Any]:
    qs = pack["questions"]
    n = len(qs)
    rec: dict[str, Any] = {
        "name": name,
        "n": n,
        "host_name": "密探k4" if pack["n_regen"] == n else "PUMA",
        "host_acc": sum(q["host_ok"] for q in qs) / n,
        "host_tok": sum(q["host_tok"] for q in qs) / n,
    }
    for rule, budget, pred in RULES:
        rec[rule] = apply(qs, budget, pred)
    return rec


def cell_txt(rec: dict[str, Any], key: str) -> str:
    door = rec[key]
    d_acc = 100.0 * (door["acc"] - rec["host_acc"])
    d_tok = door["tok"] - rec["host_tok"]
    mark = " 伤" if d_acc < -1e-12 else ""
    asag = ASAG.get(rec["name"])
    extra = ""
    if asag:
        extra = (
            " |ASAG"
            + ("Acc过" if door["acc"] + 1e-12 >= asag[0] else "Acc没过")
            + ("Tok过" if door["tok"] <= asag[1] + 1e-6 else "Tok没过")
        )
    return (
        f"{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)} "
        f"（{door['fire']}/{rec['n']}，伤 {door['hurt']}）{mark}{extra}"
    )


def auroc_on_persist(recs_qs: list[tuple[str, list[dict[str, Any]]]], key: str) -> float:
    pos: list[float] = []
    neg: list[float] = []
    for _name, qs in recs_qs:
        for q in qs:
            hit = q["hits"].get(4)
            if hit is None:
                continue
            val = feat(hit.get(key))
            if val != val:
                continue
            hurt = (not hit["ok"]) and q["host_ok"]
            (pos if hurt else neg).append(val)
    return rg.auroc(pos, neg)


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    eigen_map = load_eigen()
    recs = []
    raw_qs: list[tuple[str, list[dict[str, Any]]]] = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            if dataset in ("aime24", "aime25"):
                packs = []
                seeds = []
                for seed in dd.AIME_SEEDS:
                    pack = load_cell(model, dataset, seed, eigen_map)
                    if pack:
                        packs.append(pack)
                        seeds.append(seed)
                if not packs:
                    print(f"skip {zh} {ds_zh}", flush=True)
                    continue
                pack = merge(packs)
                name = f"{zh} {ds_zh}" if len(seeds) == 4 else f"{zh} {ds_zh}（{len(seeds)} seed）"
            else:
                pack = load_cell(model, dataset, 42, eigen_map)
                if pack is None:
                    print(f"skip {zh} {ds_zh}", flush=True)
                    continue
                name = f"{zh} {ds_zh}"
            rec = score_cell(name, pack)
            recs.append(rec)
            raw_qs.append((name, pack["questions"]))
            print(f"{name} n={rec['n']} host={rec['host_name']}", flush=True)

    seven = [r for r in recs if r["name"].startswith("7B ")]
    ranked = []
    for rule, _b, _p in RULES:
        if not seven:
            continue
        acc_vs_asag = []
        tok_vs_asag = []
        tok_vs_host = []
        d_accs = []
        fires = 0
        hurts = 0
        missing = False
        for rec in seven:
            door = rec[rule]
            asag = ASAG.get(rec["name"])
            if not asag:
                missing = True
                continue
            acc_vs_asag.append(door["acc"] + 1e-12 >= asag[0])
            tok_vs_asag.append(door["tok"] <= asag[1] + 1e-6)
            tok_vs_host.append(door["tok"] < rec["host_tok"] - 1e-6)
            d_accs.append(100.0 * (door["acc"] - rec["host_acc"]))
            fires += door["fire"]
            hurts += door["hurt"]
        if missing or not acc_vs_asag:
            continue
        ranked.append(
            {
                "rule": rule,
                "asag_acc": sum(acc_vs_asag),
                "asag_tok": sum(tok_vs_asag),
                "host_tok": sum(tok_vs_host),
                "n": len(acc_vs_asag),
                "mean_d_acc": sum(d_accs) / len(d_accs),
                "mean_d_tok": sum(
                    rec[rule]["tok"] - rec["host_tok"] for rec in seven
                )
                / len(seven),
                "fires": fires,
                "hurts": hurts,
                "all_asag_acc": all(acc_vs_asag),
                "all_save": all(tok_vs_host),
            }
        )
    ranked.sort(
        key=lambda x: (
            -int(x["all_asag_acc"] and x["all_save"] and x["asag_tok"] >= 2),
            -x["asag_acc"],
            -x["asag_tok"],
            x["mean_d_tok"],
            -x["mean_d_acc"],
        )
    )

    focus = [
        "B4 仍同答",
        "B4 把握≥0.90",
        "B4 JS≤0.60",
        "B4 DoLA升≥1",
        "B4 回看题面≥0.40",
        "B4 把握≥0.90 且 JS≤0.60",
        "B2 把握≥0.90",
        "B8 把握≥0.90",
    ]
    lines = [
        "# 连答短预算：把握 vs 层间 JS，对照密探 / 论文 ASAG",
        "",
        "第一次非高把握连答后再走 B 步，仍同答才考虑交当时试答（不重写）。",
        "层间 JS = 末层和下层下一词分布的 Jensen–Shannon（`dola_jsd_max`），",
        "DoLA 升 = 末层比一半深更像在写答案。都来自已有密探内部抽取，不用 4B。",
        "论文 ASAG 不是同盘（贪心 T=0，还会灌 logits）。这里只比绝对 Acc / token。",
        "",
        "## 1. 7B 上谁更能压过论文 ASAG 且比密探省 token",
        "",
        "| 设定 | 4 格 Acc≥ASAG | 4 格 Tok≤ASAG | 4 格比密探省 | 7B 均 ΔAcc | 7B 均 ΔTok | 开火 / 伤 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['rule']} | {row['asag_acc']}/{row['n']} | {row['asag_tok']}/{row['n']} | "
            f"{row['host_tok']}/{row['n']} | {row['mean_d_acc']:+.2f}pp | "
            f"{row['mean_d_tok']:.0f} | {row['fires']} / {row['hurts']} |"
        )
    lines += [
        "",
        "B4 持久候选上，分数越大越像「交了会伤密探」：",
        f"把握 {auroc_on_persist(raw_qs, 'c1'):.3f}，",
        f"到点 JS {auroc_on_persist(raw_qs, 'dola_jsd_max'):.3f}，",
        f"窗时 JS {auroc_on_persist(raw_qs, 'jsd0'):.3f}，",
        f"DoLA 升 {auroc_on_persist(raw_qs, 'dola_mean_rise'):.3f}，",
        f"回看题面 {auroc_on_persist(raw_qs, 'lookback_ratio'):.3f}，",
        f"停边距 {auroc_on_persist(raw_qs, 'stop_margin'):.3f}。",
        "",
        "## 2. 分格（相对宿主；附带是否压过论文 ASAG）",
        "",
        "| 集 | 宿主 | " + " | ".join(focus) + " |",
        "|---|---|" + "|".join(["---"] * len(focus)) + "|",
    ]
    for rec in recs:
        lines.append(
            "| "
            + " | ".join(
                [
                    rec["name"],
                    f"{rec['host_name']} {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f}",
                    *[cell_txt(rec, name) for name in focus],
                ]
            )
            + " |"
        )
    best = next(
        (
            x
            for x in ranked
            if x["all_asag_acc"] and x["all_save"] and x["fires"] > 0
        ),
        ranked[0] if ranked else None,
    )
    lines += ["", "## 3. 先用这条", ""]
    if best:
        lines.append(
            f"**{best['rule']}**：7B 上 Acc≥ASAG {best['asag_acc']}/{best['n']}，"
            f"Tok≤ASAG {best['asag_tok']}/{best['n']}，"
            f"相对密探均 {best['mean_d_acc']:+.2f}pp / {best['mean_d_tok']:.0f} token。"
        )
    else:
        lines.append("没有同时压过 7B 四格 ASAG Acc 又比密探省 token 的设定。")
    lines.append("")
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines[-8:]))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
