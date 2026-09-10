#!/usr/bin/env python3
"""Stop when trial equals Full-CoT or gold. Acc vs 写完 cannot drop."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg

TABLE = AE / "tables/accsafe_stop_ceiling.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("32B", "r1_32b"),
    ("30B", "qwen3_30b_a3b"),
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


def match_full(ans: Any, original_answer: Any, a_final: Any) -> bool:
    return rg.same(ans, original_answer) or rg.same(ans, a_final)


def allowed(
    ans: Any, original_answer: Any, a_final: Any, gt: Any, orig_ok: bool
) -> bool:
    if rg.same(ans, gt) or rg.same(ans, original_answer):
        return True
    if match_full(ans, original_answer, a_final):
        return (not orig_ok) or rg.same(ans, gt)
    return False


def credit(ans: Any, gt: Any, original_answer: Any, orig_ok: bool) -> int:
    if rg.same(ans, gt):
        return 1
    if rg.same(ans, original_answer):
        return int(orig_ok)
    return 0


def first_lock(
    rows: list[dict[str, Any]],
    *,
    original_answer: Any,
    a_final: Any,
    gt: Any,
    orig_ok: bool,
    mode: str,
) -> int | None:
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        if int(row["stopped_len"]) < rg.MSS:
            continue
        ans = row.get("final_answer")
        hit_full = match_full(ans, original_answer, a_final)
        hit_gt = rg.same(ans, gt)
        if mode == "af" and hit_full and allowed(ans, original_answer, a_final, gt, orig_ok):
            return end
        if mode == "safe" and (hit_full or hit_gt) and allowed(
            ans, original_answer, a_final, gt, orig_ok
        ):
            return end
    return None


def first_step(
    rows: list[dict[str, Any]],
    *,
    original_answer: Any,
    a_final: Any,
    gt: Any,
    orig_ok: bool,
) -> int | None:
    for end, row in enumerate(rows):
        if int(row["stopped_len"]) < rg.MSS:
            continue
        ans = row.get("final_answer")
        if allowed(ans, original_answer, a_final, gt, orig_ok):
            return end
    return None


def score_end(
    trials: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    end: int | None,
    orig_ok: bool,
    orig_t: int,
    gt: Any,
    original_answer: Any,
    full_tok: float,
) -> tuple[int, float, bool, bool]:
    if end is None:
        return int(orig_ok), full_tok, False, False
    sim = rg.pack(trials, rows, end, "rescue", original_tokens=orig_t)
    ok = credit(sim["answer"], gt, original_answer, orig_ok)
    return ok, sim["tokens"], True, bool(ok) and not orig_ok


def dense_trial_path(model: str, dataset: str, seed: int) -> Path:
    if dataset in ("aime24", "aime25"):
        return AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/dense_puma/trial_answers.json"
    return AE / f"results/dense_G_{model}/{dataset}/dense_puma/trial_answers.json"


def eval_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dense_trial_path(model, dataset, seed)
    if dataset in ("aime24", "aime25"):
        gpath = AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/per_sample.json"
    else:
        gpath = AE / f"results/dense_G_{model}/{dataset}/per_sample.json"
    if not puma_path.is_file() or not trials_path.is_file() or not gpath.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gpath)}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    n = orig_acc = puma_acc = regen_acc = af_acc = safe_acc = step_acc = 0
    orig_tok = puma_tok = regen_tok = af_tok = safe_tok = step_tok = 0.0
    n_af = n_safe = n_step = n_safe_lift = n_step_lift = n_regen = 0
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen_map = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
        if regen_path.is_file()
        else {}
    )
    for qi, info in sorted(official.items()):
        trials = by.get(qi)
        gg = gmap.get(qi)
        if not trials or not gg:
            continue
        original_answer = info.get("original_answer")
        a_final = gg.get("A_final") or original_answer
        gt = info.get("ground_truth")
        rows = rg.usable_rows(trials)
        orig_ok = bool(info.get("original_correct"))
        orig_t = int(info.get("original_tokens") or 0)
        full = rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=orig_t)
        kw = dict(original_answer=original_answer, a_final=a_final, gt=gt, orig_ok=orig_ok)
        puma_ok = bool(info.get("compressed_correct"))
        puma_t = int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)
        n += 1
        orig_acc += int(orig_ok)
        orig_tok += orig_t
        puma_acc += int(puma_ok)
        puma_tok += puma_t
        regen = regen_map.get(qi)
        if regen:
            n_regen += 1
            regen_acc += int(bool(regen.get("compressed_correct")))
            regen_tok += int(regen.get("compressed_tokens") or 0) + int(
                regen.get("tokens_trial_answers") or 0
            )
        a_ok, a_t, a_hit, _ = score_end(
            trials, rows, first_lock(rows, mode="af", **kw),
            orig_ok, orig_t, gt, original_answer, full["tokens"],
        )
        s_ok, s_t, s_hit, s_lift = score_end(
            trials, rows, first_lock(rows, mode="safe", **kw),
            orig_ok, orig_t, gt, original_answer, full["tokens"],
        )
        p_ok, p_t, p_hit, p_lift = score_end(
            trials, rows, first_step(rows, **kw),
            orig_ok, orig_t, gt, original_answer, full["tokens"],
        )
        af_acc += a_ok
        af_tok += a_t
        safe_acc += s_ok
        safe_tok += s_t
        step_acc += p_ok
        step_tok += p_t
        n_af += int(a_hit)
        n_safe += int(s_hit)
        n_step += int(p_hit)
        n_safe_lift += int(bool(s_ok) and not puma_ok)
        n_step_lift += int(bool(p_ok) and not puma_ok)
    if n == 0:
        return None
    return {
        "n": n,
        "n_regen": n_regen,
        "orig_acc": orig_acc / n,
        "orig_tok": orig_tok / n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "regen_acc": regen_acc / n if n_regen == n else float("nan"),
        "regen_tok": regen_tok / n if n_regen == n else float("nan"),
        "af_acc": af_acc / n,
        "af_tok": af_tok / n,
        "safe_acc": safe_acc / n,
        "safe_tok": safe_tok / n,
        "step_acc": step_acc / n,
        "step_tok": step_tok / n,
        "n_af": n_af,
        "n_safe": n_safe,
        "n_step": n_step,
        "n_safe_lift": n_safe_lift,
        "n_step_lift": n_step_lift,
        "d_af_acc": 100.0 * (af_acc / n - puma_acc / n),
        "d_af_tok": af_tok / n - puma_tok / n,
        "d_safe_acc": 100.0 * (safe_acc / n - puma_acc / n),
        "d_safe_tok": safe_tok / n - puma_tok / n,
        "d_step_acc": 100.0 * (step_acc / n - puma_acc / n),
        "d_step_tok": step_tok / n - puma_tok / n,
        "d_regen_acc": 100.0 * (regen_acc / n - puma_acc / n) if n_regen == n else float("nan"),
        "d_regen_tok": regen_tok / n - puma_tok / n if n_regen == n else float("nan"),
    }


def merge(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["n"] for r in recs)
    out: dict[str, Any] = {"n": n}
    skip = {"n"}
    for key in recs[0]:
        if key in skip or key.startswith("d_"):
            continue
        if key.startswith("n_"):
            out[key] = sum(r[key] for r in recs)
        else:
            out[key] = sum(r[key] * r["n"] for r in recs) / n
    out["d_af_acc"] = 100.0 * (out["af_acc"] - out["puma_acc"])
    out["d_af_tok"] = out["af_tok"] - out["puma_tok"]
    out["d_safe_acc"] = 100.0 * (out["safe_acc"] - out["puma_acc"])
    out["d_safe_tok"] = out["safe_tok"] - out["puma_tok"]
    out["d_step_acc"] = 100.0 * (out["step_acc"] - out["puma_acc"])
    out["d_step_tok"] = out["step_tok"] - out["puma_tok"]
    if out.get("regen_acc") == out.get("regen_acc"):
        out["d_regen_acc"] = 100.0 * (out["regen_acc"] - out["puma_acc"])
        out["d_regen_tok"] = out["regen_tok"] - out["puma_tok"]
    else:
        out["d_regen_acc"] = float("nan")
        out["d_regen_tok"] = float("nan")
    return out


def fmt(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def cell(rec: dict[str, Any], prefix: str) -> str:
    extra = ""
    if prefix == "safe":
        extra = f"；开火 {rec['n_safe']}/{rec['n']}，比PUMA多对{rec['n_safe_lift']}"
        if rec.get("regen_acc") == rec.get("regen_acc"):
            d_acc = 100.0 * (rec["safe_acc"] - rec["regen_acc"])
            d_tok = rec["safe_tok"] - rec["regen_tok"]
            extra += f"；对密探 {rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)}"
    elif prefix == "step":
        extra = f"；{rec['n_step']}/{rec['n']}"
    elif prefix == "regen":
        if rec.get("regen_acc") != rec.get("regen_acc"):
            return "—"
    return (
        f"{fmt(rec[f'{prefix}_acc'], rec[f'{prefix}_tok'])}"
        f"（{rg.fmt_pp(rec[f'd_{prefix}_acc'])} / {rg.fmt_tok(rec[f'd_{prefix}_tok'])}{extra}）"
    )


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        recs = []
        for seed in dd.AIME_SEEDS:
            rec = eval_cell(model, dataset, seed)
            if rec:
                recs.append(rec)
        if not recs:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        rec = merge(recs)
        rec["name"] = f"{zh} {ds_zh}" if len(recs) == 4 else f"{zh} {ds_zh}（{len(recs)} seed）"
    else:
        rec = eval_cell(model, dataset, 42)
        if rec is None:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        rec["name"] = f"{zh} {ds_zh}"
    print(
        f"{rec['name']} n={rec['n']} puma={rec['puma_acc']:.3f} "
        f"safe={rec['safe_acc']:.3f} {rec['d_safe_acc']:+.1f}pp",
        flush=True,
    )
    return rec


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    header = "| 集 | PUMA | 密探k4 | 先知上限（写完或金标就停） |"
    sep = "|---|---|---|---|"
    lines = [
        "# 写完或金标就停：相对官方 PUMA 的上限",
        "",
        "这不是现成方法。没有门、没有重写。先知：连续 4 步试答已经等于写完终答或金标就交试答，没锁上的题写完。",
        "括号默认相对官方 PUMA。密探k4 有重写才报，否则 —。上限格子若有密探k4，会多写一截相对密探。",
        "写完终答 = 官方 original_answer 或轨迹 A_final。交跟写完同一串时，对错跟官方写完走。",
        "只认密探轨迹。步数 < 10 不许停。AIME 默认四个 seed；缺的会标明。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for rec in recs:
        row = (
            f"| {rec['name']} | {fmt(rec['puma_acc'], rec['puma_tok'])} "
            f"| {cell(rec, 'regen')} | {cell(rec, 'safe')} |"
        )
        print(row, flush=True)
        lines.append(row)
    lines += [
        "",
        "## 读法",
        "",
        "MATH 上上限相对 PUMA 通常只有 1–3 点、几百 token，密探k4 已经贴近。",
        "GPQA / 难 AIME 上空档大：上限能比 PUMA 高好几到十几分，token 少一两千以上。",
        "奥赛常见 Acc 还能涨，但没锁上的题写完，token 不一定比 PUMA 少。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
