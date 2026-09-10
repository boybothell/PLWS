#!/usr/bin/env python3
"""Historical diagnostic (not PLWS): replay a dense-probe dual gate.

Probe: step 1, skip 2..6, dense from 7.
Consec stop: same answer for k=4, conf >= 0.995 with slack 0.03, step >= 10.
FS stop: step >= 80, length-red run >= 2, best conf >= 0.85,
         last 2 probed answers agree, pick peak-same with conf >= 0.7.
Red: adjacent step char-len ratio in [0.84, 1.16] OR |Δlen| <= 35.

Delivery = PUMA Step 4: after the gate cuts, regenerate the final answer
from the truncated reasoning prefix (`gen_prefixed_answers.py`).
This script also writes `final_candidates.json` for that regen.

Canonical PLWS does not use this delivery path. It continues from the window
prefix while suppressing Wait-class tokens in `score_leftover_suppress.py`.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
TABLE = AE / "tables/default_dense_vs_puma.md"
OUT = AE / "results/default_dense_gate/summary.json"

WARMUP_N = 7
TAU = 0.995
CONSEC = 4
EPS = 0.03
MSS = 10
FS_MIN_STEP = 80
FS_RUN = 2
FS_BEST = 0.85
FS_DEC = 0.7
FS_SAME_K = 2
LEN_LO = 0.84
LEN_HI = 1.16
ABS_DELTA = 35

MODELS = ("r1_7b", "nemotron_8b", "r1_14b", "r1_32b", "qwen3_30b_a3b")
SHORT_DS = {
    "math-500": "math",
    "olympiadbench": "oly",
    "gpqa-diamond": "gpqa",
    "aime24": "aime24",
    "aime25": "aime25",
}
AIME_SEEDS = (42, 0, 1, 123)
_WS = re.compile(r"\s+")
_LATEX = re.compile(r"\\(left|right|cdot|times|dfrac|tfrac|frac|mathrm|text)")


def norm(text: Any) -> str:
    s = _WS.sub("", str(text or "").strip().lower())
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = _LATEX.sub("", s)
    return s.replace("\\", "")


def same(a: Any, b: Any) -> bool:
    x, y = norm(a), norm(b)
    return bool(x) and x == y


def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def should_probe(step: int) -> bool:
    return step == 1 or step >= WARMUP_N


def puma_dir(model: str, dataset: str, seed: int) -> Path:
    if model == "r1_7b" and dataset == "math-500" and seed == 42:
        return AE / "results/math500_official/puma_ds7b"
    if seed == 42:
        return AE / f"results/puma_offline_{model}/{dataset}"
    return AE / f"results/puma_offline_{model}_s{seed}/{dataset}"


def puma_stat_path(model: str, dataset: str, seed: int) -> Path:
    return puma_dir(model, dataset, seed) / "statistics.json"


def cell_out_dir(model: str, dataset: str, seed: int) -> Path:
    return AE / f"results/default_dense_gate/{model}/{dataset}/s{seed}"


def regen_stat_path(model: str, dataset: str, seed: int) -> Path:
    return cell_out_dir(model, dataset, seed) / "statistics.json"


def trial_path(model: str, dataset: str, seed: int) -> Path:
    seeded = AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/dense_puma/trial_answers.json"
    flat = AE / f"results/dense_G_{model}/{dataset}/dense_puma/trial_answers.json"
    if seeded.is_file():
        dense = seeded
    elif flat.is_file():
        dense = flat
    elif dataset in ("aime24", "aime25", "amc23", "gsm8k") or seed != 42:
        dense = seeded
    else:
        dense = flat
    if dense.is_file():
        return dense
    if seed == 42:
        return AE / f"results/puma_offline_{model}/{dataset}/trial_answers.json"
    return AE / f"results/puma_offline_{model}_s{seed}/{dataset}/trial_answers.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def step_char_lens(trials: list[dict[str, Any]]) -> dict[int, int]:
    prev = ""
    out: dict[int, int] = {}
    for row in trials:
        step = int(row["stopped_len"])
        prefix = str(row.get("reasoning_prefix") or "")
        if prev and prefix.startswith(prev):
            extra = prefix[len(prev) :]
            if extra.startswith("\n\n"):
                extra = extra[2:]
            out[step] = len(extra)
        else:
            out[step] = len(prefix) if not prev else max(1, abs(len(prefix) - len(prev)))
        prev = prefix
    return out


def is_red(step: int, lens: dict[int, int]) -> bool:
    if step < 2 or step not in lens or (step - 1) not in lens:
        return False
    cur, prev = lens[step], lens[step - 1]
    if prev <= 0:
        return False
    ratio = cur / prev
    return (LEN_LO <= ratio <= LEN_HI) or abs(cur - prev) <= ABS_DELTA


def pick_peak_same(probed: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not probed:
        return None
    answer = probed[-1]["answer"]
    same_ans = [x for x in probed if x["answer"] == answer]
    if not same_ans:
        return None
    return max(same_ans, key=lambda x: x["conf"])


def trial_answer_tokens(row: dict[str, Any]) -> int:
    value = row.get("count_answer_tokens")
    if value in (None, ""):
        value = row.get("answer_tokens")
    if value in (None, ""):
        return 0
    return int(value)


def reasoning_tokens(row: dict[str, Any], *, n_steps: int, original_tokens: int) -> int:
    value = row.get("count_reasoning_tokens")
    if value not in (None, ""):
        return int(value)
    step = int(row["stopped_len"])
    if n_steps <= 0 or original_tokens <= 0:
        return 0
    return max(1, int(round(original_tokens * step / n_steps)))


def simulate(
    trials: list[dict[str, Any]], *, original_tokens: int
) -> dict[str, Any]:
    trials = sorted(trials, key=lambda x: int(x["stopped_len"]))
    lens = step_char_lens(trials)
    probed: list[dict[str, Any]] = []
    red_run = 0
    best_conf = float("-inf")
    last = trials[-1]
    n_steps = int(last["stopped_len"])

    def probed_upto(decision_step: int) -> list[dict[str, Any]]:
        out = []
        for x in trials:
            step = int(x["stopped_len"])
            if not should_probe(step) or step > decision_step:
                continue
            if not str(x.get("final_answer") or "") or not math.isfinite(finite(x.get("confidence"))):
                continue
            out.append(x)
        return out

    def pack(obs: dict[str, Any], branch: str, decision_step: int) -> dict[str, Any]:
        decision = next(x for x in trials if int(x["stopped_len"]) == decision_step)
        used = probed_upto(decision_step)
        trial_tok = sum(trial_answer_tokens(x) for x in used)
        if branch == "full":
            reason_tok = original_tokens or reasoning_tokens(
                last, n_steps=n_steps, original_tokens=original_tokens
            )
        else:
            reason_tok = reasoning_tokens(
                decision, n_steps=n_steps, original_tokens=original_tokens
            )
        return {
            "branch": branch,
            "step": decision_step,
            "answer": obs["answer"],
            "conf": obs["conf"],
            "tokens": reason_tok + trial_tok,
            "tokens_trial_answers": trial_tok,
            "generated_trial_answers": len(used),
            "original_len": n_steps,
            "early": branch != "full",
        }

    for row in trials:
        step = int(row["stopped_len"])
        conf = finite(row.get("confidence"))
        answer = str(row.get("final_answer") or "")
        usable = bool(answer) and math.isfinite(conf)
        if step >= FS_MIN_STEP:
            red_run = red_run + 1 if is_red(step, lens) else 0
        if usable:
            best_conf = max(best_conf, conf)
        if should_probe(step) and usable:
            probed.append(
                {
                    "step": step,
                    "answer": answer,
                    "conf": conf,
                    "answer_tokens": trial_answer_tokens(row),
                    "reasoning_tokens": reasoning_tokens(
                        row, n_steps=n_steps, original_tokens=original_tokens
                    ),
                }
            )
            if step >= MSS and len(probed) >= CONSEC:
                window = probed[-CONSEC:]
                steps = [x["step"] for x in window]
                if steps == list(range(steps[0], steps[0] + CONSEC)):
                    first = window[0]
                    if first["conf"] >= TAU and all(
                        x["answer"] == first["answer"] and x["conf"] >= first["conf"] - EPS
                        for x in window
                    ):
                        return pack(window[-1], "consec", step)
            if (
                step >= FS_MIN_STEP
                and red_run >= FS_RUN
                and best_conf >= FS_BEST
                and len(probed) >= FS_SAME_K
                and all(x["answer"] == probed[-1]["answer"] for x in probed[-FS_SAME_K:])
            ):
                pick = pick_peak_same(probed)
                if pick is not None and pick["conf"] >= FS_DEC:
                    return pack(pick, "fs", step)

    used = probed_upto(int(last["stopped_len"]))
    trial_tok = sum(trial_answer_tokens(x) for x in used)
    return {
        "branch": "full",
        "step": int(last["stopped_len"]),
        "answer": str(last.get("final_answer") or ""),
        "conf": finite(last.get("confidence")),
        "tokens": (original_tokens or reasoning_tokens(
            last, n_steps=n_steps, original_tokens=original_tokens
        ))
        + trial_tok,
        "tokens_trial_answers": trial_tok,
        "generated_trial_answers": len(used),
        "original_len": n_steps,
        "early": False,
    }


def export_candidates(model: str, dataset: str, seed: int) -> dict[str, Any]:
    stat_path = puma_stat_path(model, dataset, seed)
    trials_path = trial_path(model, dataset, seed)
    if not stat_path.is_file() or not trials_path.is_file():
        return {"model": model, "dataset": dataset, "seed": seed, "missing": True}
    questions_path = puma_dir(model, dataset, seed) / "filtered_steps.json"
    if not questions_path.is_file():
        return {"model": model, "dataset": dataset, "seed": seed, "missing": True}
    official = load_json(stat_path)
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in load_json(trials_path):
        by_q[int(row["question_idx"])].append(row)
    official_by_q = {int(r["question_idx"]): r for r in official}
    questions = load_json(questions_path)
    cands: list[dict[str, Any]] = []
    n_mismatch = 0
    n_consec = n_fs = n_full = 0
    for qi, info in sorted(official_by_q.items()):
        trials = by_q.get(qi)
        if not trials:
            continue
        sim = simulate(trials, original_tokens=int(info.get("original_tokens") or 0))
        question = str(info.get("question") or trials[0].get("question") or "")
        if qi - 1 >= len(questions) or questions[qi - 1].get("question") != question:
            n_mismatch += 1
        reason = "full_reasoning" if sim["branch"] == "full" else sim["branch"]
        n_consec += int(sim["branch"] == "consec")
        n_fs += int(sim["branch"] == "fs")
        n_full += int(sim["branch"] == "full")
        conf = sim["conf"]
        cands.append(
            {
                "question_idx": qi,
                "stopped_len": sim["step"],
                "original_len_reasoning_steps": sim["original_len"],
                "question": question,
                "skipped": False,
                "skip_reason": None,
                "similarity": None,
                "success": True,
                "generated_trial_answers": sim["generated_trial_answers"],
                "tokens_trial_answers": sim["tokens_trial_answers"],
                "tokens_trial_answers_online": sim["tokens_trial_answers"],
                "stop_reason": reason,
                "stop_confidence": None if not math.isfinite(conf) else conf,
                "stop_threshold": TAU if reason == "consec" else (FS_DEC if reason == "fs" else None),
                "consecutive_confidences": None,
                "confidence_trajectory": None,
                "step_similarities": None,
                "final_answer": sim["answer"],
            }
        )
    if n_mismatch:
        raise RuntimeError(
            f"{model} {dataset} s{seed}: {n_mismatch} candidates do not match "
            "filtered_steps[question_idx-1]; refuse to export"
        )
    out_dir = cell_out_dir(model, dataset, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "final_candidates.json"
    path.write_text(json.dumps(cands, indent=2, ensure_ascii=False) + "\n")
    meta = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "n": len(cands),
        "n_consec": n_consec,
        "n_fs": n_fs,
        "n_full": n_full,
        "n_early": n_consec + n_fs,
        "trials_path": str(trials_path),
        "questions_file": str(puma_dir(model, dataset, seed) / "filtered_steps.json"),
        "answers_file": str(puma_dir(model, dataset, seed) / "answers.json"),
    }
    (out_dir / "candidates_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"export {model:16} {dataset:14} s{seed:<4} "
        f"n={len(cands)} early={n_consec + n_fs} full={n_full} → {path}"
    )
    return meta


def evaluate_cell(model: str, dataset: str, seed: int) -> dict[str, Any]:
    stat_path = puma_stat_path(model, dataset, seed)
    trials_path = trial_path(model, dataset, seed)
    if not stat_path.is_file() or not trials_path.is_file():
        return {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "missing": True,
            "stat_path": str(stat_path),
            "trials_path": str(trials_path),
        }
    official = load_json(stat_path)
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in load_json(trials_path):
        by_q[int(row["question_idx"])].append(row)
    official_by_q = {int(r["question_idx"]): r for r in official}

    n = 0
    puma_ok = puma_tok = full_ok = full_tok = 0.0
    trial_ok = 0.0
    n_consec = n_fs = n_full = 0
    for qi, info in official_by_q.items():
        trials = by_q.get(qi)
        if not trials:
            continue
        n += 1
        puma_ok += int(bool(info.get("compressed_correct")))
        puma_tok += int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)
        full_ok += int(bool(info.get("original_correct")))
        full_tok += int(info.get("original_tokens") or 0)
        orig_ok = bool(info.get("original_correct"))
        orig_tok = int(info.get("original_tokens") or 0)
        sim = simulate(trials, original_tokens=orig_tok)
        gt = info.get("ground_truth")
        a_final = info.get("original_answer")
        trial_hit = same(sim["answer"], gt)
        trial_ok += int(trial_hit if sim["early"] else orig_ok)
        n_consec += int(sim["branch"] == "consec")
        n_fs += int(sim["branch"] == "fs")
        n_full += int(sim["branch"] == "full")
    if n == 0:
        return {"model": model, "dataset": dataset, "seed": seed, "missing": True, "empty": True}

    regen_path = regen_stat_path(model, dataset, seed)
    ours_acc = ours_tok = d_acc_pp = d_tok = None
    source = "pending_regen"
    if regen_path.is_file():
        regen = load_json(regen_path)
        regen_by_q = {int(r["question_idx"]): r for r in regen}
        ours_ok = ours_tokens = 0.0
        n_regen = 0
        for qi, info in official_by_q.items():
            row = regen_by_q.get(qi)
            if not row:
                continue
            n_regen += 1
            ours_ok += int(bool(row.get("compressed_correct")))
            ours_tokens += int(row.get("compressed_tokens") or 0) + int(row.get("tokens_trial_answers") or 0)
        if n_regen == n:
            ours_acc = ours_ok / n
            ours_tok = ours_tokens / n
            d_acc_pp = 100.0 * (ours_acc - puma_ok / n)
            d_tok = ours_tok - puma_tok / n
            source = "regen"
    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "missing": False,
        "n": n,
        "source": source,
        "ours_acc": ours_acc,
        "ours_tok": ours_tok,
        "trial_acc": trial_ok / n,
        "puma_acc": puma_ok / n,
        "puma_tok": puma_tok / n,
        "full_acc": full_ok / n,
        "full_tok": full_tok / n,
        "d_acc_pp": d_acc_pp,
        "d_tok": d_tok,
        "d_trial_acc_pp": 100.0 * (trial_ok / n - puma_ok / n),
        "frac_consec": n_consec / n,
        "frac_fs": n_fs / n,
        "frac_full": n_full / n,
        "trials_path": str(trials_path),
        "regen_path": str(regen_path),
    }


def mean_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [c for c in cells if not c.get("missing")]
    if not ok:
        return {"missing": True}
    n = sum(c["n"] for c in ok)
    def wavg(key: str) -> float:
        return sum(c[key] * c["n"] for c in ok) / n
    ready = [c for c in ok if c.get("source") == "regen" and c.get("ours_acc") is not None]
    out = {
        "missing": False,
        "n": n,
        "n_runs": len(ok),
        "source": "regen" if len(ready) == len(ok) else "pending_regen",
        "trial_acc": wavg("trial_acc"),
        "puma_acc": wavg("puma_acc"),
        "puma_tok": wavg("puma_tok"),
        "full_acc": wavg("full_acc"),
        "full_tok": wavg("full_tok"),
        "d_trial_acc_pp": wavg("d_trial_acc_pp"),
        "frac_consec": wavg("frac_consec"),
        "frac_fs": wavg("frac_fs"),
        "frac_full": wavg("frac_full"),
        "ours_acc": None,
        "ours_tok": None,
        "d_acc_pp": None,
        "d_tok": None,
    }
    if ready and sum(c["n"] for c in ready) == n:
        rn = n
        def rw(key: str) -> float:
            return sum(c[key] * c["n"] for c in ready) / rn
        out["ours_acc"] = rw("ours_acc")
        out["ours_tok"] = rw("ours_tok")
        out["d_acc_pp"] = rw("d_acc_pp")
        out["d_tok"] = rw("d_tok")
    return out


def fmt_delta(acc_pp: float, tok: float) -> str:
    acc_s = f"{acc_pp:+.2f}"
    tok_s = f"{tok:+.0f}"
    return f"{acc_s} / {tok_s}"


def write_table(grid: dict[str, dict[str, dict[str, Any]]]) -> str:
    ds_order = ["math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25"]
    lines = [
        "# 密探k4 vs 官方 PUMA（最新复现）",
        "",
        "密探k4：第 1 步必探，2–6 步不探，从第 7 步起每步都试答。",
        "连答停：同一答案连续 4 次、置信度 ≥ 0.995（其余次相对第一次掉不超过 0.03）、步数 ≥ 10。",
        "后路停：步数 ≥ 80，相邻步字数比在 0.84–1.16 或相差 ≤ 35 连续出现 2 次，",
        "轨迹最高置信度 ≥ 0.85，最近 2 次试答相同，选同答里置信度最高的一次且该次 ≥ 0.7。",
        "交卷和官方 PUMA 一样：闸门只截断，再从半截思路重写终答，拿重写答案判分。",
        "Token = 截断前缀 + 重写终答 + 实际探过的试答。写完全程的题复用原终答。",
        "未重写的格子显示 —。",
        "",
        "轨迹：密探 `dense_G_{model}`。非 AIME 只有 seed 42；AIME 是 42 / 0 / 1 / 123 按题数加权。",
        "7B MATH 对照盘是 `math500_official/puma_ds7b`，其余是 `puma_offline_{model}`。",
        "格子是相对官方 PUMA：正确率百分点差 / 平均 token 差。正确率差为正更好，token 差为负更省。",
        "",
        "| model | math | oly | gpqa | aime24 | aime25 |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        cells = []
        for ds in ds_order:
            cell = grid[model].get(ds)
            if not cell or cell.get("missing") or cell.get("source") != "regen":
                cells.append("—")
            else:
                cells.append(fmt_delta(cell["d_acc_pp"], cell["d_tok"]))
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += ["", "## 绝对数", ""]
    lines.append("| model | set | n | 密探k4 Acc | 密探k4 Tok | PUMA Acc | PUMA Tok | 写完全程 Acc | 连答停 | 后路停 | 写完 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        for ds in ds_order:
            cell = grid[model].get(ds)
            if not cell or cell.get("missing"):
                continue
            if cell.get("source") != "regen":
                oa = ot = "—"
            else:
                oa = f"{cell['ours_acc']:.1%}"
                ot = f"{cell['ours_tok']:.0f}"
            lines.append(
                "| {model} | {ds} | {n} | {oa} | {ot} | {pa:.1%} | {pt:.0f} | {fa:.1%} | {c:.1%} | {f:.1%} | {u:.1%} |".format(
                    model=model,
                    ds=SHORT_DS[ds],
                    n=cell["n"],
                    oa=oa,
                    ot=ot,
                    pa=cell["puma_acc"],
                    pt=cell["puma_tok"],
                    fa=cell["full_acc"],
                    c=cell["frac_consec"],
                    f=cell["frac_fs"],
                    u=cell["frac_full"],
                )
            )
    lines += [
        "",
        "## 记录",
        "",
        "8/20 对过轨迹：不是 seed 用错。旧表 Δtok 更接近「我方只算停点前缀 + 试答、对照盘把重写终答也算进去」。",
        "新表两边都算重写，所以 token 差会比旧表贵一截（例如 7B MATH 从大约 −300 变成 −44）。",
        "重写入口：`scripts/run_default_regen.sh`。重放入口：`scripts/replay_default_dense_gate.py`。",
        "",
    ]
    text = "\n".join(lines) + "\n"
    TABLE.write_text(text)
    return text


def main() -> None:
    raise SystemExit(
        "abandoned diagnostic: do not regenerate dense-gate tables. "
        "Canonical comparison is scripts/report_fullcot_puma_plws.py"
    )
    raw: list[dict[str, Any]] = []
    grid: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in MODELS}
    for model in MODELS:
        for dataset, short in SHORT_DS.items():
            seeds = AIME_SEEDS if dataset.startswith("aime") else (42,)
            for seed in seeds:
                try:
                    export_candidates(model, dataset, seed)
                except FileNotFoundError:
                    pass
            cells = [evaluate_cell(model, dataset, seed) for seed in seeds]
            raw.extend(cells)
            grid[model][dataset] = mean_cells(cells)
            cell = grid[model][dataset]
            if cell.get("missing"):
                print(f"{model:16} {short:8} MISSING")
            elif cell.get("source") != "regen":
                print(
                    f"{model:16} {short:8} pending_regen"
                    f"  consec={cell['frac_consec']:.0%} fs={cell['frac_fs']:.0%} full={cell['frac_full']:.0%}"
                )
            else:
                print(
                    f"{model:16} {short:8} {fmt_delta(cell['d_acc_pp'], cell['d_tok'])}"
                    f"  acc={cell['ours_acc']:.1%} tok={cell['ours_tok']:.0f}"
                    f"  consec={cell['frac_consec']:.0%} fs={cell['frac_fs']:.0%} full={cell['frac_full']:.0%}"
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cells": raw, "grid": grid}, indent=2, ensure_ascii=False))
    write_table(grid)
    print(f"\nwrote {TABLE}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
