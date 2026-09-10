#!/usr/bin/env python3
"""Offline rescue-R gate on frozen 7B dense trials.

High-confidence consecutive lock (k=4, first ≥ 0.995, later ≥ first − 0.03).
Additionally stop on a low-confidence consecutive lock when an internal
score clears a threshold. Delivery = the trial boxed answer (no regen).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
TABLE = AE / "tables/rescue_R_vs_puma.md"
OUT = AE / "results/rescue_R_gate/summary.json"
REGEN_ROOT = AE / "results/rescue_R_gate"

# Frozen on the trial-as-final sweep; do not retune after switching to PUMA regen.
FROZEN = {
    "math-500": ("late_rise", 8.984),
    "gpqa-diamond": ("lens_rise", 5.008),
}

K = 4
TAU = 0.995
EPS = 0.03  # 和 PUMA 一样：第一次 ≥ TAU，后面几次 ≥ 第一次 − EPS
MSS = 10  # 前这么多步不许提前停，和官方 PUMA 的 min_stop_step 对齐
# PUMA 底附加门默认（8/22 锁）：低置信窗最后一步，(末层好写−一半深)/|末层| ≥ 此数
REL_RISE_THR = 2.848
USE_FS = True  # 默认加上后路强停
FS_MIN_STEP = 80
FS_RUN = 2
FS_BEST = 0.85
FS_DEC = 0.7
FS_SAME_K = 2
_WS = re.compile(r"\s+")
_LATEX = re.compile(r"\\(left|right|cdot|times|dfrac|tfrac|frac|mathrm|text)")

SIGNALS = (
    ("late_rise", "第 20 层到第 27 层，试答词抬了多少"),
    ("dola_logp_l27", "第 27 层对试答第一个词有多像"),
    ("neg_ans_entropy", "试答用词有多集中"),
    ("lens_rise", "最后一层比第 14 层，试答好写多少"),
    ("stop_margin", "写完试答后，收口比继续 Wait 高多少"),
    ("dola_mean_rise", "浅层到深层，这段试答越来越好写的幅度"),
    ("reasoning_pmi", "有草稿比只看题，试答多顺多少"),
    ("margin", "当前试答比历史上别的试答高多少"),
)

CELLS = (
    {
        "name": "MATH",
        "dataset": "math-500",
        "trial": AE / "results/dense_G_r1_7b/math-500/dense_puma/trial_answers.json",
        "stat": AE / "results/math500_official/puma_ds7b/statistics.json",
        "gpath": AE / "results/dense_G_r1_7b/math-500/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/math-500",
            AE / "results/confcal_judge/v2/dense_lens/math-500",
            AE / "results/confcal_judge/v2/dense_solver_probes/math-500",
        ),
        "preferred": "late_rise",
    },
    {
        "name": "GPQA",
        "dataset": "gpqa-diamond",
        "trial": AE / "results/dense_G_r1_7b/gpqa-diamond/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_r1_7b/gpqa-diamond/statistics.json",
        "gpath": AE / "results/dense_G_r1_7b/gpqa-diamond/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/gpqa-diamond",
            AE / "results/confcal_judge/v2/dense_lens/gpqa-diamond",
            AE / "results/confcal_judge/v2/dense_solver_probes/gpqa-diamond",
        ),
        "preferred": "lens_rise",
    },
    {
        "name": "奥赛",
        "dataset": "olympiadbench",
        "trial": AE / "results/dense_G_r1_7b/olympiadbench/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_r1_7b/olympiadbench/statistics.json",
        "gpath": AE / "results/dense_G_r1_7b/olympiadbench/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/olympiadbench",
            AE / "results/confcal_judge/v2/dense_lens/olympiadbench",
            AE / "results/confcal_judge/v2/dense_solver_probes/olympiadbench",
        ),
        "preferred": "late_rise",
    },
)

CONF_BINS = (
    (0.0, 0.50, "0–0.5"),
    (0.50, 0.80, "0.5–0.8"),
    (0.80, 0.90, "0.8–0.9"),
    (0.90, TAU, "0.9–τ"),
)


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


def p50(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[len(xs) // 2]


def auroc(pos: list[float], neg: list[float]) -> float:
    pos = [x for x in pos if x == x]
    neg = [x for x in neg if x == x]
    if not pos or not neg:
        return float("nan")
    better = tie = 0
    for a in pos:
        for b in neg:
            if a > b:
                better += 1
            elif a == b:
                tie += 1
    return (better + 0.5 * tie) / (len(pos) * len(neg))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_scores(folders: tuple[Path, ...]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("scores_shard*.jsonl")):
            for line in path.open():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") and row.get("status") != "ok":
                    continue
                key = (int(row["question_idx"]), int(row.get("decision_step") or row.get("stopped_len") or 0))
                out.setdefault(key, {}).update(row)
    for row in out.values():
        a = finite(row.get("dola_logp_l27"))
        b = finite(row.get("dola_logp_l20"))
        if math.isfinite(a) and math.isfinite(b):
            row["late_rise"] = a - b
    return out


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


def usable_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(trials, key=lambda x: int(x["stopped_len"])):
        answer = str(row.get("final_answer") or "")
        conf = finite(row.get("confidence"))
        if not answer or not math.isfinite(conf):
            continue
        rows.append(row)
    return rows


def window_ok(rows: list[dict[str, Any]], end: int) -> bool:
    if end + 1 < K:
        return False
    window = rows[end + 1 - K : end + 1]
    steps = [int(x["stopped_len"]) for x in window]
    if steps != list(range(steps[0], steps[0] + K)):
        return False
    first = str(window[0].get("final_answer") or "")
    return bool(first) and all(same(first, x.get("final_answer")) for x in window)


def confs_of(window: list[dict[str, Any]]) -> list[float]:
    return [finite(x.get("confidence")) for x in window]


def is_high(confs: list[float]) -> bool:
    """PUMA-style: first ≥ TAU, later ≥ first − EPS."""
    if len(confs) < K or not all(math.isfinite(c) for c in confs):
        return False
    first = confs[0]
    if first < TAU:
        return False
    return all(c >= first - EPS for c in confs[1:])


def is_low(confs: list[float]) -> bool:
    return bool(confs) and all(math.isfinite(c) and c < TAU for c in confs)


def can_stop_step(row: dict[str, Any]) -> bool:
    return int(row["stopped_len"]) >= MSS


def label_windows(
    rows: list[dict[str, Any]], *, a_final: Any, gt: Any
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    labels = ["O"] * len(rows)
    first: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_tag: set[str] = set()
    for end, row in enumerate(rows):
        if not window_ok(rows, end):
            continue
        window = rows[end + 1 - K : end + 1]
        confs = confs_of(window)
        if not all(math.isfinite(c) for c in confs):
            continue
        oks = [bool(same(x.get("final_answer"), a_final) or same(x.get("final_answer"), gt)) for x in window]
        high = is_high(confs)
        low = is_low(confs)
        if all(oks) and high:
            tag = "G"
        elif all(oks) and low:
            tag = "R"
        elif (not any(oks)) and high:
            tag = "W"
        elif (not any(oks)) and low:
            tag = "L"
        else:
            continue
        for index in range(end + 1 - K, end + 1):
            if tag == "G":
                labels[index] = "G"
            elif tag == "W" and labels[index] != "G":
                labels[index] = "W"
            elif tag == "R" and labels[index] not in ("G", "W"):
                labels[index] = "R"
            elif tag == "L" and labels[index] == "O":
                labels[index] = "L"
        if tag not in seen_tag:
            first[tag].append(
                {
                    "st": int(row["stopped_len"]),
                    "conf": confs[-1],
                    "min_conf": min(confs),
                    "max_conf": max(confs),
                }
            )
            seen_tag.add(tag)
    return labels, first


def pick_peak_same(probed: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not probed:
        return None
    last = probed[-1]["answer"]
    same_ans = [x for x in probed if same(x["answer"], last)]
    if not same_ans:
        return None
    return max(same_ans, key=lambda x: x["conf"])


def pack_fs(
    trials: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    end: int,
    pick: dict[str, Any],
    original_tokens: int,
    label: str = "O",
) -> dict[str, Any]:
    rec = pack(trials, rows, end, "fs", original_tokens=original_tokens, label=label)
    rec["answer"] = pick["answer"]
    rec["conf"] = pick["conf"]
    rec["fs_pick_step"] = pick["step"]
    return rec


def fs_ready(step: int, red_run: int, best_conf: float, probed: list[dict[str, Any]]) -> dict[str, Any] | None:
    if (
        step < FS_MIN_STEP
        or red_run < FS_RUN
        or best_conf < FS_BEST
        or len(probed) < FS_SAME_K
        or not all(same(x["answer"], probed[-1]["answer"]) for x in probed[-FS_SAME_K:])
    ):
        return None
    pick = pick_peak_same(probed)
    if pick is None or pick["conf"] < FS_DEC:
        return None
    return pick


def pack(
    trials: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    end: int,
    branch: str,
    *,
    original_tokens: int,
    score: float = float("nan"),
    label: str = "O",
) -> dict[str, Any]:
    n_steps = int(trials[-1]["stopped_len"])
    if not rows:
        return {
            "branch": "full",
            "step": n_steps,
            "answer": str(trials[-1].get("final_answer") or ""),
            "conf": finite(trials[-1].get("confidence")),
            "score": float("nan"),
            "label": "O",
            "tokens": original_tokens,
            "tokens_trial_answers": 0,
            "generated_trial_answers": 0,
            "original_len": n_steps,
            "early": False,
        }
    decision = rows[end]
    used = [x for x in rows[: end + 1]]
    trial_tok = sum(trial_answer_tokens(x) for x in used)
    if branch == "full":
        reason_tok = original_tokens or reasoning_tokens(
            trials[-1], n_steps=n_steps, original_tokens=original_tokens
        )
        answer = str(trials[-1].get("final_answer") or "")
        step = n_steps
        conf = finite(trials[-1].get("confidence"))
    else:
        reason_tok = reasoning_tokens(decision, n_steps=n_steps, original_tokens=original_tokens)
        answer = str(decision.get("final_answer") or "")
        step = int(decision["stopped_len"])
        conf = finite(decision.get("confidence"))
    return {
        "branch": branch,
        "step": step,
        "answer": answer,
        "conf": conf,
        "score": score,
        "label": label,
        "tokens": reason_tok + trial_tok,
        "tokens_trial_answers": trial_tok,
        "generated_trial_answers": len(used),
        "original_len": n_steps,
        "early": branch != "full",
    }


def simulate(
    trials: list[dict[str, Any]],
    scores: dict[tuple[int, int], dict[str, Any]],
    *,
    qi: int,
    a_final: Any,
    gt: Any,
    original_tokens: int,
    signal: str | None,
    threshold: float,
    rescue_all_low: bool = False,
    oracle_r: bool = False,
    window_all: bool = True,
    use_fs: bool = USE_FS,
) -> dict[str, Any]:
    import replay_default_dense_gate as old

    trials = sorted(trials, key=lambda x: int(x["stopped_len"]))
    rows = usable_rows(trials)
    labels, _ = label_windows(rows, a_final=a_final, gt=gt)
    lens = old.step_char_lens(trials) if use_fs else {}
    red_run = 0
    best_conf = float("-inf")
    probed: list[dict[str, Any]] = []
    walked = 0
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if use_fs and tstep >= FS_MIN_STEP:
                red_run = red_run + 1 if old.is_red(tstep, lens) else 0
            walked += 1
        conf = finite(row.get("confidence"))
        answer = str(row.get("final_answer") or "")
        if answer and conf == conf:
            best_conf = max(best_conf, conf)
            probed.append({"step": step, "answer": answer, "conf": conf})
        if can_stop_step(row) and window_ok(rows, end):
            window = rows[end + 1 - K : end + 1]
            confs = confs_of(window)
            tag = labels[end]
            if is_high(confs):
                return pack(trials, rows, end, "conf", original_tokens=original_tokens, label=tag)
            if is_low(confs):
                if oracle_r:
                    if tag == "R":
                        return pack(trials, rows, end, "oracle_r", original_tokens=original_tokens, label=tag)
                elif rescue_all_low:
                    return pack(trials, rows, end, "low_all", original_tokens=original_tokens, label=tag)
                elif signal:
                    vals = [finite(scores.get((qi, int(x["stopped_len"])), {}).get(signal)) for x in window]
                    if window_all:
                        if vals and all(math.isfinite(v) and v >= threshold for v in vals):
                            return pack(
                                trials, rows, end, "rescue", original_tokens=original_tokens, score=min(vals), label=tag
                            )
                    else:
                        score = vals[-1] if vals else float("nan")
                        if math.isfinite(score) and score >= threshold:
                            return pack(
                                trials, rows, end, "rescue", original_tokens=original_tokens, score=score, label=tag
                            )
        if use_fs:
            pick = fs_ready(step, red_run, best_conf, probed)
            if pick is not None:
                tag = labels[end] if end < len(labels) else "O"
                return pack_fs(trials, rows, end, pick, original_tokens, label=tag)
    return pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=original_tokens)


def hit(answer: Any, gt: Any, a_final: Any, orig_ok: bool) -> bool:
    return same(answer, gt)


def quantiles(xs: list[float], n: int = 17) -> list[float]:
    xs = sorted(x for x in xs if x == x)
    if len(xs) < 4:
        return []
    out = []
    for i in range(n):
        q = i / (n - 1)
        out.append(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))])
    uniq: list[float] = []
    for value in out:
        if not uniq or abs(value - uniq[-1]) > 1e-9:
            uniq.append(value)
    return uniq


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(int(r["ok"]) for r in rows) / n
    tok = sum(r["tokens"] for r in rows) / n
    n_rescue = sum(int(r["branch"] == "rescue" or r["branch"] == "low_all" or r["branch"] == "oracle_r") for r in rows)
    n_conf = sum(int(r["branch"] == "conf") for r in rows)
    n_full = sum(int(r["branch"] == "full") for r in rows)
    n_fs = sum(int(r["branch"] == "fs") for r in rows)
    rescued = [r for r in rows if r["branch"] in ("rescue", "low_all", "oracle_r")]
    n_r = sum(int(r["label"] == "R") for r in rescued)
    n_l = sum(int(r["label"] == "L") for r in rescued)
    return {
        "n": n,
        "acc": acc,
        "tok": tok,
        "n_conf": n_conf,
        "n_rescue": n_rescue,
        "n_full": n_full,
        "n_fs": n_fs,
        "frac_fs": n_fs / n,
        "rescue_R": n_r,
        "rescue_L": n_l,
        "frac_conf": n_conf / n,
        "frac_rescue": n_rescue / n,
        "frac_full": n_full / n,
    }


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%" if value == value else "—"


def fmt_pp(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:+.1f}pp"


def fmt_tok(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:+.0f}"


def puma_bundle(cell: dict[str, Any]) -> dict[str, Path]:
    stat = Path(cell["stat"])
    return {
        "questions": stat.parent / "filtered_steps.json",
        "answers": stat.parent / "answers.json",
        "stat": stat,
    }


def policy_dir(dataset: str, policy: str) -> Path:
    return REGEN_ROOT / dataset / policy


def export_candidates(
    cell: dict[str, Any],
    rows: list[dict[str, Any]],
    policy: str,
) -> Path:
    bundle = puma_bundle(cell)
    questions = load_json(bundle["questions"])
    official = {int(r["question_idx"]): r for r in load_json(bundle["stat"])}
    cands = []
    n_mismatch = 0
    n_early = n_full = 0
    by_qi = {int(r["qi"]): r for r in rows}
    for qi, info in sorted(official.items()):
        sim = by_qi.get(qi)
        if not sim:
            continue
        question = str(info.get("question") or "")
        if qi - 1 >= len(questions) or questions[qi - 1].get("question") != question:
            n_mismatch += 1
        reason = "full_reasoning" if sim["branch"] == "full" else sim["branch"]
        n_full += int(reason == "full_reasoning")
        n_early += int(reason != "full_reasoning")
        conf = sim["conf"]
        cands.append(
            {
                "question_idx": qi,
                "stopped_len": sim["step"],
                "original_len_reasoning_steps": sim.get("original_len")
                or official[qi].get("original_len_reasoning_steps"),
                "question": question,
                "skipped": False,
                "skip_reason": None,
                "similarity": None,
                "success": True,
                "generated_trial_answers": sim.get("generated_trial_answers") or 0,
                "tokens_trial_answers": sim.get("tokens_trial_answers") or 0,
                "tokens_trial_answers_online": sim.get("tokens_trial_answers") or 0,
                "stop_reason": reason,
                "stop_confidence": None if not math.isfinite(conf) else conf,
                "stop_threshold": TAU if reason == "conf" else None,
                "consecutive_confidences": None,
                "confidence_trajectory": None,
                "step_similarities": None,
                "final_answer": sim["answer"],
            }
        )
    if n_mismatch:
        raise RuntimeError(
            f"{cell['dataset']} {policy}: {n_mismatch} candidates do not match "
            "filtered_steps[question_idx-1]; refuse to export"
        )
    out_dir = policy_dir(cell["dataset"], policy)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "final_candidates.json"
    path.write_text(json.dumps(cands, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "candidates_meta.json").write_text(
        json.dumps(
            {
                "dataset": cell["dataset"],
                "policy": policy,
                "n": len(cands),
                "n_early": n_early,
                "n_full": n_full,
                "questions_file": str(bundle["questions"]),
                "answers_file": str(bundle["answers"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"export {cell['name']:6} {policy:8} n={len(cands)} "
        f"early={n_early} full={n_full} → {path}",
        flush=True,
    )
    return path


def apply_regen(rows: list[dict[str, Any]], stat_path: Path) -> bool:
    if not stat_path.is_file():
        return False
    regen = {int(r["question_idx"]): r for r in load_json(stat_path)}
    n = 0
    for row in rows:
        info = regen.get(int(row["qi"]))
        if not info:
            continue
        row["ok"] = bool(info.get("compressed_correct"))
        row["tokens"] = int(info.get("compressed_tokens") or 0) + int(
            info.get("tokens_trial_answers") or 0
        )
        n += 1
    return n == len(rows)


def evaluate_cell(cell: dict[str, Any]) -> dict[str, Any]:
    trials_all = load_json(cell["trial"])
    official = {int(r["question_idx"]): r for r in load_json(cell["stat"])}
    gmap = {int(r["question_idx"]): r for r in load_json(cell["gpath"])}
    scores = load_scores(cell["scores"])
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trials_all:
        by[int(row["question_idx"])].append(row)

    questions: list[dict[str, Any]] = []
    first_low: list[dict[str, Any]] = []
    g_conf: list[float] = []
    not_g_conf: list[float] = []
    n_r_only = n_r = n_has_g = 0
    for qi, trials in by.items():
        info = official.get(qi)
        gg = gmap.get(qi)
        if not info or not gg:
            continue
        a_final = gg.get("A_final") or info.get("original_answer")
        gt = info.get("ground_truth")
        orig_ok = bool(info.get("original_correct"))
        orig_tok = int(info.get("original_tokens") or 0)
        rows = usable_rows(trials)
        labels, first = label_windows(rows, a_final=a_final, gt=gt)
        if "R" in labels:
            n_r += 1
            if "G" not in labels:
                n_r_only += 1
        if "G" in labels:
            n_has_g += 1
        for row, lab in zip(rows, labels):
            conf = finite(row.get("confidence"))
            if lab == "G":
                g_conf.append(conf)
            else:
                not_g_conf.append(conf)
        feat: dict[str, Any] = {}
        if first["R"] or first["L"]:
            src = first["R"][0] if first["R"] and (
                not first["L"] or first["R"][0]["st"] <= first["L"][0]["st"]
            ) else first["L"][0]
            inn = scores.get((qi, int(src["st"])), {})
            feat = {
                "tag": "R" if first["R"] and src is first["R"][0] else "L",
                "st": src["st"],
                "conf": src["conf"],
                **{name: finite(inn.get(name)) for name, _ in SIGNALS},
            }
            first_low.append(feat)
        questions.append(
            {
                "qi": qi,
                "trials": trials,
                "rows": rows,
                "labels": labels,
                "first": first,
                "a_final": a_final,
                "gt": gt,
                "orig_ok": orig_ok,
                "orig_tok": orig_tok,
                "puma_ok": bool(info.get("compressed_correct")),
                "puma_tok": int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0),
                "full_ok": orig_ok,
                "full_tok": orig_tok,
                "first_low": feat,
            }
        )

    def run(signal: str | None, threshold: float, **kwargs: Any) -> list[dict[str, Any]]:
        out = []
        for q in questions:
            sim = simulate(
                q["trials"],
                scores,
                qi=q["qi"],
                a_final=q["a_final"],
                gt=q["gt"],
                original_tokens=q["orig_tok"],
                signal=signal,
                threshold=threshold,
                **kwargs,
            )
            out.append(
                {
                    **sim,
                    "ok": hit(sim["answer"], q["gt"], q["a_final"], q["orig_ok"])
                    if sim["early"]
                    else q["orig_ok"],
                    "puma_ok": q["puma_ok"],
                    "puma_tok": q["puma_tok"],
                    "full_ok": q["full_ok"],
                    "full_tok": q["full_tok"],
                    "has_r": "R" in q["labels"],
                    "has_g": "G" in q["labels"],
                    "r_only": "R" in q["labels"] and "G" not in q["labels"],
                    "qi": q["qi"],
                }
            )
        return out

    conf_rows = run(None, float("inf"))
    low_all_rows = run(None, float("inf"), rescue_all_low=True)
    oracle_rows = run(None, float("inf"), oracle_r=True)
    conf_sum = summarize(conf_rows)
    puma_acc = sum(int(q["puma_ok"]) for q in questions) / max(len(questions), 1)
    puma_tok = sum(q["puma_tok"] for q in questions) / max(len(questions), 1)
    full_acc = sum(int(q["full_ok"]) for q in questions) / max(len(questions), 1)
    full_tok = sum(q["full_tok"] for q in questions) / max(len(questions), 1)

    def contrast(ours: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(ours)
        recovered = sum(
            int((not c["ok"]) and o["ok"] and o["branch"] in ("rescue", "low_all", "oracle_r"))
            for c, o in zip(conf_rows, ours)
        )
        damaged = sum(int(c["ok"] and not o["ok"]) for c, o in zip(conf_rows, ours))
        r_only_hit = sum(int(o["r_only"] and o["ok"] and o["branch"] in ("rescue", "low_all", "oracle_r")) for o in ours)
        earlier_ok = sum(
            int(c["ok"] and o["ok"] and o["step"] < c["step"]) for c, o in zip(conf_rows, ours)
        )
        g_door_same = sum(
            int(
                c["branch"] == "conf"
                and (
                    (o["branch"] == "conf" and o["step"] == c["step"])
                    or (o["ok"] and o["step"] < c["step"])
                )
            )
            for c, o in zip(conf_rows, ours)
        )
        n_conf_base = sum(int(c["branch"] == "conf") for c in conf_rows)
        s = summarize(ours)
        s.update(
            {
                "recovered": recovered,
                "damaged": damaged,
                "r_only_hit": r_only_hit,
                "earlier_ok": earlier_ok,
                "g_door_kept": g_door_same / n_conf_base if n_conf_base else float("nan"),
                "d_acc_pp_puma": 100.0 * (s["acc"] - puma_acc),
                "d_tok_puma": s["tok"] - puma_tok,
                "d_acc_pp_conf": 100.0 * (s["acc"] - conf_sum["acc"]),
                "d_tok_conf": s["tok"] - conf_sum["tok"],
                "n": n,
            }
        )
        return s

    sweeps: dict[str, list[dict[str, Any]]] = {}
    chosen: dict[str, dict[str, Any]] = {}
    for name, _zh in SIGNALS:
        xs = [finite(r.get(name)) for r in first_low]
        if sum(x == x for x in xs) < 12:
            continue
        points = []
        for thr in quantiles(xs):
            rows = run(name, thr)
            item = contrast(rows)
            item["signal"] = name
            item["threshold"] = thr
            points.append(item)
        sweeps[name] = points
        keep = [p for p in points if p["acc"] + 1e-12 >= conf_sum["acc"] and p["damaged"] == 0]
        pool = keep or points
        pool = sorted(pool, key=lambda p: (-p["acc"], p["tok"], -p["recovered"]))
        chosen[name] = pool[0]

    bin_auc: dict[str, list[dict[str, Any]]] = {}
    for name, _zh in SIGNALS:
        rows = []
        for feat in first_low:
            score = finite(feat.get(name))
            if score != score:
                continue
            rows.append((feat["tag"], feat["conf"], score))
        if not rows:
            continue
        pack_bins = []
        pos = [s for tag, _, s in rows if tag == "R"]
        neg = [s for tag, _, s in rows if tag == "L"]
        pack_bins.append({"bin": "全部低置信度连答", "n_r": len(pos), "n_l": len(neg), "auroc": auroc(pos, neg)})
        for lo, hi, lab in CONF_BINS:
            pos_b = [s for tag, c, s in rows if tag == "R" and lo <= c < hi]
            neg_b = [s for tag, c, s in rows if tag == "L" and lo <= c < hi]
            pack_bins.append(
                {"bin": lab, "n_r": len(pos_b), "n_l": len(neg_b), "auroc": auroc(pos_b, neg_b)}
            )
        bin_auc[name] = pack_bins

    pref = cell["preferred"]
    pref_choice = chosen.get(pref)
    trial_conf = contrast(conf_rows)
    trial_low = contrast(low_all_rows)
    trial_oracle = contrast(oracle_rows)
    frozen = FROZEN.get(cell["dataset"])
    source = "trial"
    conf_out = trial_conf
    if frozen:
        sig, thr = frozen
        lag_rows = run(sig, float(thr))
        export_candidates(cell, conf_rows, "conf")
        export_candidates(cell, lag_rows, "lag")
        pref_choice = contrast(lag_rows)
        pref_choice["signal"] = sig
        pref_choice["threshold"] = float(thr)
    return {
        "name": cell["name"],
        "dataset": cell["dataset"],
        "n": len(questions),
        "n_scores": len(scores),
        "n_r": n_r,
        "n_r_only": n_r_only,
        "n_has_g": n_has_g,
        "geo_g_auroc": auroc(g_conf, not_g_conf),
        "puma_acc": puma_acc,
        "puma_tok": puma_tok,
        "full_acc": full_acc,
        "full_tok": full_tok,
        "conf": conf_out,
        "low_all": trial_low,
        "oracle": trial_oracle,
        "chosen": chosen,
        "preferred": pref,
        "preferred_choice": pref_choice,
        "source": source,
        "sweeps": {
            name: [
                {k: v for k, v in p.items() if k != "frac_conf"}
                for p in points
            ]
            for name, points in sweeps.items()
        },
        "bin_auc": bin_auc,
        "first_low_n": len(first_low),
        "first_low_p50": {
            name: {
                "R": p50([finite(r.get(name)) for r in first_low if r["tag"] == "R"]),
                "L": p50([finite(r.get(name)) for r in first_low if r["tag"] == "L"]),
            }
            for name, _ in SIGNALS
        },
    }


def render(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# 用内部读数捞回低置信度连对（R）",
        "",
        "主对照是试答即终答：闸门说停，就交这次试答的 boxed 答案，不再重写。",
        "Token = 截断前缀 + 到停点为止的试答。官方 PUMA 那一行仍是它自己的重写终答，只作外对照。",
        "门槛冻结，不重扫。",
        "",
        f"高置信度门不动：同一答案连续 {K} 次、第一次 ≥ {TAU}、后面 ≥ 第一次 − {EPS} 就停。",
        f"滞后门：已经连答 4 次、每步置信度都低于 {TAU}，且这 4 步的末层读数都够高（末层已经认这个答案）。",
        "对照不是把置信度门槛降下去——那会把低置信度连错一起放进来。",
        "",
        "## 题级结果",
        "",
        "「只看置信度」= 只有高置信度连答才停。",
        f"「凡低置信度连答就停」= 不看内部读数，等于把门槛从 {TAU} 拿掉。",
        "「先知先停在对的低置信度锁」= 上界，不是能上线的门。",
        "「加滞后门」= 高置信度门不动，只放行「4 步末层都已经认这个答案」的低置信度连答；门槛选在不伤已有对题、正确率尽量高、再用更少 token。",
        "",
        "| 数据集 | 门 | 正确率 | 平均 token | 相对 PUMA 正确率 / token | 相对只看置信度 | 低置信度门放进了对的锁 / 错的锁 | 只看置信度对、这里错 | 只有低置信度连对、没有高置信度锁的题捞回 | 高置信度门还在原处 |",
        "|---|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    zh = {name: text for name, text in SIGNALS}

    def row(name: str, gate: str, s: dict[str, Any], extra: str = "") -> str:
        return (
            f"| {name} | {gate}{extra} | {fmt_pct(s['acc'])} | {s['tok']:.0f} | "
            f"{fmt_pp(s['d_acc_pp_puma'])} / {fmt_tok(s['d_tok_puma'])} | "
            f"{fmt_pp(s['d_acc_pp_conf'])} / {fmt_tok(s['d_tok_conf'])} | "
            f"{s.get('rescue_R', 0)} / {s.get('rescue_L', 0)} | "
            f"{s.get('damaged', 0)} | {s.get('r_only_hit', 0)} | "
            f"{fmt_pct(s.get('g_door_kept', float('nan')))} |"
        )

    for cell in summary:
        n = cell["n"]
        lines.append(
            row(cell["name"], "PUMA", {
                "acc": cell["puma_acc"],
                "tok": cell["puma_tok"],
                "d_acc_pp_puma": 0.0,
                "d_tok_puma": 0.0,
                "d_acc_pp_conf": 100.0 * (cell["puma_acc"] - cell["conf"]["acc"]),
                "d_tok_conf": cell["puma_tok"] - cell["conf"]["tok"],
                "rescue_R": 0,
                "rescue_L": 0,
                "damaged": 0,
                "r_only_hit": 0,
                "g_door_kept": float("nan"),
            })
        )
        lines.append(row(cell["name"], "写完全程", {
            "acc": cell["full_acc"],
            "tok": cell["full_tok"],
            "d_acc_pp_puma": 100.0 * (cell["full_acc"] - cell["puma_acc"]),
            "d_tok_puma": cell["full_tok"] - cell["puma_tok"],
            "d_acc_pp_conf": 100.0 * (cell["full_acc"] - cell["conf"]["acc"]),
            "d_tok_conf": cell["full_tok"] - cell["conf"]["tok"],
            "rescue_R": 0,
            "rescue_L": 0,
            "damaged": 0,
            "r_only_hit": 0,
            "g_door_kept": float("nan"),
        }))
        lines.append(row(cell["name"], "只看置信度", cell["conf"]))
        lines.append(row(cell["name"], "凡低置信度连答就停", cell["low_all"]))
        lines.append(row(cell["name"], "先知先停在对的低置信度锁", cell["oracle"]))
        pref = cell.get("preferred_choice")
        if pref:
            lines.append(
                row(
                    cell["name"],
                    f"加滞后门（{zh.get(pref['signal'], pref['signal'])}）",
                    pref,
                    extra=f"；门槛 {pref['threshold']:.3f}",
                )
            )
        elif cell["n_scores"] == 0:
            lines.append(f"| {cell['name']} | 加滞后门 | — | — | 还没有这套内部抽取 | — | — | — | — | — |")
        lines.append(f"|  | 题数 {n}；有低置信度连对 {cell['n_r']} 题，其中后来没有高置信度锁 {cell['n_r_only']} 题；置信度分开高置信度连对的程度 {cell['geo_g_auroc']:.3f}（没动这扇门） |  |  |  |  |  |  |  |  |")

    lines += [
        "",
        "## 同一置信度段里，内部读数还能不能分开对的锁和错的锁",
        "",
        "只看第一次出现的低置信度连答窗。0.5 是乱猜。同一段里还明显高于 0.5，才不是置信度的马甲。",
        "",
        "| 数据集 | 读数 | 全部 | 置信度 0–0.5 | 0.5–0.8 | 0.8–0.9 | 0.9–τ |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in summary:
        if not cell["bin_auc"]:
            lines.append(f"| {cell['name']} | （还没有内部读数） | — | — | — | — | — |")
            continue
        for name, _zh in SIGNALS:
            bins = cell["bin_auc"].get(name)
            if not bins:
                continue
            by = {b["bin"]: b for b in bins}

            def cell_txt(key: str) -> str:
                b = by.get(key)
                if not b or b["auroc"] != b["auroc"]:
                    return "—"
                return f"{b['auroc']:.2f}（对{b['n_r']}/错{b['n_l']}）"

            lines.append(
                f"| {cell['name']} | {zh[name]} | {cell_txt('全部低置信度连答')} | "
                f"{cell_txt('0–0.5')} | {cell_txt('0.5–0.8')} | {cell_txt('0.8–0.9')} | {cell_txt('0.9–τ')} |"
            )

    lines += [
        "",
        "## 各读数选中的工作点",
        "",
        "每个读数单独扫门槛，优先：不把只看置信度已经对的题判错，再尽量提高正确率、少用 token。",
        "",
        "| 数据集 | 读数 | 门槛 | 正确率 | 平均 token | 相对只看置信度 | 放进了对的锁 / 错的锁 | 伤到的对题 |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for cell in summary:
        for name, text in SIGNALS:
            choice = cell["chosen"].get(name)
            if not choice:
                continue
            lines.append(
                f"| {cell['name']} | {text} | {choice['threshold']:.3f} | "
                f"{fmt_pct(choice['acc'])} | {choice['tok']:.0f} | "
                f"{fmt_pp(choice['d_acc_pp_conf'])} / {fmt_tok(choice['d_tok_conf'])} | "
                f"{choice['rescue_R']} / {choice['rescue_L']} | {choice['damaged']} |"
            )

    lines += [
        "",
        "主对照入口：`scripts/replay_rescue_R_gate.py`。交卷 = 试答即终答。",
        "门槛冻结：MATH `late_rise=8.984`，GPQA `lens_rise=5.008`。奥赛还没有这套内部读数。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summary = []
    for cell in CELLS:
        if not cell["trial"].is_file() or not cell["stat"].is_file():
            print(f"skip missing {cell['name']}", flush=True)
            continue
        if not any(path.is_dir() for path in cell["scores"]):
            print(f"skip no internals {cell['name']}", flush=True)
            continue
        print(f"evaluate {cell['name']}", flush=True)
        summary.append(evaluate_cell(cell))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    slim = []
    for cell in summary:
        item = dict(cell)
        item.pop("sweeps", None)
        slim.append(item)
    OUT.write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n")
    (OUT.parent / "sweeps.json").write_text(
        json.dumps({c["dataset"]: c["sweeps"] for c in summary}, indent=2) + "\n"
    )
    TABLE.write_text(render(summary))
    print(TABLE.read_text())
    print(f"wrote {TABLE} and {OUT}", flush=True)


if __name__ == "__main__":
    main()
