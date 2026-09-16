#!/usr/bin/env python3
"""Full-CoT vs PLWS after the first same-answer window: CORE coverage, rescue, length."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.matrix import FIRSTWIN, job_rows
from plws.paths import PLWSPaths
from report_fullcot_line_start import reasoning_path, split_after_thought, texts_from_answers
from report_fullcot_puma_plws import (
    ALL_SEEDS,
    CONTEST_DATASETS,
    DATASETS,
    collect_plws_scores,
    load_json,
    SEEDS,
)

OUT_JSON = ROOT / "results" / "reports" / "postwindow_core_rescue.json"
OUT_MD = ROOT / "tables" / "firstwin_wait" / "postwindow_core_rescue.md"

MODELS = (
    ("r1_7b", "7B"),
    ("nemotron_8b", "Nemotron"),
    ("r1_14b", "14B"),
    ("qwen3_4b", "4B"),
)
CORE_RE = re.compile(
    r"(?i)(?:\bwait\b|\balternatively\b|\bhmm+\b|\bhm\b|等一下)"
)
LINE_CORE_RE = re.compile(
    r"(?m)^(?:[#>*\-\s]*)(?:\*\*)?(?:wait\b|alternatively\b|hmm+\b|hm\b|等一下)",
    re.I,
)


def nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def core_stats(text: str) -> dict:
    lines = nonempty_lines(text)
    hits = LINE_CORE_RE.findall(text)
    return {
        "has": bool(CORE_RE.search(text or "")),
        "line_has": bool(hits),
        "n_lines": len(lines),
        "n_core_lines": len(hits),
        "n_hits": len(CORE_RE.findall(text or "")),
    }


def pct(part: int, whole: int) -> float | None:
    return 100.0 * part / whole if whole else None


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def bucket(lock_ok: bool, final_ok: bool) -> str:
    if lock_ok and final_ok:
        return "ok_keep"
    if lock_ok and not final_ok:
        return "ok_ruin"
    if (not lock_ok) and final_ok:
        return "bad_save"
    return "bad_keep"


def summarize(rows: list[dict]) -> dict:
    aligned = [row for row in rows if row.get("aligned")]
    n = len(aligned)
    lock_ok = [row for row in aligned if row["lock_ok"]]
    lock_bad = [row for row in aligned if not row["lock_ok"]]
    full_core = [row for row in aligned if row["full_core"]]
    full_no = [row for row in aligned if not row["full_core"]]
    bad_core = [row for row in lock_bad if row["full_core"]]
    bad_no = [row for row in lock_bad if not row["full_core"]]
    full_saves = [row for row in aligned if row["full_bucket"] == "bad_save"]
    plws_saves = [row for row in aligned if row["plws_bucket"] == "bad_save"]

    def rate(pred, xs) -> float | None:
        return pct(sum(1 for row in xs if pred(row)), len(xs))

    return {
        "n_windowed": len(rows),
        "n_aligned": n,
        "lock_ok_pct": pct(len(lock_ok), n),
        "full_core_q": pct(sum(row["full_core"] for row in aligned), n),
        "full_core_line_q": pct(sum(row["full_core_line"] for row in aligned), n),
        "full_core_line_share": mean(
            [
                row["full_core_lines"] / row["full_post_lines"]
                for row in aligned
                if row["full_post_lines"]
            ]
        ),
        "pre_core_q": pct(sum(row["pre_core"] for row in aligned), n),
        "plws_core_q": pct(sum(row["plws_core"] for row in aligned), n),
        "plws_core_line_q": pct(sum(row["plws_core_line"] for row in aligned), n),
        "full_save_all": rate(lambda r: r["full_bucket"] == "bad_save", aligned),
        "plws_save_all": rate(lambda r: r["plws_bucket"] == "bad_save", aligned),
        "full_ruin_all": rate(lambda r: r["full_bucket"] == "ok_ruin", aligned),
        "plws_ruin_all": rate(lambda r: r["plws_bucket"] == "ok_ruin", aligned),
        "full_save_given_bad": rate(lambda r: r["full_ok"], lock_bad),
        "plws_save_given_bad": rate(lambda r: r["plws_ok"], lock_bad),
        "full_save_given_bad_core": rate(lambda r: r["full_ok"], bad_core),
        "full_save_given_bad_nocore": rate(lambda r: r["full_ok"], bad_no),
        "plws_save_given_bad_core": rate(lambda r: r["plws_ok"], bad_core),
        "plws_save_given_bad_nocore": rate(lambda r: r["plws_ok"], bad_no),
        "full_ruin_given_ok": rate(lambda r: not r["full_ok"], lock_ok),
        "plws_ruin_given_ok": rate(lambda r: not r["plws_ok"], lock_ok),
        "saves_with_core_share": pct(
            sum(1 for row in full_saves if row["full_core"]), len(full_saves)
        ),
        "plws_saves_with_core_share": pct(
            sum(1 for row in plws_saves if row["plws_core"]), len(plws_saves)
        ),
        "n_lock_bad": len(lock_bad),
        "n_lock_ok": len(lock_ok),
        "n_full_save": len(full_saves),
        "n_plws_save": len(plws_saves),
        "n_full_core": len(full_core),
        "full_post_tok": mean([row["full_post_tok"] for row in aligned if row["full_post_tok"] is not None]),
        "plws_cont_tok": mean([row["plws_cont_tok"] for row in aligned if row["plws_cont_tok"] is not None]),
        "full_post_tok_core": mean(
            [row["full_post_tok"] for row in full_core if row["full_post_tok"] is not None]
        ),
        "full_post_tok_nocore": mean(
            [row["full_post_tok"] for row in full_no if row["full_post_tok"] is not None]
        ),
        "plws_cont_tok_core": mean(
            [row["plws_cont_tok"] for row in full_core if row["plws_cont_tok"] is not None]
        ),
        "plws_cont_tok_nocore": mean(
            [row["plws_cont_tok"] for row in full_no if row["plws_cont_tok"] is not None]
        ),
        "full_post_lines": mean([row["full_post_lines"] for row in aligned]),
        "plws_post_lines": mean([row["plws_post_lines"] for row in aligned]),
        "left_tok": mean([row["n_left_tok"] for row in aligned if row["n_left_tok"] is not None]),
    }


def collect(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        official = {
            int(row["question_idx"]): row
            for row in load_json(paths.puma_statistics_path(model, dataset, seed))
        }
        jobs = {
            int(job["question_idx"]): job
            for job in job_rows(paths, model, dataset, seed)
            if job.get("uid")
        }
        scores = collect_plws_scores(paths, model, dataset, seed, [])
        answers = reasoning_path(paths, model, dataset, seed)
        qmap = {}
        if answers is not None:
            qmap = {question: text for question, text in texts_from_answers(answers)}
        for qid, job in jobs.items():
            rec = scores.get((dataset, int(qid)))
            if rec is None:
                continue
            info = official.get(int(qid), {})
            thought = str(job.get("thought") or "")
            reasoning = qmap.get((job.get("question") or "").strip(), "")
            post = split_after_thought(thought, reasoning) if reasoning else None
            text = str(rec.get("generated_text") or "")
            cont = ""
            if thought and text.startswith(thought):
                cont = text[len(thought) :].split("</think>", 1)[0]
            elif rec.get("new_text"):
                cont = str(rec.get("new_text"))
            full_ok = bool(info.get("original_correct") if "original_correct" in info else job.get("orig_ok"))
            lock_ok = bool(rec.get("left_ok") if rec.get("left_ok") is not None else job.get("left_ok"))
            plws_ok = bool(rec.get("new_gold_ok"))
            n_left = rec.get("n_left_tok")
            full_tok = float(info.get("original_tokens") or job.get("original_tokens") or 0)
            full_post_tok = None
            if n_left is not None:
                full_post_tok = max(0.0, full_tok - float(n_left))
            full_c = core_stats(post or "")
            plws_c = core_stats(cont)
            pre_c = core_stats(thought)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "qid": int(qid),
                    "aligned": post is not None,
                    "lock_ok": lock_ok,
                    "full_ok": full_ok,
                    "plws_ok": plws_ok,
                    "full_bucket": bucket(lock_ok, full_ok),
                    "plws_bucket": bucket(lock_ok, plws_ok),
                    "pre_core": pre_c["has"],
                    "full_core": full_c["has"],
                    "full_core_line": full_c["line_has"],
                    "full_core_lines": full_c["n_core_lines"],
                    "full_post_lines": full_c["n_lines"],
                    "plws_core": plws_c["has"],
                    "plws_core_line": plws_c["line_has"],
                    "plws_post_lines": plws_c["n_lines"],
                    "full_post_tok": full_post_tok,
                    "plws_cont_tok": (
                        float(rec["n_cont_tok"]) if rec.get("n_cont_tok") is not None else None
                    ),
                    "n_left_tok": float(n_left) if n_left is not None else None,
                }
            )
    return rows


def write_md(payload: dict) -> None:
    lines = [
        "# 窗后 CORE 覆盖、救回与长度",
        "",
        "有第一扇窗、且 Full-CoT 正文能对上窗前前缀的题。",
        "7B / Nemotron / 14B / 4B 是主表五集四 seed。30B 是竞赛五集五 seed。",
        "CORE = `Wait` / `Alternatively` / `Hmm`（含大小写和 `等一下`），在窗后正文里任意出现。",
        "救回 = 锁时错、该侧终答对。写毁 = 锁时对、该侧终答错。",
        "长度：Full-CoT 窗后 ≈ `original_tokens − n_left_tok`；PLWS 窗后 = `n_cont_tok`。",
        "不进主表。",
        "",
        "## 1. 窗后还在不在用 CORE，压完还剩多少",
        "",
        "| 模型 | 对齐题 | 锁已对 | 窗前有 CORE | 窗后原文有 CORE | 窗后段首有 CORE | 压后续写有 CORE | 压后段首有 CORE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for zh, cell in payload["models"].items():
        o = cell["overall"]
        lines.append(
            f"| {zh} | {o['n_aligned']} | {fmt(o['lock_ok_pct'])}% | "
            f"{fmt(o['pre_core_q'])}% | {fmt(o['full_core_q'])}% | "
            f"{fmt(o['full_core_line_q'])}% | {fmt(o['plws_core_q'])}% | "
            f"{fmt(o['plws_core_line_q'])}% |"
        )
    lines.extend(
        [
            "",
            "## 2. 错锁救回：原文靠不靠 CORE，压完还剩多少",
            "",
            "分母是错锁题。`原文+CORE` / `原文无CORE` 是按窗后原文是否出现 CORE 切开的同一批错锁。",
            "",
            "| 模型 | 错锁 | 原文救回 | 其中窗后有 CORE | 错锁且有 CORE 的救回 | 错锁且无 CORE 的救回 | 压后救回 | 压后救回里仍有 CORE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for zh, cell in payload["models"].items():
        o = cell["overall"]
        lines.append(
            f"| {zh} | {o['n_lock_bad']} | {fmt(o['full_save_given_bad'])}% "
            f"（{o['n_full_save']}） | {fmt(o['saves_with_core_share'])}% | "
            f"{fmt(o['full_save_given_bad_core'])}% | "
            f"{fmt(o['full_save_given_bad_nocore'])}% | "
            f"{fmt(o['plws_save_given_bad'])}%（{o['n_plws_save']}） | "
            f"{fmt(o['plws_saves_with_core_share'])}% |"
        )
    lines.extend(
        [
            "",
            "## 3. 好锁写毁",
            "",
            "| 模型 | 好锁 | 原文写毁 | 压后写毁 |",
            "|---|---:|---:|---:|",
        ]
    )
    for zh, cell in payload["models"].items():
        o = cell["overall"]
        lines.append(
            f"| {zh} | {o['n_lock_ok']} | {fmt(o['full_ruin_given_ok'])}% | "
            f"{fmt(o['plws_ruin_given_ok'])}% |"
        )
    lines.extend(
        [
            "",
            "## 4. 窗后长度",
            "",
            "| 模型 | 前缀 token | 原文窗后 | 压后窗后 | Δ | 原文有 CORE 的窗后 | 压后（这批） | 原文无 CORE 的窗后 | 压后（这批） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for zh, cell in payload["models"].items():
        o = cell["overall"]
        delta = None
        if o["full_post_tok"] is not None and o["plws_cont_tok"] is not None:
            delta = o["plws_cont_tok"] - o["full_post_tok"]
        lines.append(
            f"| {zh} | {fmt(o['left_tok'], 0)} | {fmt(o['full_post_tok'], 0)} | "
            f"{fmt(o['plws_cont_tok'], 0)} | {fmt(delta, 0)} | "
            f"{fmt(o['full_post_tok_core'], 0)} | {fmt(o['plws_cont_tok_core'], 0)} | "
            f"{fmt(o['full_post_tok_nocore'], 0)} | {fmt(o['plws_cont_tok_nocore'], 0)} |"
        )
    lines.extend(
        [
            "",
            "## 5. 分集（大集 + AIME）",
            "",
            "| 模型 | 集 | 对齐 | 窗后 CORE | 压后 CORE | 错锁原文救回 | 错锁压后救回 | 原文窗后 tok | 压后 tok |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for zh, cell in payload["models"].items():
        for dszh, o in cell["datasets"].items():
            lines.append(
                f"| {zh} | {dszh} | {o['n_aligned']} | {fmt(o['full_core_q'])}% | "
                f"{fmt(o['plws_core_q'])}% | {fmt(o['full_save_given_bad'])}% | "
                f"{fmt(o['plws_save_given_bad'])}% | {fmt(o['full_post_tok'], 0)} | "
                f"{fmt(o['plws_cont_tok'], 0)} |"
            )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))


def score_model(
    paths: PLWSPaths,
    model: str,
    zh: str,
    datasets: tuple[tuple[str, str, int], ...],
    seeds: tuple[int, ...],
) -> dict:
    by_ds = {}
    all_rows: list[dict] = []
    print(f"== {zh} ==", flush=True)
    for dataset, dszh, _n in datasets:
        rows = collect(paths, model, dataset, seeds=seeds)
        all_rows.extend(rows)
        cell = summarize(rows)
        by_ds[dszh] = cell
        print(
            f"{zh:8} {dszh:14} n={cell['n_aligned']} "
            f"core {fmt(cell['full_core_q'])}->{fmt(cell['plws_core_q'])} "
            f"save {fmt(cell['full_save_given_bad'])}->{fmt(cell['plws_save_given_bad'])} "
            f"tok {fmt(cell['full_post_tok'], 0)}->{fmt(cell['plws_cont_tok'], 0)}",
            flush=True,
        )
    return {"datasets": by_ds, "overall": summarize(all_rows)}


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    only = sys.argv[1:] 
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "scripts/analyze_postwindow_core_rescue.py",
        "models": {},
    }
    if only == ["30b"] and OUT_JSON.is_file():
        payload = json.loads(OUT_JSON.read_text())
        payload["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        payload["models"]["30B"] = score_model(
            paths, "qwen3_30b_a3b", "30B", CONTEST_DATASETS, ALL_SEEDS
        )
        OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        write_md(payload)
        print(f"wrote {OUT_JSON}")
        print(f"wrote {OUT_MD}")
        return
    for model, zh in MODELS:
        payload["models"][zh] = score_model(paths, model, zh, DATASETS, SEEDS)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_md(payload)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
