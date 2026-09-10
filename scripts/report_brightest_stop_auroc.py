#!/usr/bin/env python3
"""One vote per question that can stop on the lag door: brightest reachable R vs L."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as old
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp

SIGS = (("exit_minus_half", "出口减一半深"), ("stop_margin", "收口比 Wait"))
TABLE = AE / "tables/brightest_stop_auroc.md"
SEEDS = (42, 0, 1, 123)
MODELS = (
    ("7B", "r1_7b", "dense_G_r1_7b", "puma_offline_r1_7b"),
    ("8B", "nemotron_8b", "dense_G_nemotron_8b", "puma_offline_nemotron_8b"),
    ("14B", "r1_14b", "dense_G_r1_14b", "puma_offline_r1_14b"),
    ("32B", "r1_32b", "dense_G_r1_32b", "puma_offline_r1_32b"),
    ("30B", "qwen3_30b_a3b", "dense_G_qwen3_30b_a3b", "puma_offline_qwen3_30b_a3b"),
)
BIG = (("MATH", "math-500"), ("奥赛", "olympiadbench"), ("GPQA", "gpqa-diamond"))
AIME = (("AIME24", "aime24"), ("AIME25", "aime25"))
BIG_DS = {"math-500", "olympiadbench", "gpqa-diamond"}


def score_dirs(tag: str, dataset: str, seed: int | None) -> tuple[Path, ...]:
    # Wait: only dense_puma_wait (official PUMA header). Never dense_wait or
    # dense_wait_once — those are once-prompt / leftover copies.
    out: list[Path] = []
    stem = f"{dataset}_s{seed}" if seed is not None else dataset
    if tag == "r1_7b" and dataset in BIG_DS:
        for kind in ("dense_internal", "dense_lens", "dense_solver_probes"):
            path = AE / f"results/confcal_judge/v2/{kind}/{dataset}"
            if path.is_dir():
                out.append(path)
        puma_wait = AE / f"results/confcal_judge/v2/dense_puma_wait/{dataset}"
        if puma_wait.is_dir():
            out.append(puma_wait)
        return tuple(out)
    for kind in ("dense_internal", "dense_lens", "dense_solver_probes"):
        path = AE / f"results/confcal_judge/v2/{kind}/{tag}/{stem}"
        if path.is_dir():
            out.append(path)
    puma_wait = AE / f"results/confcal_judge/v2/dense_puma_wait/{tag}/{stem}"
    if puma_wait.is_dir():
        out.append(puma_wait)
    return tuple(out)


def stat_path(puma: str, dataset: str, tag: str, seed: int | None = None) -> Path:
    if dataset == "math-500" and tag == "r1_7b":
        return AE / "results/math500_official/puma_ds7b/statistics.json"
    if seed is None or seed == 42:
        return AE / "results" / puma / dataset / "statistics.json"
    return AE / "results" / f"{puma}_s{seed}" / dataset / "statistics.json"


def make_cell(zh: str, tag: str, gdir: str, puma: str, ds_zh: str, dataset: str, seed: int | None = None) -> dict[str, Any]:
    if seed is None:
        root = AE / "results" / gdir / dataset
        name = f"{zh} {ds_zh}"
        trial = root / "dense_puma/trial_answers.json"
        gpath = root / "per_sample.json"
    else:
        root = AE / "results" / gdir / dataset / f"seed_{seed}"
        name = f"{zh} {ds_zh}-s{seed}"
        trial = root / "dense_puma/trial_answers.json"
        gpath = root / "per_sample.json"
    return {
        "name": name,
        "dataset": dataset,
        "trial": trial,
        "stat": stat_path(puma, dataset, tag, seed),
        "gpath": gpath,
        "scores": score_dirs(tag, dataset, seed),
    }


def win_score(ev: dict[str, Any], sig: str) -> float:
    vals = ev["vals"].get(sig) or []
    if not vals or any(v != v for v in vals):
        return float("nan")
    return min(vals)


def brightest_vote(q: dict[str, Any], sig: str) -> tuple[float, str] | None:
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials = q["trials"]
    rows = q["rows"]
    lens = old.step_char_lens(trials) if rg.USE_FS else {}
    red_run = 0
    best_conf = float("-inf")
    probed: list[dict[str, Any]] = []
    walked = 0
    best_score: float | None = None
    best_tag: str | None = None
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if rg.USE_FS and tstep >= rg.FS_MIN_STEP:
                red_run = red_run + 1 if old.is_red(tstep, lens) else 0
            walked += 1
        conf = rg.finite(row.get("confidence"))
        answer = str(row.get("final_answer") or "")
        if answer and conf == conf:
            best_conf = max(best_conf, conf)
            probed.append({"step": step, "answer": answer, "conf": conf})
        ev = events_by_end.get(end)
        if ev is not None and step >= rg.MSS:
            if ev["high"]:
                break
            if (not ev["mixed"]) and ev["tag"] in {"R", "L"}:
                score = win_score(ev, sig)
                if score == score and (best_score is None or score > best_score):
                    best_score = score
                    best_tag = ev["tag"]
        if rg.USE_FS:
            pick = rg.fs_ready(step, red_run, best_conf, probed)
            if pick is not None:
                break
    if best_score is None or best_tag is None:
        return None
    return best_score, best_tag


def collect(pack: dict[str, Any], sig: str) -> tuple[list[float], list[float]]:
    pos, neg = [], []
    for q in pack["questions"]:
        vote = brightest_vote(q, sig)
        if vote is None:
            continue
        score, tag = vote
        if tag == "R":
            pos.append(score)
        else:
            neg.append(score)
    return pos, neg


def cell_text(pos: list[float], neg: list[float]) -> str:
    if not pos or not neg:
        return "—"
    return (
        f"{rg.auroc(pos, neg):.3f}"
        f"（停对{len(pos)}/停错{len(neg)}；中位 {rg.p50(pos):.2f} / {rg.p50(neg):.2f}）"
    )


def load_cell(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
        return None
    pack = layers.load_pack(cell)
    ids = more.discover_layers(pack["scores"])
    if len(ids) >= 2:
        cons.attach_shared(pack)
    layers.precompute_events(pack, [s for s, _ in SIGS])
    pack["_cov"] = max(cmp.cov(pack, s) for s, _ in SIGS)
    return pack


def all_jobs() -> list[tuple[str, list[dict[str, Any]], bool]]:
    jobs: list[tuple[str, list[dict[str, Any]], bool]] = []
    for zh, tag, gdir, puma in MODELS:
        for ds_zh, dataset in BIG:
            if tag == "qwen3_30b_a3b":
                continue
            jobs.append((f"{zh} {ds_zh}", [make_cell(zh, tag, gdir, puma, ds_zh, dataset)], False))
        for ds_zh, dataset in AIME:
            cells = [make_cell(zh, tag, gdir, puma, ds_zh, dataset, seed) for seed in SEEDS]
            jobs.append((f"{zh} {ds_zh}", cells, True))
    return jobs


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    header = "| 集 | 题数 | 会走滞后门 | 出口减一半深 | 收口比 Wait |"
    sep = "|---|---:|---:|---|---|"
    lines = [
        "# 一题一票：会走滞后门的题，最亮停点对不对",
        "",
        "只算还没被高置信连答、后路强停、写完先收走的题。这些题才会轮到滞后门。",
        "从前往后，只有比前面都亮的扇才可能被某条门槛停住。一题只留最亮的那一扇。",
        "这一扇连对 = 停对，连错 = 停错。分数 = 窗内 4 步最差。",
        "AUROC = 停对题的分数是否高于停错题。1 是完全分开，0.5 是猜。",
        "加粗 = 该集两个方法里更高的那个。读数未齐的集不报数。",
        "",
        "入口：`scripts/report_brightest_stop_auroc.py`",
        "",
        header,
        sep,
    ]
    print("\n" + header)
    print(sep)
    for name, cells, merge in all_jobs():
        parts = []
        missing = 0
        low = 0
        for cell in cells:
            pack = load_cell(cell)
            if pack is None:
                missing += 1
                print(f"skip {cell['name']} 无读数", flush=True)
                continue
            print(f"load {cell['name']} cov={pack['_cov']:.2f}", flush=True)
            if pack["_cov"] < 0.5:
                low += 1
                continue
            parts.append(pack)
        if merge:
            need = len(cells)
            if len(parts) < need:
                row = f"| {name} | — | — | 读数未齐（{len(parts)}/{need} seed） | 读数未齐 |"
                print(row, flush=True)
                lines.append(row)
                continue
            pack = accfirst.merge_packs(parts, name)
        elif not parts:
            why = "读数未齐" if missing or low else "缺文件"
            row = f"| {name} | — | — | {why} | {why} |"
            print(row, flush=True)
            lines.append(row)
            continue
        else:
            pack = parts[0]
        n = len(pack["questions"])
        half_pos, half_neg = collect(pack, "exit_minus_half")
        wait_pos, wait_neg = collect(pack, "stop_margin")
        n_lag = max(len(half_pos) + len(half_neg), len(wait_pos) + len(wait_neg))
        half = cell_text(half_pos, half_neg)
        wait = cell_text(wait_pos, wait_neg)
        if half != "—" and wait != "—":
            if rg.auroc(half_pos, half_neg) > rg.auroc(wait_pos, wait_neg):
                half = f"**{half}**"
            elif rg.auroc(wait_pos, wait_neg) > rg.auroc(half_pos, half_neg):
                wait = f"**{wait}**"
        row = f"| {name} | {n} | {n_lag} | {half} | {wait} |"
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"\n写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
