#!/usr/bin/env python3
"""Verify Full-CoT / PUMA / PLWS (窗后压) / aligned DEER on the five main datasets.

Seeds are 42 / 0 / 1 / 123. Overall is an equal-weight mean of the
listed datasets. 7B Overall also folds in AMC23 (shown as its own row).
GSM8K is not in the table. The token column is delivery plus trial
answers. PLWS
is Wait-suppression after the first same-answer window, not prefix
regeneration. Aligned DEER is unified-host ``puma-fullcot-32k-v2``;
a cell is filled only when all four seeds are complete.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from plws.matrix import (
    FIRSTWIN,
    TIER_KINDS,
    deer_complete,
    deer_output_dir,
    job_rows,
    reusable_score,
)
from plws.paths import PLWSPaths

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "tables" / "firstwin_wait" / "fullcot_puma_plws.md"
REPORT = ROOT / "results" / "reports" / "fullcot_puma_plws.json"
DEER_CACHE = ROOT / "results" / "reports" / "cache" / "deer_aligned_v2_cells.json"
TOKEN_POLICY = "delivery_plus_boxed_trial_v1"

MODELS = (
    ("r1_7b", "7B"),
    ("nemotron_8b", "Nemotron"),
    ("r1_14b", "14B"),
    ("qwen3_4b", "4B"),
    ("qwen3_8b", "8B"),
)
DATASETS = (
    ("math-500", "MATH", 2000),
    ("olympiadbench", "OlympiadBench", 2700),
    ("gpqa-diamond", "GPQA-Diamond", 792),
    ("aime24", "AIME24", 120),
    ("aime25", "AIME25", 120),
)
EXTRA_MODELS = frozenset({"r1_7b"})
OVERALL_ONLY_DATASETS = (
    ("amc23", "AMC23", 160),
)
EXTRA_DATASETS = OVERALL_ONLY_DATASETS
SEEDS = (42, 0, 1, 123)
KINDS = (FIRSTWIN, *TIER_KINDS)
SHARD_RE = re.compile(r"^(shard_\d+|scores|scores_shard\d+)\.jsonl$")
TIER_KIND_DIRS = {
    "low": "",
    "mix": "_mix",
    "high": "_high",
}

_GRADER = None


def load_json(path: Path):
    return json.loads(path.read_text())


def rec_key(rec: dict) -> tuple[str, int] | None:
    dataset = rec.get("dataset")
    question_idx = rec.get("question_idx")
    if dataset is None or question_idx is None:
        parts = str(rec.get("uid") or "").split(":")
        if len(parts) >= 4:
            dataset = dataset or parts[1]
            question_idx = question_idx if question_idx is not None else parts[3]
    if dataset is None or question_idx is None:
        return None
    return str(dataset), int(question_idx)


def leftover_suppress_dir(
    paths: PLWSPaths, model: str, seed: int, kind: str
) -> Path:
    suffix = TIER_KIND_DIRS[kind]
    return (
        paths.results
        / "archive"
        / "legacy_layout"
        / "plws"
        / "leftover_suppress_toend"
        / f"{model}_s{seed}_suppress{suffix}"
    )


def load_score_folder(
    folder: Path,
    *,
    require_protocol: bool = True,
    dataset: str | None = None,
) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.iterdir()):
        if not SHARD_RE.match(path.name):
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if require_protocol and not reusable_score(rec):
                continue
            if rec.get("status") not in {"ok", "too_long"} or not rec.get("uid"):
                continue
            key = rec_key(rec)
            if key is None:
                continue
            if dataset is not None and key[0] != dataset:
                continue
            out[key] = rec
    return out


def load_scores(folder: Path) -> dict[tuple[str, int], dict]:
    return load_score_folder(folder, require_protocol=True)


def collect_plws_scores(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    inputs: list[str],
    *,
    allow_legacy: bool = False,
) -> dict[tuple[str, int], dict]:
    scores: dict[tuple[str, int], dict] = {}
    for kind in KINDS:
        score_dir = paths.score_dir(model, dataset, seed, kind, k=4, lexicon="core")
        loaded = load_scores(score_dir)
        if loaded:
            inputs.append(str(score_dir.relative_to(paths.root)))
        if kind == FIRSTWIN:
            scores.update(loaded)
        else:
            for key, rec in loaded.items():
                scores.setdefault(key, rec)
    if not allow_legacy:
        return scores
    for kind in TIER_KINDS:
        folder = leftover_suppress_dir(paths, model, seed, kind)
        loaded = load_score_folder(
            folder, require_protocol=False, dataset=dataset
        )
        if loaded:
            inputs.append(str(folder.relative_to(paths.root)))
        for key, rec in loaded.items():
            scores.setdefault(key, rec)
    return scores


def load_trials_by_question(path: Path) -> dict[int, list[dict]]:
    raw = load_json(path)
    rows = raw if isinstance(raw, list) else raw.get("rows", [])
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(int(row["question_idx"]), []).append(row)
    return out


def boxed_trial_tokens_upto(trials: list[dict], end_step: int) -> float:
    """Sum boxed-interior trial tokens through the first-window end step."""
    total = 0
    for row in trials:
        if row.get("skipped"):
            continue
        if int(row["stopped_len"]) > int(end_step):
            continue
        if not str(row.get("final_answer") or ""):
            continue
        total += int(row.get("count_answer_tokens") or 0)
    return float(total)


def puma_token(row: dict) -> float:
    if "tokens_trial_answers" not in row:
        raise RuntimeError("PUMA row missing tokens_trial_answers")
    return float(row.get("compressed_tokens") or 0) + float(
        row.get("tokens_trial_answers") or 0
    )


def puma_delivery_path(paths: PLWSPaths, model: str, dataset: str, seed: int) -> Path:
    backfill = (
        paths.results
        / "baselines"
        / "puma"
        / "backfill"
        / model
        / dataset
        / f"seed_{seed}"
        / "statistics.json"
    )
    if backfill.is_file():
        return backfill
    return paths.puma_statistics_path(model, dataset, seed)


def fmt_cell(acc: float, tok: float) -> str:
    return f"{acc:.2f}% / {tok:.0f}"


def fmt_optional(cell: dict | None) -> str:
    if cell is None:
        return ""
    return fmt_cell(cell["acc"], cell["tok"])


def deer_seed_n(expect_n: int) -> int:
    return expect_n // len(SEEDS)


def four_seed_deer_complete(
    paths: PLWSPaths, model: str, dataset: str, expect_n: int
) -> bool:
    expected = deer_seed_n(expect_n)
    return all(
        deer_complete(paths, model, dataset, seed, expected=expected)[0]
        for seed in SEEDS
    )


def _extract_answer(generated_text: str, task_type: str) -> str:
    """Match PUMA ``run_vllm.extract_answer`` without importing vLLM."""
    from baselines.utils.math_util import my_answer_extraction  # noqa: PLC0415

    answer = my_answer_extraction(generated_text)
    if task_type == "gpqa" and answer not in ("A", "B", "C", "D"):
        match = re.search(r"ANSWER\s*:\s*([A-D])", generated_text or "")
        if match:
            answer = match.group(1)
    return answer


def _grader():
    global _GRADER
    if _GRADER is None:
        puma_root = ROOT.parent / "PUMA"
        sys.path.insert(0, str(puma_root))
        sys.path.insert(0, str(puma_root / "puma"))
        from math_grader import check_is_correct  # noqa: PLC0415
        from prompt_utils import get_task_type  # noqa: PLC0415

        _GRADER = (_extract_answer, get_task_type, check_is_correct)
    return _GRADER


def grade_deer_item(item: tuple[str, str, str, float]) -> tuple[bool, float]:
    dataset, text, gold, tok = item
    extract_answer, get_task_type, check_is_correct = _grader()
    pred = extract_answer(str(text or ""), get_task_type(dataset))
    try:
        ok = bool(check_is_correct(pred, gold or ""))
    except Exception:
        ok = False
    return ok, float(tok)


def _deer_token(row: dict) -> float:
    tok = float(row.get("delivery_tokens") or 0)
    if tok <= 0:
        tok = float(row.get("thinking_tokens") or 0) + float(
            row.get("answer_tokens") or 0
        )
    return tok + float(row.get("num_trial_answer_tokens") or 0)


def _deer_fingerprint(paths: PLWSPaths, model: str, dataset: str) -> str:
    parts: list[str] = []
    for seed in SEEDS:
        path = deer_output_dir(paths, model, dataset, seed) / "deer.jsonl"
        stat = path.stat()
        parts.append(f"{seed}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _load_deer_cache() -> dict:
    if not DEER_CACHE.is_file():
        return {}
    return json.loads(DEER_CACHE.read_text())


def _save_deer_cache(cache: dict) -> None:
    DEER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DEER_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")


def load_deer_cell(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    expect_n: int,
    inputs: list[str],
    cache: dict,
) -> dict | None:
    if not four_seed_deer_complete(paths, model, dataset, expect_n):
        return None
    key = f"{model}/{dataset}"
    fingerprint = f"{TOKEN_POLICY}|{_deer_fingerprint(paths, model, dataset)}"
    hit = cache.get(key)
    if (
        isinstance(hit, dict)
        and hit.get("fingerprint") == fingerprint
        and hit.get("n") == expect_n
        and hit.get("token_policy") == TOKEN_POLICY
    ):
        for seed in SEEDS:
            path = deer_output_dir(paths, model, dataset, seed) / "deer.jsonl"
            inputs.append(str(path.relative_to(paths.root)))
        return {"n": hit["n"], "acc": hit["acc"], "tok": hit["tok"]}
    items: list[tuple[str, str, str, float]] = []
    for seed in SEEDS:
        path = deer_output_dir(paths, model, dataset, seed) / "deer.jsonl"
        inputs.append(str(path.relative_to(paths.root)))
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                (
                    dataset,
                    str(row.get("generated_text") or ""),
                    str(row.get("gold_answer") or ""),
                    _deer_token(row),
                )
            )
    if len(items) != expect_n:
        return None
    with Pool(8) as pool:
        rows = pool.map(grade_deer_item, items, chunksize=16)
    cell = {
        "n": len(rows),
        "acc": 100.0 * sum(ok for ok, _ in rows) / len(rows),
        "tok": sum(tok for _, tok in rows) / len(rows),
    }
    cache[key] = {**cell, "fingerprint": fingerprint, "token_policy": TOKEN_POLICY}
    _save_deer_cache(cache)
    return cell


def _round_cell(cell: dict) -> dict:
    return {"acc": round(cell["acc"], 2), "tok": round(cell["tok"])}


def _mean_cell(rows: list[tuple[float, float]]) -> str:
    return (
        f"{sum(acc for acc, _ in rows) / len(rows):.2f}% / "
        f"{sum(tok for _, tok in rows) / len(rows):.0f}"
    )


def score_model_dataset(
    paths: PLWSPaths,
    model: str,
    zh: str,
    dataset: str,
    dszh: str,
    expect_n: int,
    inputs: list[str],
    too_long: list[dict],
    deer_cache: dict,
    *,
    allow_legacy: bool = False,
    want_deer: bool = True,
) -> dict[str, dict | None]:
    full: list[tuple[bool, float]] = []
    puma: list[tuple[bool, float]] = []
    plws: list[tuple[bool, float]] = []
    for seed in SEEDS:
        official_path = paths.puma_statistics_path(model, dataset, seed)
        if not official_path.is_file():
            raise FileNotFoundError(official_path)
        official = {int(row["question_idx"]): row for row in load_json(official_path)}
        inputs.append(str(official_path.relative_to(paths.root)))
        delivery_path = puma_delivery_path(paths, model, dataset, seed)
        delivery = load_json(delivery_path)
        delivery_rows = (
            delivery if isinstance(delivery, list) else delivery.get("rows", [])
        )
        inputs.append(str(delivery_path.relative_to(paths.root)))
        for row in official.values():
            full.append(
                (
                    bool(row.get("original_correct")),
                    float(row.get("original_tokens") or 0),
                )
            )
        for row in delivery_rows:
            puma.append((bool(row.get("compressed_correct")), puma_token(row)))
        windowed: dict[int, dict] = {}
        for job in job_rows(paths, model, dataset, seed):
            if job.get("dataset") in (None, dataset) and job.get("uid"):
                windowed[int(job["question_idx"])] = job
        trials_path = paths.dense_trial_path(model, dataset, seed)
        if windowed and not trials_path.is_file():
            raise FileNotFoundError(trials_path)
        trials_by_q = (
            load_trials_by_question(trials_path) if trials_path.is_file() else {}
        )
        if trials_path.is_file():
            inputs.append(str(trials_path.relative_to(paths.root)))
        jobs_path = paths.jobs_path(
            model, dataset, seed, FIRSTWIN, k=4, lexicon="core"
        )
        if jobs_path.is_file():
            inputs.append(str(jobs_path.relative_to(paths.root)))
        scores = collect_plws_scores(
            paths, model, dataset, seed, inputs, allow_legacy=allow_legacy
        )
        for question_idx, info in official.items():
            rec = scores.get((dataset, int(question_idx)))
            if rec is not None:
                if rec.get("status") == "too_long":
                    too_long.append(
                        {
                            "model": zh,
                            "dataset": dszh,
                            "seed": seed,
                            "question_idx": int(question_idx),
                            "uid": rec.get("uid"),
                        }
                    )
                job = windowed.get(int(question_idx))
                left_step = rec.get("left_step")
                if left_step is None and job is not None:
                    left_step = job.get("left_step")
                if left_step is None:
                    raise RuntimeError(
                        f"missing left_step {model} {dataset} seed={seed} q={question_idx}"
                    )
                plws.append(
                    (
                        bool(rec.get("new_gold_ok")),
                        float(rec.get("n_think_tok") or 0)
                        + float(rec.get("n_ans_tok") or 0)
                        + boxed_trial_tokens_upto(
                            trials_by_q.get(int(question_idx), []),
                            left_step,
                        ),
                    )
                )
            elif int(question_idx) not in windowed:
                plws.append(
                    (
                        bool(info.get("original_correct")),
                        float(info.get("original_tokens") or 0),
                    )
                )
            else:
                raise RuntimeError(
                    f"missing PLWS score {model} {dataset} seed={seed} q={question_idx}"
                )

    def agg(rows: list[tuple[bool, float]]) -> dict:
        n = len(rows)
        if n != expect_n:
            raise RuntimeError(f"{zh} {dszh} expected n={expect_n}, got {n}")
        return {
            "n": n,
            "acc": 100.0 * sum(ok for ok, _ in rows) / n,
            "tok": sum(tok for _, tok in rows) / n,
        }

    return {
        "full": agg(full),
        "puma": agg(puma),
        "plws": agg(plws),
        "deer": (
            load_deer_cell(paths, model, dataset, expect_n, inputs, deer_cache)
            if want_deer
            else None
        ),
    }


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    cells: dict[tuple[str, str], dict[str, dict | None]] = {}
    too_long: list[dict] = []
    inputs: list[str] = []
    deer_cache = _load_deer_cache()

    for model, zh in MODELS:
        for dataset, dszh, expect_n in DATASETS:
            print(f"score DEER {zh} {dszh}", flush=True)
            cells[(zh, dszh)] = score_model_dataset(
                paths,
                model,
                zh,
                dataset,
                dszh,
                expect_n,
                inputs,
                too_long,
                deer_cache,
            )
        if model in EXTRA_MODELS:
            for dataset, dszh, expect_n in EXTRA_DATASETS:
                print(f"score extra {zh} {dszh}", flush=True)
                cells[(zh, dszh)] = score_model_dataset(
                    paths,
                    model,
                    zh,
                    dataset,
                    dszh,
                    expect_n,
                    inputs,
                    too_long,
                    deer_cache,
                    allow_legacy=True,
                    want_deer=False,
                )

    lines = [
        "# Full-CoT / PUMA / DEER / 窗后压",
        "",
        "由 `scripts/report_fullcot_puma_plws.py` 从规范产物重算。",
        "四 seed：0 / 1 / 42 / 123。Overall 是表内各集等权平均 Acc / 平均 token。",
        "7B Overall 另折进 AMC23（四 seed，n=160），表中单列一行。GSM8K 不进表。",
        "AMC23 窗后压按现行规则从已有 leftover 分重算，与主表同一套 Acc / 交付+试答。",
        "PLWS 主五行只收 `puma-fullcot-32k-v2` firstwin 分；无第一扇窗的题贴 Full-CoT。",
        "7B / Nemotron / 14B 的 Full-CoT 与 PUMA 沿用已有 host，未按 v2 重跑；",
        "PLWS 与 DEER 是 v2（37888）。4B / 8B 的 Full-CoT / PUMA 来自 `puma_offline`。",
        "DEER 是统一 host 的 aligned v2，不是官方 greedy / 16k。",
        "某个模型×数据集的 DEER 只有四 seed 都齐才填，否则留空。",
        "Overall 的 DEER 仅当五个数据集都齐时才填。",
        "预实验数字不进本表。",
        "Token 只有一列：交付 + 试答。试答只数 boxed 内部（PUMA / 窗后压），",
        "DEER 用记录的 `num_trial_answer_tokens`（R1/Nemotron 约等于 boxed；",
        "Qwen3 probe 会写到 `</think>`，试答比 boxed 大）。",
        "",
        "- Full-CoT：`original_correct` / `original_tokens`（无试答）",
        "- PUMA：`compressed_tokens + tokens_trial_answers`",
        "- DEER：`delivery_tokens + num_trial_answer_tokens`；四 seed 齐才填",
        "- 窗后压：思考 + 终答 + 第一扇窗末步及之前的 `count_answer_tokens`；无窗贴 Full-CoT、试答记 0；固定最后一列",
        "",
        "| 模型 | 集 | n | Full-CoT（原始 CoT） | PUMA | DEER | 窗后压 |",
        "|---|---|---:|---|---|---|---|",
    ]
    payload_cells = []
    for model, zh in MODELS:
        shown: dict[str, list[tuple[float, float]]] = {
            "full": [],
            "puma": [],
            "plws": [],
            "deer": [],
        }
        hidden = OVERALL_ONLY_DATASETS if model in EXTRA_MODELS else ()
        for dataset, dszh, expect_n in (*DATASETS, *hidden):
            hidden_row = (dataset, dszh, expect_n) in OVERALL_ONLY_DATASETS
            cell = cells[(zh, dszh)]
            row = {
                "model": zh,
                "dataset": dszh,
                "n": expect_n,
                "full": _round_cell(cell["full"]),
                "puma": _round_cell(cell["puma"]),
                "plws": _round_cell(cell["plws"]),
                "deer": None if cell["deer"] is None else _round_cell(cell["deer"]),
                "overall_only": hidden_row,
            }
            payload_cells.append(row)
            for key in ("full", "puma", "plws"):
                shown[key].append((row[key]["acc"], row[key]["tok"]))
            if row["deer"] is not None:
                shown["deer"].append((row["deer"]["acc"], row["deer"]["tok"]))
            lines.append(
                f"| {zh} | {dszh} | {expect_n} | "
                f"{fmt_cell(cell['full']['acc'], cell['full']['tok'])} | "
                f"{fmt_cell(cell['puma']['acc'], cell['puma']['tok'])} | "
                f"{fmt_optional(cell['deer'])} | "
                f"{fmt_cell(cell['plws']['acc'], cell['plws']['tok'])} |"
            )
        deer_overall = _mean_cell(shown["deer"]) if len(shown["deer"]) == 5 else ""
        lines.append(
            f"| {zh} | Overall（等权） | — | "
            f"{_mean_cell(shown['full'])} | "
            f"{_mean_cell(shown['puma'])} | "
            f"{deer_overall} | "
            f"{_mean_cell(shown['plws'])} |"
        )

    if too_long:
        lines.extend(
            [
                "",
                "窗后压有 3 题 `too_long`：计为错，token 按缺失的思考/终答计 0，与原汇总一致。",
                "",
            ]
        )
        for item in too_long:
            lines.append(
                f"- {item['model']} {item['dataset']} seed {item['seed']} q{item['question_idx']}"
            )
    TABLE.write_text("\n".join(lines) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "script": "scripts/report_fullcot_puma_plws.py",
                "table": str(TABLE.relative_to(ROOT)),
                "seeds": list(SEEDS),
                "column_order": ["full", "puma", "deer", "plws"],
                "token_policy": TOKEN_POLICY,
                "deer_protocol": "puma-fullcot-32k-v2",
                "deer_rule": "fill only when all four seeds are complete",
                "overall_only_datasets": [dszh for _ds, dszh, _n in OVERALL_ONLY_DATASETS],
                "extra_rule": "7B Overall folds AMC23; AMC23 is a table row; GSM8K is omitted",
                "too_long": too_long,
                "cells": payload_cells,
                "inputs": sorted(set(inputs)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines))
    print(f"wrote {TABLE}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
