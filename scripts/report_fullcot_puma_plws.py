#!/usr/bin/env python3
"""Verify Full-CoT / PUMA / PLWS (窗后压) / aligned DEER on the main table.

Base seeds are 42 / 0 / 1 / 123. Seed 7 is folded in when that seed's
official, PUMA, and PLWS artifacts are complete. Overall is an
equal-weight mean of the listed datasets. Finished contest extras
(AIME26 / AMC23 / BRUMO25 / HMMT25) are shown and folded in. All listed
models use the same dataset set, including MATH / Olympiad / GPQA; a row
is omitted until its four-seed cell is complete. GSM8K is not in the table.
The token column is delivery plus trial answers. PLWS is Wait-suppression
after the first same-answer window, not prefix regeneration. Official CORE
is the leakfix list (``WAIT`` / ``Hmmm`` / leading spaces included).
Official leftover lives in ``shard_*.jsonl``; a leftover
``scores_leakfix.jsonl`` sidecar is still overlaid if present. Aligned DEER
is unified-host ``puma-fullcot-32k-v2``; a cell is filled when the
base four seeds are complete, and seed 7 DEER is added only when that
seed is already in the row. Overall Acc is an equal-weight mean.
Overall token is shown twice: equal-weight mean of dataset tokens, and
the same tokens weighted by each row's n.
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
    ("r1_1p5b", "1.5B"),
    ("r1_llama_8b", "Llama-8B"),
    ("r1_32b", "R1-32B"),
    ("qwen3_4b", "4B"),
    ("qwen3_8b", "8B"),
    ("qwen3_30b_a3b", "30B"),
)
DATASETS = (
    ("math-500", "MATH", 2000),
    ("olympiadbench", "OlympiadBench", 2700),
    ("gpqa-diamond", "GPQA-Diamond", 792),
    ("aime24", "AIME24", 120),
    ("aime25", "AIME25", 120),
)
EXTRA_MODELS = frozenset(
    {"r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b"}
)
OVERALL_ONLY_DATASETS = (
    ("amc23", "AMC23", 160),
)
EXTRA_DATASETS = OVERALL_ONLY_DATASETS
CONTEST_MODELS = frozenset(
    {"qwen3_30b_a3b", "r1_1p5b", "r1_llama_8b", "r1_32b"}
)
CONTEST_FOLLOWON_MODELS = frozenset(
    {"r1_7b", "nemotron_8b", "qwen3_4b", "qwen3_8b", "r1_14b"}
)
CONTEST_DATASETS = (
    ("brumo25", "BRUMO25", 120),
    ("hmmt25", "HMMT25", 120),
    ("aime24", "AIME24", 120),
    ("aime25", "AIME25", 120),
    ("aime26", "AIME26", 120),
)
CONTEST_EXTRA_DATASETS = (
    ("aime26", "AIME26", 120),
    ("brumo25", "BRUMO25", 120),
    ("hmmt25", "HMMT25", 120),
)
SEEDS = (42, 0, 1, 123)
FIFTH_SEED = 7
ALL_SEEDS = SEEDS + (FIFTH_SEED,)
KINDS = (FIRSTWIN, *TIER_KINDS)


def datasets_for(model: str) -> tuple[tuple[str, str, int], ...]:
    wants_contest_tail = (
        model in CONTEST_FOLLOWON_MODELS or model in CONTEST_MODELS
    )
    rows: list[tuple[str, str, int]] = []
    for item in DATASETS:
        rows.append(item)
        if wants_contest_tail and item[0] == "aime25":
            rows.append(("aime26", "AIME26", 120))
    if model in EXTRA_MODELS or model in CONTEST_MODELS:
        rows.extend(OVERALL_ONLY_DATASETS)
    if wants_contest_tail:
        rows.extend(
            (
                ("brumo25", "BRUMO25", 120),
                ("hmmt25", "HMMT25", 120),
            )
        )
    return tuple(rows)


SHARD_RE = re.compile(
    r"^(shard_\d+|scores|scores_shard\d+)\.jsonl$"
)
LEAKFIX_SCORE = "scores_leakfix.jsonl"
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
    def ingest(path: Path, *, overwrite: bool) -> None:
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
            if overwrite or key not in out:
                out[key] = rec

    for path in sorted(folder.iterdir(), key=lambda item: item.name):
        if path.name == LEAKFIX_SCORE:
            continue
        if not SHARD_RE.match(path.name):
            continue
        ingest(path, overwrite=True)
    leakfix = folder / LEAKFIX_SCORE
    if leakfix.is_file():
        ingest(leakfix, overwrite=True)
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


COMPARE = ("full", "puma", "deer", "plws")
OVERALL_EQ_LABEL = "Overall（等权）"
OVERALL_NW_LABEL = "Overall（按题加权）"


def fmt_cell(acc: float, tok: float) -> str:
    return f"{acc:.2f}% / {tok:.0f}"


def fmt_optional(cell: dict | None) -> str:
    if cell is None:
        return ""
    return fmt_cell(cell["acc"], cell["tok"])


def winners(parts: dict) -> tuple[set[str], set[str]]:
    keys = [key for key in COMPARE if parts.get(key)]
    if not keys:
        return set(), set()
    accs = {key: parts[key]["acc"] for key in keys}
    toks = {key: parts[key]["tok"] for key in keys}
    max_acc = max(accs.values())
    min_tok = min(toks.values())
    return (
        {key for key, value in accs.items() if value == max_acc},
        {key for key, value in toks.items() if value == min_tok},
    )


def fmt_highlighted(acc: float, tok: float, acc_win: bool, tok_win: bool) -> str:
    acc_s = f"{acc:.2f}%"
    tok_s = f"{tok:.0f}"
    if acc_win:
        acc_s = f"**{acc_s}**"
    if tok_win:
        tok_s = f"**{tok_s}**"
    return f"{acc_s} / {tok_s}"


def fmt_methods(parts: dict) -> dict[str, str]:
    acc_w, tok_w = winners(parts)
    out: dict[str, str] = {}
    for key in COMPARE:
        cell = parts.get(key)
        if cell is None:
            out[key] = ""
        else:
            out[key] = fmt_highlighted(
                cell["acc"], cell["tok"], key in acc_w, key in tok_w
            )
    return out


def _mean_pair(rows: list[tuple]) -> dict:
    return {
        "acc": sum(row[0] for row in rows) / len(rows),
        "tok": sum(row[1] for row in rows) / len(rows),
    }


def _n_weighted_pair(rows: list[tuple[float, float, int]]) -> dict:
    total_n = sum(n for _acc, _tok, n in rows)
    return {
        "acc": sum(acc for acc, _tok, _n in rows) / len(rows),
        "tok": sum(tok * n for _acc, tok, n in rows) / total_n,
        "n": total_n,
    }


def method_n(row: dict, key: str) -> int:
    """Per-method question count for Overall token weights.

    DEER can stay on four seeds after seed 7 is folded into the row.
    """
    if key != "deer" or row.get("deer") is None:
        return int(row["n"])
    deer = row["deer"]
    if deer.get("n") is not None:
        return int(deer["n"])
    deer_seeds = row.get("deer_seeds") or []
    row_seeds = row.get("seeds") or []
    if deer_seeds and row_seeds and len(deer_seeds) != len(row_seeds):
        return int(round(int(row["n"]) * len(deer_seeds) / len(row_seeds)))
    return int(row["n"])


def shown_row(row: dict, key: str) -> tuple[float, float, int] | None:
    cell = row.get(key)
    if cell is None:
        return None
    return (float(cell["acc"]), float(cell["tok"]), method_n(row, key))


def collect_shown(rows: list[dict]) -> dict[str, list[tuple[float, float, int]]]:
    shown: dict[str, list[tuple[float, float, int]]] = {key: [] for key in COMPARE}
    for row in rows:
        for key in COMPARE:
            item = shown_row(row, key)
            if item is not None:
                shown[key].append(item)
    return shown


def overall_method_cells(
    shown: dict[str, list[tuple[float, float, int]]],
) -> tuple[dict, dict]:
    def one(fn, key: str, require_five: bool = False):
        rows = shown[key]
        if not rows:
            return None
        if require_five and len(rows) != 5:
            return None
        return fn(rows)

    equal = {
        key: one(_mean_pair, key, require_five=(key == "deer")) for key in COMPARE
    }
    weighted = {
        key: one(_n_weighted_pair, key, require_five=(key == "deer"))
        for key in COMPARE
    }
    return equal, weighted


def format_overall_md_row(
    zh: str,
    label: str,
    cells: dict,
    *,
    extra: list[str] | None = None,
) -> str:
    texts = fmt_methods(cells)
    n = "—" if label == OVERALL_EQ_LABEL else str(cells["full"]["n"])
    mid = f" {' | '.join(extra)} |" if extra else ""
    return (
        f"| {zh} | {label} | {n} |{mid} "
        f"{texts['full']} | "
        f"{texts['puma']} | "
        f"{texts['deer']} | "
        f"{texts['plws']} |"
    )


def append_overall_md_rows(
    lines: list[str],
    zh: str,
    rows: list[dict],
    *,
    extra: list[str] | None = None,
) -> None:
    equal, weighted = overall_method_cells(collect_shown(rows))
    lines.append(format_overall_md_row(zh, OVERALL_EQ_LABEL, equal, extra=extra))
    lines.append(format_overall_md_row(zh, OVERALL_NW_LABEL, weighted, extra=extra))


def deer_seed_n(expect_n: int) -> int:
    return expect_n // len(SEEDS)


def seed_official_puma_ready(
    paths: PLWSPaths, model: str, dataset: str, seed: int, per_n: int
) -> bool:
    official_path = paths.puma_statistics_path(model, dataset, seed)
    if not official_path.is_file():
        return False
    official = load_json(official_path)
    if len(official) != per_n:
        return False
    delivery_path = puma_delivery_path(paths, model, dataset, seed)
    if not delivery_path.is_file():
        return False
    delivery = load_json(delivery_path)
    rows = delivery if isinstance(delivery, list) else delivery.get("rows", [])
    return len(rows) == per_n


def seed_plws_ready(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    *,
    allow_legacy: bool = False,
) -> bool:
    official_path = paths.puma_statistics_path(model, dataset, seed)
    if not official_path.is_file():
        return False
    official = {int(row["question_idx"]): row for row in load_json(official_path)}
    windowed = {
        int(job["question_idx"]): job
        for job in job_rows(paths, model, dataset, seed)
        if job.get("uid")
    }
    scores = collect_plws_scores(
        paths, model, dataset, seed, [], allow_legacy=allow_legacy
    )
    for question_idx in official:
        if scores.get((dataset, int(question_idx))) is not None:
            continue
        if int(question_idx) not in windowed:
            continue
        return False
    return True


def four_seed_cell_complete(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    expect_n: int,
    *,
    allow_legacy: bool = False,
) -> bool:
    """Official + PUMA + leftover must all be ready on the base four seeds."""
    per_n = deer_seed_n(expect_n)
    return all(
        seed_official_puma_ready(paths, model, dataset, seed, per_n)
        and seed_plws_ready(
            paths, model, dataset, seed, allow_legacy=allow_legacy
        )
        for seed in SEEDS
    )


def seeds_for_cell(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    expect_n: int,
    *,
    allow_legacy: bool = False,
) -> tuple[int, ...]:
    """Base four seeds, plus seed 7 when that seed is fully scored."""
    per_n = deer_seed_n(expect_n)
    used = list(SEEDS)
    if seed_official_puma_ready(
        paths, model, dataset, FIFTH_SEED, per_n
    ) and seed_plws_ready(
        paths, model, dataset, FIFTH_SEED, allow_legacy=allow_legacy
    ):
        used.append(FIFTH_SEED)
    return tuple(used)


def four_seed_deer_complete(
    paths: PLWSPaths, model: str, dataset: str, expect_n: int
) -> bool:
    expected = deer_seed_n(expect_n)
    return all(
        deer_complete(paths, model, dataset, seed, expected=expected)[0]
        for seed in SEEDS
    )


def deer_seeds_for_cell(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    expect_n: int,
    used_seeds: tuple[int, ...],
) -> tuple[int, ...]:
    if not four_seed_deer_complete(paths, model, dataset, expect_n):
        return ()
    seeds = list(SEEDS)
    per_n = deer_seed_n(expect_n)
    if FIFTH_SEED in used_seeds and deer_complete(
        paths, model, dataset, FIFTH_SEED, expected=per_n
    )[0]:
        seeds.append(FIFTH_SEED)
    return tuple(seeds)


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


def _deer_fingerprint(
    paths: PLWSPaths, model: str, dataset: str, seeds: tuple[int, ...]
) -> str:
    parts: list[str] = []
    for seed in seeds:
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
    used_seeds: tuple[int, ...],
) -> dict | None:
    deer_seeds = deer_seeds_for_cell(paths, model, dataset, expect_n, used_seeds)
    if not deer_seeds:
        return None
    deer_n = deer_seed_n(expect_n) * len(deer_seeds)
    key = f"{model}/{dataset}"
    fingerprint = f"{TOKEN_POLICY}|{_deer_fingerprint(paths, model, dataset, deer_seeds)}"
    hit = cache.get(key)
    if (
        isinstance(hit, dict)
        and hit.get("fingerprint") == fingerprint
        and hit.get("n") == deer_n
        and hit.get("token_policy") == TOKEN_POLICY
    ):
        for seed in deer_seeds:
            path = deer_output_dir(paths, model, dataset, seed) / "deer.jsonl"
            inputs.append(str(path.relative_to(paths.root)))
        return {"n": hit["n"], "acc": hit["acc"], "tok": hit["tok"]}
    items: list[tuple[str, str, str, float]] = []
    for seed in deer_seeds:
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
    if len(items) != deer_n:
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
    out = {"acc": round(cell["acc"], 2), "tok": round(cell["tok"])}
    if cell.get("n") is not None:
        out["n"] = int(cell["n"])
    return out


def _mean_cell(rows: list[tuple[float, float]]) -> str:
    return fmt_cell(
        sum(acc for acc, _ in rows) / len(rows),
        sum(tok for _, tok in rows) / len(rows),
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
    used_seeds: tuple[int, ...] | None = None,
) -> dict[str, dict | None]:
    full: list[tuple[bool, float]] = []
    puma: list[tuple[bool, float]] = []
    plws: list[tuple[bool, float]] = []
    if used_seeds is None:
        used_seeds = seeds_for_cell(
            paths, model, dataset, expect_n, allow_legacy=allow_legacy
        )
    actual_n = deer_seed_n(expect_n) * len(used_seeds)
    for seed in used_seeds:
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
        if n != actual_n:
            raise RuntimeError(f"{zh} {dszh} expected n={actual_n}, got {n}")
        return {
            "n": n,
            "acc": 100.0 * sum(ok for ok, _ in rows) / n,
            "tok": sum(tok for _, tok in rows) / n,
        }

    deer = (
        load_deer_cell(
            paths, model, dataset, expect_n, inputs, deer_cache, used_seeds
        )
        if want_deer
        else None
    )
    return {
        "full": agg(full),
        "puma": agg(puma),
        "plws": agg(plws),
        "deer": deer,
        "seeds": list(used_seeds),
        "deer_seeds": (
            list(deer_seeds_for_cell(paths, model, dataset, expect_n, used_seeds))
            if deer is not None
            else []
        ),
    }


def render_markdown_from_payload(payload: dict) -> str:
    header = [
        "# Full-CoT / PUMA / DEER / 窗后压",
        "",
        "由 `scripts/update_fullcot_puma_plws.sh` 从规范产物重算并写飞书表。",
        "默认四 seed：0 / 1 / 42 / 123。seed 7 的官方 / PUMA / 窗后压都齐了再折进该行，n 跟着变成五 seed。",
        "齐的格子才填；缺四 seed 的 DEER 留空。五行已是五 seed、但 seed 7 DEER 未齐时，DEER 仍按四 seed。",
        "Overall Acc 仍是表内各集等权平均。Overall token 先并列两行：等权平均 token，以及按该行 n 加权的平均 token。",
        "已齐的 AIME26 / AMC23 / BRUMO25 / HMMT25 进表并折进 Overall。GSM8K 不进表。",
        "1.5B / Llama-8B / R1-32B / 30B 与主五行同一套集（含 MATH / Olympiad / GPQA）；四 seed 未齐的行不出现。",
        "Qwen3-32B 未齐，不进。30B 与三个新 R1 蒸馏这轮没跑 DEER，格子留空。",
        "AMC23 窗后压按现行规则从已有 leftover 分重算，与主表同一套 Acc / 交付+试答。",
        "PLWS 主五行只收 `puma-fullcot-32k-v2` firstwin 分；正式 leftover 只有修复 CORE 的 `shard_*.jsonl`。无第一扇窗的题贴 Full-CoT。",
        "7B / Nemotron / 14B 的 Full-CoT 与 PUMA 沿用已有 host，未按 v2 重跑；",
        "1.5B / Llama-8B / R1-32B 的 Full-CoT / PUMA / PLWS 都是 v2。",
        "PLWS 与 DEER 是 v2（37888）。4B / 8B 的 Full-CoT / PUMA 来自 `puma_offline`。",
        "DEER 是统一 host 的 aligned v2，不是官方 greedy / 16k。",
        "某个模型×数据集的 DEER 只有四 seed 都齐才填，否则留空。",
        "Overall 的 DEER 仅当五个数据集都齐时才填。",
        "预实验数字不进本表。",
        "Token 只有一列：交付 + 试答。试答只数 boxed 内部（PUMA / 窗后压），",
        "DEER 用记录的 `num_trial_answer_tokens`（R1/Nemotron 约等于 boxed；",
        "Qwen3 probe 会写到 `</think>`，试答比 boxed 大）。",
        "同行 Acc 最高、token 最低加粗；Full-CoT 也进比较，并列都加粗。",
        "",
        "- Full-CoT：`original_correct` / `original_tokens`（无试答）",
        "- PUMA：`compressed_tokens + tokens_trial_answers`",
        "- DEER：`delivery_tokens + num_trial_answer_tokens`；四 seed 齐才填，seed 7 齐了再折进",
        "- 窗后压：思考 + 终答 + 第一扇窗末步及之前的 `count_answer_tokens`；无窗贴 Full-CoT、试答记 0；固定最后一列",
        "",
        "| 模型 | 集 | n | Full-CoT（原始 CoT） | PUMA | DEER | 窗后压 |",
        "|---|---|---:|---|---|---|---|",
    ]
    lines = list(header)
    by_model: dict[str, list[dict]] = {}
    for row in payload["cells"]:
        by_model.setdefault(row["model"], []).append(row)
    for _tag, zh in MODELS:
        rows = by_model.get(zh, [])
        if not rows:
            continue
        for row in rows:
            texts = fmt_methods(row)
            lines.append(
                f"| {zh} | {row['dataset']} | {row['n']} | "
                f"{texts['full']} | "
                f"{texts['puma']} | "
                f"{texts['deer']} | "
                f"{texts['plws']} |"
            )
        append_overall_md_rows(lines, zh, rows)
    too_long = payload.get("too_long") or []
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
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--from-json":
        payload = json.loads(REPORT.read_text())
        TABLE.write_text(render_markdown_from_payload(payload))
        print(f"rewrote {TABLE} from {REPORT}")
        return
    paths = PLWSPaths.discover(ROOT)
    cells: dict[tuple[str, str], dict[str, dict | None]] = {}
    too_long: list[dict] = []
    inputs: list[str] = []
    deer_cache = _load_deer_cache()

    for model, zh in MODELS:
        for dataset, dszh, expect_n in datasets_for(model):
            allow_legacy = (dataset, dszh, expect_n) in OVERALL_ONLY_DATASETS
            if not four_seed_cell_complete(
                paths,
                model,
                dataset,
                expect_n,
                allow_legacy=allow_legacy,
            ):
                print(f"skip {zh} {dszh}: four-seed cell incomplete", flush=True)
                continue
            print(f"score {zh} {dszh}", flush=True)
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
                allow_legacy=allow_legacy,
                want_deer=not allow_legacy,
            )

    lines = [
        "# Full-CoT / PUMA / DEER / 窗后压",
        "",
        "由 `scripts/update_fullcot_puma_plws.sh` 从规范产物重算并写飞书表。",
        "默认四 seed：0 / 1 / 42 / 123。seed 7 的官方 / PUMA / 窗后压都齐了再折进该行，n 跟着变成五 seed。",
        "齐的格子才填；缺四 seed 的 DEER 留空。五行已是五 seed、但 seed 7 DEER 未齐时，DEER 仍按四 seed。",
        "Overall Acc 仍是表内各集等权平均。Overall token 先并列两行：等权平均 token，以及按该行 n 加权的平均 token。",
        "已齐的 AIME26 / AMC23 / BRUMO25 / HMMT25 进表并折进 Overall。GSM8K 不进表。",
        "1.5B / Llama-8B / R1-32B / 30B 与主五行同一套集（含 MATH / Olympiad / GPQA）；四 seed 未齐的行不出现。",
        "Qwen3-32B 未齐，不进。30B 与三个新 R1 蒸馏这轮没跑 DEER，格子留空。",
        "AMC23 窗后压按现行规则从已有 leftover 分重算，与主表同一套 Acc / 交付+试答。",
        "PLWS 主五行只收 `puma-fullcot-32k-v2` firstwin 分；正式 leftover 只有修复 CORE 的 `shard_*.jsonl`。无第一扇窗的题贴 Full-CoT。",
        "7B / Nemotron / 14B 的 Full-CoT 与 PUMA 沿用已有 host，未按 v2 重跑；",
        "1.5B / Llama-8B / R1-32B 的 Full-CoT / PUMA / PLWS 都是 v2。",
        "PLWS 与 DEER 是 v2（37888）。4B / 8B 的 Full-CoT / PUMA 来自 `puma_offline`。",
        "DEER 是统一 host 的 aligned v2，不是官方 greedy / 16k。",
        "某个模型×数据集的 DEER 只有四 seed 都齐才填，否则留空。",
        "Overall 的 DEER 仅当五个数据集都齐时才填。",
        "预实验数字不进本表。",
        "Token 只有一列：交付 + 试答。试答只数 boxed 内部（PUMA / 窗后压），",
        "DEER 用记录的 `num_trial_answer_tokens`（R1/Nemotron 约等于 boxed；",
        "Qwen3 probe 会写到 `</think>`，试答比 boxed 大）。",
        "同行 Acc 最高、token 最低加粗；Full-CoT 也进比较，并列都加粗。",
        "",
        "- Full-CoT：`original_correct` / `original_tokens`（无试答）",
        "- PUMA：`compressed_tokens + tokens_trial_answers`",
        "- DEER：`delivery_tokens + num_trial_answer_tokens`；四 seed 齐才填，seed 7 齐了再折进",
        "- 窗后压：思考 + 终答 + 第一扇窗末步及之前的 `count_answer_tokens`；无窗贴 Full-CoT、试答记 0；固定最后一列",
        "",
        "| 模型 | 集 | n | Full-CoT（原始 CoT） | PUMA | DEER | 窗后压 |",
        "|---|---|---:|---|---|---|---|",
    ]
    payload_cells = []
    for model, zh in MODELS:
        model_rows = []
        for dataset, dszh, expect_n in datasets_for(model):
            hidden_row = (dataset, dszh, expect_n) in OVERALL_ONLY_DATASETS
            cell = cells.get((zh, dszh))
            if cell is None:
                continue
            n = int(cell["full"]["n"])
            row = {
                "model": zh,
                "dataset": dszh,
                "n": n,
                "seeds": cell["seeds"],
                "deer_seeds": cell["deer_seeds"],
                "full": _round_cell(cell["full"]),
                "puma": _round_cell(cell["puma"]),
                "plws": _round_cell(cell["plws"]),
                "deer": None if cell["deer"] is None else _round_cell(cell["deer"]),
                "overall_only": hidden_row,
            }
            payload_cells.append(row)
            model_rows.append(row)
            texts = fmt_methods(
                {
                    "full": row["full"],
                    "puma": row["puma"],
                    "deer": row["deer"],
                    "plws": row["plws"],
                }
            )
            lines.append(
                f"| {zh} | {dszh} | {n} | "
                f"{texts['full']} | "
                f"{texts['puma']} | "
                f"{texts['deer']} | "
                f"{texts['plws']} |"
            )
        if model_rows:
            append_overall_md_rows(lines, zh, model_rows)

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
                "fifth_seed": FIFTH_SEED,
                "column_order": ["full", "puma", "deer", "plws"],
                "token_policy": TOKEN_POLICY,
                "deer_protocol": "puma-fullcot-32k-v2",
                "deer_rule": "fill when four seeds are complete; fold seed 7 when that seed is already in the row",
                "overall_only_datasets": [dszh for _ds, dszh, _n in OVERALL_ONLY_DATASETS],
                "contest_datasets": [dszh for _ds, dszh, _n in CONTEST_DATASETS],
                "extra_rule": "finished AIME26/AMC23/BRUMO/HMMT fold into Overall; Acc equal-weight; token shown as equal-weight and n-weighted; all models include MATH/Olympiad/GPQA when four-seed complete; Qwen3-32B omitted; GSM8K omitted",
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
