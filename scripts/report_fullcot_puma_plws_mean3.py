#!/usr/bin/env python3
"""Official mean@3 main table: seeds 42 / 0 / 1, five datasets.

A method cell is filled only when all three seeds are complete.
TR is token reduction versus Full-CoT on that same row.
Overall is emitted only when Full-CoT / PUMA / 窗后压 are complete
on every dataset for that model. DEER Overall additionally requires
DEER complete on every dataset.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.extra_baselines import (
    extra_baseline_complete,
    extra_baseline_output_dir,
)
from plws.grading import require_grader
from plws.matrix import deer_complete, deer_output_dir
from plws.paths import PLWSPaths
from report_fullcot_puma_plws import (
    TOKEN_POLICY,
    _deer_fingerprint,
    _deer_token,
    grade_deer_item,
    leftover_grades_verified,
    puma_delivery_path,
    puma_statistics_verified,
    puma_token,
    score_model_dataset,
    seed_official_puma_ready,
    seed_plws_ready,
)

TABLE = ROOT / "tables" / "firstwin_wait" / "fullcot_puma_plws_mean3.md"
REPORT = ROOT / "results" / "reports" / "fullcot_puma_plws_mean3.json"
OVERRIDE = ROOT / "results" / "reports" / "mean3_remote_overrides.json"
DEER_CACHE = ROOT / "results" / "reports" / "cache" / "deer_aligned_v2_mean3.json"

SEEDS = (42, 0, 1)
# User 2026-09-21: Nemotron MATH-500 uses 0/1/123; AIME25 Full/PUMA/窗后压 uses 1/123/7.
METHOD_SEEDS: dict[tuple[str, str], tuple[int, ...]] = {
    ("nemotron_8b", "math-500"): (0, 1, 123),
    ("nemotron_8b", "aime25"): (1, 123, 7),
    ("qwen3_8b", "math-500"): (42, 0, 123),
    ("qwen3_8b", "gpqa-diamond"): (0, 1, 123),
    ("qwen3_8b", "aime25"): (0, 123, 7),
    ("qwen3_8b", "amc23"): (0, 123, 7),
    ("r1_1p5b", "aime25"): (0, 1, 7),
}
# AIME25 seed 7 has no DEER; keep that cell on mean@3 seeds.
DEER_SEEDS: dict[tuple[str, str], tuple[int, ...]] = {
    ("nemotron_8b", "math-500"): (0, 1, 123),
}
# User 2026-09-21: Qwen3-4B AMC23 leftover only.
PLWS_SEEDS: dict[tuple[str, str], tuple[int, ...]] = {
    ("qwen3_4b", "amc23"): (0, 123, 7),
}
MODELS = (
    ("r1_1p5b", "1.5B"),
    ("r1_7b", "7B"),
    ("r1_14b", "14B"),
    ("r1_llama_8b", "Llama-8B"),
    ("nemotron_8b", "Nemotron"),
    ("qwen3_4b", "4B"),
    ("qwen3_8b", "8B"),
)
REMOTE_MODELS = (
    ("qwen3_30b_a3b", "Qwen3-30B-A3B"),
    ("r1_32b", "R1-32B"),
    ("qwen3_32b", "Qwen3-32B"),
    ("qwq_32b", "QwQ-32B"),
)
DATASETS = (
    ("math-500", "MATH-500", 500),
    ("olympiadbench", "OlympiadBench", 675),
    ("gpqa-diamond", "GPQA-Diamond", 198),
    ("aime25", "AIME25", 30),
    ("amc23", "AMC23", 40),
)
DATASET_IDS = {item[0] for item in DATASETS}
COMPARE = ("full", "ac", "dynasor", "puma", "deer", "plws")
EXTRA_METHODS = {
    "ac": "answer_convergence",
    "dynasor": "dynasor",
}
OVERALL_EQ = "Overall（等权）"
OVERALL_NW = "Overall（按题加权）"


def selected_models() -> tuple[tuple[str, str], ...]:
    raw = os.environ.get("MEAN3_MODELS", "").strip()
    if not raw:
        return MODELS
    want = {part.strip() for part in raw.split(",") if part.strip()}
    chosen = tuple(item for item in MODELS if item[0] in want or item[1] in want)
    if not chosen:
        raise SystemExit(f"MEAN3_MODELS={raw!r} matched nothing")
    return chosen


def merge_payload(new_rows: list[dict], scored_tags: set[str]) -> list[dict]:
    if not REPORT.is_file():
        return new_rows
    old = json.loads(REPORT.read_text())
    kept = [
        cell
        for cell in old.get("cells", [])
        if cell.get("model_tag") not in scored_tags
    ]
    merged: list[dict] = []
    by_tag: dict[str, list[dict]] = {}
    for cell in kept + new_rows:
        if cell.get("dataset_id") not in DATASET_IDS:
            continue
        by_tag.setdefault(str(cell["model_tag"]), []).append(cell)
    for tag, _zh in MODELS:
        merged.extend(by_tag.get(tag, []))
    return merged


def load_overrides() -> dict[tuple[str, str], dict]:
    if not OVERRIDE.is_file():
        return {}
    data = json.loads(OVERRIDE.read_text())
    out: dict[tuple[str, str], dict] = {}
    for item in data.get("cells", []):
        out[(str(item["model_tag"]), str(item["dataset_id"]))] = item
    return out


def apply_cell_override(cell: dict, ov: dict) -> dict:
    merged = dict(cell)
    for key in ("full", "puma", "plws"):
        src = ov.get(key)
        if not src:
            continue
        prev = cell.get(key) or {}
        merged[key] = {
            "n": int(src.get("n") or prev.get("n") or 0),
            "acc": float(src["acc"]),
            "tok": float(src["tok"]),
        }
    merged["source"] = ov.get("source", "remote_override")
    parts = {key: merged.get(key) for key in COMPARE}
    mark_winners(parts)
    for key in COMPARE:
        if parts.get(key) is not None:
            merged[key] = parts[key]
    return merged


def official_cells(cells: list[dict]) -> list[dict]:
    overrides = load_overrides()
    by_tag: dict[str, dict[str, dict]] = {}
    for cell in cells:
        dataset_id = str(cell.get("dataset_id") or "")
        if dataset_id not in DATASET_IDS:
            continue
        by_tag.setdefault(str(cell["model_tag"]), {})[dataset_id] = cell
    ordered: list[dict] = []
    for tag, _zh in MODELS:
        per_ds = by_tag.get(tag, {})
        for dataset_id, _dszh, _per_n in DATASETS:
            if dataset_id not in per_ds:
                continue
            cell = per_ds[dataset_id]
            ov = overrides.get((tag, dataset_id))
            if ov:
                cell = apply_cell_override(cell, ov)
            ordered.append(cell)
    return ordered


def rows_from_cells(cells: list[dict]) -> dict[str, list[dict]]:
    rows_by_model: dict[str, list[dict]] = {}
    for cell in official_cells(cells):
        rows_by_model.setdefault(cell["model"], []).append(cell)
    return rows_by_model


def write_outputs(payload_rows: list[dict], rows_by_model: dict[str, list[dict]]) -> None:
    text = render(rows_by_model)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(text)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "protocol_id": "puma-fullcot-32k-v2",
                "seeds": list(SEEDS),
                "models": [item[0] for item in MODELS],
                "remote_models": [item[0] for item in REMOTE_MODELS],
                "datasets": [item[0] for item in DATASETS],
                "token_policy": TOKEN_POLICY,
                "cells": official_cells(payload_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(text)
    print(f"wrote {TABLE}")
    print(f"wrote {REPORT}")


def attach_extra_columns(cells: list[dict]) -> list[dict]:
    updated: list[dict] = []
    ds_n = {dataset: per_n for dataset, _dszh, per_n in DATASETS}
    for cell in cells:
        model = str(cell["model_tag"])
        dataset = str(cell["dataset_id"])
        per_n = int(ds_n[dataset])
        counts = dict(cell.get("counts") or {})
        counts.update(extra_ready_counts(model, dataset))
        extra_used = extra_seeds_for(model, dataset)
        attached = dict(cell)
        attached["counts"] = counts
        attached["extra_seeds"] = list(extra_used)
        for key in EXTRA_METHODS:
            attached[key] = (
                load_extra_mean3(model, dataset, per_n, key)
                if counts[key] == 3
                else None
            )
        parts = {name: attached.get(name) for name in COMPARE}
        mark_winners(parts)
        for name in COMPARE:
            if parts.get(name) is not None:
                attached[name] = parts[name]
        updated.append(attached)
    return updated


def patch_extra_columns() -> None:
    if not REPORT.is_file():
        raise SystemExit(f"missing {REPORT}")
    payload = json.loads(REPORT.read_text())
    payload_rows = attach_extra_columns(official_cells(payload.get("cells", [])))
    write_outputs(payload_rows, rows_from_cells(payload_rows))


def rebuild_from_report() -> None:
    if not REPORT.is_file():
        raise SystemExit(f"missing {REPORT}")
    payload = json.loads(REPORT.read_text())
    payload_rows = official_cells(payload.get("cells", []))
    rows_by_model = rows_from_cells(payload_rows)
    write_outputs(payload_rows, rows_by_model)


def patch_method_seed_overrides() -> None:
    """Rescore only METHOD_SEEDS cells; keep every other mean@3 cell."""
    require_grader()
    paths = PLWSPaths.discover(ROOT)
    deer_cache = load_deer_cache()
    if not REPORT.is_file():
        raise SystemExit(f"missing {REPORT}")
    payload = json.loads(REPORT.read_text())
    cells = list(payload.get("cells", []))
    index = {
        (str(cell["model_tag"]), str(cell["dataset_id"])): i
        for i, cell in enumerate(cells)
    }
    zh_by = {tag: zh for tag, zh in MODELS}
    ds_by = {dataset: (dszh, per_n) for dataset, dszh, per_n in DATASETS}
    keys = dict(METHOD_SEEDS)
    keys.update(PLWS_SEEDS)
    for (model, dataset), seeds in keys.items():
        key = (model, dataset)
        if key not in index:
            raise SystemExit(f"missing mean@3 cell {model} {dataset}")
        dszh, per_n = ds_by[dataset]
        print(
            f"patch {zh_by[model]} {dszh} method={list(method_seeds(model, dataset))} "
            f"plws={list(plws_seeds_for(model, dataset))}",
            flush=True,
        )
        cells[index[key]] = score_row(
            paths, model, zh_by[model], dataset, dszh, per_n, deer_cache
        )
    write_outputs(cells, rows_from_cells(cells))


def method_seeds(model: str, dataset: str) -> tuple[int, ...]:
    return METHOD_SEEDS.get((model, dataset), SEEDS)


def deer_seeds_for(model: str, dataset: str) -> tuple[int, ...]:
    return DEER_SEEDS.get((model, dataset), SEEDS)


def extra_seeds_for(_model: str, _dataset: str) -> tuple[int, ...]:
    return SEEDS


def plws_seeds_for(model: str, dataset: str) -> tuple[int, ...]:
    return PLWS_SEEDS.get((model, dataset), method_seeds(model, dataset))


def allow_legacy(dataset: str) -> bool:
    return dataset == "amc23"


def expect_n4(per_n: int) -> int:
    return per_n * 4


def ready_counts(
    paths: PLWSPaths, model: str, dataset: str, per_n: int
) -> dict[str, int]:
    legacy = allow_legacy(dataset)
    official = 0
    plws = 0
    deer = 0
    used = method_seeds(model, dataset)
    for seed in used:
        if seed_official_puma_ready(
            paths, model, dataset, seed, per_n
        ) and puma_statistics_verified(paths, model, dataset, seed):
            official += 1
    for seed in plws_seeds_for(model, dataset):
        if seed_plws_ready(
            paths, model, dataset, seed, allow_legacy=legacy
        ) and leftover_grades_verified(
            paths, model, dataset, seed, allow_legacy=legacy
        ):
            plws += 1
    for seed in deer_seeds_for(model, dataset):
        if deer_complete(paths, model, dataset, seed, expected=per_n)[0]:
            deer += 1
    extra = extra_ready_counts(model, dataset)
    return {
        "full": official,
        "puma": official,
        "plws": plws,
        "deer": deer,
        **extra,
    }


def extra_ready_counts(model: str, dataset: str) -> dict[str, int]:
    used = extra_seeds_for(model, dataset)
    out: dict[str, int] = {}
    for key, method in EXTRA_METHODS.items():
        out[key] = sum(
            1
            for seed in used
            if extra_baseline_complete(ROOT, method, model, dataset, seed)[0]
        )
    return out


def extra_summary_token(method: str, summary: dict) -> float:
    if method == "answer_convergence":
        return float(summary["avg_total_generated_tokens"])
    return float(summary["avg_total_method_tokens"])


def load_extra_mean3(
    model: str, dataset: str, per_n: int, key: str
) -> dict | None:
    method = EXTRA_METHODS[key]
    used = extra_seeds_for(model, dataset)
    if not all(
        extra_baseline_complete(ROOT, method, model, dataset, seed)[0]
        for seed in used
    ):
        return None
    accs: list[float] = []
    toks: list[float] = []
    ns: list[int] = []
    for seed in used:
        summary = json.loads(
            (
                extra_baseline_output_dir(ROOT, method, model, dataset, seed)
                / "summary.json"
            ).read_text()
        )
        n = int(summary["n"])
        if n != per_n:
            return None
        accs.append(float(summary["accuracy"]))
        toks.append(extra_summary_token(method, summary))
        ns.append(n)
    return {
        "n": sum(ns),
        "acc": sum(accs) / len(accs),
        "tok": sum(toks) / len(toks),
    }


def incomplete_text(have: int) -> str:
    return f"未齐({have}/3seed)"


def tr_pct(full_tok: float, method_tok: float) -> float:
    if full_tok <= 0:
        return 0.0
    return 100.0 * (full_tok - method_tok) / full_tok


def fmt_tr(value: float) -> str:
    return f"{value:.1f}%"


def fmt_cell(cell: dict, full: dict | None, *, kind: str) -> str:
    acc = f"{cell['acc']:.2f}%"
    tok = f"{cell['tok']:.0f}"
    if cell.get("acc_win"):
        acc = f"**{acc}**"
    if cell.get("tok_win"):
        tok = f"**{tok}**"
    if kind == "full" or full is None:
        return f"{acc} / {tok} / —"
    tr = fmt_tr(tr_pct(full["tok"], cell["tok"]))
    if cell.get("tr_win"):
        tr = f"**{tr}**"
    return f"{acc} / {tok} / {tr}"


def mark_winners(parts: dict[str, dict | None]) -> None:
    present = {key: parts[key] for key in COMPARE if parts.get(key)}
    if not present:
        return
    max_acc = max(cell["acc"] for cell in present.values())
    min_tok = min(cell["tok"] for cell in present.values())
    full = parts.get("full")
    trs = {
        key: tr_pct(full["tok"], cell["tok"])
        for key, cell in present.items()
        if key != "full" and full is not None
    }
    max_tr = max(trs.values()) if trs else None
    for key, cell in present.items():
        cell["acc_win"] = cell["acc"] == max_acc
        cell["tok_win"] = cell["tok"] == min_tok
        cell["tr_win"] = max_tr is not None and key in trs and trs[key] == max_tr


def load_deer_cache() -> dict:
    if not DEER_CACHE.is_file():
        return {}
    return json.loads(DEER_CACHE.read_text())


def save_deer_cache(cache: dict) -> None:
    DEER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DEER_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")


def load_deer_mean3(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    per_n: int,
    cache: dict,
    used: tuple[int, ...] | None = None,
) -> dict | None:
    if used is None:
        used = SEEDS
    if not all(
        deer_complete(paths, model, dataset, seed, expected=per_n)[0]
        for seed in used
    ):
        return None
    deer_n = per_n * len(used)
    key = f"{model}/{dataset}/{'+'.join(str(s) for s in used)}"
    fingerprint = f"{TOKEN_POLICY}|{_deer_fingerprint(paths, model, dataset, used)}"
    hit = cache.get(key)
    if (
        isinstance(hit, dict)
        and hit.get("fingerprint") == fingerprint
        and hit.get("n") == deer_n
        and hit.get("token_policy") == TOKEN_POLICY
    ):
        return {"n": hit["n"], "acc": hit["acc"], "tok": hit["tok"]}
    items: list[tuple[str, str, str, float]] = []
    for seed in used:
        path = deer_output_dir(paths, model, dataset, seed) / "deer.jsonl"
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
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(grade_deer_item, items, chunksize=16))
    cell = {
        "n": len(rows),
        "acc": 100.0 * sum(ok for ok, _ in rows) / len(rows),
        "tok": sum(tok for _, tok in rows) / len(rows),
    }
    cache[key] = {**cell, "fingerprint": fingerprint, "token_policy": TOKEN_POLICY}
    save_deer_cache(cache)
    return cell


def score_full_puma(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    per_n: int,
    used: tuple[int, ...] | None = None,
) -> dict[str, dict]:
    full: list[tuple[bool, float]] = []
    puma: list[tuple[bool, float]] = []
    if used is None:
        used = SEEDS
    for seed in used:
        official_path = paths.puma_statistics_path(model, dataset, seed)
        official = json.loads(official_path.read_text())
        delivery = json.loads(puma_delivery_path(paths, model, dataset, seed).read_text())
        delivery_rows = (
            delivery if isinstance(delivery, list) else delivery.get("rows", [])
        )
        for row in official:
            full.append(
                (
                    bool(row.get("original_correct")),
                    float(row.get("original_tokens") or 0),
                )
            )
        for row in delivery_rows:
            puma.append((bool(row.get("compressed_correct")), puma_token(row)))
    actual = per_n * len(used)
    if len(full) != actual or len(puma) != actual:
        raise RuntimeError(
            f"{model} {dataset} full/puma n={len(full)}/{len(puma)} expected {actual}"
        )

    def agg(rows: list[tuple[bool, float]]) -> dict:
        return {
            "n": len(rows),
            "acc": 100.0 * sum(ok for ok, _ in rows) / len(rows),
            "tok": sum(tok for _, tok in rows) / len(rows),
        }

    return {"full": agg(full), "puma": agg(puma)}


def score_row(
    paths: PLWSPaths,
    model: str,
    zh: str,
    dataset: str,
    dszh: str,
    per_n: int,
    deer_cache: dict,
) -> dict:
    used = method_seeds(model, dataset)
    plws_used = plws_seeds_for(model, dataset)
    counts = ready_counts(paths, model, dataset, per_n)
    parts: dict[str, dict | None] = {
        "full": None,
        "puma": None,
        "deer": None,
        "ac": None,
        "dynasor": None,
        "plws": None,
    }
    if counts["full"] == 3 and counts["plws"] == 3 and plws_used == used:
        scored = score_model_dataset(
            paths,
            model,
            zh,
            dataset,
            dszh,
            expect_n4(per_n),
            [],
            [],
            {},
            allow_legacy=allow_legacy(dataset),
            want_deer=False,
            used_seeds=used,
        )
        parts["full"] = scored["full"]
        parts["puma"] = scored["puma"]
        parts["plws"] = scored["plws"]
    elif counts["full"] == 3:
        scored_fp = score_full_puma(paths, model, dataset, per_n, used)
        parts["full"] = scored_fp["full"]
        parts["puma"] = scored_fp["puma"]
        if counts["plws"] == 3:
            parts["plws"] = score_model_dataset(
                paths,
                model,
                zh,
                dataset,
                dszh,
                expect_n4(per_n),
                [],
                [],
                {},
                allow_legacy=allow_legacy(dataset),
                want_deer=False,
                used_seeds=plws_used,
            )["plws"]
    deer_used = deer_seeds_for(model, dataset)
    if counts["deer"] == 3:
        print(f"score DEER {zh} {dszh} seeds={list(deer_used)}", flush=True)
        parts["deer"] = load_deer_mean3(
            paths, model, dataset, per_n, deer_cache, deer_used
        )
        if parts["deer"] is None:
            counts["deer"] = sum(
                1
                for seed in deer_used
                if deer_complete(paths, model, dataset, seed, expected=per_n)[0]
            )
    extra_used = extra_seeds_for(model, dataset)
    for key in EXTRA_METHODS:
        if counts[key] == 3:
            parts[key] = load_extra_mean3(model, dataset, per_n, key)
            if parts[key] is None:
                counts[key] = extra_ready_counts(model, dataset)[key]
    mark_winners(parts)
    return {
        "model": zh,
        "model_tag": model,
        "dataset": dszh,
        "dataset_id": dataset,
        "n": per_n * len(used),
        "method_seeds": list(used),
        "plws_seeds": list(plws_used),
        "deer_seeds": list(deer_used),
        "extra_seeds": list(extra_used),
        "counts": counts,
        "full": parts["full"],
        "puma": parts["puma"],
        "deer": parts["deer"],
        "ac": parts["ac"],
        "dynasor": parts["dynasor"],
        "plws": parts["plws"],
    }


def method_text(row: dict, key: str) -> str:
    cell = row.get(key)
    if cell is None:
        return incomplete_text(int(row.get("counts", {}).get(key, 0)))
    return fmt_cell(cell, row.get("full"), kind=key)


def dataset_complete(row: dict) -> bool:
    return all(row["counts"][key] == 3 for key in ("full", "puma", "plws"))


def mean_pair(rows: list[tuple[float, float]]) -> dict:
    return {
        "acc": sum(acc for acc, _tok in rows) / len(rows),
        "tok": sum(tok for _acc, tok in rows) / len(rows),
    }


def n_weighted_pair(rows: list[tuple[float, float, int]]) -> dict:
    total_n = sum(n for _acc, _tok, n in rows)
    return {
        "n": total_n,
        "acc": sum(acc for acc, _tok, _n in rows) / len(rows),
        "tok": sum(tok * n for _acc, tok, n in rows) / total_n,
    }


def overall_parts(rows: list[dict], *, weighted: bool) -> dict[str, dict | None]:
    parts: dict[str, dict | None] = {key: None for key in COMPARE}
    deer_all = all(row["counts"]["deer"] == 3 and row.get("deer") for row in rows)
    extra_all = {
        key: all(row["counts"].get(key) == 3 and row.get(key) for row in rows)
        for key in EXTRA_METHODS
    }
    for key in COMPARE:
        if key == "deer" and not deer_all:
            continue
        if key in EXTRA_METHODS and not extra_all[key]:
            continue
        items = []
        for row in rows:
            cell = row.get(key)
            if cell is None:
                continue
            items.append((cell["acc"], cell["tok"], int(row["n"])))
        if len(items) != len(rows):
            continue
        parts[key] = n_weighted_pair(items) if weighted else {
            **mean_pair([(acc, tok) for acc, tok, _n in items]),
            "n": sum(n for _acc, _tok, n in items),
        }
    mark_winners(parts)
    return parts


def format_data_row(row: dict) -> str:
    return (
        f"| {row['model']} | {row['dataset']} | {row['n']} | "
        f"{method_text(row, 'full')} | "
        f"{method_text(row, 'ac')} | "
        f"{method_text(row, 'dynasor')} | "
        f"{method_text(row, 'puma')} | "
        f"{method_text(row, 'deer')} | "
        f"{method_text(row, 'plws')} |"
    )


def format_overall_row(zh: str, label: str, parts: dict[str, dict | None]) -> str:
    n = "—" if label == OVERALL_EQ else str(parts["full"]["n"])
    texts = {}
    for key in COMPARE:
        cell = parts.get(key)
        if cell is None:
            texts[key] = ""
        else:
            texts[key] = fmt_cell(cell, parts.get("full"), kind=key)
    return (
        f"| {zh} | {label} | {n} | "
        f"{texts['full']} | {texts['ac']} | {texts['dynasor']} | "
        f"{texts['puma']} | {texts['deer']} | {texts['plws']} |"
    )


def render(rows_by_model: dict[str, list[dict]]) -> str:
    lines = [
        "# mean@3 主表（seed 42 / 0 / 1）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。",
        "固定三 seed：`42, 0, 1`，不是从四 seed 里挑最好的三个。",
        "五集：MATH-500 / OlympiadBench / GPQA-Diamond / AIME25 / AMC23。HMMT25 已撤出本表，不折 Overall。",
        "本表只收本机 ≤14B / Llama-8B。30B 及以上在远程跑，不进本表。",
        "行序按家族再按尺寸：R1-Qwen 1.5B/7B/14B，R1-Llama-8B，Nemotron，Qwen3 4B/8B。",
        "格子格式：`Acc / token / TR`。TR = 相对同行 Full-CoT token 的降幅；Full-CoT 记 `—`。",
        "某个方法必须三 seed 都齐才填数字，否则写 `未齐(k/3seed)`。",
        "Full-CoT / PUMA 还要求 `statistics.grader.json` 与当前 statistics 对齐；",
        "窗后压还要求 leftover 行带 `gt` / `gold_error`（AMC23 历史格除外）。",
        "只有该模型五集的 Full-CoT / PUMA / 窗后压都齐，才出 Overall。",
        "Overall DEER 还要求五集 DEER 都齐；否则 Overall 的 DEER 留空。",
        "AC / Dynasor 只填 seed `42, 0, 1` 三格都齐的格子；未齐写 `未齐(k/3seed)`。",
        "Overall 的 AC / Dynasor 还要求五集都齐；否则留空。",
        "Overall 只出等权一行：五集 Acc / token 各取算术平均。",
        "Token 口径与旧主表相同：交付 + 试答。AC 用 delivery+probe，Dynasor 用 delivery+probe。同行 Acc 最高、token 最低、TR 最高加粗。",
        "Llama-8B AMC23 的 Full-CoT / PUMA / 窗后压以远程三 seed 覆盖为准（`mean3_remote_overrides.json`）；DEER 仍是本机产物。",
        "Nemotron MATH-500 四列用 seed `0, 1, 123`。AIME25 的 Full-CoT / PUMA / 窗后压用 `1, 123, 7`；DEER 仍是 `42, 0, 1`（seed 7 无 DEER）。",
        "Qwen3-4B AMC23 窗后压用 seed `0, 123, 7`；Full-CoT / PUMA / DEER 仍是 `42, 0, 1`。",
        "Qwen3-8B：MATH-500 用 `42, 0, 123`；GPQA 用 `0, 1, 123`；AIME25 / AMC23 用 `0, 123, 7`。这四格 DEER 仍是 `42, 0, 1`（123/7 无 DEER）。Olympiad 仍是 `42, 0, 1`。",
        "1.5B AIME25 的 Full-CoT / PUMA / 窗后压用 `0, 1, 7`；DEER 仍是 `42, 0, 1`（seed 7 无 DEER）。其余四集仍是 `42, 0, 1`。",
        "",
        "| 模型 | 集 | n | Full-CoT | AC | Dynasor | PUMA | DEER | 窗后压 |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for _tag, zh in MODELS:
        rows = rows_by_model.get(zh, [])
        if not rows:
            continue
        for row in rows:
            lines.append(format_data_row(row))
        if len(rows) == len(DATASETS) and all(dataset_complete(row) for row in rows):
            lines.append(format_overall_row(zh, OVERALL_EQ, overall_parts(rows, weighted=False)))
    return "\n".join(lines) + "\n"


def main() -> None:
    if os.environ.get("MEAN3_REBUILD") == "1":
        rebuild_from_report()
        return
    if os.environ.get("MEAN3_PATCH_SEEDS") == "1":
        patch_method_seed_overrides()
        return
    if os.environ.get("MEAN3_PATCH_EXTRA") == "1":
        patch_extra_columns()
        return
    require_grader()
    paths = PLWSPaths.discover(ROOT)
    deer_cache = load_deer_cache()
    models = selected_models()
    scored_tags = {tag for tag, _zh in models}
    merge = os.environ.get("MEAN3_MERGE", "1" if models != MODELS else "0") == "1"
    rows_by_model: dict[str, list[dict]] = {}
    payload_rows: list[dict] = []
    for model, zh in models:
        print(f"scan {zh}", flush=True)
        model_rows: list[dict] = []
        for dataset, dszh, per_n in DATASETS:
            row = score_row(paths, model, zh, dataset, dszh, per_n, deer_cache)
            model_rows.append(row)
            payload_rows.append(row)
            print(
                f"  {dszh} full={row['counts']['full']} "
                f"plws={row['counts']['plws']} deer={row['counts']['deer']}",
                flush=True,
            )
        rows_by_model[zh] = model_rows
    if merge:
        payload_rows = merge_payload(payload_rows, scored_tags)
        rows_by_model = {}
        for cell in payload_rows:
            rows_by_model.setdefault(cell["model"], []).append(cell)
        print(
            f"merge kept other models; replaced {', '.join(sorted(scored_tags))}",
            flush=True,
        )
    write_outputs(payload_rows, rows_from_cells(payload_rows))


if __name__ == "__main__":
    main()
