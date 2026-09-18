#!/usr/bin/env python3
"""Official mean@3 main table: seeds 42 / 0 / 1, six datasets.

A method cell is filled only when all three seeds are complete.
TR is token reduction versus Full-CoT on that same row.
Overall is emitted only when Full-CoT / PUMA / 窗后压 are complete
on every dataset for that model. DEER Overall additionally requires
DEER complete on every dataset.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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
DEER_CACHE = ROOT / "results" / "reports" / "cache" / "deer_aligned_v2_mean3.json"

SEEDS = (42, 0, 1)
MODELS = (
    ("r1_7b", "7B"),
    ("nemotron_8b", "Nemotron"),
    ("r1_14b", "14B"),
    ("qwen3_4b", "4B"),
    ("qwen3_8b", "8B"),
    ("r1_1p5b", "1.5B"),
    ("r1_llama_8b", "Llama-8B"),
    ("qwen3_30b_a3b", "Qwen3-30B-A3B"),
    ("r1_32b", "R1-32B"),
    ("qwen3_32b", "Qwen3-32B"),
)
DATASETS = (
    ("math-500", "MATH-500", 500),
    ("olympiadbench", "OlympiadBench", 675),
    ("gpqa-diamond", "GPQA-Diamond", 198),
    ("aime25", "AIME25", 30),
    ("hmmt25", "HMMT25", 30),
    ("amc23", "AMC23", 40),
)
COMPARE = ("full", "puma", "deer", "plws")
OVERALL_EQ = "Overall（等权）"
OVERALL_NW = "Overall（按题加权）"


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
    for seed in SEEDS:
        if seed_official_puma_ready(
            paths, model, dataset, seed, per_n
        ) and puma_statistics_verified(paths, model, dataset, seed):
            official += 1
        if seed_plws_ready(
            paths, model, dataset, seed, allow_legacy=legacy
        ) and leftover_grades_verified(
            paths, model, dataset, seed, allow_legacy=legacy
        ):
            plws += 1
        if deer_complete(paths, model, dataset, seed, expected=per_n)[0]:
            deer += 1
    return {"full": official, "puma": official, "plws": plws, "deer": deer}


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
) -> dict | None:
    if not all(
        deer_complete(paths, model, dataset, seed, expected=per_n)[0]
        for seed in SEEDS
    ):
        return None
    deer_n = per_n * len(SEEDS)
    key = f"{model}/{dataset}/{'+'.join(str(s) for s in SEEDS)}"
    fingerprint = f"{TOKEN_POLICY}|{_deer_fingerprint(paths, model, dataset, SEEDS)}"
    hit = cache.get(key)
    if (
        isinstance(hit, dict)
        and hit.get("fingerprint") == fingerprint
        and hit.get("n") == deer_n
        and hit.get("token_policy") == TOKEN_POLICY
    ):
        return {"n": hit["n"], "acc": hit["acc"], "tok": hit["tok"]}
    items: list[tuple[str, str, str, float]] = []
    for seed in SEEDS:
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
    paths: PLWSPaths, model: str, dataset: str, per_n: int
) -> dict[str, dict]:
    full: list[tuple[bool, float]] = []
    puma: list[tuple[bool, float]] = []
    for seed in SEEDS:
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
    actual = per_n * len(SEEDS)
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
    counts = ready_counts(paths, model, dataset, per_n)
    parts: dict[str, dict | None] = {
        "full": None,
        "puma": None,
        "deer": None,
        "plws": None,
    }
    if counts["plws"] == 3 and counts["full"] == 3:
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
            used_seeds=SEEDS,
        )
        parts["full"] = scored["full"]
        parts["puma"] = scored["puma"]
        parts["plws"] = scored["plws"]
    elif counts["full"] == 3:
        scored = score_full_puma(paths, model, dataset, per_n)
        parts["full"] = scored["full"]
        parts["puma"] = scored["puma"]
    if counts["deer"] == 3:
        print(f"score DEER {zh} {dszh}", flush=True)
        parts["deer"] = load_deer_mean3(paths, model, dataset, per_n, deer_cache)
        if parts["deer"] is None:
            counts["deer"] = sum(
                1
                for seed in SEEDS
                if deer_complete(paths, model, dataset, seed, expected=per_n)[0]
            )
    mark_winners(parts)
    return {
        "model": zh,
        "model_tag": model,
        "dataset": dszh,
        "dataset_id": dataset,
        "n": per_n * len(SEEDS),
        "counts": counts,
        "full": parts["full"],
        "puma": parts["puma"],
        "deer": parts["deer"],
        "plws": parts["plws"],
    }


def method_text(row: dict, key: str) -> str:
    cell = row.get(key)
    if cell is None:
        return incomplete_text(int(row["counts"][key]))
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
    for key in COMPARE:
        if key == "deer" and not deer_all:
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
        f"{texts['full']} | {texts['puma']} | {texts['deer']} | {texts['plws']} |"
    )


def render(rows_by_model: dict[str, list[dict]]) -> str:
    lines = [
        "# mean@3 主表（seed 42 / 0 / 1）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。",
        "固定三 seed：`42, 0, 1`，不是从四 seed 里挑最好的三个。",
        "六集：MATH-500 / OlympiadBench / GPQA-Diamond / AIME25 / HMMT25 / AMC23。",
        "格子格式：`Acc / token / TR`。TR = 相对同行 Full-CoT token 的降幅；Full-CoT 记 `—`。",
        "某个方法必须三 seed 都齐才填数字，否则写 `未齐(k/3seed)`。",
        "Full-CoT / PUMA 还要求 `statistics.grader.json` 与当前 statistics 对齐；",
        "窗后压还要求 leftover 行带 `gt` / `gold_error`（AMC23 历史格除外）。",
        "只有该模型六集的 Full-CoT / PUMA / 窗后压都齐，才出 Overall。",
        "Overall DEER 还要求六集 DEER 都齐；否则 Overall 的 DEER 留空。",
        "Overall Acc 等权；Overall token 并列等权平均和按题数加权。",
        "Token 口径与旧主表相同：交付 + 试答。同行 Acc 最高、token 最低、TR 最高加粗。",
        "",
        "| 模型 | 集 | n | Full-CoT | PUMA | DEER | 窗后压 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for _tag, zh in MODELS:
        rows = rows_by_model.get(zh, [])
        if not rows:
            continue
        for row in rows:
            lines.append(format_data_row(row))
        if len(rows) == len(DATASETS) and all(dataset_complete(row) for row in rows):
            lines.append(format_overall_row(zh, OVERALL_EQ, overall_parts(rows, weighted=False)))
            lines.append(format_overall_row(zh, OVERALL_NW, overall_parts(rows, weighted=True)))
    return "\n".join(lines) + "\n"


def main() -> None:
    require_grader()
    paths = PLWSPaths.discover(ROOT)
    deer_cache = load_deer_cache()
    rows_by_model: dict[str, list[dict]] = {}
    payload_rows: list[dict] = []
    for model, zh in MODELS:
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
                "token_policy": TOKEN_POLICY,
                "cells": payload_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(text)
    print(f"wrote {TABLE}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
