#!/usr/bin/env python3
"""导出 PLWS 第一扇窗 jobs。

锁点是第一扇 k 步同答窗。`firstwin` 是规范产物；low / mix / high 只是
同一扇窗的把握标签，不再作为分跑方法。
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_HINT = Path(os.environ.get("PLWS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT_HINT / "src"))

from plws.artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    utc_now,
)
from plws.inputs import (  # noqa: E402
    answer_credit,
    finite,
    load_json,
    same_answer,
    usable_trial_rows,
)
from plws.paths import PLWSPaths  # noqa: E402
from plws.window import (  # noqa: E402
    EPS,
    K,
    MSS,
    TAU,
    Observation,
    WindowLevel,
    classify_window,
)

PATHS = PLWSPaths.discover(__file__)
AE = PATHS.root

DATASETS = ("gpqa-diamond", "math-500", "olympiadbench", "aime24", "aime25", "amc23", "gsm8k")
AIME_ONLY = ("aime24", "aime25")
MODELS = ("r1_7b", "nemotron_8b", "r1_14b", "r1_32b", "qwen3_30b_a3b")
TIER_KINDS = ("low", "mix", "high")
FIRSTWIN = "firstwin"
KINDS = (FIRSTWIN, *TIER_KINDS)


def window_ok_k(rows: list[dict], end: int, k: int) -> bool:
    if end + 1 < k:
        return False
    window = rows[end + 1 - k : end + 1]
    steps = [int(row["stopped_len"]) for row in window]
    if steps != list(range(steps[0], steps[0] + k)):
        return False
    first = str(window[0].get("final_answer") or "")
    return bool(first) and all(
        same_answer(first, row.get("final_answer")) for row in window
    )


def kind_of_k(confs: list[float], k: int) -> str:
    if len(confs) < k or not all(math.isfinite(value) for value in confs):
        return "mix"
    observations = [
        Observation(step=index + 1, answer="", confidence=value)
        for index, value in enumerate(confs)
    ]
    return {
        WindowLevel.H: "high",
        WindowLevel.M: "mix",
        WindowLevel.L: "low",
    }[classify_window(observations, tau=TAU, eps=EPS)]


def same_windows_k(rows: list[dict], k: int) -> list[dict]:
    out = []
    for end, row in enumerate(rows):
        if not window_ok_k(rows, end, k):
            continue
        if int(row["stopped_len"]) < MSS:
            continue
        window = rows[end + 1 - k : end + 1]
        confs = [finite(item.get("confidence")) for item in window]
        if not confs or not all(math.isfinite(c) for c in confs):
            continue
        out.append(
            {
                "end": end,
                "step": int(row["stopped_len"]),
                "ans": row.get("final_answer"),
                "c": confs[-1],
                "kind": kind_of_k(confs, k),
            }
        )
    return out


def pick_window(rows: list[dict], kind: str, k: int | None = None) -> dict | None:
    k = K if k is None else int(k)
    wins = same_windows_k(rows, k)
    if not wins:
        return None
    win = wins[0]
    if kind == FIRSTWIN:
        return win
    if win.get("kind") != kind:
        return None
    return win


def build_jobs(
    model: str,
    kind: str,
    seed: int = 42,
    datasets: tuple[str, ...] = DATASETS,
    k: int | None = None,
) -> list[dict]:
    jobs: list[dict] = []
    for dataset in datasets:
        trials_path = PATHS.dense_trial_path(model, dataset, seed)
        if not trials_path.is_file():
            continue
        official_path = PATHS.puma_statistics_path(model, dataset, seed)
        official = (
            {int(r["question_idx"]): r for r in load_json(official_path)}
            if official_path.is_file()
            else {}
        )
        gp = PATHS.dense_g_path(model, dataset, seed)
        gmap = {int(r["question_idx"]): r for r in load_json(gp)} if gp.is_file() else {}
        by: dict[int, list] = {}
        for row in load_json(trials_path):
            by.setdefault(int(row["question_idx"]), []).append(row)
        qis = sorted(set(by) | set(official) | set(gmap))
        for qi in qis:
            trials = by.get(qi)
            if not trials:
                continue
            rows = usable_trial_rows(trials)
            if not rows:
                continue
            win = pick_window(rows, kind, k=k)
            if win is None:
                continue
            left_row = rows[win["end"]]
            thought = str(left_row.get("reasoning_prefix") or "")
            question = str(left_row.get("question") or "")
            if not thought or not question:
                continue
            info = official.get(qi) or {}
            g = gmap.get(qi) or {}
            last = max(trials, key=lambda x: int(x["stopped_len"]))
            gt = info.get("ground_truth") or g.get("ground_truth")
            original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
            orig_ok = (
                bool(info.get("original_correct"))
                if "original_correct" in info
                else bool(answer_credit(original, gt, original, True))
            )
            wins = same_windows_k(rows, K if k is None else int(k))
            high_after = next(
                (w for w in wins if w["kind"] == "high" and w["step"] > win["step"]),
                None,
            )
            left_ok = bool(answer_credit(win["ans"], gt, original, orig_ok))
            job = {
                "uid": f"{model}:{dataset}:{seed}:{qi}",
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "question_idx": int(qi),
                "left_step": int(win["step"]),
                "kind": str(win.get("kind") or kind),
                "lock": FIRSTWIN if kind == FIRSTWIN else kind,
                "confidence": win.get("c"),
                "left_ok": left_ok,
                "wait_helps": False,
                "will_change": bool(
                    high_after is not None
                    and not same_answer(win["ans"], high_after["ans"])
                ),
                "never_high": high_after is None,
                "same_as_high": bool(
                    high_after is not None
                    and same_answer(win["ans"], high_after["ans"])
                ),
                "host_ok": orig_ok,
                "old_answer": str(win.get("ans") or ""),
                "question": question,
                "thought": thought,
                "gt": gt,
                "original": original,
                "orig_ok": orig_ok,
            }
            window_k = K if k is None else int(k)
            if window_k != K:
                job["k"] = window_k
            jobs.append(job)
    return jobs


def _puma_dir(model: str, dataset: str, seed: int) -> Path:
    return PATHS.puma_output_dir(model, dataset, seed)


def cell_ready(model: str, dataset: str, seed: int) -> bool:
    d = _puma_dir(model, dataset, seed)
    trials = PATHS.dense_trial_path(model, dataset, seed)
    return (
        (d / "statistics.json").is_file()
        and (d / "prefixed_answers.json").is_file()
        and trials.is_file()
    )


def workers_live() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "run_puma_official|run_dense_trials|gen_trial_answers|plws4fill"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return any("export_leftover_suppress_jobs" not in line for line in out.splitlines())


def wait_amc_gsm(model: str, seeds: list[int], datasets: tuple[str, ...]) -> None:
    need = [ds for ds in datasets if ds in ("amc23", "gsm8k")]
    if not need:
        return
    idle = 0
    while True:
        missing = [f"{ds}s{seed}" for seed in seeds for ds in need if not cell_ready(model, ds, seed)]
        if not missing:
            print(f"export wait: amc23/gsm8k ready {need} seeds={seeds}", flush=True)
            return
        live = workers_live()
        print(f"export wait: missing {missing} workers={'yes' if live else 'no'}", flush=True)
        if not live:
            idle += 1
            if idle >= 4:
                print("export wait: no workers, export what exists", flush=True)
                return
        else:
            idle = 0
        time.sleep(30)


def jobs_path(
    model: str,
    dataset: str,
    kind: str,
    seed: int = 42,
    *,
    k: int = K,
    lexicon: str = "core",
) -> Path:
    return PATHS.jobs_path(
        model, dataset, seed, kind, k=k, lexicon=lexicon
    )


def _merge_cell_metadata(
    *,
    cell: Path,
    model: str,
    dataset: str,
    seed: int,
    kind: str,
    k: int,
    lexicon: str,
    artifact: Path,
    count: int,
) -> None:
    atomic_write_json(
        cell / "jobs" / f"manifest_{kind}.json",
        {
            "schema_version": 1,
            "method": "plws",
            "policy": "window_first",
            "k": k,
            "lexicon": lexicon,
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "tier": kind,
            "artifact": str(artifact),
            "rows": count,
            "created_at": utc_now(),
        },
    )
    atomic_write_json(
        cell / "jobs" / f"status_{kind}.json",
        {
            "schema_version": 1,
            "state": "succeeded",
            "tier": kind,
            "rows": count,
            "updated_at": utc_now(),
        },
    )


def write_jobs(
    model: str,
    kind: str,
    seed: int = 42,
    datasets: tuple[str, ...] = DATASETS,
    stem: str = "",
    k: int | None = None,
    lexicon: str = "core",
    out_root: Path | None = None,
) -> None:
    jobs = build_jobs(model, kind, seed, datasets=datasets, k=k)
    window_k = K if k is None else int(k)
    if out_root is not None:
        folder = out_root / f"{model}_s{seed}"
        base = stem or "jobs"
        out = folder / (
            f"{base}.jsonl" if kind == "low" else f"{base}_{kind}.jsonl"
        )
        atomic_write_jsonl(out, jobs)
        print(f"wrote compatibility aggregate {len(jobs)} {kind} -> {out}", flush=True)
        return

    by_dataset = {
        dataset: [job for job in jobs if job["dataset"] == dataset]
        for dataset in datasets
        if PATHS.dense_trial_path(model, dataset, seed).is_file()
    }
    for dataset, rows in by_dataset.items():
        out = jobs_path(
            model,
            dataset,
            kind,
            seed,
            k=window_k,
            lexicon=lexicon,
        )
        atomic_write_jsonl(out, rows)
        _merge_cell_metadata(
            cell=PATHS.cell_dir(
                model, dataset, seed, k=window_k, lexicon=lexicon
            ),
            model=model,
            dataset=dataset,
            seed=seed,
            kind=kind,
            k=window_k,
            lexicon=lexicon,
            artifact=out,
            count=len(rows),
        )
        print(f"wrote {len(rows)} {kind} -> {out}", flush=True)

    if stem:
        work = PATHS.work_jobs_path(
            model,
            seed,
            kind,
            k=window_k,
            lexicon=lexicon,
            stem=stem,
        )
        atomic_write_jsonl(work, jobs)
        atomic_write_json(
            work.with_suffix(".manifest.json"),
            {
                "schema_version": 1,
                "role": "work_aggregate",
                "canonical": False,
                "model": model,
                "seed": seed,
                "kind": kind,
                "k": window_k,
                "lexicon": lexicon,
                "datasets": list(by_dataset),
                "rows": len(jobs),
                "created_at": utc_now(),
            },
        )
        print(f"wrote noncanonical work aggregate {len(jobs)} -> {work}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag")
    parser.add_argument("--kind", choices=KINDS, default=FIRSTWIN)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--kinds",
        default="",
        help="comma list, firstwin, or all (firstwin only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="", help="comma list, e.g. 0,1,123")
    parser.add_argument(
        "--datasets",
        default="",
        help="comma list。非 42 的 seed 默认只导出 AIME，避免误用 seed42 的 MATH/oly 平铺轨迹。",
    )
    parser.add_argument(
        "--out-stem",
        default="",
        help="可选：另写 run-root/work/ 下明确标记为非 canonical 的聚合任务文件。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="第一扇同答窗连续步数。每个 k 使用独立 canonical cell。",
    )
    parser.add_argument("--lexicon", default="core")
    parser.add_argument(
        "--out-root",
        type=Path,
        help="Compatibility override; writes the historical model/seed layout below this root.",
    )
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be at least 2")
    kinds = [args.kind]
    if args.kinds in {"all", FIRSTWIN}:
        kinds = [FIRSTWIN]
    elif args.kinds:
        kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    seeds = [args.seed]
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.datasets:
        datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    else:
        datasets = DATASETS
    models = MODELS if args.all or not args.model_tag else (args.model_tag,)
    if args.model_tag:
        wait_amc_gsm(args.model_tag, seeds, datasets)
    for seed in seeds:
        ds = datasets if seed == 42 or args.datasets else AIME_ONLY
        for model in models:
            for kind in kinds:
                write_jobs(
                    model,
                    kind,
                    seed=seed,
                    datasets=ds,
                    stem=args.out_stem,
                    k=args.k,
                    lexicon=args.lexicon,
                    out_root=args.out_root,
                )


if __name__ == "__main__":
    main()
