#!/usr/bin/env python3
"""Rewrite SentencePiece dumps in Full-CoT samples and stash broken PUMA/PLWS.

Does not resample Full-CoT. GPU PUMA / leftover rebuild is a later queue.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from plws.paths import PLWSPaths  # noqa: E402
from plws.piece_text import row_has_tokenizer_pieces, sanitize_rows  # noqa: E402

STAMP = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def load_answers(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} is not a list")
    return [row for row in payload if isinstance(row, dict)]


def answers_have_pieces(path: Path) -> bool:
    if not path.is_file():
        return False
    return any(row_has_tokenizer_pieces(row) for row in load_answers(path))


def iter_sample_answers(root: Path) -> list[Path]:
    return sorted(root.glob("samples/*/*/seed_*/answers.json"))


def cell_from_sample(path: Path) -> tuple[str, str, int]:
    seed_dir = path.parent
    dataset_dir = seed_dir.parent
    model_dir = dataset_dir.parent
    return model_dir.name, dataset_dir.name, int(seed_dir.name.split("_", 1)[1])


def piece_repair_models(root: Path) -> list[str]:
    hits = scan_samples(root)
    if hits:
        return sorted({str(hit["model"]) for hit in hits})
    backups = sorted(root.glob("samples/*/*/seed_*/answers.piece.json"))
    return sorted({path.parents[2].name for path in backups})


def scan_samples(root: Path) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for path in iter_sample_answers(root):
        rows = load_answers(path)
        n_piece = sum(1 for row in rows if row_has_tokenizer_pieces(row))
        if not n_piece:
            continue
        model, dataset, seed = cell_from_sample(path)
        hits.append(
            {
                "path": str(path.relative_to(root)),
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "n": len(rows),
                "n_piece": n_piece,
            }
        )
    return hits


def stash(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = dest.with_name(f"{dest.name}_{STAMP}")
    shutil.move(str(src), str(dest))
    print(f"stash {src} -> {dest}", flush=True)


def rewrite_sample(path: Path) -> bool:
    rows = load_answers(path)
    if not any(row_has_tokenizer_pieces(row) for row in rows):
        return False
    backup = path.with_name("answers.piece.json")
    if not backup.is_file():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    fixed = sanitize_rows(rows)
    if any(row_has_tokenizer_pieces(row) for row in fixed):
        raise RuntimeError(f"sanitize left pieces in {path}")
    tmp = path.with_name(f".answers.json.tmp.{path.stat().st_mtime_ns}")
    tmp.write_text(json.dumps(fixed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"rewrote {path} n={len(fixed)}", flush=True)
    return True


def quarantine_cell(paths: PLWSPaths, model: str, dataset: str, seed: int) -> None:
    stale = paths.results / "baselines" / "puma" / "_stale_llama_pieces"
    puma = paths.puma_output_dir(model, dataset, seed)
    stash(puma, stale / f"{model}_s{seed}" / f"{dataset}_{STAMP}")
    dense = (
        paths.results
        / "upstream"
        / "dense_trials"
        / f"dense_G_{model}"
        / dataset
        / f"seed_{seed}"
    )
    stash(
        dense,
        paths.results
        / "upstream"
        / "dense_trials"
        / "_stale_llama_pieces"
        / f"dense_G_{model}"
        / dataset
        / f"seed_{seed}_{STAMP}",
    )
    cell = paths.cell_dir(model, dataset, seed, k=4, lexicon="core")
    stash(
        cell,
        paths.results
        / "runs"
        / "plws"
        / "_stale_llama_pieces"
        / model
        / dataset
        / f"seed_{seed}_{STAMP}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    hits = scan_samples(root)
    print(json.dumps({"hits": hits, "n": len(hits)}, ensure_ascii=False, indent=2))
    if args.scan_only or not hits:
        return 0
    paths = PLWSPaths(root)
    seen: set[tuple[str, str, int]] = set()
    for hit in hits:
        path = root / str(hit["path"])
        rewrite_sample(path)
        key = (str(hit["model"]), str(hit["dataset"]), int(hit["seed"]))
        if key in seen:
            continue
        seen.add(key)
        quarantine_cell(paths, *key)
    print(f"repaired {len(seen)} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
