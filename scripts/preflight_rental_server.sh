#!/usr/bin/env bash
# Validate code, patched PUMA, datasets, models, and the GPU Python runtime.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck source=lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init
PUMA_ROOT="$(cd "$PUMA_ROOT" && pwd)"
plws_export_cuda_runtime
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" - "$ROOT" "$PUMA_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

from plws.runtime import dataset_path, model_path

root, puma = map(Path, sys.argv[1:3])
models = ("qwen3_30b_a3b", "r1_32b", "qwen3_32b", "qwq_32b")
datasets = {
    "math-500": 500,
    "olympiadbench": 675,
    "gpqa-diamond": 198,
    "aime24": 30,
    "aime25": 30,
    "aime26": 30,
    "brumo25": 30,
    "hmmt25": 30,
    "amc23": 40,
}

errors: list[str] = []
for tag in models:
    config = model_path(tag) / "config.json"
    if not config.is_file():
        errors.append(f"missing model: {config}")

required_puma = (
    puma / "puma" / "run_vllm.py",
    puma / "puma" / "gen_trial_answers.py",
    puma / "puma" / "vllm_shutdown.py",
    puma / "baselines" / "deer" / "vllm_deer.py",
    puma / "baselines" / "deer" / "canonical_protocol.py",
)
for path in required_puma:
    if not path.is_file():
        errors.append(f"missing patched PUMA file: {path}")

checksums = {}
sums = root / "data" / "SHA256SUMS"
if not sums.is_file():
    errors.append(f"missing pinned checksums: {sums}")
else:
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        checksums[name] = digest

for slug, expected in datasets.items():
    name = f"{slug}_test.jsonl"
    pinned = root / "data" / name
    runtime = dataset_path(slug, puma)
    puma_data = puma / "data" / name
    expected_digest = checksums.get(name)
    if expected_digest is None:
        errors.append(f"dataset not pinned: {name}")
    seen: set[Path] = set()
    for path in (pinned, runtime, puma_data):
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            errors.append(f"missing dataset: {path}")
            continue
        actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        if actual != expected:
            errors.append(f"dataset size mismatch: {path} ({actual} != {expected})")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_digest and digest != expected_digest:
            errors.append(f"dataset hash mismatch: {path}")
        if pinned.is_file() and path != pinned and path.read_bytes() != pinned.read_bytes():
            errors.append(f"dataset differs from repo pin: {path}")

for module in ("torch", "transformers", "vllm", "plws.contest"):
    try:
        loaded = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot import {module}: {exc}")
    else:
        print(f"[ok] import {module} {getattr(loaded, '__version__', '')}")

for script in (
    root / "scripts" / "run_large_model_cell.sh",
    root / "scripts" / "run_contest_prereq_cell.sh",
    root / "scripts" / "run_contest_plws_cell.sh",
    root / "scripts" / "run_deer_official.sh",
):
    if not script.is_file():
        errors.append(f"missing runner: {script}")

if errors:
    print("\n".join(f"[error] {error}" for error in errors), file=sys.stderr)
    raise SystemExit(1)
print("[ok] rental server preflight passed")
PY
