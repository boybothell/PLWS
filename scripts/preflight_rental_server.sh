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

from plws.deploy import load_deployment_profile
from plws.puma_official_conf import EMBEDDING_DIRNAME, embedding_model_path
from plws.runtime import dataset_path, model_path

root, puma = map(Path, sys.argv[1:3])
profile = load_deployment_profile(root=root)
models = profile.allowed_models
print(f"[ok] deployment profile {profile.name}: {','.join(models)}")
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
    puma / "puma" / "gen_prefixed_answers.py",
    puma / "puma" / "math_grader.py",
    puma / "puma" / "vllm_shutdown.py",
    puma / "baselines" / "deer" / "vllm_deer.py",
    puma / "baselines" / "deer" / "canonical_protocol.py",
    puma / "configs" / "DS-32B.conf",
    puma / "configs" / "DS-7B.conf",
    puma / "configs" / "Q30B-T.conf",
)
embed = embedding_model_path()
if not (embed / "config.json").is_file():
    errors.append(
        f"missing PUMA embedding model: {embed} "
        f"(need {EMBEDDING_DIRNAME} under PLWS_MODELS_ROOT)"
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

for module in (
    "torch",
    "transformers",
    "vllm",
    "nltk",
    "plws.contest",
    "plws.deploy",
    "plws.dynasor",
):
    try:
        loaded = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot import {module}: {exc}")
    else:
        print(f"[ok] import {module} {getattr(loaded, '__version__', '')}")

try:
    from plws.grading import require_grader

    require_grader()
    print("[ok] math grader self-test")
except Exception as exc:  # noqa: BLE001
    errors.append(f"math grader self-test failed: {exc}")

for script in (
    root / "scripts" / "run_large_model_cell.sh",
    root / "scripts" / "run_contest_prereq_cell.sh",
    root / "scripts" / "run_contest_plws_cell.sh",
    root / "scripts" / "run_deer_official.sh",
    root / "baselines" / "deer" / "run_cell.sh",
    root / "baselines" / "answer_convergence" / "run_cell.sh",
    root / "baselines" / "dynasor" / "run_cell.sh",
    root / "baselines" / "dynasor" / "runner.py",
):
    if not script.is_file():
        errors.append(f"missing runner: {script}")

try:
    import nltk

    nltk.sent_tokenize("Preflight sentence.")
except Exception as exc:  # noqa: BLE001
    errors.append(f"NLTK punkt preflight failed: {exc}")

if errors:
    print("\n".join(f"[error] {error}" for error in errors), file=sys.stderr)
    raise SystemExit(1)
print("[ok] rental server preflight passed")
PY
