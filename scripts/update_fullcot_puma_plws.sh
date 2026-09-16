#!/usr/bin/env bash
# Frozen main-table refresh. Schema lives in report_fullcot_puma_plws.py:
# 7B / Nemotron / 14B / 4B / 8B × MATH / Olympiad / GPQA / AIME24 / AIME25,
# plus finished AIME26 / AMC23 / BRUMO / HMMT. 1.5B / Llama-8B / R1-32B /
# 30B use the same sets, including MATH / Olympiad / GPQA when the
# four-seed cell is complete. Qwen3-32B stays out. Overall Acc is
# equal-weight; Overall token is shown as equal-weight and n-weighted.
# Base seeds are
# 42/0/1/123; seed 7 folds in when that seed is complete. DEER fills
# when four seeds are complete. Only complete cells are written;
# holes stay blank.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

REPORT_PY="${REPORT_PY:-python3}"
for candidate in \
  "$REPORT_PY" \
  /mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python \
  "$ROOT/.venv/bin/python"
do
  if "$candidate" -c 'from plws.paths import PLWSPaths' 2>/dev/null; then
    REPORT_PY="$candidate"
    break
  fi
done

XLSX_PY=python3
if ! "$XLSX_PY" -c 'import openpyxl' 2>/dev/null; then
  echo "ERROR: need a python with openpyxl to write the Feishu xlsx" >&2
  exit 1
fi

echo "recount artifacts -> markdown + json"
"$REPORT_PY" "$ROOT/scripts/report_fullcot_puma_plws.py"
echo "write Feishu xlsx"
"$XLSX_PY" "$ROOT/scripts/export_fullcot_puma_plws_feishu.py"
echo "done $ROOT/tables/firstwin_wait/fullcot_puma_plws.md"
echo "done $ROOT/tables/firstwin_wait/fullcot_puma_plws_feishu.xlsx"
