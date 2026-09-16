#!/usr/bin/env bash
# Clone the pinned PUMA source and apply PLWS's reproducibility patch.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PUMA_ROOT="${PUMA_ROOT:-$ROOT/../PUMA}"
PUMA_URL="${PUMA_URL:-https://github.com/giovanni-vaccarino/PUMA.git}"
PUMA_COMMIT="ae8922c6c495bf778926a07694cfb88740fd3791"
PATCH="$ROOT/vendor/patches/puma-fullcot-32k-v2.patch"

if [[ ! -d "$PUMA_ROOT/.git" ]]; then
  git clone "$PUMA_URL" "$PUMA_ROOT"
  git -C "$PUMA_ROOT" checkout --detach "$PUMA_COMMIT"
fi

actual="$(git -C "$PUMA_ROOT" rev-parse HEAD)"
if [[ "$actual" != "$PUMA_COMMIT" ]]; then
  echo "ERROR: PUMA must be at $PUMA_COMMIT, got $actual" >&2
  exit 2
fi

if git -C "$PUMA_ROOT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "[skip] PUMA protocol patch is already applied"
elif git -C "$PUMA_ROOT" apply --check "$PATCH"; then
  git -C "$PUMA_ROOT" apply "$PATCH"
  echo "[ok] applied $PATCH"
else
  echo "ERROR: PUMA worktree is neither clean nor already patched" >&2
  exit 2
fi
