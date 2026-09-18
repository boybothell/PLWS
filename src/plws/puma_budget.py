"""Audit PUMA final regenerations against the shared 32K host budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plws.protocol import FULLCOT_GENERATION_TOKENS


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    fullcot_generation_tokens: int = FULLCOT_GENERATION_TOKENS,
) -> dict[str, int]:
    """Validate regenerated rows; unchanged Full-CoT rows are already host-checked."""

    regenerated = 0
    legacy = 0
    maximum_used = 0
    violations: list[tuple[Any, int]] = []
    for row in rows:
        if not str(row.get("reasoning_prefix") or "").strip():
            continue
        regenerated += 1
        if "prefix_generated_tokens" in row:
            prefix = int(row["prefix_generated_tokens"])
        else:
            prefix = int(row.get("count_reasoning_tokens") or 0)
            legacy += 1
        generated = int(row.get("count_generated_tokens") or 0)
        used = prefix + generated
        maximum_used = max(maximum_used, used)
        if used > fullcot_generation_tokens:
            violations.append((row.get("question_idx"), used))
    if violations:
        preview = ", ".join(f"q{idx}={used}" for idx, used in violations[:10])
        raise ValueError(
            f"{len(violations)} PUMA rows exceed {fullcot_generation_tokens}: "
            f"{preview}"
        )
    return {
        "rows": len(rows),
        "regenerated_rows": regenerated,
        "legacy_rows": legacy,
        "violations": 0,
        "max_prefix_plus_generation": maximum_used,
        "fullcot_generation_tokens": fullcot_generation_tokens,
    }


def audit_file(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"PUMA prefixed output must be a JSON list: {path}")
    return audit_rows(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixed_answers", type=Path)
    args = parser.parse_args()
    report = audit_file(args.prefixed_answers)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
