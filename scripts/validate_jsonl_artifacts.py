#!/usr/bin/env python3
"""Stream-validate JSONL artifacts without loading them into memory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def validate(path: Path, key: str | None) -> dict[str, Any]:
    rows = valid = invalid = duplicate_keys = 0
    last_valid_offset = 0
    trailing_partial = False
    keys: set[str] | None = set() if key else None
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            rows += 1
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid += 1
                trailing_partial = bool(handle.peek(1) == b"" and not raw.endswith(b"\n"))
                continue
            valid += 1
            last_valid_offset = handle.tell()
            if keys is not None:
                value = row.get(key)
                if value is not None:
                    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
                    if rendered in keys:
                        duplicate_keys += 1
                    keys.add(rendered)
    return {
        "path": str(path),
        "rows": rows,
        "valid_rows": valid,
        "invalid_rows": invalid,
        "duplicate_keys": duplicate_keys,
        "trailing_partial": trailing_partial,
        "last_valid_offset": last_valid_offset,
        "valid": invalid == 0 and duplicate_keys == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--key", help="Optional field that must be unique.")
    parser.add_argument(
        "--repair-trailing-partial",
        action="store_true",
        help="Explicitly truncate one invalid final line after creating a .bak copy.",
    )
    args = parser.parse_args()
    if not args.path.is_file():
        raise SystemExit(f"not a file: {args.path}")
    report = validate(args.path, args.key)
    if args.repair_trailing_partial:
        if report["invalid_rows"] != 1 or not report["trailing_partial"]:
            raise SystemExit("refusing repair: exactly one invalid trailing line is required")
        backup = args.path.with_suffix(args.path.suffix + ".bak")
        if backup.exists():
            raise SystemExit(f"refusing repair: backup already exists: {backup}")
        shutil.copy2(args.path, backup)
        with args.path.open("r+b") as handle:
            handle.truncate(report["last_valid_offset"])
            handle.flush()
            os.fsync(handle.fileno())
        report = validate(args.path, args.key)
        report["repaired_from"] = str(backup)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
