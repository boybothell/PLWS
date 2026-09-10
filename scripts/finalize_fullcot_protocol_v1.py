#!/usr/bin/env python3
"""Merge successful protocol-repair rows back into canonical PLWS shards."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPAIR_ROOT = ROOT / "results" / "runs" / "plws_protocol_repair_v1"
REPORT = ROOT / "results" / "reports" / "plws_protocol_v1_affected.json"
PROTOCOL = {
    "protocol_id": "puma-fullcot-32k-v2",
    "fullcot_generation_tokens": 32768,
    "truncated_answer_fix_tokens": 2048,
    "max_model_len": 37888,
}


def records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = {row["uid"]: row for row in report["full_rerun"]}
    repaired: dict[str, dict[str, Any]] = {}
    for path in sorted((REPAIR_ROOT / "outputs").glob("lane_*.jsonl")):
        for row in records(path):
            if row.get("status") != "ok":
                raise RuntimeError(f"repair did not succeed: {row}")
            uid = str(row["uid"])
            if uid in repaired:
                raise RuntimeError(f"duplicate repaired UID: {uid}")
            repaired[uid] = row
    missing = sorted(set(expected) - set(repaired))
    extra = sorted(set(repaired) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"repair output mismatch: missing={len(missing)} extra={len(extra)}"
        )

    by_target: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for uid, row in repaired.items():
        info = expected[uid]
        row["kind"] = info["kind"]
        row["protocol_repair"] = True
        by_target[ROOT / info["target"]].append(row)

    for target, additions in by_target.items():
        current = records(target) if target.is_file() else []
        current_uids = {str(row.get("uid")) for row in current}
        overlap = current_uids & {str(row["uid"]) for row in additions}
        if overlap:
            raise RuntimeError(f"canonical shard already contains repaired rows: {overlap}")
        merged = sorted(
            [*current, *additions],
            key=lambda row: str(row.get("uid") or ""),
        )
        write_jsonl(target, merged)
        artifact = target.stem
        manifest = {
            "schema_version": 2,
            "method": "plws",
            "experiment": "window_first",
            **PROTOCOL,
            "output": str(target),
            "rows": len(merged),
            "compatibility": (
                "legacy rows naturally terminated below old caps; "
                "bounded rows regenerated under canonical protocol"
            ),
        }
        (target.parent / f"manifest_{artifact}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (target.parent / f"status_{artifact}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "state": "succeeded",
                    "message": "canonical Full-CoT 32K protocol repair complete",
                    **PROTOCOL,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    (REPAIR_ROOT / "completed.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "repaired_uids": len(repaired),
                "canonical_shards": len(by_target),
                **PROTOCOL,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"merged {len(repaired)} UIDs into {len(by_target)} canonical shards")


if __name__ == "__main__":
    main()
