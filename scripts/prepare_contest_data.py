#!/usr/bin/env python3
"""Write PUMA-format contest jsonl. Does not rewrite aime24/aime25."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUMA_DATA = ROOT.parent / "PUMA" / "data"

LOCAL_SOURCES = {
    "aime26": Path(
        "/mnt/d/lfy/UQ/evaluation/Qwen2.5-Math/evaluation/data/aime26/test.jsonl"
    ),
    "hmmt25": Path(
        "/mnt/d/lfy/UQ/evaluation/Qwen2.5-Math/evaluation/data/hmmt_25/hmmt_25.jsonl"
    ),
}
KEEP = ("aime24", "aime25")
BRUMO_HF = "MathArena/brumo_2025"
BRUMO_LOCAL = Path(__file__).resolve().parent / "_brumo25_source.json"


def puma_row(item: dict) -> dict[str, str]:
    question = item.get("question") or item.get("problem")
    answer = item.get("answer")
    if question is None or answer is None:
        raise ValueError(f"missing question/answer keys: {sorted(item)}")
    return {"question": str(question).strip(), "answer": str(answer).strip()}


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_brumo_hub() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(BRUMO_HF, split="train")
    return [dict(row) for row in dataset]


def load_brumo_viewer() -> list[dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=MathArena/brumo_2025&config=default&split=train"
        "&offset=0&length=30"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = [item["row"] for item in payload.get("rows") or []]
    if len(rows) != 30:
        raise RuntimeError(f"BRUMO viewer returned {len(rows)} rows, expected 30")
    return rows


def load_brumo_local() -> list[dict]:
    if not BRUMO_LOCAL.is_file():
        raise FileNotFoundError(BRUMO_LOCAL)
    rows = json.loads(BRUMO_LOCAL.read_text(encoding="utf-8"))
    if len(rows) != 30:
        raise RuntimeError(f"local BRUMO source has {len(rows)} rows, expected 30")
    return rows


def load_brumo() -> list[dict]:
    try:
        return load_brumo_hub()
    except Exception:
        try:
            return load_brumo_viewer()
        except Exception:
            return load_brumo_local()


def convert_local(tag: str) -> list[dict[str, str]]:
    source = LOCAL_SOURCES[tag]
    if not source.is_file():
        raise FileNotFoundError(source)
    return [puma_row(item) for item in load_jsonl(source)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    written: dict[str, int] = {}
    for tag in KEEP:
        dest = PUMA_DATA / f"{tag}_test.jsonl"
        if not dest.is_file():
            raise FileNotFoundError(f"refusing to invent {dest}")
        written[tag] = sum(1 for line in dest.read_text().splitlines() if line.strip())
    for tag in ("aime26", "hmmt25"):
        dest = PUMA_DATA / f"{tag}_test.jsonl"
        if dest.is_file() and not args.force:
            written[tag] = sum(1 for line in dest.read_text().splitlines() if line.strip())
            continue
        rows = convert_local(tag)
        write_jsonl(dest, rows)
        written[tag] = len(rows)
    dest = PUMA_DATA / "brumo25_test.jsonl"
    if dest.is_file() and not args.force:
        written["brumo25"] = sum(
            1 for line in dest.read_text().splitlines() if line.strip()
        )
    else:
        rows = [puma_row(item) for item in load_brumo()]
        write_jsonl(dest, rows)
        written["brumo25"] = len(rows)
    print(json.dumps({"puma_data": str(PUMA_DATA), "counts": written}, indent=2))
    missing = [tag for tag, count in written.items() if count <= 0]
    if missing:
        print(f"ERROR: empty contest sets: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
