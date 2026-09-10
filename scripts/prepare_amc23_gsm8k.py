#!/usr/bin/env python3
"""从本机 DEER 数据写出 PUMA 格式 amc23 / gsm8k：每行 {question, answer}。"""
from __future__ import annotations

import json
from pathlib import Path

DEER = Path("/mnt/d/lsj/visual-latent-tts/repos/DEER/data")
PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA/data")


def fmt_answer(raw) -> str:
    if isinstance(raw, float) and raw == int(raw):
        return str(int(raw))
    return str(raw).strip()


def convert(src: Path, dest: Path, expect: int) -> None:
    rows = []
    for line in src.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        q = rec.get("question") or rec.get("problem")
        a = rec.get("answer")
        if q is None or a is None:
            raise SystemExit(f"bad row in {src}: keys={list(rec)}")
        rows.append({"question": str(q).strip(), "answer": fmt_answer(a)})
    if len(rows) != expect:
        print(f"WARN {dest.name} n={len(rows)} expected {expect}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {dest}", flush=True)


def main() -> None:
    convert(DEER / "amc/test.jsonl", PUMA / "amc23_test.jsonl", 40)
    convert(DEER / "gsm8k/test.jsonl", PUMA / "gsm8k_test.jsonl", 1319)


if __name__ == "__main__":
    main()

