#!/usr/bin/env python3
"""Summarize ASAG vs DEER reproduction: Acc, tokens, how often each action fired."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

AE = Path(__file__).resolve().parents[1]


def load_dir(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("shard*.jsonl")):
        for line in path.open():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"{name}: 还没有结果\n"
    n = len(rows)
    acc = sum(int(r.get("correct") or 0) for r in rows) / n
    tok = sum(int(r.get("n_tokens") or 0) for r in rows) / n
    waits = sum(int(r.get("n_wait") or 0) for r in rows)
    exits = sum(int(r.get("n_exit_decisions") or 0) for r in rows)
    injects = sum(int(r.get("n_inject_decisions") or 0) for r in rows)
    jumps = sum(int(r.get("n_jump_decisions") or 0) for r in rows)
    cont = sum(int(r.get("n_continue_decisions") or 0) for r in rows)
    probes = exits + injects + jumps + cont
    q_inj = sum(1 for r in rows if int(r.get("n_inject_decisions") or 0) > 0)
    q_jump = sum(1 for r in rows if int(r.get("n_jump_decisions") or 0) > 0)
    q_exit = sum(1 for r in rows if int(r.get("exited") or 0))
    lines = [
        f"## {name}（{n} 题）",
        f"- 对标准答案：{acc:.1%}（{sum(int(r.get('correct') or 0) for r in rows)}/{n}）",
        f"- 人均生成 token：{tok:.0f}",
        f"- 在 Wait 处做了 {probes} 次探，其中停 {exits}、灌试答 {injects}、插入换路 {jumps}、继续 {cont}",
        f"- 至少灌过一次试答的题：{q_inj}；至少插入过一次换路的题：{q_jump}；早停过的题：{q_exit}",
        f"- 一共碰到 Wait {waits} 次",
        "",
    ]
    acts = Counter()
    for row in rows:
        for dec in row.get("decisions") or []:
            acts[str(dec.get("action"))] += 1
            if dec.get("action") == "inject":
                lines.append(
                    f"  灌试答：题 {row['question_idx']} 把握 {dec.get('conf'):.3f} 熵变化 {dec.get('delta_h')}"
                )
            if dec.get("action") == "jump":
                lines.append(
                    f"  换路：题 {row['question_idx']} 把握 {dec.get('conf'):.3f} 熵变化 {dec.get('delta_h')}"
                )
    return "\n".join(lines)


def main() -> None:
    root = AE / "results/asag_r1_7b/math-500"
    chunks = []
    for mode in ("asag", "deer", "asag_no_inject", "asag_no_jump", "vanilla"):
        folder = root / mode
        if folder.exists():
            chunks.append(summarize(mode, load_dir(folder)))
    text = "# ASAG 复现（R1-7B，MATH 子集，贪心）\n\n" + "\n".join(chunks)
    out = AE / "tables/probe_asag_repro_math-500.md"
    out.write_text(text + "\n")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
