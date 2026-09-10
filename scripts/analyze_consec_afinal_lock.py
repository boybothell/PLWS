#!/usr/bin/env python3
"""How many consecutive A_final trials lock the rest of the trajectory."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE))

from attn_early_exit.answers import answers_equal  # noqa: E402

_WS = re.compile(r"\s+")
KS = (1, 2, 3, 4, 5, 6)
DATASETS = (
    ("math-500", "math-500"),
    ("olympiadbench", "olympiadbench"),
    ("gpqa-diamond", "gpqa-diamond"),
)
TABLE = AE / "tables/probe_consec_afinal_lock.md"


def fast_eq(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    x = _WS.sub("", str(a).strip().lower())
    y = _WS.sub("", str(b).strip().lower())
    return bool(x) and x == y


def dense_root(dataset: str) -> Path:
    if dataset in {"math-500", "olympiadbench", "gpqa-diamond"}:
        return AE / f"results/dense_G_r1_7b/{dataset}/dense_puma"
    return AE / f"results/dense_G_r1_7b/{dataset}/s42/dense_puma"


def load_a_final(dataset: str) -> dict[int, str]:
    path = AE / f"results/dense_G_r1_7b/{dataset}/per_sample.json"
    rows = json.loads(path.read_text())
    return {int(r["question_idx"]): r.get("A_final") for r in rows}


def load_trials(dataset: str) -> dict[int, list[tuple[int, str | None]]]:
    path = dense_root(dataset) / "trial_answers.json"
    print(f"  load {path}", flush=True)
    raw = json.loads(path.read_text())
    by: dict[int, list[tuple[int, str | None]]] = defaultdict(list)
    for row in raw:
        by[int(row["question_idx"])].append(
            (int(row["stopped_len"]), row.get("final_answer"))
        )
    for qi in by:
        by[qi].sort(key=lambda x: x[0])
    print(f"  questions={len(by)} trials={len(raw)}", flush=True)
    return by


def match_seq(a_final: str | None, seq: list[tuple[int, str | None]]) -> list[bool]:
    known_pos: list[str] = []
    out: list[bool] = []
    cache: dict[str, bool] = {}
    for _, ans in seq:
        key = "" if ans is None else str(ans)
        if key in cache:
            out.append(cache[key])
            continue
        hit = fast_eq(ans, a_final)
        if not hit:
            hit = any(fast_eq(ans, prev) for prev in known_pos)
        if not hit and ans and a_final:
            hit = bool(answers_equal(ans, a_final))
        cache[key] = hit
        if hit and ans is not None:
            known_pos.append(str(ans))
        out.append(hit)
    return out


def first_k_run(match: list[bool], k: int) -> int | None:
    run = 0
    for i, ok in enumerate(match):
        run = run + 1 if ok else 0
        if run >= k:
            return i
    return None


def eval_dataset(name: str) -> dict:
    a_finals = load_a_final(name)
    trials = load_trials(name)
    rows = []
    n_has_g = 0
    for qi, seq in sorted(trials.items()):
        match = match_seq(a_finals.get(qi), seq)
        if any(match):
            n_has_g += 1
        rows.append(match)
        if len(rows) % 50 == 0:
            print(f"  scored {len(rows)}/{len(trials)}", flush=True)

    n = len(rows)
    by_k = {}
    for k in KS:
        reached = later = flip = 0
        for match in rows:
            end = first_k_run(match, k)
            if end is None:
                continue
            reached += 1
            tail = match[end + 1 :]
            if tail:
                later += 1
                if not all(tail):
                    flip += 1
        by_k[k] = {
            "reached": reached,
            "reached_frac": reached / n if n else 0.0,
            "with_later": later,
            "flip": flip,
            "flip_frac_reached": flip / reached if reached else 0.0,
            "stable_frac_reached": (reached - flip) / reached if reached else 0.0,
        }

    g_flip = g_n = 0
    for match in rows:
        if not any(match):
            continue
        g_n += 1
        tail = match[match.index(True) + 1 :]
        if tail and not all(tail):
            g_flip += 1
    return {
        "dataset": name,
        "n": n,
        "n_has_g": n_has_g,
        "g_flip": g_flip,
        "g_flip_frac": g_flip / g_n if g_n else 0.0,
        "by_k": by_k,
    }


def fmt(p: float) -> str:
    return f"{100 * p:.1f}%"


def main() -> None:
    results = []
    for name, _ in DATASETS:
        if not (AE / f"results/dense_G_r1_7b/{name}/per_sample.json").exists():
            print(f"skip {name}", flush=True)
            continue
        print(f"eval {name}", flush=True)
        results.append(eval_dataset(name))

    lines = [
        "# 连续试答 = Full-CoT `A_final` 之后还会不会变",
        "",
        "dense 每步试答。`A_final` 来自 `per_sample.json`；匹配与 `compute_dense_G.py` 相同（先规范化，再 ≈）。",
        "对每个 k：第一次连续 k 次试答 = `A_final` 之后，后面是否还出现 ≠ `A_final`。",
        "",
        "| 数据集 | n / 有 G | 首次 G 后仍离开 |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['n']} / {r['n_has_g']} | "
            f"{r['g_flip']} / {r['n_has_g']} ({fmt(r['g_flip_frac'])}) |"
        )
    lines += [
        "",
        "| 数据集 | k | 能达到 | 占全题 | 有后续 | 后续离开 | 达到后稳定 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        for k in KS:
            b = r["by_k"][k]
            lines.append(
                f"| {r['dataset']} | {k} | {b['reached']} | {fmt(b['reached_frac'])} | "
                f"{b['with_later']} | {b['flip']} ({fmt(b['flip_frac_reached'])}) | "
                f"{fmt(b['stable_frac_reached'])} |"
            )
    lines += [
        "",
        "## 判定",
        "",
        "对照只对 PUMA。方法 work = Acc 不差于 PUMA（对 GT），并且停点落在",
        "「连续 k 次试答已经是 `A_final`」——过了这个 k，后面再试答基本不变。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
