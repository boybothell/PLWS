#!/usr/bin/env python3
"""剩窗写到收口：压 Wait vs 不压的 Acc 和 token。

Acc 只报有完整 generated_text 的题，抽取 / 判分走官方 PUMA：
`run_vllm.extract_answer` + `math_grader.check_is_correct`。
没有全文的旧残片不进正确率。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(PUMA / "puma"))

from score_confcal_v1 import nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()

from math_grader import check_is_correct  # noqa: E402
from prompt_utils import get_task_type  # noqa: E402
from run_vllm import extract_answer  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402

ROOT = AE / "results/leftover_suppress_toend"
JOBS = AE / "results/leftover_jump/r1_7b_s42/jobs.jsonl"
TABLE = AE / "tables/leftover_suppress_tok.md"


def load_jobs() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not JOBS.is_file():
        return rows
    for line in JOBS.read_text().splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        rows[job["uid"]] = job
    return rows


def official_ok(answer: Any, gt: Any) -> bool:
    try:
        return bool(check_is_correct(answer, gt))
    except Exception:
        return False


def load_rows(folder: Path, jobs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            job = jobs.get(row["uid"], {})
            row = dict(row)
            text = row.get("generated_text")
            row["official"] = bool(text)
            row["orig_ok"] = bool(job.get("orig_ok", False))
            if text:
                task_type = row.get("task_type") or get_task_type(row.get("dataset") or job.get("dataset") or "")
                answer = extract_answer(text, task_type)
                row["new_answer"] = answer
                row["new_gold_ok"] = official_ok(answer, job.get("gt"))
                row["keep"] = bool(rg.same(row.get("old_answer"), answer))
            else:
                row["new_gold_ok"] = False
            rows.append(row)
    return rows


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def med(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def pct(xs: list[dict[str, Any]], key: str) -> str:
    if not xs:
        return "—"
    return f"{100.0 * mean([int(bool(x.get(key))) for x in xs]):.1f}%"


def line(name: str, xs: list[dict[str, Any]]) -> str:
    if not xs:
        return f"| {name} | 0 | — | — | — | — | — | — | — | — |"
    official = [x for x in xs if x.get("official")]
    acc = pct(official, "new_gold_ok") if official else "—"
    cont = [float(x.get("n_cont_tok") or 0) for x in xs]
    think = [float(x.get("n_think_tok") or 0) for x in xs]
    cap = sum(1 for x in xs if x.get("hit_cap"))
    return (
        f"| {name} | {len(xs)} | {acc} | {pct(xs, 'left_ok')} | "
        f"{pct(xs, 'host_ok')} | {pct(xs, 'orig_ok')} | "
        f"{mean(cont):.0f} / {med(cont):.0f} | {mean(think):.0f} | "
        f"{cap}/{len(xs)} | {len(official)}/{len(xs)} |"
    )


def main() -> None:
    tag = "r1_7b"
    jobs = load_jobs()
    suppress = load_rows(ROOT / f"{tag}_s42_suppress", jobs)
    free = load_rows(ROOT / f"{tag}_s42_free", jobs)
    official_sup = [x for x in suppress if x.get("official")]
    official_free = [x for x in free if x.get("official")]
    both_ids = {x["uid"] for x in official_sup} & {x["uid"] for x in official_free}
    sup_b = [x for x in official_sup if x["uid"] in both_ids]
    free_b = [x for x in official_free if x["uid"] in both_ids]
    free_by = {x["uid"]: x for x in free_b}
    d_acc = d_cont = float("nan")
    if sup_b:
        d_acc = 100.0 * (
            mean([int(bool(x.get("new_gold_ok"))) for x in sup_b])
            - mean([int(bool(free_by[x["uid"]].get("new_gold_ok"))) for x in sup_b])
        )
        d_cont = mean([float(x.get("n_cont_tok") or 0) for x in sup_b]) - mean(
            [float(free_by[x["uid"]].get("n_cont_tok") or 0) for x in sup_b]
        )
    pair_txt = (
        f"官方成对题 {len(sup_b)}。压 Wait 相对不压：正确率 {d_acc:+.1f}pp，续写 {d_cont:+.0f} token。"
        if sup_b
        else "还没有官方全文成对题。正确率空着，不报残片 Acc。"
    )
    lines = [
        "# 剩窗写到收口：压 Wait 的 Acc / token",
        "",
        "7B 低把握四步同答窗。从锁点前缀写思考；自己收到 `</think>` 或顶满后强行塞 `</think>`，再写终答。",
        "压 Wait：思考阶段禁止 `Wait` / `Alternatively` / `Hmm`。不压：同一预算，这些词可以出现。",
        "新答正确率只走官方 Full-CoT 协议：对完整 `generated_text` 调用 `PUMA/puma/run_vllm.py` 的 `extract_answer`，再用 `math_grader.check_is_correct` 对金标。",
        "锁点试答 / 密探仍是试答协议，只作对照。原 Full-CoT 是官方 `original_correct`。",
        "续写 token = 锁点之后新写出的思考词。思考 token = 锁点前缀 + 续写。",
        pair_txt,
        "",
        "| 做法 | 题 | 新答正确率 | 锁点试答 | 密探 | 原 Full-CoT | 续写 token 均值/中位 | 思考 token | 顶到上限 | 官方全文 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|",
        line("压 Wait", suppress),
        line("不压", free),
        line("压 Wait（成对）", sup_b),
        line("不压（成对）", free_b),
        "",
    ]
    for ds, zh in (("math-500", "MATH"), ("olympiadbench", "奥赛"), ("gpqa-diamond", "GPQA")):
        lines += [
            f"## {zh}",
            "",
            "| 做法 | 题 | 新答正确率 | 锁点试答 | 密探 | 原 Full-CoT | 续写 token 均值/中位 | 思考 token | 顶到上限 | 官方全文 |",
            "|---|---:|---:|---:|---:|---:|---|---:|---|---|",
            line("压 Wait", [x for x in suppress if x.get("dataset") == ds]),
            line("不压", [x for x in free if x.get("dataset") == ds]),
        ]
        ds_sup_b = [x for x in sup_b if x.get("dataset") == ds]
        ds_free_b = [x for x in free_b if x.get("dataset") == ds]
        if ds_sup_b or ds_free_b:
            lines += [
                line("压 Wait（成对）", ds_sup_b),
                line("不压（成对）", ds_free_b),
            ]
        lines += [""]
    if sup_b and not math.isnan(d_acc):
        if d_acc < -1.0:
            read = "正确率掉了，方向 1 不能当方法。"
        elif d_cont > -50:
            read = "正确率还在，但续写几乎没短下来。锁后压 Wait 省不了这摊题的后半段。"
        else:
            read = "正确率还在，续写短了。这只是剩窗后半段，不是从题面重写整集。"
    else:
        read = "正确率等官方全文。残片 Acc 不再报。"
    lines += ["## 读法", "", read, ""]
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
