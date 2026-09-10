#!/usr/bin/env python3
"""Official DEER host (Wait-ATP, conf > 0.95) from already-run aligned dumps.

Host is repos/DEER/vllm-deer.py: probe only at Wait (≤10), first pred_prob > 0.95
then still writes the final answer after </think>. Trial is a sensor, not the delivery.
Do not use every-step dense conf as DEER.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/deer_official_plus.md"
FIG = AE / "figures/deer"
LAMB = 0.95
K = 4
LOW = 0.995

JOBS = (
    ("7B MATH", "math500_ds7b", "math-500"),
    ("7B 奥赛", "olympiadbench_ds7b", "olympiadbench"),
    ("7B GPQA", "gpqa_diamond_ds7b", "gpqa-diamond"),
    ("7B AIME24", "aime24_ds7b", "aime24"),
    ("7B AIME25", "aime25_ds7b", "aime25"),
)


def puma_stat(dataset: str) -> Path:
    if dataset == "math-500":
        return AE / "results/math500_official/puma_ds7b/statistics.json"
    return AE / "results/puma_offline_r1_7b" / dataset / "statistics.json"


def puma_pair(dataset: str) -> tuple[float, float] | None:
    path = puma_stat(dataset)
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())
    n = max(len(rows), 1)
    acc = sum(int(r.get("compressed_correct")) for r in rows) / n
    tok = sum(
        int(r.get("compressed_tokens") or 0) + int(r.get("tokens_trial_answers") or 0)
        for r in rows
    ) / n
    return acc, tok


def host_stop(row: dict[str, Any]) -> dict[str, Any]:
    checks = list(row.get("checks") or [])
    early = None
    for i, ch in enumerate(checks):
        if ch.get("exit_reason") == "conf_exit" or (
            rg.finite(ch.get("conf")) == rg.finite(ch.get("conf"))
            and float(ch["conf"]) > LAMB
        ):
            early = i
            break
    last = checks[early] if early is not None else (checks[-1] if checks else None)
    tok = int((last or {}).get("think_token_len") or 0)
    return {
        "n_checks": len(checks),
        "early": early is not None,
        "stop_i": early if early is not None else len(checks),
        "tok": tok,
        "ok": bool(row.get("compressed_correct")),
        "has_g": row.get("G") is not None,
        "g": row.get("G"),
        "checks": checks,
        "a_final": row.get("model_answer"),
        "gt": row.get("ground_truth") or row.get("gt") or row.get("answer"),
    }


def low_window_before(host: dict[str, Any]) -> list[int]:
    """Ends of k=4 Wait probes all < 0.995, strictly before DEER conf_exit."""
    checks = host["checks"][: host["stop_i"]]
    out = []
    for end in range(K - 1, len(checks)):
        window = checks[end + 1 - K : end + 1]
        confs = [rg.finite(c.get("conf")) for c in window]
        if not all(c == c and c < LOW for c in confs):
            continue
        out.append(end)
    return out


def oracle_before(host: dict[str, Any]) -> dict[str, Any] | None:
    for i, ch in enumerate(host["checks"][: host["stop_i"]]):
        if rg.same(ch.get("trial_answer"), host.get("gt")):
            return {"i": i, "tok": int(ch.get("think_token_len") or 0), "conf": ch.get("conf")}
    return None


def main() -> None:
    lines = [
        "# 官方 DEER 底（已跑的 Wait-ATP）+ 附加门有没有空间",
        "",
        "实现按 `repos/DEER/vllm-deer.py`：生成碰到 Wait 才探（最多 10 次），把握用试答 boxed 的几何平均概率，",
        f"第一次 **> {LAMB}** 就截断思考，再写终答。试答只当传感器，交卷不是试答。",
        "数据：`figures/deer/*_ds7b/data/per_sample.json`（deer_aligned_wait_checks，7B 五集）。",
        "AIME 这里只有 30 题（一个 seed），不是四 seed。8B / 14B / 32B 没有这套对齐转储。",
        "默认对比率门要中间层好写，这些 Wait 探点上还没有层分数，所以这张表只报：官方 DEER 自己、停点前有没有低置信连探、连对上界。",
        "层门等 14B/32B/Qwen3 齐了、并且 DEER Wait 探点打了 lens，再用 `screen_shared_rel_layers.py` 同一套扫描。",
        "",
        "## 官方 DEER 自己",
        "",
        "| 集 | 题 | PUMA | DEER | 早停 | 平均探几次 | 停点思考token | 有G | 停在G之后才截 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    print(lines[-2], flush=True)
    rows_out = []
    for name, tag, dataset in JOBS:
        path = FIG / tag / "data/per_sample.json"
        if not path.is_file():
            print(f"missing {path}", flush=True)
            continue
        raw = json.loads(path.read_text())
        hosts = [host_stop(r) for r in raw]
        n = len(hosts)
        deer_acc = sum(int(h["ok"]) for h in hosts) / n
        deer_tok = sum(h["tok"] for h in hosts) / n
        n_early = sum(int(h["early"]) for h in hosts)
        n_probe = sum(h["n_checks"] for h in hosts) / n
        n_g = sum(int(h["has_g"]) for h in hosts)
        after_g = sum(
            int(h["has_g"] and int(h["g"]) < (h["stop_i"] + 1))
            for h in hosts
            if h["g"] is not None
        )
        puma = puma_pair(dataset)
        puma_s = sw.fmt_pair(*puma) if puma else "—"
        d_acc = ""
        d_tok = ""
        if puma:
            d_acc = rg.fmt_pp(100.0 * (deer_acc - puma[0]))
            d_tok = rg.fmt_tok(deer_tok - puma[1])
        row = (
            f"| {name} | {n} | {puma_s} | {sw.fmt_pair(deer_acc, deer_tok)} "
            f"（{d_acc} / {d_tok}） | {n_early}/{n} | {n_probe:.1f} | {deer_tok:.0f} | "
            f"{n_g}/{n} | {after_g} |"
        )
        print(row, flush=True)
        lines.append(row)
        rows_out.append((name, hosts, puma))

    lines += [
        "",
        "## 停点前还能不能挂附加门（只看官方 Wait 探点）",
        "",
        "| 集 | 停点前有探 | 停点前有 k=4 低置信连探 | 停点前试答已对金标 | 若先知交对金标的试答 vs DEER |",
        "|---|---:|---:|---:|---|",
    ]
    print(lines[-2], flush=True)
    for name, hosts, puma in rows_out:
        n = len(hosts)
        has_pre = room_k4 = ora_n = 0
        acc = tok = base_acc = base_tok = 0.0
        for h in hosts:
            base_acc += int(h["ok"])
            base_tok += h["tok"]
            pre = h["checks"][: h["stop_i"]]
            if pre:
                has_pre += 1
            if low_window_before(h):
                room_k4 += 1
            ora = oracle_before(h)
            if ora:
                ora_n += 1
                acc += 1
                tok += ora["tok"]
            else:
                acc += int(h["ok"])
                tok += h["tok"]
        d_acc = 100.0 * (acc / n - base_acc / n)
        d_tok = tok / n - base_tok / n
        row = (
            f"| {name} | {has_pre}/{n} | {room_k4}/{n} | {ora_n}/{n} | "
            f"{sw.fmt_pair(acc / n, tok / n)}（{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)}） |"
        )
        print(row, flush=True)
        lines.append(row)

    lines += [
        "",
        "读法：官方 DEER 早停少的集（GPQA、AIME），停点前几乎整段都能挂门，上界大。",
        "MATH / 奥赛大半题第一次 Wait 把握已经 >0.95，k=4 低置信窗根本凑不齐，默认对比率也开不了。",
        "上界是「停点前某个 Wait 试答已经对金标就交试答」；官方 DEER 不会这么交，只说明附加门有没有题可救。正确率只对金标。",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
