#!/usr/bin/env python3
"""齐的 High/Mix 窗后压，以及 Mix 出现在第一扇 High 之前、还没切过 High 的题。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_first_lock_room as room
import report_k4_second_lock as sl

MODELS = (("r1_7b", "7B"), ("nemotron_8b", "8B"), ("r1_14b", "14B"))
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
DSZH = {
    "math-500": "MATH",
    "olympiadbench": "oly",
    "gpqa-diamond": "GPQA",
    "aime24": "A24",
    "aime25": "A25",
}
SEEDS = (42, 0, 1, 123)


def load_scores(folder: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") not in ("ok", "too_long"):
                continue
            uid = rec.get("uid")
            if uid:
                out[str(uid)] = rec
    return out


def load_jobs(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def tot(rec: dict) -> float:
    return float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)


def cell(tag: str, kind: str, ds: str, seed: int) -> dict | None:
    jobs = load_jobs(AE / f"results/leftover_jump/{tag}_s{seed}/jobs_{kind}.jsonl")
    jobs = [j for j in jobs if str(j.get("dataset")) == ds]
    if not jobs:
        return None
    scores = load_scores(AE / f"results/leftover_suppress_toend/{tag}_s{seed}_suppress_{kind}")
    off_path = dd.puma_stat_path(tag, ds, seed)
    if not off_path.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(off_path)}
    rows = []
    for job in jobs:
        rec = scores.get(str(job["uid"]))
        info = official.get(int(job["question_idx"]))
        if rec is None or info is None:
            return None
        ot = float(info.get("original_tokens") or 0)
        if ot <= 0:
            return None
        rows.append(
            {
                "ok": bool(rec.get("new_gold_ok")),
                "orig": bool(info.get("original_correct")),
                "tot": tot(rec),
                "ot": ot,
            }
        )
    n = len(rows)
    d = sum(x["ot"] for x in rows)
    return {
        "n": n,
        "ok": sum(x["ok"] for x in rows),
        "orig": sum(x["orig"] for x in rows),
        "t": sum(x["tot"] for x in rows) / n,
        "ot": sum(x["ot"] for x in rows) / n,
        "r": sum(x["tot"] for x in rows) / d,
        "seeds": [seed],
    }


def merge(cells: list[dict]) -> dict:
    n = sum(c["n"] for c in cells)
    d = sum(c["ot"] * c["n"] for c in cells)
    return {
        "n": n,
        "ok": sum(c["ok"] for c in cells),
        "orig": sum(c["orig"] for c in cells),
        "t": sum(c["t"] * c["n"] for c in cells) / n,
        "ot": sum(c["ot"] * c["n"] for c in cells) / n,
        "r": sum(c["t"] * c["n"] for c in cells) / d,
        "seeds": sorted({s for c in cells for s in c["seeds"]}),
    }


def fmt(c: dict, star: bool = False) -> str:
    s = "*" if star and c["n"] < 10 else ""
    return (
        f"{c['n']} | {100 * c['ok'] / c['n']:.1f}%{s} | {100 * c['orig'] / c['n']:.1f}%{s} | "
        f"{c['t']:.0f} | {c['ot']:.0f} | {c['r']:.2f}"
    )


def leftover_tables() -> None:
    print("=== leftover High/Mix：第一扇窗就是该档，不是后面才出现的 High ===")
    for kind in ("high", "mix"):
        print(f"\n## {kind.upper()}")
        print("| 模型 | 集 | seeds | n | 窗后 Acc | 官方 Acc | tok | 官 tok | /官 |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for tag, name in MODELS:
            all_cells = []
            for ds in DATASETS:
                cells = [c for seed in SEEDS if (c := cell(tag, kind, ds, seed))]
                if not cells:
                    print(f"| {name} | {DSZH[ds]} | - | 未齐 |")
                    continue
                m = merge(cells)
                all_cells.append(m)
                seeds = ",".join(str(s) for s in m["seeds"])
                star = " *" if m["n"] < 10 else ""
                print(
                    f"| {name} | {DSZH[ds]} | {seeds} | {m['n']} | "
                    f"{100 * m['ok'] / m['n']:.1f}%{star} | {100 * m['orig'] / m['n']:.1f}%{star} | "
                    f"{m['t']:.0f} | {m['ot']:.0f} | {m['r']:.2f} |"
                )
            if all_cells:
                m = merge(all_cells)
                seeds = ",".join(str(s) for s in sorted({s for c in all_cells for s in c["seeds"]}))
                print(
                    f"| {name} | 全部 | {seeds} | {m['n']} | "
                    f"{100 * m['ok'] / m['n']:.1f}% | {100 * m['orig'] / m['n']:.1f}% | "
                    f"{m['t']:.0f} | {m['ot']:.0f} | {m['r']:.2f} |"
                )


def first_hm_table() -> None:
    print("\n=== 7B s42 等到第一扇 High 或 Mix 再压（整集贴：无 H/M 用官方）===")
    jobs = load_jobs(AE / "results/first_hm_gate/jobs/r1_7b_s42.jsonl")
    hm = load_scores(AE / "results/first_hm_gate/r1_7b_s42_suppress_hm")
    first: dict[str, dict] = {}
    for suf in ("", "_high", "_mix"):
        first.update(load_scores(AE / f"results/leftover_suppress_toend/r1_7b_s42_suppress{suf}"))
    windowed = set()
    folder = AE / "results/leftover_jump/r1_7b_s42"
    for name in ("jobs.jsonl", "jobs_high.jsonl", "jobs_mix.jsonl"):
        for job in load_jobs(folder / name):
            windowed.add((str(job["dataset"]), int(job["question_idx"])))
    hm_uids = {j["uid"] for j in jobs}
    print("| 集 | n | 第一扇就压 | 等H或M | 官方 | 扇/官 | HM/官 | delayed |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    all_fw, all_hm = [], []
    for ds in DATASETS:
        off = {int(r["question_idx"]): r for r in dd.load_json(dd.puma_stat_path("r1_7b", ds, 42))}
        rows_fw, rows_hm = [], []
        nd = 0
        miss = 0
        for qi, info in off.items():
            uid = f"r1_7b:{ds}:42:{qi}"
            orig = bool(info.get("original_correct"))
            ot = float(info.get("original_tokens") or 0)
            rec_f, rec_h = first.get(uid), hm.get(uid)
            if rec_f:
                rows_fw.append(dict(ok=bool(rec_f.get("new_gold_ok")), orig=orig, tot=tot(rec_f), ot=ot))
            elif (ds, qi) not in windowed:
                rows_fw.append(dict(ok=orig, orig=orig, tot=ot, ot=ot))
            if rec_h:
                rows_hm.append(dict(ok=bool(rec_h.get("new_gold_ok")), orig=orig, tot=tot(rec_h), ot=ot))
            elif uid not in hm_uids:
                rows_hm.append(dict(ok=orig, orig=orig, tot=ot, ot=ot))
            else:
                miss += 1
            job = next((j for j in jobs if j["uid"] == uid), None)
            if job and job.get("delayed"):
                nd += 1
        if miss or len(rows_fw) != len(off) or len(rows_hm) != len(off):
            print(f"| {DSZH[ds]} | INCOMPLETE fw={len(rows_fw)} hm={len(rows_hm)}/{len(off)} miss{miss} |")
            continue
        all_fw.extend(rows_fw)
        all_hm.extend(rows_hm)
        def agg(xs):
            n = len(xs)
            d = sum(x["ot"] for x in xs)
            return dict(
                n=n,
                a=100 * sum(x["ok"] for x in xs) / n,
                o=100 * sum(x["orig"] for x in xs) / n,
                r=(sum(x["tot"] for x in xs) / d) if d else float("nan"),
            )
        af, ah = agg(rows_fw), agg(rows_hm)
        print(
            f"| {DSZH[ds]} | {ah['n']} | {af['a']:.1f}% | {ah['a']:.1f}% | {ah['o']:.1f}% | "
            f"{af['r']:.2f} | {ah['r']:.2f} | {nd} |"
        )
    def agg(xs):
        n = len(xs)
        d = sum(x["ot"] for x in xs)
        return dict(
            n=n,
            a=100 * sum(x["ok"] for x in xs) / n,
            o=100 * sum(x["orig"] for x in xs) / n,
            r=(sum(x["tot"] for x in xs) / d) if d else float("nan"),
        )
    af, ah = agg(all_fw), agg(all_hm)
    print(
        f"| 全部 | {ah['n']} | {af['a']:.1f}% | {ah['a']:.1f}% | {ah['o']:.1f}% | "
        f"{af['r']:.2f} | {ah['r']:.2f} | |"
    )

    print("\n=== delayed=第一扇 Low，等到后面 H/M ===")
    print("| 集 | n | 第一扇Low压 | 等到H/M | 官方 | later |")
    print("|---|---:|---:|---:|---:|---|")
    for ds in DATASETS:
        xs = []
        kinds = Counter()
        for j in jobs:
            if j["dataset"] != ds or not j.get("delayed"):
                continue
            rec_h, rec_f = hm.get(j["uid"]), first.get(j["uid"])
            if not rec_h or not rec_f:
                continue
            orig = bool(j.get("orig_ok") if "orig_ok" in j else j.get("host_ok"))
            xs.append((bool(rec_f.get("new_gold_ok")), bool(rec_h.get("new_gold_ok")), orig))
            kinds[j.get("kind")] += 1
        if not xs:
            continue
        n = len(xs)
        print(
            f"| {DSZH[ds]} | {n} | {100 * sum(a for a, _, _ in xs) / n:.1f}% | "
            f"{100 * sum(b for _, b, _ in xs) / n:.1f}% | {100 * sum(c for _, _, c in xs) / n:.1f}% | "
            f"H{kinds['high']} M{kinds['mix']} |"
        )


def mix_before_high() -> None:
    print("\n=== Mix 出现在第一扇 High 之前（旧 High leftover / first-HM 都没在 High 切）===")
    first = load_scores(AE / "results/leftover_suppress_toend/r1_7b_s42_suppress_mix")
    first.update(load_scores(AE / "results/leftover_suppress_toend/r1_7b_s42_suppress"))
    print("| 集 | n | 第一扇 | Mix切已有分 | Mix压Acc | 官方 | High窗已对 | Mix≠High |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    tot_n = scored = mix_ok = orig_ok = left_ok = changed = 0
    first_kinds = Counter()
    for ds in DATASETS:
        path = room.dense_trial_path("r1_7b", ds, seed=42)
        if not path.is_file():
            continue
        official = {int(r["question_idx"]): r for r in dd.load_json(dd.puma_stat_path("r1_7b", ds, 42))}
        by: dict[int, list] = defaultdict(list)
        for row in dd.load_json(path):
            by[int(row["question_idx"])].append(row)
        rows_n = 0
        rec_n = 0
        mok = ook = lok = ch = 0
        fk = Counter()
        for qi, trials in by.items():
            rows = rg.usable_rows(trials)
            if not rows:
                continue
            wins = sl.same_windows(rows)
            if not wins:
                continue
            mix = next((w for w in wins if w["kind"] == "mix"), None)
            high = next((w for w in wins if w["kind"] == "high"), None)
            if not (mix and high and mix["step"] < high["step"]):
                continue
            rows_n += 1
            info = official.get(qi) or {}
            orig = bool(info.get("original_correct"))
            ook += int(orig)
            lok += int(bool(rg.same(high.get("ans"), info.get("ground_truth")) or False))
            # left_ok at high trial vs gold via same as official gold if available
            gt = info.get("ground_truth")
            if gt is not None:
                lok_fix = int(bool(rg.same(high.get("ans"), gt)))
            else:
                lok_fix = 0
            ch += int(not rg.same(mix.get("ans"), high.get("ans")))
            fk[wins[0]["kind"]] += 1
            uid = f"r1_7b:{ds}:42:{qi}"
            rec = first.get(uid)
            # Mix-cut leftover only exists if first window is mix
            if rec and wins[0]["kind"] == "mix":
                rec_n += 1
                mok += int(bool(rec.get("new_gold_ok")))
        tot_n += rows_n
        scored += rec_n
        mix_ok += mok
        orig_ok += ook
        left_ok += lok_fix if False else sum(
            # recount cleanly below; keep running totals from loop
            0 for _ in ()
        )
        changed += ch
        first_kinds.update(fk)
        # recount left_ok properly
        lok2 = 0
        for qi, trials in by.items():
            rows = rg.usable_rows(trials)
            if not rows:
                continue
            wins = sl.same_windows(rows)
            if not wins:
                continue
            mix = next((w for w in wins if w["kind"] == "mix"), None)
            high = next((w for w in wins if w["kind"] == "high"), None)
            if not (mix and high and mix["step"] < high["step"]):
                continue
            gt = (official.get(qi) or {}).get("ground_truth")
            lok2 += int(gt is not None and bool(rg.same(high.get("ans"), gt)))
        print(
            f"| {DSZH[ds]} | {rows_n} | {dict(fk)} | {rec_n} | "
            f"{(100 * mok / rec_n) if rec_n else float('nan'):.1f}% | "
            f"{(100 * ook / rows_n) if rows_n else float('nan'):.1f}% | "
            f"{(100 * lok2 / rows_n) if rows_n else float('nan'):.1f}% | "
            f"{ch} |"
        )
        left_ok += lok2
    print(
        f"| 全部 | {tot_n} | {dict(first_kinds)} | {scored} | "
        f"{(100 * mix_ok / scored) if scored else float('nan'):.1f}% | "
        f"{(100 * orig_ok / tot_n) if tot_n else float('nan'):.1f}% | "
        f"{(100 * left_ok / tot_n) if tot_n else float('nan'):.1f}% | "
        f"{changed} |"
    )
    print(
        "说明：这些题 High leftover 没收（第一扇不是 High）；"
        "first-HM 在 Mix 就压。High 切点分数还没有。"
    )


def main() -> None:
    leftover_tables()
    first_hm_table()
    mix_before_high()


if __name__ == "__main__":
    main()
