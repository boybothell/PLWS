#!/usr/bin/env python3
"""7B s42：等到第一扇非 Low（High/Mix）再压 vs 第一扇就压 vs Full-CoT。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
DSZH = {
    "math-500": "MATH",
    "olympiadbench": "oly",
    "gpqa-diamond": "GPQA",
    "aime24": "A24",
    "aime25": "A25",
}
TAG = "r1_7b"
SEED = 42


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


def tot(rec: dict) -> float:
    return float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)


def agg(xs: list[dict]) -> dict:
    n = len(xs)
    d = sum(x["ot"] for x in xs)
    return {
        "n": n,
        "a": 100 * sum(x["ok"] for x in xs) / n,
        "o": 100 * sum(x["orig"] for x in xs) / n,
        "t": sum(x["tot"] for x in xs) / n,
        "ot": sum(x["ot"] for x in xs) / n,
        "r": (sum(x["tot"] for x in xs) / d) if d else float("nan"),
    }


def main() -> None:
    jobs = [json.loads(l) for l in (AE / "results/first_hm_gate/jobs/r1_7b_s42.jsonl").open() if l.strip()]
    hm = load_scores(AE / "results/first_hm_gate/r1_7b_s42_suppress_hm")
    first: dict[str, dict] = {}
    for suf in ("", "_high", "_mix"):
        first.update(load_scores(AE / f"results/leftover_suppress_toend/{TAG}_s{SEED}_suppress{suf}"))
    windowed = set()
    folder = AE / f"results/leftover_jump/{TAG}_s{SEED}"
    for name in ("jobs.jsonl", "jobs_high.jsonl", "jobs_mix.jsonl"):
        path = folder / name
        if not path.is_file():
            continue
        for line in path.open():
            if not line.strip():
                continue
            j = json.loads(line)
            windowed.add((str(j["dataset"]), int(j["question_idx"])))
    hm_uids = {j["uid"] for j in jobs}
    job_by = {j["uid"]: j for j in jobs}
    miss = [j["uid"] for j in jobs if j["uid"] not in hm]
    print(f"first-HM jobs={len(jobs)} scored={len(hm)} pending={len(miss)}")

    print("\n## 整集（无非Low窗的题贴官方）")
    print("| 集 | n | 第一扇就压 | 等非Low再压 | Full-CoT | Δ扇 | Δ等 | 扇/官 | 等/官 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    all_fw, all_hm, all_hmq, all_del = [], [], [], []
    for ds in DATASETS:
        off = {int(r["question_idx"]): r for r in dd.load_json(dd.puma_stat_path(TAG, ds, SEED))}
        rows_fw, rows_hm, rows_hmq, rows_del = [], [], [], []
        for qi, info in off.items():
            uid = f"{TAG}:{ds}:{SEED}:{qi}"
            orig = bool(info.get("original_correct"))
            ot = float(info.get("original_tokens") or 0)
            rec_f, rec_h = first.get(uid), hm.get(uid)
            if rec_f:
                rows_fw.append(dict(ok=bool(rec_f.get("new_gold_ok")), orig=orig, tot=tot(rec_f), ot=ot))
            elif (ds, qi) not in windowed:
                rows_fw.append(dict(ok=orig, orig=orig, tot=ot, ot=ot))
            if rec_h:
                row = dict(ok=bool(rec_h.get("new_gold_ok")), orig=orig, tot=tot(rec_h), ot=ot)
                rows_hm.append(row)
                rows_hmq.append(row)
                if job_by[uid].get("delayed"):
                    if rec_f:
                        rows_del.append(
                            dict(
                                ok=bool(rec_h.get("new_gold_ok")),
                                fw=bool(rec_f.get("new_gold_ok")),
                                orig=orig,
                                tot=tot(rec_h),
                                ftot=tot(rec_f),
                                ot=ot,
                            )
                        )
            elif uid not in hm_uids:
                rows_hm.append(dict(ok=orig, orig=orig, tot=ot, ot=ot))
        af, ah = agg(rows_fw), agg(rows_hm)
        all_fw.extend(rows_fw)
        all_hm.extend(rows_hm)
        all_hmq.extend(rows_hmq)
        all_del.extend(rows_del)
        print(
            f"| {DSZH[ds]} | {ah['n']} | {af['a']:.1f}% | {ah['a']:.1f}% | {ah['o']:.1f}% | "
            f"{af['a']-ah['o']:+.1f} | {ah['a']-ah['o']:+.1f} | {af['r']:.2f} | {ah['r']:.2f} |"
        )
    af, ah = agg(all_fw), agg(all_hm)
    print(
        f"| 全部 | {ah['n']} | {af['a']:.1f}% | {ah['a']:.1f}% | {ah['o']:.1f}% | "
        f"{af['a']-ah['o']:+.1f} | {ah['a']-ah['o']:+.1f} | {af['r']:.2f} | {ah['r']:.2f} |"
    )

    print("\n## 只看出现过非Low窗的题（1027）")
    print("| 集 | n | 第一扇就压 | 等非Low再压 | Full-CoT | 等-扇 | 等-官 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    by_ds: dict[str, list] = {ds: [] for ds in DATASETS}
    reuse_rows, delay_cmp = [], []
    for j in jobs:
        rec_h, rec_f = hm.get(j["uid"]), first.get(j["uid"])
        if not rec_h:
            continue
        info_ok = bool(j.get("orig_ok") if "orig_ok" in j else j.get("host_ok"))
        ot = None
        # official tokens from puma if needed later; skip if missing rec_f for delayed cmp
        row = dict(
            ok=bool(rec_h.get("new_gold_ok")),
            orig=info_ok,
            tot=tot(rec_h),
            ot=1.0,
            fw=bool(rec_f.get("new_gold_ok")) if rec_f else None,
            delayed=bool(j.get("delayed")),
            kind=j.get("kind"),
            first_kind=j.get("first_kind"),
        )
        by_ds[j["dataset"]].append(row)
        if rec_f:
            if j.get("delayed"):
                delay_cmp.append(
                    dict(
                        ok=bool(rec_h.get("new_gold_ok")),
                        fw=bool(rec_f.get("new_gold_ok")),
                        orig=info_ok,
                    )
                )
            else:
                reuse_rows.append(row)

    for ds in DATASETS:
        xs = by_ds[ds]
        if not xs:
            continue
        n = len(xs)
        hm_a = 100 * sum(x["ok"] for x in xs) / n
        fw_xs = [x for x in xs if x["fw"] is not None]
        fw_a = 100 * sum(x["fw"] for x in fw_xs) / len(fw_xs) if fw_xs else float("nan")
        o = 100 * sum(x["orig"] for x in xs) / n
        print(f"| {DSZH[ds]} | {n} | {fw_a:.1f}% | {hm_a:.1f}% | {o:.1f}% | {hm_a-fw_a:+.1f} | {hm_a-o:+.1f} |")
    xs = [x for ds in DATASETS for x in by_ds[ds]]
    n = len(xs)
    hm_a = 100 * sum(x["ok"] for x in xs) / n
    fw_xs = [x for x in xs if x["fw"] is not None]
    fw_a = 100 * sum(x["fw"] for x in fw_xs) / len(fw_xs)
    o = 100 * sum(x["orig"] for x in xs) / n
    print(f"| 全部 | {n} | {fw_a:.1f}% | {hm_a:.1f}% | {o:.1f}% | {hm_a-fw_a:+.1f} | {hm_a-o:+.1f} |")

    print("\n## 拆开：第一扇已是非Low（复用） vs 第一扇Low等到非Low（新跑）")
    print("| 子集 | n | 第一扇就压 | 等非Low | Full-CoT |")
    print("|---|---:|---:|---:|---:|")
    already = [x for x in xs if not x["delayed"] and x["fw"] is not None]
    delayed = [x for x in xs if x["delayed"] and x["fw"] is not None]
    for name, ys in (("第一扇已是H/M", already), ("第一扇Low，等到H/M", delayed)):
        n = len(ys)
        print(
            f"| {name} | {n} | {100*sum(x['fw'] for x in ys)/n:.1f}% | "
            f"{100*sum(x['ok'] for x in ys)/n:.1f}% | {100*sum(x['orig'] for x in ys)/n:.1f}% |"
        )

    print("\n## delayed 分集")
    print("| 集 | n | 第一扇Low压 | 等到H/M | Full-CoT | later |")
    print("|---|---:|---:|---:|---:|---|")
    for ds in DATASETS:
        ys = [x for x in by_ds[ds] if x["delayed"] and x["fw"] is not None]
        if not ys:
            continue
        n = len(ys)
        later = ""
        from collections import Counter
        c = Counter(x["kind"] for x in ys)
        later = f"H{c['high']} M{c['mix']}"
        print(
            f"| {DSZH[ds]} | {n} | {100*sum(x['fw'] for x in ys)/n:.1f}% | "
            f"{100*sum(x['ok'] for x in ys)/n:.1f}% | {100*sum(x['orig'] for x in ys)/n:.1f}% | {later} |"
        )


if __name__ == "__main__":
    main()
