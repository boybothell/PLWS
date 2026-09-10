#!/usr/bin/env python3
"""7B 四种子：等第一扇非 Low 再压 vs 第一扇就压 vs Full-CoT。"""
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


def tot(rec: dict) -> float:
    return float(rec.get("n_think_tok") or 0) + float(rec.get("n_ans_tok") or 0)


def agg(xs: list[dict]) -> dict:
    n = len(xs)
    d = sum(x["ot"] for x in xs)
    return {
        "n": n,
        "ok": sum(x["ok"] for x in xs),
        "orig": sum(x["orig"] for x in xs),
        "fw": sum(x["fw"] for x in xs),
        "a": 100 * sum(x["ok"] for x in xs) / n,
        "o": 100 * sum(x["orig"] for x in xs) / n,
        "f": 100 * sum(x["fw"] for x in xs) / n,
        "t": sum(x["tot"] for x in xs) / n,
        "ot": sum(x["ot"] for x in xs) / n,
        "ft": sum(x["ftot"] for x in xs) / n,
        "r": (sum(x["tot"] for x in xs) / d) if d else float("nan"),
        "fr": (sum(x["ftot"] for x in xs) / d) if d else float("nan"),
    }


def main() -> None:
    first: dict[str, dict] = {}
    hm: dict[str, dict] = {}
    jobs: dict[str, dict] = {}
    windowed: set[tuple] = set()
    for seed in SEEDS:
        hm.update(load_scores(AE / f"results/first_hm_gate/{TAG}_s{seed}_suppress_hm"))
        for suf in ("", "_high", "_mix"):
            first.update(load_scores(AE / f"results/leftover_suppress_toend/{TAG}_s{seed}_suppress{suf}"))
        jp = AE / f"results/first_hm_gate/jobs/{TAG}_s{seed}.jsonl"
        for line in jp.open():
            if not line.strip():
                continue
            j = json.loads(line)
            jobs[j["uid"]] = j
        folder = AE / f"results/leftover_jump/{TAG}_s{seed}"
        for name in ("jobs.jsonl", "jobs_high.jsonl", "jobs_mix.jsonl"):
            path = folder / name
            if not path.is_file():
                continue
            for line in path.open():
                if not line.strip():
                    continue
                j = json.loads(line)
                windowed.add((str(j["dataset"]), int(j["question_idx"]), int(j["seed"])))

    miss = [uid for uid, j in jobs.items() if uid not in hm]
    print(f"first-HM jobs={len(jobs)} scored={len(hm)} pending={len(miss)}")

    print("\n## 整集（无非Low窗贴官方）")
    print("| 集 | n | 第一扇就压 | 等非Low再压 | Full-CoT | 等-官 | 等-扇 | 扇/官 | 等/官 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    all_rows: list[dict] = []
    hmq: list[dict] = []
    delayed: list[dict] = []
    already: list[dict] = []
    by_ds: dict[str, list] = {ds: [] for ds in DATASETS}
    by_ds_hmq: dict[str, list] = {ds: [] for ds in DATASETS}
    by_ds_del: dict[str, list] = {ds: [] for ds in DATASETS}
    for seed in SEEDS:
        for ds in DATASETS:
            off = {int(r["question_idx"]): r for r in dd.load_json(dd.puma_stat_path(TAG, ds, seed))}
            for qi, info in off.items():
                uid = f"{TAG}:{ds}:{seed}:{qi}"
                orig = bool(info.get("original_correct"))
                ot = float(info.get("original_tokens") or 0)
                rec_f, rec_h = first.get(uid), hm.get(uid)
                if rec_f:
                    fw, ftot = bool(rec_f.get("new_gold_ok")), tot(rec_f)
                elif (ds, qi, seed) not in windowed:
                    fw, ftot = orig, ot
                else:
                    continue
                if rec_h:
                    ok, t = bool(rec_h.get("new_gold_ok")), tot(rec_h)
                elif uid not in jobs:
                    ok, t = orig, ot
                else:
                    continue
                row = dict(ok=ok, orig=orig, fw=fw, tot=t, ftot=ftot, ot=ot, delayed=False)
                if uid in jobs:
                    row["delayed"] = bool(jobs[uid].get("delayed"))
                    hmq.append(row)
                    by_ds_hmq[ds].append(row)
                    (delayed if row["delayed"] else already).append(row)
                    if row["delayed"]:
                        by_ds_del[ds].append(row)
                all_rows.append(row)
                by_ds[ds].append(row)

    for ds in DATASETS:
        a = agg(by_ds[ds])
        print(
            f"| {DSZH[ds]} | {a['n']} | {a['f']:.1f}% | {a['a']:.1f}% | {a['o']:.1f}% | "
            f"{a['a']-a['o']:+.1f} | {a['a']-a['f']:+.1f} | {a['fr']:.2f} | {a['r']:.2f} |"
        )
    a = agg(all_rows)
    print(
        f"| 全部 | {a['n']} | {a['f']:.1f}% | {a['a']:.1f}% | {a['o']:.1f}% | "
        f"{a['a']-a['o']:+.1f} | {a['a']-a['f']:+.1f} | {a['fr']:.2f} | {a['r']:.2f} |"
    )

    print("\n## 只看出现过非Low窗的题")
    print("| 集 | n | 第一扇就压 | 等非Low再压 | Full-CoT | 等-扇 | 等-官 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        xs = by_ds_hmq[ds]
        if not xs:
            continue
        a = agg(xs)
        print(
            f"| {DSZH[ds]} | {a['n']} | {a['f']:.1f}% | {a['a']:.1f}% | {a['o']:.1f}% | "
            f"{a['a']-a['f']:+.1f} | {a['a']-a['o']:+.1f} |"
        )
    a = agg(hmq)
    print(
        f"| 全部 | {a['n']} | {a['f']:.1f}% | {a['a']:.1f}% | {a['o']:.1f}% | "
        f"{a['a']-a['f']:+.1f} | {a['a']-a['o']:+.1f} |"
    )

    print("\n## 拆开：第一扇已是非Low vs 第一扇Low等到非Low")
    print("| 子集 | n | 第一扇就压 | 等非Low | Full-CoT |")
    print("|---|---:|---:|---:|---:|")
    for name, xs in (("第一扇已是H/M", already), ("第一扇Low，等到H/M", delayed)):
        a = agg(xs)
        print(f"| {name} | {a['n']} | {a['f']:.1f}% | {a['a']:.1f}% | {a['o']:.1f}% |")

    print("\n## delayed 分集")
    print("| 集 | n | 第一扇Low压 | 等到H/M | Full-CoT |")
    print("|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        xs = by_ds_del[ds]
        if not xs:
            continue
        a = agg(xs)
        print(f"| {DSZH[ds]} | {a['n']} | {a['f']:.1f}% | {a['a']:.1f}% | {a['o']:.1f}% |")


if __name__ == "__main__":
    main()
