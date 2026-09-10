#!/usr/bin/env python3
"""7B s42：第一扇就压 vs 等到 High/Mix 再压 vs 官方。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402
from score_leftover_jump import done_uids  # noqa: E402

TAG = "r1_7b"
SEED = 42
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
DSZH = {
    "math-500": "MATH",
    "olympiadbench": "oly",
    "gpqa-diamond": "GPQA",
    "aime24": "A24",
    "aime25": "A25",
}
HM_JOBS = AE / "results/first_hm_gate/jobs/r1_7b_s42.jsonl"
HM_DIR = AE / "results/first_hm_gate/r1_7b_s42_suppress_hm"
OLD = AE / "results/leftover_suppress_toend"


def load_folder(folder: Path) -> dict[str, dict]:
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


def leftover_windowed() -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    folder = AE / f"results/leftover_jump/{TAG}_s{SEED}"
    for name in ("jobs.jsonl", "jobs_high.jsonl", "jobs_mix.jsonl"):
        path = folder / name
        if not path.is_file():
            continue
        for line in path.open():
            if not line.strip():
                continue
            job = json.loads(line)
            keys.add((str(job["dataset"]), int(job["question_idx"])))
    return keys


def agg(xs: list[dict]) -> dict | None:
    if not xs:
        return None
    n = len(xs)
    denom = sum(x["ot"] for x in xs)
    return {
        "n": n,
        "a": 100.0 * sum(x["ok"] for x in xs) / n,
        "o": 100.0 * sum(x["orig"] for x in xs) / n,
        "t": sum(x["tot"] for x in xs) / n,
        "ot": sum(x["ot"] for x in xs) / n,
        "r": (sum(x["tot"] for x in xs) / denom) if denom else float("nan"),
    }


def line(name: str, s: dict) -> str:
    return (
        f"| {name} | {s['n']} | {s['a']:.1f}% | {s['o']:.1f}% | "
        f"{s['t']:.0f} | {s['ot']:.0f} | {s['r']:.2f} |"
    )


def main() -> None:
    jobs = [json.loads(line) for line in HM_JOBS.read_text().splitlines() if line.strip()]
    hm = load_folder(HM_DIR)
    first: dict[str, dict] = {}
    for suf in ("", "_high", "_mix"):
        first.update(load_folder(OLD / f"{TAG}_s{SEED}_suppress{suf}"))
    windowed = leftover_windowed()
    pending = [job["uid"] for job in jobs if job["uid"] not in hm]
    print(f"hm scores={len(hm)} jobs={len(jobs)} pending={len(pending)}", flush=True)

    rows_hm: dict[str, list[dict]] = {ds: [] for ds in DATASETS}
    rows_fw: dict[str, list[dict]] = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        off = {
            int(r["question_idx"]): r
            for r in dd.load_json(dd.puma_stat_path(TAG, ds, SEED))
        }
        for qi, info in off.items():
            uid = f"{TAG}:{ds}:{SEED}:{qi}"
            orig = bool(info.get("original_correct"))
            ot = float(info.get("original_tokens") or 0)
            rec_h = hm.get(uid)
            rec_f = first.get(uid)
            if rec_h:
                rows_hm[ds].append({"ok": bool(rec_h.get("new_gold_ok")), "orig": orig, "tot": tot(rec_h), "ot": ot})
            elif (ds, qi) not in windowed or uid not in {j["uid"] for j in jobs}:
                rows_hm[ds].append({"ok": orig, "orig": orig, "tot": ot, "ot": ot})
            if rec_f:
                rows_fw[ds].append({"ok": bool(rec_f.get("new_gold_ok")), "orig": orig, "tot": tot(rec_f), "ot": ot})
            elif (ds, qi) not in windowed:
                rows_fw[ds].append({"ok": orig, "orig": orig, "tot": ot, "ot": ot})

    print("| 方法 | 集 | n | Acc | 官方 | tok | 官 tok | /官 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for title, store in (("窗后压第一扇", rows_fw), ("等到 High/Mix", rows_hm)):
        pooled: list[dict] = []
        for ds in DATASETS:
            s = agg(store[ds])
            if not s or s["n"] < len(dd.load_json(dd.puma_stat_path(TAG, ds, SEED))):
                continue
            pooled.extend(store[ds])
            print(f"| {title} | {DSZH[ds]} | {s['n']} | {s['a']:.1f}% | {s['o']:.1f}% | {s['t']:.0f} | {s['ot']:.0f} | {s['r']:.2f} |")
        all_s = agg(pooled)
        if all_s:
            print(f"| {title} | 全部 | {all_s['n']} | {all_s['a']:.1f}% | {all_s['o']:.1f}% | {all_s['t']:.0f} | {all_s['ot']:.0f} | {all_s['r']:.2f} |")


if __name__ == "__main__":
    main()
