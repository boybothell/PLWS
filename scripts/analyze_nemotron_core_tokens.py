#!/usr/bin/env python3
"""Why Nemotron CORE-suppress saves few tokens and when Acc moves."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from plws.matrix import FIRSTWIN, TIER_KINDS, job_rows
from plws.paths import PLWSPaths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_fullcot_puma_plws import (  # noqa: E402
    DATASETS,
    KINDS,
    SEEDS,
    load_json,
    load_scores,
    puma_delivery_path,
)

OUT = ROOT / "results" / "reports" / "nemotron_core_token_acc.json"

MODELS = (("nemotron_8b", "Nemotron"), ("r1_7b", "7B"))
BUT_RE = re.compile(r"\bbut\b", re.I)
SO_RE = re.compile(r"\bso\b", re.I)
THEREFORE_RE = re.compile(r"\btherefore\b", re.I)
WAIT_RE = re.compile(r"\bwait\b", re.I)
ALT_RE = re.compile(r"\balternatively\b", re.I)
HMM_RE = re.compile(r"\bhm+\b", re.I)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pct(part: int, whole: int) -> float | None:
    return 100.0 * part / whole if whole else None


def summarize(rows: list[dict]) -> dict:
    def take(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    win = [row for row in rows if row["windowed"]]
    un = [row for row in rows if not row["windowed"]]
    grew = sum(1 for row in win if row["plws_tok"] > row["full_tok"] + 32)
    saved = sum(1 for row in win if row["plws_tok"] + 32 < row["full_tok"])
    sameish = len(win) - grew - saved
    leftover_full = [
        max(0.0, row["full_tok"] - row["n_left_tok"])
        for row in win
        if row.get("n_left_tok") is not None
    ]
    late = sum(
        1
        for row in win
        if row.get("n_left_tok") is not None
        and row["full_tok"] > 0
        and row["n_left_tok"] / row["full_tok"] >= 0.8
    )
    return {
        "n": len(rows),
        "windowed": len(win),
        "window_rate": pct(len(win), len(rows)),
        "unwindowed": len(un),
        "full_acc": pct(sum(row["full_ok"] for row in rows), len(rows)),
        "plws_acc": pct(sum(row["plws_ok"] for row in rows), len(rows)),
        "puma_acc": pct(sum(row["puma_ok"] for row in rows), len(rows)),
        "full_tok": mean([row["full_tok"] for row in rows]),
        "plws_tok": mean([row["plws_tok"] for row in rows]),
        "puma_tok": mean([row["puma_tok"] for row in rows]),
        "unwindowed_tok": mean([row["full_tok"] for row in un]),
        "windowed_full_tok": mean([row["full_tok"] for row in win]),
        "windowed_plws_tok": mean([row["plws_tok"] for row in win]),
        "n_left_tok": mean(take("n_left_tok")),
        "n_cont_tok": mean(take("n_cont_tok")),
        "leftover_full_tok": mean(leftover_full),
        "left_frac": mean(
            [
                row["n_left_tok"] / row["full_tok"]
                for row in win
                if row.get("n_left_tok") is not None and row["full_tok"] > 0
            ]
        ),
        "late_window_pct": pct(late, len(win)),
        "hit_cap": sum(1 for row in win if row.get("hit_cap")),
        "hit_cap_pct": pct(sum(1 for row in win if row.get("hit_cap")), len(win)),
        "grew_pct": pct(grew, len(win)),
        "saved_pct": pct(saved, len(win)),
        "sameish_pct": pct(sameish, len(win)),
        "keep_pct": pct(sum(1 for row in win if row.get("keep")), len(win)),
        "n_wait": mean(take("n_wait")),
        "has_wait_pct": pct(sum(1 for row in win if row.get("has_wait")), len(win)),
        "but_in_cont": mean(take("n_but")),
        "so_in_cont": mean(take("n_so")),
        "therefore_in_cont": mean(take("n_therefore")),
        "flips_up": sum(1 for row in win if row["plws_ok"] and not row["full_ok"]),
        "flips_down": sum(1 for row in win if row["full_ok"] and not row["plws_ok"]),
        "lock_ok_keep": sum(
            1 for row in win if row.get("left_ok") and row["plws_ok"]
        ),
        "lock_ok_ruin_full": sum(
            1
            for row in win
            if row.get("left_ok") and row["full_ok"] is False and row["plws_ok"]
        ),
        "lock_ok_ruin_plws": sum(
            1
            for row in win
            if row.get("left_ok") and row["full_ok"] and not row["plws_ok"]
        ),
        "bad_lock_save_full": sum(
            1
            for row in win
            if not row.get("left_ok") and row["full_ok"] and not row["plws_ok"]
        ),
        "bad_lock_save_plws": sum(
            1
            for row in win
            if not row.get("left_ok") and row["plws_ok"] and not row["full_ok"]
        ),
        "bad_lock_both_ok": sum(
            1
            for row in win
            if not row.get("left_ok") and row["full_ok"] and row["plws_ok"]
        ),
        "windowed_full_acc": pct(sum(row["full_ok"] for row in win), len(win)),
        "windowed_plws_acc": pct(sum(row["plws_ok"] for row in win), len(win)),
        "left_ok_acc": pct(sum(1 for row in win if row.get("left_ok")), len(win)),
    }


def collect(paths: PLWSPaths, model: str, dataset: str) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        official = {
            int(row["question_idx"]): row
            for row in load_json(paths.puma_statistics_path(model, dataset, seed))
        }
        delivery = load_json(puma_delivery_path(paths, model, dataset, seed))
        delivery_rows = (
            delivery if isinstance(delivery, list) else delivery.get("rows", [])
        )
        puma = {int(row["question_idx"]): row for row in delivery_rows}
        windowed = {
            int(job["question_idx"]): job
            for job in job_rows(paths, model, dataset, seed)
            if job.get("uid")
        }
        scores: dict[tuple[str, int], dict] = {}
        for kind in (FIRSTWIN, *TIER_KINDS):
            loaded = load_scores(
                paths.score_dir(model, dataset, seed, kind, k=4, lexicon="core")
            )
            if kind == FIRSTWIN:
                scores.update(loaded)
            else:
                for key, rec in loaded.items():
                    scores.setdefault(key, rec)
        for qid, info in official.items():
            rec = scores.get((dataset, int(qid)))
            full_ok = bool(info.get("original_correct"))
            full_tok = float(info.get("original_tokens") or 0)
            prow = puma.get(int(qid), {})
            if rec is None:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "qid": int(qid),
                        "windowed": False,
                        "full_ok": full_ok,
                        "plws_ok": full_ok,
                        "puma_ok": bool(prow.get("compressed_correct")),
                        "full_tok": full_tok,
                        "plws_tok": full_tok,
                        "puma_tok": float(prow.get("compressed_tokens") or 0),
                    }
                )
                continue
            cont = ""
            text = str(rec.get("generated_text") or "")
            thought = str(windowed.get(int(qid), {}).get("thought") or "")
            if thought and text.startswith(thought):
                cont = text[len(thought) :].split("</think>", 1)[0]
            elif rec.get("new_text"):
                cont = str(rec.get("new_text"))
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "qid": int(qid),
                    "uid": rec.get("uid"),
                    "windowed": True,
                    "full_ok": full_ok,
                    "plws_ok": bool(rec.get("new_gold_ok")),
                    "puma_ok": bool(prow.get("compressed_correct")),
                    "full_tok": full_tok,
                    "plws_tok": float(rec.get("n_think_tok") or 0)
                    + float(rec.get("n_ans_tok") or 0),
                    "puma_tok": float(prow.get("compressed_tokens") or 0),
                    "n_left_tok": rec.get("n_left_tok"),
                    "n_cont_tok": rec.get("n_cont_tok"),
                    "n_wait": rec.get("n_wait"),
                    "has_wait": rec.get("has_wait"),
                    "hit_cap": rec.get("hit_cap"),
                    "keep": rec.get("keep"),
                    "left_ok": rec.get("left_ok"),
                    "left_step": rec.get("left_step"),
                    "n_but": len(BUT_RE.findall(cont)),
                    "n_so": len(SO_RE.findall(cont)),
                    "n_therefore": len(THEREFORE_RE.findall(cont)),
                    "n_wait_text": len(WAIT_RE.findall(cont)),
                    "n_alt": len(ALT_RE.findall(cont)),
                    "n_hmm": len(HMM_RE.findall(cont)),
                }
            )
    return rows


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    payload: dict = {"models": {}}
    for model, zh in MODELS:
        by_ds = {}
        all_rows: list[dict] = []
        print(f"== {zh} ==", flush=True)
        for dataset, dszh, _n in DATASETS:
            rows = collect(paths, model, dataset)
            all_rows.extend(rows)
            cell = summarize(rows)
            by_ds[dszh] = cell
            print(
                f"{zh:8} {dszh:14} win={cell['window_rate']:.1f}% "
                f"left_frac={cell['left_frac'] or 0:.2f} "
                f"leftover={cell['leftover_full_tok'] or 0:.0f} "
                f"cont={cell['n_cont_tok'] or 0:.0f} "
                f"save={cell['saved_pct']} grew={cell['grew_pct']} "
                f"cap={cell['hit_cap_pct']} "
                f"up={cell['flips_up']} down={cell['flips_down']} "
                f"ruin_full={cell['lock_ok_ruin_full']} "
                f"block_save={cell['bad_lock_save_full']}",
                flush=True,
            )
        payload["models"][zh] = {
            "datasets": by_ds,
            "overall": summarize(all_rows),
        }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
