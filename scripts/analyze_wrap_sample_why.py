#!/usr/bin/env python3
"""Why CORE+But/So/Therefore loses Acc on finished wrap-sample cells."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from plws.matrix import FIRSTWIN, TIER_KINDS, reusable_score
from plws.paths import PLWSPaths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_fullcot_puma_plws import DATASETS, SHARD_RE  # noqa: E402

JOBS_DIR = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
WRAP_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation" / "wrap_sample"
OUT = ROOT / "results" / "reports" / "wrap_sample_why.json"
LEXICON = "core_plus_but_so_therefore"
MODELS = (("qwen3_4b", "4B"), ("nemotron_8b", "Nemotron"))
DSZH = {name: zh for name, zh, _n in DATASETS}
TZ = ZoneInfo("Asia/Shanghai")
BANNED = re.compile(r"\b(but|so|therefore)\b", re.I)
SUBS = [
    ("However", re.compile(r"\bhowever\b", re.I)),
    ("Thus", re.compile(r"\bthus\b", re.I)),
    ("Hence", re.compile(r"\bhence\b", re.I)),
    ("Then", re.compile(r"\bthen\b", re.I)),
    ("Now", re.compile(r"\bnow\b", re.I)),
    ("And", re.compile(r"\band\b", re.I)),
    ("Finally", re.compile(r"\bfinally\b", re.I)),
    ("Therefore-banned-leak", re.compile(r"\btherefore\b", re.I)),
    ("So-banned-leak", re.compile(r"\bso\b", re.I)),
    ("But-banned-leak", re.compile(r"\bbut\b", re.I)),
    ("Let me", re.compile(r"\blet me\b", re.I)),
    ("The answer", re.compile(r"\bthe answer\b", re.I)),
]
FIRST_LINE = re.compile(
    r"^\s*(However|Thus|Hence|Then|Now|And|Finally|Also|Still|"
    r"Meanwhile|Although|Yet|Next|The|This|That|We|I|Let|If|"
    r"Since|Because|First|Okay|Ok|Yes|No|Wait|But|So|Therefore)\b",
    re.I,
)
WORD = {
    "but": re.compile(r"\bbut\b", re.I),
    "so": re.compile(r"\bso\b", re.I),
    "therefore": re.compile(r"\btherefore\b", re.I),
}


def leftover_of(rec: dict[str, Any], thought: str) -> str:
    generated = str(rec.get("generated_text") or "")
    if thought and generated.startswith(thought):
        rest = generated[len(thought) :]
    else:
        rest = generated
    return rest.split("</think>", 1)[0]


def slim_rec(rec: dict[str, Any], thought: str) -> dict[str, Any]:
    text = leftover_of(rec, thought)
    return {
        "ok": bool(rec.get("new_gold_ok")),
        "keep": bool(rec.get("keep")),
        "left_ok": bool(rec.get("left_ok")),
        "hit_cap": bool(rec.get("hit_cap")),
        "natural_close": bool(rec.get("natural_close")),
        "finish": rec.get("finish_reason"),
        "stop": rec.get("stop_reason"),
        "n_cont": float(rec.get("n_cont_tok") or 0),
        "n_think": float(rec.get("n_think_tok") or 0),
        "n_ans": float(rec.get("n_ans_tok") or 0),
        "n_left": float(rec.get("n_left_tok") or 0),
        "new_answer": rec.get("new_answer"),
        "old_answer": rec.get("old_answer"),
        "text": text[:4000],
    }


def load_jsonl_slim(
    path: Path, thoughts: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not reusable_score(rec):
                continue
            uid = str(rec.get("uid") or "")
            if not uid:
                continue
            thought = (thoughts or {}).get(uid, "")
            out[uid] = slim_rec(rec, thought)
    return out


def wrap_scores(model: str, thoughts: dict[str, str]) -> dict[str, dict[str, Any]]:
    folder = WRAP_ROOT / model / LEXICON
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        out.update(load_jsonl_slim(path, thoughts))
    return out


def core_scores(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    thoughts: dict[str, str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for kind in (FIRSTWIN, *TIER_KINDS):
        folder = paths.score_dir(model, dataset, seed, kind, k=4, lexicon="core")
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not SHARD_RE.match(path.name):
                continue
            for uid, rec in load_jsonl_slim(path, thoughts).items():
                out.setdefault(uid, rec)
    return out


def first_token(text: str) -> str:
    line = (text or "").lstrip()
    if not line:
        return "(empty)"
    match = FIRST_LINE.match(line)
    if match:
        return match.group(1).capitalize()
    word = re.split(r"\s+", line, maxsplit=1)[0]
    return re.sub(r"[^A-Za-z']+", "", word)[:16] or "(punct)"


def counts(text: str) -> dict[str, int]:
    return {name: len(pat.findall(text or "")) for name, pat in WORD.items()}


def sub_hits(text: str) -> dict[str, int]:
    return {name: len(pat.findall(text or "")) for name, pat in SUBS}


def bucket(job: dict[str, Any], core: dict[str, Any], wrap: dict[str, Any]) -> str:
    core_ok = bool(job.get("core_gold_ok"))
    wrap_ok = bool(wrap.get("ok"))
    left_ok = bool(job.get("left_ok") or core.get("left_ok"))
    if core_ok and not wrap_ok:
        return "ruin_good_core" if left_ok else "ruin_saved_or_other"
    if wrap_ok and not core_ok:
        return "recover"
    if core_ok and wrap_ok:
        return "both_right"
    return "both_wrong"


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pct(part: int, whole: int) -> float | None:
    return 100.0 * part / whole if whole else None


def top(counter: Counter[str], n: int = 8) -> dict[str, int]:
    return dict(counter.most_common(n))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    by_bucket = Counter(row["bucket"] for row in rows)
    ruin = [row for row in rows if row["bucket"].startswith("ruin")]
    recover = [row for row in rows if row["bucket"] == "recover"]
    keep_long = [row for row in rows if row["reason"] == "keep_right_long"]
    flip = [row for row in rows if row["reason"] == "flip"]

    def pack(subset: list[dict[str, Any]], *, extra: bool = False) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        core_first = Counter(row["core_first"] for row in subset)
        wrap_first = Counter(row["wrap_first"] for row in subset)
        grew = sum(1 for row in subset if row["d_cont"] > 32)
        cut = sum(1 for row in subset if row["d_cont"] < -32)
        wrap_keep = sum(1 for row in subset if row["wrap_keep"])
        hit = sum(1 for row in subset if row["wrap_hit"])
        close = sum(1 for row in subset if row["wrap_close"])
        core_has_close = sum(
            1 for row in subset if row["core_so"] or row["core_therefore"]
        )
        core_has_but = sum(1 for row in subset if row["core_but"])
        wrap_has_however = sum(1 for row in subset if row["wrap_however"])
        wrap_has_thus = sum(1 for row in subset if row["wrap_thus"] or row["wrap_hence"])
        wrap_has_letme = sum(1 for row in subset if row["wrap_letme"])
        leak = sum(1 for row in subset if row["wrap_leak"])
        out = {
            "n": len(subset),
            "core_first": top(core_first),
            "wrap_first": top(wrap_first),
            "d_cont": mean([row["d_cont"] for row in subset]),
            "core_cont": mean([row["core_cont"] for row in subset]),
            "wrap_cont": mean([row["wrap_cont"] for row in subset]),
            "grew_pct": pct(grew, len(subset)),
            "cut_pct": pct(cut, len(subset)),
            "wrap_keep_lock_pct": pct(wrap_keep, len(subset)),
            "wrap_hit_cap_pct": pct(hit, len(subset)),
            "wrap_natural_close_pct": pct(close, len(subset)),
            "core_used_so_therefore_pct": pct(core_has_close, len(subset)),
            "core_used_but_pct": pct(core_has_but, len(subset)),
            "wrap_however_pct": pct(wrap_has_however, len(subset)),
            "wrap_thus_hence_pct": pct(wrap_has_thus, len(subset)),
            "wrap_let_me_pct": pct(wrap_has_letme, len(subset)),
            "wrap_banned_leak_pct": pct(leak, len(subset)),
            "mean_core_but": mean([row["core_but"] for row in subset]),
            "mean_core_so": mean([row["core_so"] for row in subset]),
            "mean_core_therefore": mean([row["core_therefore"] for row in subset]),
        }
        if extra:
            out["examples"] = [
                {
                    "uid": row["uid"],
                    "reason": row["reason"],
                    "bucket": row["bucket"],
                    "core_first": row["core_first"],
                    "wrap_first": row["wrap_first"],
                    "core_ans": row["core_ans"],
                    "wrap_ans": row["wrap_ans"],
                    "old_ans": row["old_ans"],
                    "d_cont": row["d_cont"],
                    "core_head": row["core_head"],
                    "wrap_head": row["wrap_head"],
                }
                for row in subset[:6]
            ]
        return out

    return {
        "n": n,
        "buckets": dict(by_bucket),
        "ruin_good_core": by_bucket.get("ruin_good_core", 0),
        "ruin_other": by_bucket.get("ruin_saved_or_other", 0),
        "recover": by_bucket.get("recover", 0),
        "net": by_bucket.get("recover", 0)
        - by_bucket.get("ruin_good_core", 0)
        - by_bucket.get("ruin_saved_or_other", 0),
        "all": pack(rows),
        "ruin": pack(ruin, extra=True),
        "recover": pack(recover, extra=True),
        "keep_right_long": pack(keep_long),
        "flip": pack(flip),
    }


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    jobs_by_model: dict[str, list[dict[str, Any]]] = {}
    wrap_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    thoughts_by_model: dict[str, dict[str, str]] = {}
    for model, _zh in MODELS:
        jobs_by_model[model] = [
            json.loads(line)
            for line in (JOBS_DIR / f"wrap_sample_{model}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        thoughts_by_model[model] = {
            str(job["uid"]): str(job.get("thought") or "")
            for job in jobs_by_model[model]
        }
        wrap_by_model[model] = wrap_scores(model, thoughts_by_model[model])

    cells: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    core_cache: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}

    for model, zh in MODELS:
        jobs = jobs_by_model[model]
        wrap = wrap_by_model[model]
        by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for job in jobs:
            by_ds[str(job["dataset"])].append(job)
        for dataset, ds_jobs in sorted(by_ds.items()):
            missing = [job for job in ds_jobs if str(job["uid"]) not in wrap]
            progress = {
                "model": model,
                "model_zh": zh,
                "dataset": dataset,
                "dataset_zh": DSZH.get(dataset, dataset),
                "jobs": len(ds_jobs),
                "scored": len(ds_jobs) - len(missing),
                "missing": len(missing),
                "complete": not missing,
            }
            if missing:
                pending.append(progress)
                continue
            rows: list[dict[str, Any]] = []
            for job in ds_jobs:
                seed = int(job["seed"])
                key = (model, dataset, seed)
                if key not in core_cache:
                    core_cache[key] = core_scores(
                        paths, model, dataset, seed, thoughts_by_model[model]
                    )
                core = core_cache[key].get(str(job["uid"]), {})
                wrec = wrap[str(job["uid"])]
                core_text = str(core.get("text") or "")
                wrap_text = str(wrec.get("text") or "")
                core_c = counts(core_text)
                wrap_c = counts(wrap_text)
                wrap_sub = sub_hits(wrap_text)
                row = {
                    "uid": job["uid"],
                    "model": model,
                    "dataset": dataset,
                    "reason": job.get("sample_reason") or "",
                    "bucket": bucket(job, core, wrec),
                    "core_ok": bool(job.get("core_gold_ok")),
                    "wrap_ok": bool(wrec.get("ok")),
                    "left_ok": bool(job.get("left_ok")),
                    "core_first": first_token(core_text),
                    "wrap_first": first_token(wrap_text),
                    "core_cont": float(job.get("core_n_cont_tok") or core.get("n_cont") or 0),
                    "wrap_cont": float(wrec.get("n_cont") or 0),
                    "d_cont": float(wrec.get("n_cont") or 0)
                    - float(job.get("core_n_cont_tok") or core.get("n_cont") or 0),
                    "core_but": core_c["but"],
                    "core_so": core_c["so"],
                    "core_therefore": core_c["therefore"],
                    "wrap_however": wrap_sub["However"],
                    "wrap_thus": wrap_sub["Thus"],
                    "wrap_hence": wrap_sub["Hence"],
                    "wrap_letme": wrap_sub["Let me"],
                    "wrap_leak": bool(
                        wrap_sub["Therefore-banned-leak"]
                        or wrap_sub["So-banned-leak"]
                        or wrap_sub["But-banned-leak"]
                    ),
                    "wrap_keep": bool(wrec.get("keep")),
                    "wrap_hit": bool(wrec.get("hit_cap")),
                    "wrap_close": bool(wrec.get("natural_close")),
                    "core_ans": core.get("new_answer") or "",
                    "wrap_ans": wrec.get("new_answer") or "",
                    "old_ans": job.get("old_answer") or wrec.get("old_answer") or "",
                    "core_head": core_text[:180].replace("\n", " / "),
                    "wrap_head": wrap_text[:180].replace("\n", " / "),
                }
                rows.append(row)
                all_rows.append(row)
            summary = summarize(rows)
            summary.update(progress)
            cells.append(summary)

    payload = {
        "generated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "script": "scripts/analyze_wrap_sample_why.py",
        "lexicon": LEXICON,
        "note": "Finished wrap-sample datasets only. Texts are leftover continuations.",
        "overall": summarize(all_rows),
        "by_model": {
            zh: summarize([row for row in all_rows if row["model"] == model])
            for model, zh in MODELS
        },
        "cells": cells,
        "pending": pending,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")
    ov = payload["overall"]
    print(
        f"overall n={ov['n']} ruin_good={ov['ruin_good_core']} "
        f"ruin_other={ov['ruin_other']} recover={ov['recover']} net={ov['net']}"
    )
    print("ruin first CORE", ov["ruin"].get("core_first"))
    print("ruin first wrap", ov["ruin"].get("wrap_first"))
    print("ruin so/therefore in CORE", ov["ruin"].get("core_used_so_therefore_pct"))
    print("keep_right_long", ov["keep_right_long"])


if __name__ == "__main__":
    main()
