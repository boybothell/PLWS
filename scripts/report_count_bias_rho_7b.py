#!/usr/bin/env python3
"""R1-7B seed-42 Soft count-bias rho ablation: Acc/Tok on the five main sets.

Uses written new_gold_ok on count_bias shards, Full-CoT for unwindowed items,
and delivery-plus-boxed-trial tokens. Overall is the equal-weight mean of the
five already-rounded cells. Only rhos with all five datasets are reported.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.artifacts import load_jsonl  # noqa: E402
from plws.grading import require_grader  # noqa: E402
from plws.matrix import FIRSTWIN  # noqa: E402
from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import (  # noqa: E402
    SHARD_RE,
    boxed_trial_tokens_upto,
    load_json,
    load_trials_by_question,
    rec_key,
)

MODEL = "r1_7b"
SEED = 42
K = 4
LEXICON = "core"
RHOS = (0.80, 0.90, 0.95, 0.98)
DATASETS = (
    ("math-500", "MATH-500", 500),
    ("olympiadbench", "Olympiad", 675),
    ("gpqa-diamond", "GPQA", 198),
    ("aime25", "AIME25", 30),
    ("amc23", "AMC23", 40),
)
COUNT_ROOT = ROOT / "results" / "runs" / "plws" / "count_bias"
TABLE = ROOT / "tables" / "count_bias_rho_seed42.md"
REPORT = ROOT / "results" / "reports" / "count_bias_rho_seed42.json"
OVERLEAF_FIG = Path(
    "/mnt/d/lsj/visual-latent-tts/overleaf/figures/rho_soft.pdf"
)


def rho_tag(rho: float) -> str:
    return f"rho_{rho:.2f}".replace(".", "p")


def load_shard_scores(folder: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    found = False
    for path in sorted(folder.iterdir()):
        if not SHARD_RE.match(path.name):
            continue
        found = True
        for rec in load_jsonl(path):
            if rec.get("status") not in {"ok", "too_long"} or not rec.get("uid"):
                continue
            key = rec_key(rec)
            if key is None:
                raise RuntimeError(f"unkeyed row in {path}")
            if key in out:
                raise RuntimeError(f"duplicate {key} in {path}")
            out[key] = rec
    if not found:
        raise FileNotFoundError(f"no shard jsonl in {folder}")
    return out


def load_jobs(paths: PLWSPaths, dataset: str) -> dict[int, dict]:
    path = paths.jobs_path(
        MODEL, dataset, SEED, FIRSTWIN, k=K, lexicon=LEXICON
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[int, dict] = {}
    for job in load_jsonl(path):
        if not job.get("uid"):
            continue
        qid = int(job["question_idx"])
        if qid in out:
            raise RuntimeError(f"duplicate job {dataset} q={qid}")
        out[qid] = job
    return out


def tok_fmt(n: int) -> str:
    return f"{n:,}"


def md_acc(acc: float, win: bool) -> str:
    s = f"{acc:.2f}"
    return f"\\textbf{{{s}}}" if win else s


def md_tok(tok: int, win: bool) -> str:
    s = tok_fmt(tok)
    return f"\\textbf{{{s}}}" if win else s


def round_cell(acc: float, tok: float) -> tuple[float, int]:
    return round(acc, 2), round(tok)


def plot_pdf(display: dict) -> None:
    import matplotlib.pyplot as plt

    rhos = list(RHOS)
    accs = [display[rho]["overall"]["acc"] for rho in rhos]
    toks = [display[rho]["overall"]["tok"] for rho in rhos]
    full_acc = display["full"]["overall"]["acc"]
    full_tok = display["full"]["overall"]["tok"]
    highlight = 0.98

    acc_lo = min(accs + [full_acc]) - 1.0
    acc_hi = max(accs + [full_acc]) + 1.0
    tok_lo = min(toks) - 80
    tok_hi = max(toks) + 80
    del full_tok

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax_tok = ax.twinx()
    acc_color = "#2171b5"
    tok_color = "#cb181d"
    ax.plot(rhos, accs, color=acc_color, lw=1.6, zorder=3)
    ax_tok.plot(rhos, toks, color=tok_color, lw=1.6, zorder=3)
    for rho, acc, tok in zip(rhos, accs, toks):
        filled = abs(rho - highlight) < 1e-9
        ax.scatter(
            [rho],
            [acc],
            s=38 if filled else 28,
            facecolors=acc_color if filled else "white",
            edgecolors=acc_color,
            linewidths=1.4,
            zorder=4,
        )
        ax_tok.scatter(
            [rho],
            [tok],
            s=38 if filled else 28,
            marker="s",
            facecolors=tok_color if filled else "white",
            edgecolors=tok_color,
            linewidths=1.4,
            zorder=4,
        )
    ax.axhline(full_acc, color=acc_color, ls=":", lw=1.0, zorder=2)
    ax.set_xlim(0.775, 1.005)
    ax.set_xticks(rhos)
    ax.set_xticklabels(["0.80", "0.90", "0.95", "0.98"])
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Accuracy (%)", color=acc_color)
    ax_tok.set_ylabel("Tokens", color=tok_color)
    ax.tick_params(axis="y", colors=acc_color)
    ax_tok.tick_params(axis="y", colors=tok_color)
    ax.set_ylim(acc_lo, acc_hi)
    ax_tok.set_ylim(tok_lo, tok_hi)
    ax.spines["top"].set_visible(False)
    ax_tok.spines["top"].set_visible(False)
    ax.legend(
        [
            plt.Line2D([0], [0], color=acc_color, lw=1.6, marker="o", ms=4),
            plt.Line2D([0], [0], color=tok_color, lw=1.6, marker="s", ms=4),
        ],
        ["Accuracy", "Length"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        fontsize=8,
        handlelength=1.6,
    )
    fig.tight_layout()
    ax.set_ylim(acc_lo, acc_hi)
    ax_tok.set_ylim(tok_lo, tok_hi)
    OVERLEAF_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OVERLEAF_FIG, format="pdf")
    plt.close(fig)


def main() -> None:
    require_grader()
    paths = PLWSPaths.discover(ROOT)
    raw: dict[str, dict] = {}
    sources: list[str] = []
    cells: dict[str, dict] = {"full": {}}

    for dataset, dszh, expect_n in DATASETS:
        official_path = paths.puma_statistics_path(MODEL, dataset, SEED)
        if not official_path.is_file():
            raise FileNotFoundError(official_path)
        official = {
            int(row["question_idx"]): row for row in load_json(official_path)
        }
        if len(official) != expect_n:
            raise RuntimeError(
                f"{dataset} n={len(official)} expected {expect_n}"
            )
        sources.append(str(official_path.relative_to(ROOT)))
        trials_path = paths.dense_trial_path(MODEL, dataset, SEED)
        if not trials_path.is_file():
            raise FileNotFoundError(trials_path)
        trials = load_trials_by_question(trials_path)
        sources.append(str(trials_path.relative_to(ROOT)))
        jobs = load_jobs(paths, dataset)
        sources.append(
            str(
                paths.jobs_path(
                    MODEL, dataset, SEED, FIRSTWIN, k=K, lexicon=LEXICON
                ).relative_to(ROOT)
            )
        )
        full_ok = sum(int(bool(row.get("original_correct"))) for row in official.values())
        full_tok = sum(float(row.get("original_tokens") or 0) for row in official.values())
        n = len(official)
        cells["full"][dataset] = {
            "acc": 100.0 * full_ok / n,
            "tok": full_tok / n,
            "n": n,
            "windowed": 0,
            "unwindowed": n,
        }
        for rho in RHOS:
            folder = COUNT_ROOT / rho_tag(rho) / MODEL / dataset / f"seed_{SEED}"
            scores = load_shard_scores(folder)
            sources.append(str(folder.relative_to(ROOT)))
            missing_jobs = [qid for qid in jobs if (dataset, qid) not in scores]
            extra = [key for key in scores if key[1] not in jobs]
            if missing_jobs or extra:
                raise RuntimeError(
                    f"{rho_tag(rho)} {dataset}: missing={len(missing_jobs)} extra={len(extra)}"
                )
            ok = tok = 0.0
            win = 0
            for qid, info in official.items():
                rec = scores.get((dataset, int(qid)))
                job = jobs.get(int(qid))
                if rec is not None:
                    if rec.get("protocol_id") != "puma-fullcot-32k-v2":
                        raise RuntimeError(f"bad protocol {rho} {dataset} q={qid}")
                    if float(rec.get("bias_rho")) != rho:
                        raise RuntimeError(f"bias_rho {rec.get('bias_rho')} != {rho}")
                    if rec.get("bias_formula") != "one_minus_rho_pow_n":
                        raise RuntimeError(f"bad formula {dataset} q={qid}")
                    left = rec.get("left_step")
                    if left is None and job is not None:
                        left = job.get("left_step")
                    if left is None:
                        raise RuntimeError(f"missing left_step {dataset} q={qid}")
                    win += 1
                    ok += int(bool(rec.get("new_gold_ok")))
                    tok += (
                        float(rec.get("n_think_tok") or 0)
                        + float(rec.get("n_ans_tok") or 0)
                        + boxed_trial_tokens_upto(
                            trials.get(int(qid), []), int(left)
                        )
                    )
                elif job is None:
                    ok += int(bool(info.get("original_correct")))
                    tok += float(info.get("original_tokens") or 0)
                else:
                    raise RuntimeError(f"missing score {rho} {dataset} q={qid}")
            cells.setdefault(str(rho), {})[dataset] = {
                "acc": 100.0 * ok / n,
                "tok": tok / n,
                "n": n,
                "windowed": win,
                "unwindowed": n - win,
            }

    display: dict = {"full": {"datasets": {}, "overall": {}}}
    shown_full = []
    shown_rho: dict[float, list] = {rho: [] for rho in RHOS}
    for dataset, _, _ in DATASETS:
        acc, tok = round_cell(
            cells["full"][dataset]["acc"], cells["full"][dataset]["tok"]
        )
        display["full"]["datasets"][dataset] = {"acc": acc, "tok": tok}
        shown_full.append((acc, tok))
        for rho in RHOS:
            racc, rtok = round_cell(
                cells[str(rho)][dataset]["acc"], cells[str(rho)][dataset]["tok"]
            )
            display.setdefault(rho, {"datasets": {}, "overall": {}})
            display[rho]["datasets"][dataset] = {
                "acc": racc,
                "tok": rtok,
                "windowed": cells[str(rho)][dataset]["windowed"],
                "unwindowed": cells[str(rho)][dataset]["unwindowed"],
            }
            shown_rho[rho].append((racc, rtok))
    display["full"]["overall"] = {
        "acc": round(sum(a for a, _ in shown_full) / 5, 2),
        "tok": round(sum(t for _, t in shown_full) / 5),
    }
    for rho in RHOS:
        display[rho]["overall"] = {
            "acc": round(sum(a for a, _ in shown_rho[rho]) / 5, 2),
            "tok": round(sum(t for _, t in shown_rho[rho]) / 5),
        }

    rho_accs = {rho: display[rho]["overall"]["acc"] for rho in RHOS}
    rho_toks = {rho: display[rho]["overall"]["tok"] for rho in RHOS}
    # per-dataset winners among rhos only
    winners: dict[str, dict] = {}
    keys = ["full", *[str(r) for r in RHOS]]
    for dataset, _, _ in DATASETS:
        accs = {rho: display[rho]["datasets"][dataset]["acc"] for rho in RHOS}
        toks = {rho: display[rho]["datasets"][dataset]["tok"] for rho in RHOS}
        winners[dataset] = {
            "acc": {rho for rho, v in accs.items() if v == max(accs.values())},
            "tok": {rho for rho, v in toks.items() if v == min(toks.values())},
        }
    winners["overall"] = {
        "acc": {rho for rho, v in rho_accs.items() if v == max(rho_accs.values())},
        "tok": {rho for rho, v in rho_toks.items() if v == min(rho_toks.values())},
    }

    lines = [
        "# Soft count-bias rho on R1-7B seed 42",
        "",
        "Formula `b(n)=-10*(1-rho^n)`. Acc uses written `new_gold_ok`; unwindowed items copy Full-CoT.",
        "Token = think + answer + boxed trial tokens up to `left_step`.",
        "Overall = equal-weight mean of five already-rounded cells.",
        "",
        "| Variant | MATH-500 Acc | MATH-500 Tok | Olympiad Acc | Olympiad Tok | GPQA Acc | GPQA Tok | AIME25 Acc | AIME25 Tok | AMC23 Acc | AMC23 Tok | Overall Acc | Overall Tok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, block: dict, *, bold: bool) -> str:
        parts = [name]
        for dataset, _, _ in DATASETS:
            cell = block["datasets"][dataset]
            acc_b = bold and name != "Full CoT" and (
                float(name.split("=")[-1]) in winners[dataset]["acc"]
                if name.startswith("rho=")
                else False
            )
            tok_b = bold and name.startswith("rho=") and (
                float(name.split("=")[-1]) in winners[dataset]["tok"]
            )
            parts.append(f"**{cell['acc']:.2f}**" if acc_b else f"{cell['acc']:.2f}")
            parts.append(f"**{tok_fmt(cell['tok'])}**" if tok_b else tok_fmt(cell["tok"]))
        oacc = block["overall"]["acc"]
        otok = block["overall"]["tok"]
        if name.startswith("rho="):
            rho = float(name.split("=")[-1])
            parts.append(f"**{oacc:.2f}**" if rho in winners["overall"]["acc"] else f"{oacc:.2f}")
            parts.append(f"**{tok_fmt(otok)}**" if rho in winners["overall"]["tok"] else tok_fmt(otok))
        else:
            parts.append(f"{oacc:.2f}")
            parts.append(tok_fmt(otok))
        return "| " + " | ".join(parts) + " |"

    lines.append(row("Full CoT", display["full"], bold=False))
    for rho in RHOS:
        lines.append(row(f"rho={rho:.2f}", display[rho], bold=True))
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "seed": SEED,
        "rhos": list(RHOS),
        "token_policy": "delivery_plus_boxed_trial_v1",
        "overall_rule": "equal_weight_of_rounded_cells",
        "grader": "require_grader + written new_gold_ok",
        "cells_raw": cells,
        "display": {
            "full": display["full"],
            **{str(rho): display[rho] for rho in RHOS},
        },
        "sources": sorted(set(sources)),
        "figure": str(OVERLEAF_FIG),
        "table": str(TABLE),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    plot_pdf(display)
    print(TABLE.read_text())
    print(f"wrote {TABLE}")
    print(f"wrote {REPORT}")
    print(f"wrote {OVERLEAF_FIG}")


if __name__ == "__main__":
    main()
