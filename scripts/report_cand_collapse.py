#!/usr/bin/env python3
"""Do leftover low-conf windows collapse to one candidate in the thought text?"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg

TABLE = AE / "tables/cand_collapse.md"
MODELS = (("7B", "r1_7b"), ("8B", "nemotron_8b"))
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)
SKIP = {
    "k", "n", "i", "x", "y", "z", "the", "and", "we", "so", "if", "or", "to",
    "be", "is", "of", "in", "it", "as", "at", "by", "for", "this", "that",
}
LATEX = (
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"\\\((.+?)\\\)"),
    re.compile(r"\\\[(.+?)\\\]"),
    re.compile(r"\$([^$]+)\$"),
)
LETTER = re.compile(
    r"(?i)(?:option|answer|choice|choose|pick|select|so|thus|therefore|hence|final|is)\s+"
    r"[:\s]*\(?([ABCD])\)?"
    r"|(?:^|[\s\(\[\"'])([ABCD])(?=[\s\)\]\.\,:]|$)"
)
CUE = re.compile(
    r"(?is)(?:therefore|thus|hence|so the|so we|we (?:get|have|find)|the answer(?: is)?|"
    r"final answer|equals|所以|因此|答案(?:是|为)?)\s*[:\s]*([^\n.]{1,80})"
)
CUE_LINE = re.compile(
    r"(?i)(\btherefore\b|\bthus\b|\bhence\b|\bso\b|\banswer\b|\bfinal\b|"
    r"\bequals\b|\bwe get\b|\bwe have\b|\bwe find\b|所以|因此|答案)"
)
OPT_LIST = re.compile(r"(?i)(?:options? are|[ABCD]\s*:).{0,40}[ABCD]\s*:")


def gpath(model: str, dataset: str, seed: int) -> Path:
    if dataset in ("aime24", "aime25"):
        return AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/per_sample.json"
    return AE / f"results/dense_G_{model}/{dataset}/per_sample.json"


def keep_expr(text: str) -> str | None:
    s = rg.norm(text)
    if not s or s in SKIP or len(s) > 48:
        return None
    return s


def latex_bits(text: str) -> list[str]:
    out = []
    for pat in LATEX:
        for m in pat.finditer(text):
            bit = next((g for g in m.groups() if g), "")
            kept = keep_expr(bit)
            if kept:
                out.append(kept)
    return out


def letter_bits(text: str) -> list[str]:
    if OPT_LIST.search(text):
        return []
    out = []
    for m in LETTER.finditer(text):
        ch = next((g for g in m.groups() if g), "")
        if ch:
            out.append(ch.upper())
    return out


def math_bits(text: str) -> list[str]:
    out = latex_bits(text)
    for m in CUE.finditer(text):
        chunk = m.group(1) or ""
        out.extend(latex_bits(chunk))
        kept = keep_expr(chunk)
        if kept and (
            any(ch.isdigit() for ch in kept)
            or "frac" in kept
            or "pi" in kept
            or re.fullmatch(r"[0-9a-z./+-]+", kept)
        ):
            out.append(kept)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?:\n\n+|(?<=[.!?])\s+)", text) if s.strip()]


def recent_text(prefix: str, n_chars: int) -> str:
    return prefix[-n_chars:] if prefix else ""


def extract(text: str, *, gpqa: bool) -> list[str]:
    if not text:
        return []
    if gpqa:
        bits = []
        for sent in sentences(text):
            bits.extend(letter_bits(sent))
        return bits
    bits = []
    for sent in sentences(text):
        if CUE_LINE.search(sent) or LATEX[0].search(sent) or LATEX[1].search(sent):
            bits.extend(math_bits(sent))
    if not bits:
        bits = math_bits(text)
    return bits


def unique(bits: list[str]) -> list[str]:
    seen = []
    for b in bits:
        if b not in seen:
            seen.append(b)
    return seen


def hits_trial(bits: list[str], trial: str) -> int:
    return sum(1 for b in bits if rg.same(b, trial))


def text_has_trial(text: str, trial: str) -> bool:
    raw = str(trial or "").strip()
    n_ans = rg.norm(raw)
    if not raw or not n_ans:
        return False
    if n_ans in rg.norm(text):
        return True
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(raw)}(?![A-Za-z0-9])", text))


def scores_of(prefix: str, trial: str, *, gpqa: bool) -> dict[str, float]:
    near = recent_text(prefix, 400)
    mid = recent_text(prefix, 1200)
    early = prefix[-2400:-1200] if prefix and len(prefix) > 1200 else ""
    near_b = extract(near, gpqa=gpqa)
    mid_b = extract(mid, gpqa=gpqa)
    early_b = extract(early, gpqa=gpqa)
    near_u = unique(near_b)
    mid_u = unique(mid_b)
    early_u = unique(early_b)
    n_near = len(near_u)
    n_mid = len(mid_u)
    n_early = len(early_u)
    share = hits_trial(mid_b, trial) / len(mid_b) if mid_b else 0.0
    only = float(n_near == 1 and rg.same(near_u[0], trial))
    written = float(hits_trial(mid_b, trial) > 0 or text_has_trial(mid, trial))
    drop = float(n_early - n_near) if n_early or n_near else 0.0
    return {
        "only_trial": only,
        "trial_share": share,
        "neg_n_unique": -float(n_mid if n_mid else 9),
        "mentioned": written,
        "written": written,
        "collapse_drop": drop,
        "n_unique_near": float(n_near),
        "n_unique_mid": float(n_mid),
        "n_bits_mid": float(len(mid_b)),
    }


def leftover_windows(
    rows: list[dict[str, Any]], host_step: int
) -> list[int]:
    out = []
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        window = rows[end + 1 - rg.K : end + 1]
        confs = rg.confs_of(window)
        if rg.is_high(confs) or not rg.is_low(confs):
            continue
        step = int(row["stopped_len"])
        if step < rg.MSS or step >= host_step:
            continue
        out.append(end)
    return out


def stoppable(ans: Any, gt: Any, original: Any, a_final: Any) -> bool:
    return rg.same(ans, gt) or rg.same(ans, original) or rg.same(ans, a_final)


def gold_ok(ans: Any, gt: Any, original: Any, orig_ok: bool) -> bool:
    if rg.same(ans, gt):
        return True
    if rg.same(ans, original):
        return bool(orig_ok)
    return False


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    regen_path = dd.regen_stat_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    gp = gpath(model, dataset, seed)
    if not regen_path.is_file() or not puma_path.is_file() or not trials_path.is_file():
        return None
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    questions = []
    gpqa = dataset == "gpqa-diamond"
    for qi, info in sorted(official.items()):
        host = regen.get(qi)
        trials = by.get(qi)
        if not host or not trials:
            continue
        rows = rg.usable_rows(trials)
        host_step = int(host.get("stopped_len") or 10**9)
        ends = leftover_windows(rows, host_step)
        windows = []
        for end in ends:
            row = rows[end]
            ans = row.get("final_answer")
            sc = scores_of(str(row.get("reasoning_prefix") or ""), str(ans or ""), gpqa=gpqa)
            windows.append(
                {
                    "end": end,
                    "step": int(row["stopped_len"]),
                    "ans": ans,
                    "pos": stoppable(ans, info.get("ground_truth"), info.get("original_answer"), (gmap.get(qi) or {}).get("A_final") or info.get("original_answer")),
                    **sc,
                }
            )
        questions.append(
            {
                "qi": qi,
                "rows": rows,
                "trials": trials,
                "gt": info.get("ground_truth"),
                "original": info.get("original_answer"),
                "a_final": (gmap.get(qi) or {}).get("A_final") or info.get("original_answer"),
                "orig_ok": bool(info.get("original_correct")),
                "orig_tok": int(info.get("original_tokens") or 0),
                "host_ok": bool(host.get("compressed_correct")),
                "host_tok": int(host.get("compressed_tokens") or 0) + int(host.get("tokens_trial_answers") or 0),
                "puma_ok": bool(info.get("compressed_correct")),
                "puma_tok": int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0),
                "windows": windows,
            }
        )
    return {"questions": questions, "gpqa": gpqa}


def auroc_of(windows: list[dict[str, Any]], field: str) -> tuple[float, int, int]:
    pos = [float(w[field]) for w in windows if w["pos"]]
    neg = [float(w[field]) for w in windows if not w["pos"]]
    return rg.auroc(pos, neg), len(pos), len(neg)


def rate(windows: list[dict[str, Any]], field: str, pos: bool) -> float:
    xs = [float(w[field]) for w in windows if w["pos"] is pos and w[field] == w[field]]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def fire_end(q: dict[str, Any], field: str, thr: float, binary: bool) -> int | None:
    for w in q["windows"]:
        val = float(w[field])
        if binary:
            if val >= 0.5:
                return int(w["end"])
        elif val >= thr:
            return int(w["end"])
    return None


def eval_door(pack: dict[str, Any], field: str, thr: float, binary: bool = False) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = host_acc = host_tok = puma_acc = puma_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        host_acc += int(q["host_ok"])
        host_tok += q["host_tok"]
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        end = fire_end(q, field, thr, binary)
        if end is None:
            acc += int(q["host_ok"])
            tok += q["host_tok"]
            continue
        sim = rg.pack(q["trials"], q["rows"], end, "rescue", original_tokens=q["orig_tok"])
        ok = gold_ok(sim["answer"], q["gt"], q["original"], q["orig_ok"])
        acc += int(ok)
        tok += sim["tokens"]
        n_fire += 1
        n_gain += int(ok and not q["host_ok"])
        n_hurt += int((not ok) and q["host_ok"])
    return {
        "n": n,
        "threshold": thr,
        "acc": acc / n,
        "tok": tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "d_host_acc": 100.0 * (acc / n - host_acc / n),
        "d_host_tok": tok / n - host_tok / n,
    }


def peek(pack: dict[str, Any], field: str) -> dict[str, Any]:
    host = eval_door(pack, field, float("inf"))
    xs = [float(w[field]) for q in pack["questions"] for w in q["windows"] if w[field] == w[field]]
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=21))
    points = [eval_door(pack, field, thr) for thr in thrs]
    ok = [p for p in points if p["acc"] + 1e-12 >= host["host_acc"]]
    return min(ok, key=lambda p: (p["tok"], -p["acc"])) if ok else host


def merge_packs(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs = []
    for pack in packs:
        qs.extend(pack["questions"])
    return {"questions": qs, "gpqa": packs[0]["gpqa"]}


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def cell(rec: dict[str, Any]) -> str:
    if rec["n_fire"] == 0:
        return f"{pair(rec['acc'], rec['tok'])}（不开）"
    return (
        f"{pair(rec['acc'], rec['tok'])}"
        f"（{rg.fmt_pp(rec['d_host_acc'])} / {rg.fmt_tok(rec['d_host_tok'])}；"
        f"{rec['n_fire']}火/{rec['n_gain']}救/{rec['n_hurt']}伤）"
    )


def fmt_auc(v: float) -> str:
    return "—" if v != v else f"{v:.3f}"


def fmt_rate(v: float) -> str:
    return "—" if v != v else f"{100.0 * v:.0f}%"


def examples(pack: dict[str, Any], n: int = 2) -> list[str]:
    pos = [w for q in pack["questions"] for w in q["windows"] if w["pos"]]
    neg = [w for q in pack["questions"] for w in q["windows"] if not w["pos"]]
    lines = []
    for tag, bucket in (("对上", pos[:n]), ("没对上", neg[:n])):
        for w in bucket:
            lines.append(
                f"  {tag} step={w['step']} only={w['only_trial']:.0f} share={w['trial_share']:.2f} "
                f"unique={w['n_unique_mid']:.0f}/{w['n_unique_near']:.0f} ans={w['ans']}"
            )
    return lines


def eval_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = []
        for seed in dd.AIME_SEEDS:
            pack = load_cell(model, dataset, seed)
            if pack:
                packs.append(pack)
        if len(packs) != 4:
            print(f"skip {zh} {ds_zh} seeds={len(packs)}/4", flush=True)
            return None
        pack = merge_packs(packs)
    else:
        pack = load_cell(model, dataset, 42)
        if pack is None:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
    wins = [w for q in pack["questions"] for w in q["windows"]]
    firsts = [q["windows"][0] for q in pack["questions"] if q["windows"]]
    rec = {
        "name": f"{zh} {ds_zh}",
        "n_q": len(pack["questions"]),
        "n_left": sum(1 for q in pack["questions"] if q["windows"]),
        "n_win": len(wins),
        "n_first": len(firsts),
        "pack": pack,
        "wins": wins,
        "firsts": firsts,
    }
    for field in ("only_trial", "trial_share", "neg_n_unique", "written", "collapse_drop"):
        rec[f"auc_{field}"] = auroc_of(wins, field)[0]
        rec[f"fauc_{field}"] = auroc_of(firsts, field)[0]
    rec["only_pos"] = rate(wins, "only_trial", True)
    rec["only_neg"] = rate(wins, "only_trial", False)
    rec["share_pos"] = rate(wins, "trial_share", True)
    rec["share_neg"] = rate(wins, "trial_share", False)
    rec["written_pos"] = rate(wins, "written", True)
    rec["written_neg"] = rate(wins, "written", False)
    rec["uniq_pos"] = rate(wins, "n_unique_mid", True)
    rec["uniq_neg"] = rate(wins, "n_unique_mid", False)
    rec["door_only"] = eval_door(pack, "only_trial", 0.5, binary=True)
    rec["peek_share"] = peek(pack, "trial_share")
    rec["peek_only"] = peek(pack, "only_trial")
    return rec


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    lines = [
        "# 低把握窗里，思路正文的候选有没有收成一个",
        "",
        "只看密探k4 停点前、连续 4 步同一试答、把握都 < 0.995 的窗。",
        "从最近思路里抽候选：GPQA 抽 A–D，数学抽收口句和公式。",
        "只剩试答 = 最近一小段只抽出这一个、且等于当前试答。正类 = 试答已经等于金标或写完终答。",
        "门开火就交试答、不重写；没开留密探k4。Acc 只对金标。同集偷看会塌，只当对照。",
        "",
        "## 分得开吗",
        "",
        "| 集 | 有剩窗的题 / 窗数 | 只剩试答 对上/没对上 | 正文已写出 对上/没对上 | 窗级AUROC 只剩/占比/已写出 | 第一扇占比AUROC |",
        "|---|---:|---|---|---|---|",
    ]
    door_lines = [
        "",
        "## 当门（相对密探k4）",
        "",
        "| 集 | 密探k4 | 只剩试答就交 | 同集偷看占比 |",
        "|---|---|---|---|",
    ]
    print(lines[-2], flush=True)
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = eval_named(zh, model, ds_zh, dataset)
            if rec is None:
                continue
            recs.append(rec)
            row = (
                f"| {rec['name']} | {rec['n_left']} / {rec['n_win']} "
                f"| {fmt_rate(rec['only_pos'])} / {fmt_rate(rec['only_neg'])} "
                f"| {fmt_rate(rec['written_pos'])} / {fmt_rate(rec['written_neg'])} "
                f"| {fmt_auc(rec['auc_only_trial'])} / {fmt_auc(rec['auc_trial_share'])} / {fmt_auc(rec['auc_written'])} "
                f"| {fmt_auc(rec['fauc_trial_share'])} |"
            )
            drow = (
                f"| {rec['name']} | {pair(rec['door_only']['host_acc'], rec['door_only']['host_tok'])} "
                f"| {cell(rec['door_only'])} | {cell(rec['peek_share'])} |"
            )
            print(row, flush=True)
            print(" ", drow, flush=True)
            for ex in examples(rec["pack"]):
                print(ex, flush=True)
            lines.append(row)
            door_lines.append(drow)
    TABLE.write_text("\n".join(lines + door_lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
