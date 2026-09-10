#!/usr/bin/env python3
"""Default rel-rise door on the saved dense+regen host. Gold Acc."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_brightest_stop_auroc as jobs
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import report_puma_plus_r_oracle as base

TABLE = AE / "tables/dense_regen_default_door.md"
HALF_READY = 0.85
SIG = "exit_minus_half"
MODELS = (
    ("7B", "r1_7b", "dense_G_r1_7b", "puma_offline_r1_7b"),
    ("8B", "nemotron_8b", "dense_G_nemotron_8b", "puma_offline_nemotron_8b"),
)
DS = (
    ("MATH", "math-500", False),
    ("奥赛", "olympiadbench", False),
    ("GPQA", "gpqa-diamond", False),
    ("AIME24", "aime24", True),
    ("AIME25", "aime25", True),
)


def attach_host(pack: dict[str, Any], model: str, dataset: str, seed: int) -> bool:
    regen_path = dd.regen_stat_path(model, dataset, seed)
    if not regen_path.is_file():
        return False
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
    n = 0
    for q in pack["questions"]:
        info = regen.get(int(q["qi"]))
        if not info:
            continue
        q["host_step"] = int(info.get("stopped_len") or 10**9)
        q["host_ok"] = bool(info.get("compressed_correct"))
        q["host_tok"] = int(info.get("compressed_tokens") or 0) + int(
            info.get("tokens_trial_answers") or 0
        )
        n += 1
    return n == len(pack["questions"])


def fire_door(q: dict[str, Any], thr: float) -> dict[str, Any] | None:
    host_step = int(q.get("host_step") or 10**9)
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= host_step:
            continue
        rise = base.last_val(ev["vals"].get(SIG))
        exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
        if rise != rise or exit_s != exit_s:
            continue
        if rise / (abs(exit_s) + 1e-6) >= thr:
            return ev
    return None


def eval_pack(pack: dict[str, Any], thr: float | None = None) -> dict[str, Any] | None:
    if float(pack.get("_half_cov") or 0.0) < HALF_READY:
        return None
    if thr is None:
        thr = rg.REL_RISE_THR
    n = len(pack["questions"])
    acc = tok = host_acc = host_tok = puma_acc = puma_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        host_ok = bool(q["host_ok"])
        host_t = float(q["host_tok"])
        puma_ok = bool(q["puma_ok"])
        puma_t = float(q["puma_tok"])
        host_acc += int(host_ok)
        host_tok += host_t
        puma_acc += int(puma_ok)
        puma_tok += puma_t
        ev = fire_door(q, thr)
        if ev is None:
            acc += int(host_ok)
            tok += host_t
            continue
        sim = rg.pack(
            q["trials"],
            q["rows"],
            ev["end"],
            "rescue",
            original_tokens=q["orig_tok"],
            label=ev.get("tag") or "O",
        )
        ok = rg.same(sim["answer"], q["gt"])
        acc += int(ok)
        tok += sim["tokens"]
        n_fire += 1
        if ok and not host_ok:
            n_gain += 1
        if (not ok) and host_ok:
            n_hurt += 1
    return {
        "n": n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "acc": acc / n,
        "tok": tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "d_host_acc": 100.0 * (acc / n - host_acc / n),
        "d_host_tok": tok / n - host_tok / n,
        "d_puma_acc": 100.0 * (acc / n - puma_acc / n),
        "d_puma_tok": tok / n - puma_tok / n,
        "half_cov": pack["_half_cov"],
        "threshold": thr,
    }


def load_one(zh: str, tag: str, gdir: str, puma: str, ds_zh: str, dataset: str, seed: int | None) -> dict[str, Any] | None:
    cell = jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset, seed)
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
        return None
    pack = layers.load_pack(cell)
    ids = more.discover_layers(pack["scores"])
    if len(ids) < 2:
        return None
    cons.attach_shared(pack)
    layers.precompute_events(pack, [SIG, "last_mean_logp"])
    pack["_half_cov"] = cmp.cov(pack, SIG)
    pack["_cov"] = pack["_half_cov"]
    use_seed = 42 if seed is None else seed
    if not attach_host(pack, tag, dataset, use_seed):
        return None
    return pack


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    header = (
        "| 集 | 官方 PUMA | 密探+重写 | 现行门 | 门比重写 | 门比PUMA | 开火 / 救回 / 伤 |"
    )
    sep = "|---|---|---|---|---|---|---:|"
    lines = [
        "# 密探+重写 + 现行对比率门（正确率只对金标）",
        "",
        "宿主是已存密探双闸门：k=4、0.995、后路强停，截断后再重写终答。",
        f"附加门和挂在官方 PUMA 上的一样：低置信连答窗最后一步，"
        f"(末层好写 − 一半深好写) / |末层好写| ≥ {rg.REL_RISE_THR:.2f}，必须在密探宿主停点之前，交试答。",
        "十集同一条，不按集选门槛。中间层覆盖不到 85% 的不报。AIME 四个 seed。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for zh, tag, gdir, puma in MODELS:
        for ds_zh, dataset, is_aime in DS:
            if is_aime:
                packs = []
                for seed in jobs.SEEDS:
                    pack = load_one(zh, tag, gdir, puma, ds_zh, dataset, seed)
                    if pack is None:
                        continue
                    print(
                        f"load {zh} {ds_zh}-s{seed} n={len(pack['questions'])} "
                        f"half={pack['_half_cov']:.2f}",
                        flush=True,
                    )
                    packs.append(pack)
                if len(packs) != 4:
                    print(f"skip {zh} {ds_zh} seeds={len(packs)}/4", flush=True)
                    continue
                pack = accfirst.merge_packs(packs, f"{zh} {ds_zh}")
                pack["_half_cov"] = min(p["_half_cov"] for p in packs)
            else:
                pack = load_one(zh, tag, gdir, puma, ds_zh, dataset, None)
                if pack is None:
                    print(f"skip {zh} {ds_zh}", flush=True)
                    continue
                print(
                    f"load {zh} {ds_zh} n={len(pack['questions'])} half={pack['_half_cov']:.2f}",
                    flush=True,
                )
            rec = eval_pack(pack)
            if rec is None:
                print(f"| {zh} {ds_zh} | 未齐 |", flush=True)
                continue
            row = (
                f"| {zh} {ds_zh} | {fmt_pair(rec['puma_acc'], rec['puma_tok'])} "
                f"| {fmt_pair(rec['host_acc'], rec['host_tok'])} "
                f"| {fmt_pair(rec['acc'], rec['tok'])} "
                f"| {rg.fmt_pp(rec['d_host_acc'])} / {rg.fmt_tok(rec['d_host_tok'])} "
                f"| {rg.fmt_pp(rec['d_puma_acc'])} / {rg.fmt_tok(rec['d_puma_tok'])} "
                f"| {rec['n_fire']} / {rec['n_gain']} / {rec['n_hurt']} |"
            )
            print(row, flush=True)
            lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
