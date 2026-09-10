#!/usr/bin/env python3
"""PUMA stays the base. Low-conf R is an add-on.

Oracle: stop at first R before official PUMA exit, else keep PUMA.
Half-depth: same add-on, but the actual last-minus-half door. Same-set
threshold vs PUMA (Acc not drop, then max Acc, then min tokens).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/puma_plus_r_oracle.md"
SIG = "exit_minus_half"
DOLA = "dola_last"
HALF_READY = 0.85


def attach_puma_step(pack: dict[str, Any], stat: Path) -> None:
    official = {int(row["question_idx"]): row for row in json.loads(stat.read_text())}
    for q in pack["questions"]:
        info = official.get(int(q["qi"])) or {}
        q["puma_step"] = int(info.get("stopped_len") or 10**9)
        q["puma_reason"] = str(info.get("stop_reason") or "")


def trial_ok(q: dict[str, Any], answer: Any) -> bool:
    return rg.same(answer, q["gt"])


def attach_dola_last(pack: dict[str, Any]) -> str | None:
    ids = more.discover_layers(pack["scores"])
    if not ids:
        pack["_dola_sig"] = None
        return None
    src = f"dola_mean_l{max(ids)}"
    for row in pack["scores"].values():
        row[DOLA] = rg.finite(row.get(src))
    pack["_dola_sig"] = src
    pack["_dola_layer"] = max(ids)
    return src


def load_jobs() -> list[tuple[str, dict[str, Any]]]:
    layers.enrich = more.enrich_all
    out: list[tuple[str, dict[str, Any]]] = []
    for name, cells in sw.all_jobs():
        packs: list[dict[str, Any]] = []
        for cell in cells:
            cell = dict(cell)
            if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
                continue
            has_scores = any(Path(p).is_dir() for p in cell.get("scores") or ())
            if not has_scores:
                cell["scores"] = ()
            pack = layers.load_pack(cell)
            attach_puma_step(pack, Path(cell["stat"]))
            names = []
            if has_scores:
                cons.attach_shared(pack)
                attach_dola_last(pack)
                names = [SIG, DOLA, "last_mean_logp"]
                layers.precompute_events(pack, names)
            else:
                pack.setdefault("scores", {})
                pack["_dola_sig"] = None
                layers.precompute_events(pack, [])
            pack["_half_cov"] = cmp.cov(pack, SIG) if has_scores else 0.0
            pack["_dola_cov"] = cmp.cov(pack, DOLA) if has_scores and pack.get("_dola_sig") else 0.0
            pack["_cov"] = pack["_half_cov"]
            packs.append(pack)
        if not packs:
            continue
        if len(cells) > 1 and len(packs) != len(cells):
            if len(packs) < 3:
                print(f"skip {name} seeds={len(packs)}/{len(cells)}", flush=True)
                continue
            name = f"{name}（{len(packs)} seed）"
            print(f"partial {name} seeds={len(packs)}/{len(cells)}", flush=True)
        pack = accfirst.merge_packs(packs, name) if len(packs) > 1 else packs[0]
        pack["_half_cov"] = min(p["_half_cov"] for p in packs)
        pack["_dola_cov"] = min(p["_dola_cov"] for p in packs)
        pack["_half_layer"] = packs[0].get("_half_layer")
        pack["_dola_layer"] = packs[0].get("_dola_layer")
        out.append((name, pack))
        print(
            f"load {name} n={len(pack['questions'])} half_cov={pack['_half_cov']:.2f} "
            f"dola_cov={pack['_dola_cov']:.2f} half_layer={pack.get('_half_layer')} "
            f"dola_layer={pack.get('_dola_layer')}",
            flush=True,
        )
    return out


def first_r(q: dict[str, Any]) -> dict[str, Any] | None:
    """First low-conf consecutive window whose trial matches gold."""
    rows = q["rows"]
    puma_step = int(q.get("puma_step") or 10**9)
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(rows[ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= puma_step:
            continue
        if rg.same(rows[ev["end"]].get("final_answer"), q["gt"]):
            return ev
    return None


def fire_sig(q: dict[str, Any], sig: str, thr: float, need: str) -> dict[str, Any] | None:
    puma_step = int(q.get("puma_step") or 10**9)
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= puma_step:
            continue
        vals = ev["vals"].get(sig) or []
        if layers.passed(vals, thr, need):
            return ev
    return None


def fire_half(q: dict[str, Any], thr: float) -> dict[str, Any] | None:
    return fire_sig(q, SIG, thr, "all")


def last_val(vals: list[float] | None) -> float:
    if not vals:
        return float("nan")
    val = vals[-1]
    return val if val == val else float("nan")


def fire_ratio(q: dict[str, Any], thr: float) -> dict[str, Any] | None:
    puma_step = int(q.get("puma_step") or 10**9)
    for ev in q["events"]:
        if ev.get("high") or ev.get("mixed"):
            continue
        step = int(q["rows"][ev["end"]]["stopped_len"])
        if step < rg.MSS or step >= puma_step:
            continue
        rise = last_val(ev["vals"].get(SIG))
        exit_s = last_val(ev["vals"].get("last_mean_logp"))
        if rise != rise or exit_s != exit_s:
            continue
        if rise / (abs(exit_s) + 1e-6) >= thr:
            return ev
    return None


def eval_ratio(pack: dict[str, Any], thr: float) -> dict[str, Any] | None:
    if float(pack.get("_half_cov") or 0.0) < HALF_READY:
        return None
    pack["_fire_sig"] = SIG
    pack["_fire_need"] = "last"
    n = len(pack["questions"])
    acc = tok = puma_acc = puma_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        ev = fire_ratio(q, thr)
        if ev is None:
            acc += int(q["puma_ok"])
            tok += q["puma_tok"]
            continue
        sim = rg.pack(
            q["trials"],
            q["rows"],
            ev["end"],
            "rescue",
            original_tokens=q["orig_tok"],
            label=ev.get("tag") or "O",
        )
        ok = trial_ok(q, sim["answer"])
        acc += int(ok)
        tok += sim["tokens"]
        n_fire += 1
        if ok and not q["puma_ok"]:
            n_gain += 1
        if (not ok) and q["puma_ok"]:
            n_hurt += 1
    return {
        "threshold": thr,
        "n": n,
        "acc": acc / max(n, 1),
        "tok": tok / max(n, 1),
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "puma_acc": puma_acc / max(n, 1),
        "puma_tok": puma_tok / max(n, 1),
        "d_acc": 100.0 * (acc / max(n, 1) - puma_acc / max(n, 1)),
        "d_tok": tok / max(n, 1) - puma_tok / max(n, 1),
    }


def eval_oracle(pack: dict[str, Any]) -> dict[str, Any]:
    rg.K = 4
    rg.TAU = 0.995
    n = len(pack["questions"])
    has_r = 0
    before = 0
    gain = 0
    acc = 0
    tok = 0.0
    puma_acc = 0
    puma_tok = 0.0
    for q in pack["questions"]:
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        ev = first_r(q)
        if ev is None:
            acc += int(q["puma_ok"])
            tok += q["puma_tok"]
            continue
        has_r += 1
        before += 1
        sim = rg.pack(q["trials"], q["rows"], ev["end"], "rescue", original_tokens=q["orig_tok"], label="R")
        acc += 1
        tok += sim["tokens"]
        if not q["puma_ok"]:
            gain += 1
    return {
        "n": n,
        "has_r": has_r,
        "before": before,
        "gain": gain,
        "puma_acc": puma_acc / max(n, 1),
        "puma_tok": puma_tok / max(n, 1),
        "acc": acc / max(n, 1),
        "tok": tok / max(n, 1),
        "d_acc": 100.0 * (acc / max(n, 1) - puma_acc / max(n, 1)),
        "d_tok": tok / max(n, 1) - puma_tok / max(n, 1),
    }


def eval_half_thr(pack: dict[str, Any], thr: float) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = 0
    tok = 0.0
    n_fire = 0
    n_gain = 0
    n_hurt = 0
    n_r = 0
    n_l = 0
    puma_acc = 0
    puma_tok = 0.0
    for q in pack["questions"]:
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        ev = fire_sig(q, pack.get("_fire_sig") or SIG, thr, pack.get("_fire_need") or "all")
        if ev is None:
            acc += int(q["puma_ok"])
            tok += q["puma_tok"]
            continue
        sim = rg.pack(
            q["trials"],
            q["rows"],
            ev["end"],
            "rescue",
            original_tokens=q["orig_tok"],
            label=ev.get("tag") or "O",
        )
        ok = trial_ok(q, sim["answer"])
        acc += int(ok)
        tok += sim["tokens"]
        n_fire += 1
        if ev.get("tag") == "R":
            n_r += 1
        elif ev.get("tag") == "L":
            n_l += 1
        if ok and not q["puma_ok"]:
            n_gain += 1
        if (not ok) and q["puma_ok"]:
            n_hurt += 1
    return {
        "threshold": thr,
        "n": n,
        "acc": acc / max(n, 1),
        "tok": tok / max(n, 1),
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "rescue_R": n_r,
        "rescue_L": n_l,
        "puma_acc": puma_acc / max(n, 1),
        "puma_tok": puma_tok / max(n, 1),
        "d_acc": 100.0 * (acc / max(n, 1) - puma_acc / max(n, 1)),
        "d_tok": tok / max(n, 1) - puma_tok / max(n, 1),
    }


def sweep_sig(pack: dict[str, Any], sig: str, need: str, cov_key: str) -> dict[str, Any] | None:
    if float(pack.get(cov_key) or 0.0) < HALF_READY:
        return None
    pack["_fire_sig"] = sig
    pack["_fire_need"] = need
    xs = cmp.low_xs(pack, sig)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = [eval_half_thr(pack, thr) for thr in thrs]
    puma_acc = pack.get("puma_acc")
    puma_tok = pack.get("puma_tok")
    if puma_acc is None or puma_tok is None:
        n = max(len(pack["questions"]), 1)
        puma_acc = sum(int(q["puma_ok"]) for q in pack["questions"]) / n
        puma_tok = sum(q["puma_tok"] for q in pack["questions"]) / n
    return accfirst.pick_acc_first_both(points, puma_acc, puma_tok)


def sweep_half(pack: dict[str, Any]) -> dict[str, Any] | None:
    return sweep_sig(pack, SIG, "all", "_half_cov")


def sweep_dola(pack: dict[str, Any]) -> dict[str, Any] | None:
    return sweep_sig(pack, DOLA, "last", "_dola_cov")


def fmt_actual(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "未齐"
    if rec["threshold"] == float("inf"):
        return "不开"
    return sw.fmt_pair(rec["acc"], rec["tok"])


def fmt_actual_delta(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "—"
    if rec["threshold"] == float("inf"):
        return "不开"
    return f"{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])}"


def main() -> None:
    head = (AE / "tables/puma_plus_r_oracle.md").read_text()
    # keep formula block if present; otherwise write a short gold-only header
    cut = head.find("| 集 |")
    prefix = head[:cut] if cut >= 0 else ""
    if "只对金标" not in prefix:
        prefix = (
            "# PUMA 底：默认共用对比率 + 同集对照 + 低置信上界\n\n"
            "正确率只对金标。试答对齐写完终答、金标不对，算错。\n"
            "默认附加门公式见下。上限 = 停点前若有低置信连答且试答对金标，先知交这步；否则留 PUMA。不是能冻的门。\n\n"
        )
    else:
        # refresh the 上限 sentence in the existing prefix
        prefix = prefix.replace(
            "上限 = 低置信连对若出现在 PUMA 停点之前就先知停。不是能冻的门。",
            "上限 = 停点前若有低置信连答且试答对金标，先知交这步；否则留 PUMA。不是能冻的门。正确率只对金标。",
        )
    lines = [
        prefix.rstrip(),
        "",
        "| 集 | PUMA | 默认共用 | 默认比PUMA | 同集一半深 | 一半深比PUMA | 最后一层好写 | 好写比PUMA | 上限 | 上限比PUMA |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    print(lines[-2], flush=True)
    for name, pack in load_jobs():
        rec = eval_oracle(pack)
        default = eval_ratio(pack, rg.REL_RISE_THR)
        half = sweep_half(pack)
        dola = sweep_dola(pack)
        if default is not None:
            print(
                f"  default {name} fire={default['n_fire']} "
                f"gain={default['n_gain']} hurt={default['n_hurt']} {fmt_actual_delta(default)}",
                flush=True,
            )
        print(
            f"  oracle {name} n={rec['n']} before={rec['before']} gain={rec['gain']} "
            f"{sw.fmt_pair(rec['acc'], rec['tok'])} {rg.fmt_pp(rec['d_acc'])}",
            flush=True,
        )
        row = (
            f"| {name} | {sw.fmt_pair(rec['puma_acc'], rec['puma_tok'])} | "
            f"{fmt_actual(default)} | {fmt_actual_delta(default)} | "
            f"{fmt_actual(half)} | {fmt_actual_delta(half)} | "
            f"{fmt_actual(dola)} | {fmt_actual_delta(dola)} | "
            f"{sw.fmt_pair(rec['acc'], rec['tok'])} | "
            f"{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
