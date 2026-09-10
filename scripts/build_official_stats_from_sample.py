#!/usr/bin/env python3
"""从已采 Full-CoT 写出官方 statistics.json（original_*）。不跑密探、不重写。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
PUMA = AE.parent / "PUMA" / "puma"
sys.path.insert(0, str(PUMA))
import statistics_puma as st  # noqa: E402

MODELS = {
    "r1_7b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B",
    "nemotron_8b": "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_14b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B",
}
CELLS = [
    ("r1_7b", "math-500", 0),
    ("r1_7b", "math-500", 1),
    ("r1_7b", "math-500", 123),
    ("r1_7b", "olympiadbench", 0),
    ("r1_7b", "olympiadbench", 1),
    ("r1_7b", "olympiadbench", 123),
    ("r1_7b", "gpqa-diamond", 0),
    ("nemotron_8b", "math-500", 0),
    ("nemotron_8b", "math-500", 123),
    ("r1_14b", "math-500", 1),
    ("r1_14b", "math-500", 123),
    ("r1_14b", "olympiadbench", 0),
    ("r1_14b", "olympiadbench", 1),
    ("r1_14b", "olympiadbench", 123),
    ("r1_14b", "gpqa-diamond", 123),
]


def puma_dir(tag: str, ds: str, seed: int) -> Path:
    if tag == "r1_7b" and ds == "math-500" and seed == 42:
        return AE / "results/math500_official/puma_ds7b"
    if seed == 42:
        return AE / f"results/puma_offline_{tag}/{ds}"
    return AE / f"results/puma_offline_{tag}_s{seed}/{ds}"


def gold(text: str) -> str:
    text = str(text or "").strip()
    m = re.fullmatch(r"\\boxed\{(.+)\}", text)
    return m.group(1).strip() if m else text


def build_one(tag: str, ds: str, seed: int, tokenizer) -> Path:
    folder = puma_dir(tag, ds, seed)
    answers_path = folder / "answers.json"
    if not answers_path.is_file():
        src = AE / f"samples/{tag}/{ds}/seed_{seed}/answers.json"
        if not src.is_file():
            raise FileNotFoundError(src)
        folder.mkdir(parents=True, exist_ok=True)
        answers_path.write_text(src.read_text())
    answers = json.loads(answers_path.read_text())
    indexed = st.process_original_data(answers, tokenizer)
    steps = []
    filt = folder / "filtered_steps.json"
    if filt.is_file():
        steps = json.loads(filt.read_text())
    out = []
    n_ok = 0
    for qi, obj in indexed.items():
        ans = str(obj.get("model_answer") or "").strip()
        gt = gold(str(obj.get("ground_truth_answer") or ""))
        ok = bool(st._grade_one((qi, ans, gt, "orig"))[2])
        n_ok += int(ok)
        n_steps = 0
        if 0 <= qi - 1 < len(steps):
            n_steps = len(steps[qi - 1].get("reasoning_steps") or [])
        tok = int(obj["total_token_count"])
        out.append(
            {
                "question_idx": qi,
                "question": obj.get("question") or "",
                "ground_truth": gt,
                "original_answer": ans,
                "compressed_answer": ans,
                "original_correct": ok,
                "compressed_correct": ok,
                "transition": "R→R" if ok else "W→W",
                "original_tokens": tok,
                "compressed_tokens": tok,
                "stopped_len": n_steps,
                "original_len_reasoning_steps": n_steps,
                "stop_reason": "full_reasoning",
                "stop_confidence": None,
                "tokens_trial_answers": 0,
                "generated_trial_answers": 0,
                "online_overhead_tokens": 0,
            }
        )
    dest = folder / "statistics.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{tag} {ds} s{seed} n={len(out)} acc={100.0 * n_ok / len(out):.1f}% -> {dest}",
        flush=True,
    )
    return dest


def verify_match() -> None:
    tag, ds, seed = "r1_7b", "aime24", 0
    st._LOCAL_TOKENIZER = None
    tok = st.load_tokenizer(MODELS[tag])
    gold_stats = {int(r["question_idx"]): r for r in json.loads(puma_dir(tag, ds, seed).joinpath("statistics.json").read_text())}
    answers = json.loads(puma_dir(tag, ds, seed).joinpath("answers.json").read_text())
    indexed = st.process_original_data(answers, tok)
    n = miss = 0
    for qi, obj in indexed.items():
        off = gold_stats.get(qi)
        if not off:
            continue
        n += 1
        if int(obj["total_token_count"]) != int(off["original_tokens"]):
            miss += 1
    print(f"verify {tag} {ds} s{seed} token_match={n - miss}/{n}", flush=True)
    if n == 0 or miss:
        raise SystemExit(f"token mismatch {miss}/{n}")


def main() -> None:
    verify_match()
    last = None
    for tag, ds, seed in CELLS:
        dest = puma_dir(tag, ds, seed) / "statistics.json"
        if dest.is_file() and dest.stat().st_size > 10:
            print(f"skip have {tag} {ds} s{seed}", flush=True)
            continue
        if last != tag:
            st._LOCAL_TOKENIZER = None
            tokenizer = st.load_tokenizer(MODELS[tag])
            last = tag
        else:
            tokenizer = st._LOCAL_TOKENIZER
        build_one(tag, ds, seed, tokenizer)


if __name__ == "__main__":
    main()
