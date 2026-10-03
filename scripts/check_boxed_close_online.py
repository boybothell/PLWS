"""Greedy check: one 30-token trial vs step-by-step matching-brace stop.

Does not change the live trial generator. Uses two questions and two prefixes
per single-GPU model. 32B-class models are omitted because this card is one GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA/puma")
sys.path.insert(0, str(PUMA))

from gen_trial_answers import (  # noqa: E402
    build_prompt,
    extract_first_braced_content,
    find_boxed_content_token_span,
    generate_until_matching_brace,
    parse_response,
)
from prompt_utils import boxed_close_keep_length  # noqa: E402
from vllm_shutdown import shutdown_llm  # noqa: E402

ROOT = Path("/mnt/d/lsj/visual-latent-tts")
MODELS = Path("/mnt/d/lsj/models")
TRIALS = ROOT / "repos/plws/results/upstream/dense_trials"
OUT = ROOT / "repos/plws/results/runs/plws/boxed_close_check/online_greedy.jsonl"

TAGS = {
    "r1_1p5b": "DeepSeek-R1-Distill-Qwen-1.5B",
    "r1_7b": "DeepSeek-R1-Distill-Qwen-7B",
    "r1_14b": "DeepSeek-R1-Distill-Qwen-14B",
    "r1_llama_8b": "DeepSeek-R1-Distill-Llama-8B",
    "nemotron_8b": "Llama-3.1-Nemotron-Nano-8B-v1",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}
MAX_MODEL_LEN = 12288
N_QUESTIONS = 2
MAX_TOKENS = 30


def trial_file(tag: str) -> Path | None:
    for dataset in ("math-500", "amc23"):
        path = (
            TRIALS
            / f"dense_G_{tag}"
            / dataset
            / "seed_42"
            / "dense_puma"
            / "trial_answers.json"
        )
        if path.is_file():
            return path
    return None


def load_prompts(path: Path) -> list[dict]:
    buckets: dict[int, list[dict]] = {}
    current: dict = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().rstrip(",")
            if stripped.startswith(
                ('"question_idx"', '"stopped_len"', '"question"', '"reasoning_prefix"')
            ):
                current.update(json.loads("{" + stripped + "}"))
            elif stripped in ("}", "},"):
                qid = current.get("question_idx")
                current_row = dict(current)
                current = {}
                if qid is None or "reasoning_prefix" not in current_row:
                    continue
                qid = int(qid)
                if qid > N_QUESTIONS:
                    break
                buckets.setdefault(qid, []).append(current_row)
    chosen = []
    for qid in sorted(buckets)[:N_QUESTIONS]:
        rows = sorted(buckets[qid], key=lambda row: int(row["stopped_len"]))
        picks = [rows[0]]
        if len(rows) > 1:
            picks.append(rows[min(7, len(rows) - 1)])
        chosen.extend(picks)
    return chosen


def score(token_ids: list[int], tokenizer) -> tuple[str, int]:
    text = tokenizer.decode(token_ids)
    answer = extract_first_braced_content(text)
    if not answer:
        answer = parse_response(text)
    start, end = find_boxed_content_token_span(token_ids, tokenizer)
    count = (end - start) if start < end else len(token_ids)
    return answer, count


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as sink:
        for tag, folder in TAGS.items():
            model = MODELS / folder
            source = trial_file(tag)
            if not model.is_dir() or source is None:
                print(f"skip {tag}", flush=True)
                continue
            rows = load_prompts(source)
            print(f"load {tag} prompts={len(rows)} from {source.name}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            llm = LLM(
                model=str(model),
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=0.9,
                seed=0,
            )
            try:
                for row in rows:
                    prompt = build_prompt(
                        tokenizer,
                        str(model),
                        row["question"],
                        row["reasoning_prefix"],
                        "math",
                        "default",
                        True,
                        "math-500",
                    )
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    if len(prompt_ids) + MAX_TOKENS > MAX_MODEL_LEN:
                        print(
                            f"skip long {tag} q={row['question_idx']} "
                            f"step={row['stopped_len']} n={len(prompt_ids)}",
                            flush=True,
                        )
                        continue
                    base = TokensPrompt(prompt_token_ids=prompt_ids)
                    oneshot = llm.generate(
                        [base],
                        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
                        use_tqdm=False,
                    )[0].outputs[0]
                    oneshot_ids = list(oneshot.token_ids)
                    step_ids = generate_until_matching_brace(
                        llm,
                        [prompt_ids],
                        tokenizer,
                        max_tokens=MAX_TOKENS,
                        sampling_kwargs={"temperature": 0.0},
                    )[0]
                    oneshot_answer, oneshot_count = score(oneshot_ids, tokenizer)
                    step_answer, step_count = score(step_ids, tokenizer)
                    texts = [tokenizer.decode([tid]) for tid in oneshot_ids]
                    keep = boxed_close_keep_length(texts)
                    prefix_ok = oneshot_ids[: len(step_ids)] == step_ids
                    record = {
                        "model": tag,
                        "question_idx": int(row["question_idx"]),
                        "stopped_len": int(row["stopped_len"]),
                        "oneshot_len": len(oneshot_ids),
                        "step_len": len(step_ids),
                        "keep_on_oneshot": keep,
                        "prefix_match": prefix_ok,
                        "answer_same": oneshot_answer == step_answer,
                        "count_same": oneshot_count == step_count,
                        "oneshot_answer": oneshot_answer,
                        "step_answer": step_answer,
                        "oneshot_count": oneshot_count,
                        "step_count": step_count,
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                    print(
                        f"{tag} q={record['question_idx']} step={record['stopped_len']} "
                        f"prefix={prefix_ok} answer={record['answer_same']} "
                        f"count={record['count_same']} "
                        f"len={record['oneshot_len']}/{record['step_len']}",
                        flush=True,
                    )
            finally:
                shutdown_llm(llm)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
