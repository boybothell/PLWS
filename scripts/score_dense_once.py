#!/usr/bin/env python3
"""One HF forward: default lag scores + per-layer lens/DoLA + other internals.

Writes the dense_lens jsonl (last_mean_logp, lens_l*) and the same row to
dense_internal so later analysis does not need a second prefix forward.
Same-question steps reuse the prefix KV cache. Boxed tokens are not kept
in the cache. Raw hidden tensors are not dumped (too large); every layer
in the grid is stored as scalars.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import score_dense_internal_metrics as inn  # noqa: E402
from layer_grid import layer_grid, parse_layers  # noqa: E402
from score_confcal_keytoken import INSTRUCTION, SOLVER  # noqa: E402
from score_confcal_v1 import nvidia_lib_path  # noqa: E402

os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()


def _lag_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))


def _lens_ids(row: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for key in row:
        if key.startswith("lens_l") and key[6:].isdigit():
            out.add(int(key[6:]))
    return out


def _scored_for_lag(row: dict[str, Any]) -> bool:
    """Done only when this row has the full layer grid.

    Wait-only rows (stop_margin, no last_mean_logp) must be extracted again.
    New dumps write n_layers and every lens_l*; sparse last+half rows do not
    count as done. Old rows without n_layers still skip so 7B/8B MATH is not
    re-extracted.
    """
    status = row.get("status")
    if status in {"too_long", "oom", "no_answer"}:
        return True
    if status != "ok":
        return False
    have = _lens_ids(row)
    n_layers = row.get("n_layers")
    if n_layers:
        return set(layer_grid(int(n_layers))) <= have
    return bool(row.get("once") or "last_mean_logp" in row or have)


def once_done_keys(path: Path) -> set[tuple[int, int, str]]:
    """Skip only rows that already have half-depth scalars, or were skipped.

    Wait-only (stop_margin, no last_mean_logp) is not half-depth done.
    """
    if not path.exists():
        return set()
    return {_lag_key(row) for row in inn.load_jsonl(path) if _scored_for_lag(row)}


def twin_internal(lens_out: Path) -> Path:
    text = str(lens_out)
    if "/dense_lens/" in text:
        return Path(text.replace("/dense_lens/", "/dense_internal/"))
    if text.endswith("/dense_lens"):
        return Path(text[: -len("dense_lens")] + "dense_internal")
    return lens_out.parent.parent / "dense_internal" / lens_out.parent.name / lens_out.name


def crop_past(past: Any, keep: int) -> Any:
    if past is None or keep <= 0:
        return None
    if hasattr(past, "crop"):
        past.crop(keep)
        return past
    cropped = []
    for layer in past:
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            key, value = layer[0], layer[1]
            cropped.append((key[..., :keep, :], value[..., :keep, :], *layer[2:]))
        else:
            cropped.append(layer)
    return tuple(cropped)


def cache_len(past: Any) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[-2])


def is_prefix(prev: list[int], cur: list[int]) -> bool:
    return len(cur) >= len(prev) and cur[: len(prev)] == prev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--internal-out", type=Path)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=SOLVER)
    parser.add_argument("--layers", default="")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--no-kv", action="store_true")
    parser.add_argument("--prompt-mode", choices=("once", "puma"), default="once")
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = inn.load_jsonl(args.candidates)
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for i, qi in enumerate(qis) if i % args.num_shards == args.shard_id}
    jobs = [row for row in rows if int(row["question_idx"]) in keep]
    jobs.sort(key=lambda row: (int(row["question_idx"]), int(row["decision_step"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    internal_out = args.internal_out or twin_internal(args.out)
    internal_out.parent.mkdir(parents=True, exist_ok=True)
    done = once_done_keys(args.out)
    done_internal = once_done_keys(internal_out)
    pending = [
        row
        for row in jobs
        if (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in done
    ]
    if not pending:
        print(f"dense-once shard={args.shard_id}/{args.num_shards} pending=0 -> {args.out}", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    load_kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    if args.device_map:
        n_vis = torch.cuda.device_count()
        load_kw["device_map"] = args.device_map
        load_kw["max_memory"] = {i: "36GiB" for i in range(n_vis)}
        model = AutoModelForCausalLM.from_pretrained(str(args.model), **load_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(args.model), **load_kw).to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    head_device = model.lm_head.weight.device
    n_layers = int(model.config.num_hidden_layers)
    layer_ids = parse_layers(args.layers, n_layers)

    def first_id(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0]) if ids else -1

    stop_id = first_id("</think>")
    wait_id = first_id("Wait")
    alt_id = first_id("Alternatively")
    revise_pats = [tokenizer.encode(text, add_special_tokens=False) for text in inn.REVISE_STRINGS]
    print(
        f"dense-once shard={args.shard_id}/{args.num_shards} pending={len(pending)} "
        f"model={Path(args.model).name} layers={layer_ids} kv={not args.no_kv} "
        f"-> {args.out} + {internal_out}",
        flush=True,
    )

    started = time.perf_counter()
    prev_qi = None
    prev_h = None
    prev_prefix: list[int] = []
    past = None
    last_hiddens: list[torch.Tensor] = []
    mid_hiddens: list[torch.Tensor] = []
    prefix_hin: torch.Tensor | None = None

    with args.out.open("a") as handle, internal_out.open("a") as handle_in:
        for index, row in enumerate(pending, start=1):
            if args.prompt_mode == "puma":
                sys.path.insert(0, str(AE.parent / "PUMA"))
                from puma.prompt_utils import build_base_prompt, get_task_type

                # Folder names like aime24_s42 are not PUMA dataset ids (they
                # become task type nq). Prefer --dataset, then the candidate row.
                dataset = str(args.dataset or row.get("dataset") or "")
                if not dataset:
                    raise SystemExit(f"puma prompt needs --dataset for {args.out}")
                base = build_base_prompt(
                    tokenizer, str(args.model), str(row["question"]), get_task_type(dataset), "default"
                )
                prompt = base + "<think>"
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{INSTRUCTION}\n{row['question']}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prefix = tokenizer.encode(prompt + str(row["reasoning_prefix"]) + "\n", add_special_tokens=False)
            answer = tokenizer.encode(f"\\boxed{{{row['answer']}}}", add_special_tokens=False)
            ids = prefix + answer
            prompt_len = len(prompt_ids)
            if len(ids) + 1 > args.max_context:
                overflow = len(ids) + 1 - args.max_context
                if overflow >= len(prefix):
                    slim = {k: v for k, v in row.items() if k != "reasoning_prefix"}
                    slim["status"] = "too_long"
                    line = json.dumps(slim) + "\n"
                    handle.write(line)
                    key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
                    if key not in done_internal:
                        handle_in.write(line)
                    past = None
                    prev_prefix = []
                    prefix_hin = None
                    continue
                ids = prefix[overflow:] + answer
                prefix = prefix[overflow:]
                prompt_len = max(0, prompt_len - overflow)
                past = None
                prev_prefix = []
                prefix_hin = None

            qi = int(row["question_idx"])
            reuse = (
                (not args.no_kv)
                and past is not None
                and qi == prev_qi
                and is_prefix(prev_prefix, prefix)
                and cache_len(past) == len(prev_prefix)
            )
            cache_keep = len(prev_prefix) if reuse else 0
            if reuse and cache_keep == len(prefix) and prefix:
                past = crop_past(past, len(prefix) - 1)
                cache_keep = len(prefix) - 1
            chunk = (prefix[cache_keep:] + answer) if reuse else ids
            if not chunk:
                past = None
                reuse = False
                cache_keep = 0
                chunk = ids

            if len(answer) <= 0:
                slim = {k: v for k, v in row.items() if k != "reasoning_prefix"}
                slim["status"] = "no_answer"
                line = json.dumps(slim) + "\n"
                handle.write(line)
                key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
                if key not in done_internal:
                    handle_in.write(line)
                continue

            boxed_start = len(chunk) - len(answer)
            start = len(prefix)
            end = start + len(answer)
            keep_n = min(len(chunk), max(len(answer) + 1, 1))

            def forward(chunk_ids: list[int], cache: Any, use_cache: bool):
                tensor = torch.tensor([chunk_ids], device=device)
                with torch.inference_mode():
                    return model(
                        tensor,
                        past_key_values=cache,
                        use_cache=use_cache,
                        output_hidden_states=True,
                        logits_to_keep=min(len(chunk_ids), keep_n),
                    )

            try:
                out = forward(chunk, past if reuse else None, not args.no_kv)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                past = None
                reuse = False
                cache_keep = 0
                chunk = ids
                boxed_start = len(chunk) - len(answer)
                keep_n = min(len(chunk), max(len(answer) + 1, 1))
                try:
                    out = forward(chunk, None, False)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    slim = {k: v for k, v in row.items() if k != "reasoning_prefix"}
                    slim["status"] = "oom"
                    line = json.dumps(slim) + "\n"
                    handle.write(line)
                    key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
                    if key not in done_internal:
                        handle_in.write(line)
                    prev_prefix = []
                    prefix_hin = None
                    continue
            hidden = out.hidden_states
            logit_off = len(chunk) - int(out.logits.shape[1])
            logits = out.logits[0, boxed_start - 1 - logit_off : boxed_start + len(answer) - 1 - logit_off].float()
            target = torch.tensor(answer, device=logits.device)
            last_logp = torch.log_softmax(logits, dim=-1)
            gather = last_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            last_mean = float(gather.mean().item())
            last_min = float(gather.min().item())
            ent = float((-(last_logp.exp() * last_logp).sum(-1)).mean().item())

            content_start = 2 if len(answer) > 3 else 0
            content_end = len(answer) - 1 if len(answer) > 3 else len(answer)
            if content_end <= content_start:
                content_start, content_end = 0, len(answer)
            first_tok = int(answer[content_start])
            last_prefix_h = hidden[-1][0, boxed_start - 1].float()
            ans_h = hidden[-1][0, -1].float()
            cos = float(torch.nn.functional.cosine_similarity(last_prefix_h, ans_h, dim=0).item())
            step_cos = (
                float(torch.nn.functional.cosine_similarity(last_prefix_h, prev_h, dim=0).item())
                if prev_qi == qi and prev_h is not None
                else float("nan")
            )

            layer_logp: dict[int, float] = {}
            layer_mean: dict[int, float] = {}
            top1_match: dict[int, int] = {}
            jsds: dict[int, float] = {}
            lens: dict[str, float] = {}
            layer_ent: dict[str, float] = {}
            layer_stop: dict[str, float] = {}
            pre_logits = []
            last_log = None
            best = float("-inf")
            best_layer = float("nan")
            half_id = n_layers // 2
            for layer in layer_ids:
                if layer + 1 >= len(hidden):
                    continue
                full = hidden[layer + 1][0, boxed_start - 1 :].to(head_device)
                logp_all = torch.log_softmax(model.lm_head(model.model.norm(full)).float(), dim=-1)
                logp = logp_all[: len(answer)]
                nxt_lp = logp_all[-1]
                first_logp = logp[content_start]
                pre_logits.append((layer, first_logp))
                layer_logp[layer] = float(first_logp[first_tok].item())
                content_lp = logp[content_start:content_end]
                content_tgt = target[content_start:content_end].to(content_lp.device)
                layer_mean[layer] = float(content_lp.gather(-1, content_tgt.unsqueeze(-1)).squeeze(-1).mean().item())
                top1_match[layer] = int(int(first_logp.argmax().item()) == first_tok)
                lens_score = float(logp.gather(-1, target.to(logp.device).unsqueeze(-1)).squeeze(-1).mean().item())
                lens[f"lens_l{layer}"] = lens_score
                layer_ent[f"lens_ent_l{layer}"] = float((-(logp.exp() * logp).sum(-1)).mean().item())
                stop_l = float(nxt_lp[stop_id].item()) if 0 <= stop_id < nxt_lp.numel() else float("nan")
                wait_l = float(nxt_lp[wait_id].item()) if 0 <= wait_id < nxt_lp.numel() else float("nan")
                alt_l = float(nxt_lp[alt_id].item()) if 0 <= alt_id < nxt_lp.numel() else float("nan")
                layer_stop[f"stop_logp_l{layer}"] = stop_l
                layer_stop[f"wait_logp_l{layer}"] = wait_l
                layer_stop[f"alt_logp_l{layer}"] = alt_l
                layer_stop[f"stop_margin_l{layer}"] = stop_l - wait_l
                if lens_score > best:
                    best = lens_score
                    best_layer = float(layer)
                if layer == layer_ids[-1]:
                    last_log = first_logp
            if last_log is not None:
                for layer, logp in pre_logits:
                    jsds[layer] = inn.js_divergence(last_log, logp)
            emerge = next((layer for layer in layer_ids if top1_match.get(layer)), float("nan"))
            if jsds:
                pre_layer = max(jsds, key=jsds.get)
                dola_score = layer_logp.get(layer_ids[-1], float("nan")) - layer_logp.get(pre_layer, float("nan"))
                dola_jsd = jsds[pre_layer]
            else:
                pre_layer = float("nan")
                dola_score = float("nan")
                dola_jsd = float("nan")

            nxt = out.logits[0, -1].float()
            nxt_logp = torch.log_softmax(nxt, dim=-1)
            stop_logp = float(nxt_logp[stop_id].item()) if 0 <= stop_id < nxt_logp.numel() else float("nan")
            wait_logp = float(nxt_logp[wait_id].item()) if 0 <= wait_id < nxt_logp.numel() else float("nan")
            alt_logp = float(nxt_logp[alt_id].item()) if 0 <= alt_id < nxt_logp.numel() else float("nan")
            cont_ids = [idx for idx in (wait_id, alt_id) if 0 <= idx < nxt_logp.numel()]
            cont = torch.logsumexp(nxt_logp[cont_ids], dim=0) if cont_ids else nxt_logp.new_tensor(float("nan"))

            last_h = hidden[-1][0, -1].float().cpu()
            mid_idx = n_layers // 2
            mid_h = hidden[mid_idx][0, -1].float().cpu() if mid_idx < len(hidden) else last_h
            if qi != prev_qi:
                last_hiddens = []
                mid_hiddens = []
            last_hiddens.append(last_h)
            mid_hiddens.append(mid_h)
            last_hiddens = last_hiddens[-inn.EIGEN_K :]
            mid_hiddens = mid_hiddens[-inn.EIGEN_K :]

            if not args.no_kv:
                past = crop_past(out.past_key_values, len(prefix))
            else:
                past = None
            delta = len(prefix) - cache_keep
            try:
                hin_new = hidden[-2][0].detach()
                if qi != prev_qi or prefix_hin is None or not reuse:
                    prefix_hin = hin_new[:delta].cpu() if delta > 0 else hin_new[:0].cpu()
                elif prefix_hin.shape[0] == len(prefix) and delta == 1:
                    prefix_hin = torch.cat([prefix_hin[:-1], hin_new[:1].cpu()], dim=0)
                elif delta > 0:
                    prefix_hin = torch.cat([prefix_hin, hin_new[:delta].cpu()], dim=0)
                full_hin = torch.cat([prefix_hin, hin_new[delta:].cpu()], dim=0)
                revise = []
                for pattern in revise_pats:
                    revise.extend(inn.find_spans(ids, pattern))
                layer_dev = inn.states_device(model.model.layers[-1])
                attn_end = inn.last_layer_weights(model, full_hin.to(layer_dev), end - 1)
                attn_pre = inn.last_layer_weights(model, full_hin.to(layer_dev), start - 1)
                lb = inn.lookback_bundle(attn_end, prompt_len, start, end, sorted(set(revise)))
                pre = inn.lookback_bundle(attn_pre, prompt_len, start, start, sorted(set(revise)))
                pre_lb = {
                    "pre_lb_q": pre["lb_q"],
                    "pre_lb_recency5": pre["lb_recency5"],
                    "pre_lookback_ratio": pre["lookback_ratio"],
                }
            except Exception:
                lb = {key: float("nan") for key in ("lb_q", "lb_cot", "lb_ans", "lb_rev", "lb_recency5", "lookback_ratio")}
                pre_lb = {f"pre_{key}": float("nan") for key in ("lb_q", "lb_recency5", "lookback_ratio")}
                if qi != prev_qi:
                    prefix_hin = None

            slim = {key: value for key, value in row.items() if key not in {"reasoning_prefix", "question"}}
            slim.update(
                {
                    "status": "ok",
                    "once": True,
                    "n_layers": n_layers,
                    "layer_ids": list(layer_ids),
                    "last_mean_logp": last_mean,
                    "last_min_logp": last_min,
                    "ans_entropy": ent,
                    "neg_ans_entropy": -ent,
                    "hidden_norm": float(last_prefix_h.norm().item()),
                    "prefix_ans_cos": cos,
                    "hidden_nn_cos": step_cos,
                    "lens_best": best if best > float("-inf") else float("nan"),
                    "lens_best_layer": best_layer,
                    "lens_rise": last_mean - lens.get(f"lens_l{half_id}", float("nan")),
                    **lens,
                    **layer_ent,
                    **layer_stop,
                    **{f"dola_logp_l{layer}": layer_logp.get(layer, float("nan")) for layer in layer_ids},
                    **{f"dola_mean_l{layer}": layer_mean.get(layer, float("nan")) for layer in layer_ids},
                    **{f"dola_jsd_l{layer}": jsds.get(layer, float("nan")) for layer in layer_ids},
                    **{f"dola_top1_l{layer}": top1_match.get(layer, 0) for layer in layer_ids},
                    "dola_mean_rise": layer_mean.get(layer_ids[-1], float("nan")) - layer_mean.get(layer_ids[0], float("nan"))
                    if layer_ids
                    else float("nan"),
                    "dola_jsd_max": dola_jsd,
                    "dola_pre_layer": float(pre_layer) if pre_layer == pre_layer else float("nan"),
                    "dola_score": dola_score,
                    "emerge_layer": float(emerge) if emerge == emerge else float("nan"),
                    "neg_emerge": (-float(emerge)) if emerge == emerge else float("nan"),
                    "layer_agree": float(sum(top1_match.get(layer, 0) for layer in layer_ids) / max(len(layer_ids), 1)),
                    "stop_logp": stop_logp,
                    "wait_logp": wait_logp,
                    "alt_logp": alt_logp,
                    "stop_margin": stop_logp - wait_logp,
                    "stop_margin_alt": stop_logp - alt_logp,
                    "stop_vs_cont": stop_logp - float(cont.item()),
                    "eigen_k2": inn.eigenscore(last_hiddens[-2:]),
                    "eigen_k4": inn.eigenscore(last_hiddens),
                    "neg_eigen_k2": -inn.eigenscore(last_hiddens[-2:]),
                    "neg_eigen_k4": -inn.eigenscore(last_hiddens),
                    "mid_neg_eigen_k4": -inn.eigenscore(mid_hiddens),
                    **lb,
                    **pre_lb,
                }
            )
            line = json.dumps(slim) + "\n"
            handle.write(line)
            key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
            if key not in done_internal:
                handle_in.write(line)
                done_internal.add(key)
            handle.flush()
            handle_in.flush()
            prev_qi = qi
            prev_h = last_prefix_h.detach()
            prev_prefix = prefix
            if index % 20 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
