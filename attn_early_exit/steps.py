"""Split think-span tokens into step spans."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


# Qwen3 </think>
THINK_END_ID = 151668

# repos/PUMA on sys.path (workspace: visual-latent-tts/repos/{attn-early-exit,PUMA})
_PUMA_ROOT = Path(__file__).resolve().parents[2] / "PUMA"


@dataclass(frozen=True)
class StepSpan:
    """Half-open [start, end) indices into full_ids."""

    start: int
    end: int
    text: str

    @property
    def n(self) -> int:
        return self.end - self.start


def think_end_index(
    gen_ids: list[int], think_end_id: int | None = THINK_END_ID
) -> int | None:
    if think_end_id is None:
        return None
    for i, t in enumerate(gen_ids):
        if t == think_end_id:
            return i
    return None


def _vocab_size(tokenizer) -> int:
    # Prefer full tokenizer length (includes added specials); fall back to vocab_size.
    n = len(tokenizer)
    vs = getattr(tokenizer, "vocab_size", None)
    if isinstance(vs, int) and vs > 0:
        return max(n, vs)
    return n


def resolve_think_end_id(tokenizer) -> int | None:
    """Return single-token </think> id if it lies in this model's vocab; else None.

    Do not fall back to the Qwen hardcode (151668) when the tokenizer encodes
    </think> as multiple tokens (e.g. Llama/Nemotron) — that id is OOB there.
    """
    vocab_n = _vocab_size(tokenizer)

    def _ok(tid: int) -> bool:
        return 0 <= int(tid) < vocab_n

    ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(ids) == 1 and _ok(ids[0]):
        return int(ids[0])
    tid = tokenizer.convert_tokens_to_ids("</think>")
    if isinstance(tid, int) and _ok(tid):
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is None or tid != unk:
            return int(tid)
    if _ok(THINK_END_ID):
        return int(THINK_END_ID)
    return None


def _decode_piece(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def split_steps_by_delim(
    tokenizer,
    full_ids: list[int],
    prompt_len: int,
    *,
    delim: str = "\n\n",
    only_think: bool = True,
    think_end_id: int | None = None,
) -> list[StepSpan]:
    """Split generation into steps by `delim` (default paragraph breaks).

    If only_think, truncate at </think> (exclusive of that token).
    Falls back to single-newline split when fewer than 2 paragraphs.
    """
    if think_end_id is None:
        think_end_id = resolve_think_end_id(tokenizer)
    gen_ids = full_ids[prompt_len:]
    te = think_end_index(gen_ids, think_end_id)
    if only_think and te is not None:
        gen_ids = gen_ids[:te]
    if not gen_ids:
        return []

    spans = _split_ids(tokenizer, gen_ids, prompt_len, delim)
    if len(spans) < 2 and delim == "\n\n":
        spans = _split_ids(tokenizer, gen_ids, prompt_len, "\n")
    return spans


def _split_ids(
    tokenizer, gen_ids: list[int], prompt_len: int, delim: str
) -> list[StepSpan]:
    spans: list[StepSpan] = []
    start_local = 0
    buf: list[int] = []
    for j, tid in enumerate(gen_ids):
        buf.append(tid)
        piece = _decode_piece(tokenizer, buf)
        # Boundary when the accumulated decode ends with delim, or last token.
        at_end = j == len(gen_ids) - 1
        if piece.endswith(delim) or at_end:
            # Drop trailing delim-only chunks
            abs_start = prompt_len + start_local
            abs_end = prompt_len + j + 1
            text = _decode_piece(tokenizer, gen_ids[start_local : j + 1]).strip()
            if text:
                spans.append(StepSpan(start=abs_start, end=abs_end, text=text))
            start_local = j + 1
            buf = []
    return spans


def step_token_ranges(steps: list[StepSpan]) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in steps]


def split_steps_puma(
    tokenizer,
    full_ids: list[int],
    prompt_len: int,
    *,
    only_think: bool = True,
    think_end_id: int | None = None,
    min_step_chars: int = 200,
    max_step_chars: int = 1000,
) -> list[StepSpan]:
    """Split via PUMA `separate_steps`, mapped back to token spans.

    Mainline step cut for attn-early-exit (preferred over raw ``\\n\\n``).
    Token ends are allocated by cumulative char length of each step text
    (fast; good enough for probes / G mapping). Exact char↔token decode
    alignment is unnecessary for step-level features.
    """
    if think_end_id is None:
        think_end_id = resolve_think_end_id(tokenizer)
    gen_ids = full_ids[prompt_len:]
    te = think_end_index(gen_ids, think_end_id)
    if only_think and te is not None:
        gen_ids = gen_ids[:te]
    if not gen_ids:
        return []

    if str(_PUMA_ROOT) not in sys.path:
        sys.path.insert(0, str(_PUMA_ROOT))
    from puma.separate_steps import separate_steps  # type: ignore

    think_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    step_texts = separate_steps(
        think_text, min_step_chars=min_step_chars, max_step_chars=max_step_chars
    )
    if not step_texts:
        return []

    weights = [max(len(st), 1) for st in step_texts]
    total_w = float(sum(weights))
    n_tok = len(gen_ids)
    spans: list[StepSpan] = []
    tok_cursor = 0
    cum_w = 0.0
    for i, st in enumerate(step_texts):
        cum_w += weights[i]
        if i == len(step_texts) - 1:
            tok_end = n_tok
        else:
            tok_end = int(round(cum_w / total_w * n_tok))
            tok_end = max(tok_cursor + 1, min(n_tok, tok_end))
        spans.append(
            StepSpan(
                start=prompt_len + tok_cursor,
                end=prompt_len + tok_end,
                text=st,
            )
        )
        tok_cursor = tok_end
    return spans


def map_nn_index_to_puma(
    nn_steps: list[StepSpan],
    nn_index: int,
    puma_steps: list[StepSpan],
) -> int | None:
    """Map a ``\\n\\n``-space step index (e.g. golden G) into PUMA step index."""
    if nn_index is None or nn_index < 0 or nn_index >= len(nn_steps) or not puma_steps:
        return None
    target_end = nn_steps[nn_index].end
    for i, s in enumerate(puma_steps):
        if s.end >= target_end:
            return i
    return len(puma_steps) - 1


def split_steps_mainline(
    tokenizer,
    full_ids: list[int],
    prompt_len: int,
    *,
    only_think: bool = True,
    think_end_id: int | None = None,
) -> list[StepSpan]:
    """Default mainline cut: PUMA separate_steps (+ token spans)."""
    return split_steps_puma(
        tokenizer,
        full_ids,
        prompt_len,
        only_think=only_think,
        think_end_id=think_end_id,
    )
