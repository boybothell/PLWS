"""Cross-step attention Ratio + per-step confidence."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from attn_early_exit.steps import StepSpan


def late_layer_indices(n_layers: int, n_late: int = 3) -> list[int]:
    """Last n_late layers (0-based). E.g. 32-layer → [29,30,31] (= paper's 30–32)."""
    n_late = max(1, min(n_late, n_layers))
    return list(range(n_layers - n_late, n_layers))


def sink_renorm(A: np.ndarray, sink_k: int) -> np.ndarray:
    """Zero first sink_k key columns, row-renormalize. A: [T,T]."""
    if sink_k <= 0:
        return A
    out = A.copy()
    k = min(sink_k, out.shape[1])
    out[:, :k] = 0.0
    row = out.sum(axis=-1, keepdims=True)
    out = out / np.clip(row, 1e-8, None)
    return out


def mean_attn_matrix(
    attentions: tuple,
    layer_ids: list[int],
) -> np.ndarray:
    """Mean over selected layers and heads → [T, T] float32 numpy."""
    acc = None
    for li in layer_ids:
        a = attentions[li]  # [1, H, T, T] or tuple
        m = a[0].float().mean(dim=0)  # [T, T]
        acc = m if acc is None else acc + m
    assert acc is not None
    return (acc / max(len(layer_ids), 1)).detach().cpu().numpy().astype(np.float32)


def ratio_matrices(
    A: np.ndarray,
    steps: list[StepSpan],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mass [S,S], dens [S,S]) with Ratio(t→i) at [t, i].

    mass(t→i) = (1/N_t) * sum_q sum_k A[q,k]
    dens(t→i) = mass / N_i
    Causal: only i <= t filled; others nan.
    """
    S = len(steps)
    mass = np.full((S, S), np.nan, dtype=np.float32)
    dens = np.full((S, S), np.nan, dtype=np.float32)
    if S == 0:
        return mass, dens
    for t, st in enumerate(steps):
        qs = slice(st.start, st.end)
        nt = max(st.n, 1)
        for i, si in enumerate(steps):
            if i > t:
                continue
            block = A[qs, si.start : si.end]
            m = float(block.sum() / nt)
            mass[t, i] = m
            dens[t, i] = m / max(si.n, 1)
    return mass, dens


def max_offdiag_incoming(mass: np.ndarray) -> np.ndarray:
    """For each t, max_i<t mass[t,i] (nan if t==0)."""
    S = mass.shape[0]
    out = np.full(S, np.nan, dtype=np.float32)
    for t in range(1, S):
        out[t] = float(np.nanmax(mass[t, :t]))
    return out


def step_avg_confidence(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    steps: list[StepSpan],
) -> np.ndarray:
    """Avg next-token prob of actual tokens in each step.

    logits: [1, T, V], input_ids: [1, T]
    conf for position j (token j) uses logits[0, j-1].
    """
    # p[j] = softmax(logits[j])[ids[j+1]] → align to token index j+1
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    ids = input_ids[0, 1:]
    tok_logp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # [T-1]
    tok_p = tok_logp.exp().detach().cpu().numpy()

    out = np.full(len(steps), np.nan, dtype=np.float32)
    for t, st in enumerate(steps):
        # tokens in [start, end); conf defined for index >=1
        lo = max(st.start, 1)
        hi = st.end
        if hi <= lo:
            continue
        # tok_p index = token_index - 1
        vals = tok_p[lo - 1 : hi - 1]
        if vals.size:
            out[t] = float(vals.mean())
    return out


def summarize_dual_signal(
    mass: np.ndarray,
    conf: np.ndarray,
    *,
    tau_ratio: float | None = None,
    tau_conf: float | None = None,
) -> dict:
    max_r = max_offdiag_incoming(mass)
    trig = None
    if tau_ratio is not None and tau_conf is not None:
        for t in range(len(max_r)):
            if (
                np.isfinite(max_r[t])
                and np.isfinite(conf[t])
                and max_r[t] > tau_ratio
                and conf[t] > tau_conf
            ):
                trig = int(t)
                break
    def _clean(xs: np.ndarray) -> list:
        out = []
        for v in xs.tolist():
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                out.append(None)
            else:
                out.append(float(v))
        return out

    return {
        "max_ratio_per_step": _clean(max_r),
        "avg_conf_per_step": _clean(conf),
        "first_dual_trigger_step": trig,
        "tau_ratio": tau_ratio,
        "tau_conf": tau_conf,
    }
