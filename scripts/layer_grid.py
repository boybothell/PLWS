"""Which transformer layers to score after a hidden-state forward.

The forward already returns every layer. Extra lm_head on a short boxed
span is cheap. Keep all layers so later analysis can pick a lag pair or
another layer score; last-minus-half is only one candidate.
"""
from __future__ import annotations


def layer_grid(n_layers: int) -> tuple[int, ...]:
    if n_layers <= 0:
        return ()
    return tuple(range(n_layers))


def parse_layers(text: str, n_layers: int) -> tuple[int, ...]:
    if text.strip():
        return tuple(int(x) for x in text.split(",") if x.strip())
    return layer_grid(n_layers)
