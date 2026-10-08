"""Guard FlashInfer MoE autotune under tensor parallel.

vLLM 0.29 writes a shared autotune cache from rank 0. The next TP engine
in the same process, or a later cell that reuses the same cache, can
cache-hit on rank 0 while ranks 1-3 start profiling. That desync hangs
Qwen3-30B-A3B: GPU0 100% NCCL spin, GPU1-3 0% util.

1. Unique cache dir per LLM() so every engine's ranks miss together.
2. Honor VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 to skip the tuner entirely.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable


def _truthy_off(raw: str) -> bool:
    return raw.lower() in ("0", "false", "no")


def isolate_flashinfer_cache() -> str | None:
    base = os.environ.get("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR")
    if not base:
        return None
    unique = Path(base) / f"engine_{os.getpid()}_{time.time_ns()}"
    unique.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = str(unique)
    return str(unique)


def apply_flashinfer_guard(
    llm_kwargs: dict[str, Any],
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    cache = isolate_flashinfer_cache()
    if cache and log is not None:
        log(f"FlashInfer autotune cache isolated: {cache}")
    if _truthy_off(os.environ.get("VLLM_ENABLE_FLASHINFER_AUTOTUNE", "1")):
        llm_kwargs["enable_flashinfer_autotune"] = False
        if log is not None:
            log("enable_flashinfer_autotune=False (env)")
    return llm_kwargs
