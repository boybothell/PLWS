"""Shared paths and launch guards for local or rental vLLM workers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_MODELS_ROOT = Path("/mnt/d/lsj/models")
MODELS_ROOT = Path(os.environ.get("PLWS_MODELS_ROOT", DEFAULT_MODELS_ROOT))
VLLM_VENV = Path(os.environ.get("PLWS_VLLM_VENV", sys.prefix))

MODEL_DIRS = {
    "qwen3_30b_a3b": "Qwen3-30B-A3B-Thinking-2507",
    "qwq_32b": "QwQ-32B",
    "qwen3_32b": "Qwen3-32B",
    "r1_7b": "DeepSeek-R1-Distill-Qwen-7B",
    "nemotron_8b": "Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_14b": "DeepSeek-R1-Distill-Qwen-14B",
    "r1_1p5b": "DeepSeek-R1-Distill-Qwen-1.5B",
    "r1_llama_8b": "DeepSeek-R1-Distill-Llama-8B",
    "r1_32b": "DeepSeek-R1-Distill-Qwen-32B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}


def model_path(model_tag: str) -> Path:
    """Resolve a model under ``PLWS_MODELS_ROOT``."""

    try:
        directory = MODEL_DIRS[model_tag]
    except KeyError as exc:
        raise ValueError(f"unknown model tag: {model_tag}") from exc
    return MODELS_ROOT / directory


def nvidia_ld_library_path(current: str | None = None) -> str:
    """CUDA 13 libs must be on LD_LIBRARY_PATH before the Python process starts."""

    extra = ":".join(
        sorted(
            str(path)
            for path in (VLLM_VENV / "lib").glob("**/nvidia/*/lib")
            if path.is_dir()
        )
    )
    existing = current if current is not None else os.environ.get("LD_LIBRARY_PATH", "")
    if extra and existing:
        return f"{extra}:{existing}"
    return extra or existing


def allow_cold_start(any_loading: bool) -> bool:
    """At most one new engine may start while another is still loading."""

    return not any_loading
