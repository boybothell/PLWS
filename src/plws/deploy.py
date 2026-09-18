"""Hardware-only deployment profiles shared by local and rental runners."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROFILE = "local_48g"
_PROFILE_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True, slots=True)
class DeploymentProfile:
    """Validated deployment knobs that must not change method semantics."""

    name: str
    allowed_models: tuple[str, ...]
    tensor_parallel: dict[str, int]
    gpu_memory_utilization: float
    max_num_seqs: int
    serialize_cold_load: bool

    def require_model(self, model_tag: str) -> None:
        if model_tag not in self.allowed_models:
            raise ValueError(
                f"model {model_tag!r} is not allowed by deployment profile "
                f"{self.name!r}; allowed={','.join(self.allowed_models)}"
            )

    def tp_for(self, model_tag: str) -> int:
        self.require_model(model_tag)
        try:
            return self.tensor_parallel[model_tag]
        except KeyError as exc:
            raise ValueError(
                f"deployment profile {self.name!r} has no TP for {model_tag!r}"
            ) from exc


def _repo_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    configured = os.environ.get("PLWS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def profile_name(name: str | None = None) -> str:
    value = name or os.environ.get("PLWS_DEPLOY_PROFILE") or DEFAULT_PROFILE
    if not _PROFILE_RE.fullmatch(value):
        raise ValueError(f"invalid deployment profile name: {value!r}")
    return value


def load_deployment_profile(
    name: str | None = None,
    *,
    root: str | Path | None = None,
) -> DeploymentProfile:
    """Load and validate ``configs/deploy/<name>.toml``."""

    selected = profile_name(name)
    path = _repo_root(root) / "configs" / "deploy" / f"{selected}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"missing deployment profile: {path}")
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    if raw.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    if raw.get("name") != selected:
        raise ValueError(f"{path}: name must be {selected!r}")

    allowed = tuple(str(value) for value in raw.get("allowed_models", ()))
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError(f"{path}: allowed_models must be non-empty and unique")

    raw_tp = raw.get("tensor_parallel")
    if not isinstance(raw_tp, dict):
        raise ValueError(f"{path}: [tensor_parallel] table is required")
    tensor_parallel = {str(key): int(value) for key, value in raw_tp.items()}
    if set(tensor_parallel) != set(allowed):
        raise ValueError(
            f"{path}: tensor_parallel keys must exactly match allowed_models"
        )
    if any(value < 1 for value in tensor_parallel.values()):
        raise ValueError(f"{path}: all tensor_parallel values must be positive")

    gpu_memory_utilization = float(raw.get("gpu_memory_utilization", 0.90))
    if not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError(f"{path}: gpu_memory_utilization must be in (0, 1)")
    max_num_seqs = int(raw.get("max_num_seqs", 64))
    if max_num_seqs < 1:
        raise ValueError(f"{path}: max_num_seqs must be positive")

    return DeploymentProfile(
        name=selected,
        allowed_models=allowed,
        tensor_parallel=tensor_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        serialize_cold_load=bool(raw.get("serialize_cold_load", True)),
    )


def configured_tensor_parallel(
    model_tag: str,
    *,
    name: str | None = None,
    root: str | Path | None = None,
) -> int:
    """Resolve TP, allowing an explicit launch override."""

    override = os.environ.get("PLWS_LARGE_TP") or os.environ.get("PLWS_TP")
    if override:
        value = int(override)
        if value < 1:
            raise ValueError(f"tensor parallel override must be positive, got {value}")
        load_deployment_profile(name, root=root).require_model(model_tag)
        return value
    return load_deployment_profile(name, root=root).tp_for(model_tag)


def deployment_manifest(
    model_tag: str,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int | None = None,
    name: str | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    profile = load_deployment_profile(name, root=root)
    profile.require_model(model_tag)
    payload: dict[str, Any] = {
        "profile": profile.name,
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_memory_utilization),
    }
    if max_num_seqs is not None:
        payload["max_num_seqs"] = int(max_num_seqs)
    return payload
