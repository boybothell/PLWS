"""Where a machine writes its queue ledger (status / events / logs).

Cell artifacts stay on the canonical result paths. Only the ledger is
namespaced. This module does not assign work to machines.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

MACHINE_ENV = "PLWS_MACHINE"
RUNS_DIRNAME = "runs"
MACHINES_DIRNAME = "machines"
_MACHINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def machine_id(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    raw = (env.get(MACHINE_ENV) or "").strip()
    return raw or None


def machines_root(repo_root: Path) -> Path:
    return Path(repo_root).resolve() / "results" / RUNS_DIRNAME / MACHINES_DIRNAME


def _validate_machine_name(name: str) -> str:
    if not _MACHINE_NAME.fullmatch(name):
        raise ValueError(
            f"{MACHINE_ENV}={name!r} must be a single label "
            r"like box2 or a800 (letters, digits, . _ -)"
        )
    return name


def resolve_run_root(
    repo_root: Path,
    *,
    env_name: str,
    default_name: str,
    environ: dict[str, str] | None = None,
) -> Path:
    """Return the ledger directory for one queue.

    Unset PLWS_MACHINE: env override or results/runs/<default_name>.
    Set PLWS_MACHINE: must stay under results/runs/machines/<machine>/.
    """

    env = os.environ if environ is None else environ
    repo = Path(repo_root).resolve()
    explicit = (env.get(env_name) or "").strip()
    machine = machine_id(env)
    if machine is None:
        if explicit:
            return Path(explicit).expanduser().resolve()
        return (repo / "results" / RUNS_DIRNAME / default_name).resolve()

    name = _validate_machine_name(machine)
    allowed = (machines_root(repo) / name).resolve()
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        root = (allowed / default_name).resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(
            f"{env_name}={root} must be under {allowed}. "
            "Worker ledgers go to results/runs/machines/<PLWS_MACHINE>/<queue>/."
        ) from exc
    return root
