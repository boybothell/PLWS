#!/usr/bin/env python3
"""Move legacy top-level results into the canonical categorized layout.

The migration is metadata-only on one filesystem: artifacts are moved, not
copied or rewritten.  It is idempotent and refuses to merge conflicting
directories.  Dry-run is the default; pass ``--apply`` to make changes.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DENSE = (
    "dense_G_nemotron_8b",
    "dense_G_qwen3_30b_a3b",
    "dense_G_qwen3_4b",
    "dense_G_qwen3_8b",
    "dense_G_r1_14b",
    "dense_G_r1_32b",
    "dense_G_r1_7b",
)

EXPERIMENTS = {
    "gating": (
        "default_dense_gate",
        "first_hm_gate",
        "fromstart_core",
        "mix_then_high",
        "rescue_R_gate",
        "window4_7b",
        "window4_mid",
    ),
    "lexicon_safety": (
        "fs_process_support_any_safe",
        "full_suppress_safe",
        "fw_lex_safe",
    ),
    "probes": (
        "asag_entropy",
        "asag_r1_7b",
        "commit_probe",
        "hc_ans_logp",
        "hc_verbal",
        "hc_verbal_ind",
        "hc_verbal_sum1",
        "samewin_challenge",
        "samewin_firsttok",
        "samewin_mask",
        "samewin_reprobe",
        "samewin_resample",
    ),
    "alternative_methods": (
        "leftover_attn",
        "leftover_dtsr",
        "leftover_eigen_dola",
        "leftover_final_h",
        "leftover_regen",
        "leftover_token_pivot",
        "leftover_waithelp",
        "s1_2x_wait",
        "s1_on_traj",
    ),
}

LEGACY_PLWS = (
    "leftover_jump",
    "leftover_suppress_kablate",
    "leftover_suppress_toend",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class Migration:
    def __init__(self, results: Path, apply: bool) -> None:
        self.results = results.resolve()
        self.apply = apply
        self.operations: list[dict[str, str]] = []

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.results))

    def mkdir(self, path: Path) -> None:
        self.operations.append({"action": "mkdir", "path": self.relative(path)})
        if self.apply:
            path.mkdir(parents=True, exist_ok=True)

    def move(self, source: Path, destination: Path) -> None:
        if not source.exists() and not source.is_symlink():
            if destination.exists() or destination.is_symlink():
                self.operations.append(
                    {
                        "action": "already_moved",
                        "source": self.relative(source),
                        "destination": self.relative(destination),
                    }
                )
                return
            self.operations.append(
                {"action": "missing", "source": self.relative(source)}
            )
            return

        if destination.is_symlink():
            try:
                same = destination.resolve() == source.resolve()
            except FileNotFoundError:
                same = False
            if not same:
                raise RuntimeError(
                    f"destination symlink does not reference source: {destination}"
                )
            self.operations.append(
                {"action": "remove_view", "path": self.relative(destination)}
            )
            if self.apply:
                destination.unlink()
        elif destination.exists():
            raise RuntimeError(
                f"refusing to merge existing destination: {destination}"
            )

        self.operations.append(
            {
                "action": "move",
                "source": self.relative(source),
                "destination": self.relative(destination),
            }
        )
        if self.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

    def symlink(self, link: Path, target: Path) -> None:
        relative_target = os.path.relpath(target, link.parent)
        self.operations.append(
            {
                "action": "symlink",
                "path": self.relative(link),
                "target": relative_target,
            }
        )
        if not self.apply:
            return
        if link.is_symlink():
            if os.readlink(link) == relative_target:
                return
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"refusing to replace non-symlink: {link}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(relative_target)

    def run(self) -> None:
        r = self.results
        for directory in (
            r / "baselines",
            r / "baselines" / "official",
            r / "reports",
            r / "cache",
            r / "archive" / "calibration",
            r / "archive" / "experiments",
            r / "archive" / "legacy_layout" / "plws",
            r / "archive" / "logs",
            r / "registry" / "migrations",
        ):
            self.mkdir(directory)

        self.move(r / "upstream" / "puma_official", r / "baselines" / "puma")
        self.move(r / "upstream" / "deer", r / "baselines" / "deer")

        for name in DENSE:
            self.move(r / name, r / "upstream" / "dense_trials" / name)

        for source in sorted(r.glob("puma_offline_*")):
            self.move(source, r / "baselines" / "puma" / source.name)

        self.move(
            r / "math500_official",
            r / "baselines" / "official" / "math500_official",
        )
        self.symlink(
            r / "baselines" / "puma" / "math500_official__puma_ds7b",
            r / "baselines" / "official" / "math500_official" / "puma_ds7b",
        )
        self.symlink(
            r / "baselines" / "deer" / "math500_official__deer",
            r / "baselines" / "official" / "math500_official" / "deer",
        )
        self.symlink(
            r / "baselines" / "deer" / "math500_official__deer_pro",
            r / "baselines" / "official" / "math500_official" / "deer_pro",
        )

        self.move(
            r / "archive" / "confcal",
            r / "archive" / "calibration" / "keytoken_compact",
        )
        self.move(
            r / "confcal_judge",
            r / "archive" / "calibration" / "confcal_judge",
        )

        for category, names in EXPERIMENTS.items():
            for name in names:
                self.move(
                    r / name,
                    r / "archive" / "experiments" / category / name,
                )

        for name in LEGACY_PLWS:
            self.move(
                r / name,
                r / "archive" / "legacy_layout" / "plws" / name,
            )

        self.move(r / "_logs", r / "archive" / "logs" / "legacy_jobs")
        self.move(
            r / "dense_fill_qwen3.log",
            r / "archive" / "logs" / "dense_fill_qwen3.log",
        )
        self.move(
            r / "migration_report.json",
            r / "registry" / "migrations" / "plws_layout.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    migration = Migration(args.results_root, args.apply)
    migration.run()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": args.apply,
        "results_root": str(args.results_root.resolve()),
        "operations": migration.operations,
    }
    if args.apply:
        atomic_json(
            args.results_root / "registry" / "migrations" / "results_layout.json",
            report,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
