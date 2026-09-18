from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plws.machine_runs import resolve_run_root


class MachineRunRootTests(unittest.TestCase):
    def test_unset_machine_keeps_historical_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = resolve_run_root(
                root,
                env_name="CONTEST_FILL_RUN_ROOT",
                default_name="contest_fill",
                environ={},
            )
            self.assertEqual(path, (root / "results" / "runs" / "contest_fill").resolve())

    def test_named_machine_defaults_under_machines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = resolve_run_root(
                root,
                env_name="EXTRA_BASELINE_RUN_ROOT",
                default_name="extra_baseline_fill",
                environ={"PLWS_MACHINE": "box2"},
            )
            self.assertEqual(
                path,
                (
                    root
                    / "results"
                    / "runs"
                    / "machines"
                    / "box2"
                    / "extra_baseline_fill"
                ).resolve(),
            )

    def test_named_machine_rejects_ledger_outside_its_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "results" / "runs" / "contest_fill"
            with self.assertRaises(RuntimeError):
                resolve_run_root(
                    root,
                    env_name="CONTEST_FILL_RUN_ROOT",
                    default_name="contest_fill",
                    environ={
                        "PLWS_MACHINE": "box2",
                        "CONTEST_FILL_RUN_ROOT": str(outside),
                    },
                )


if __name__ == "__main__":
    unittest.main()
