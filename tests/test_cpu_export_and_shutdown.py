from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "PUMA" / "puma"))

from run_matrix_queue import export_cell_from_cmd  # noqa: E402
from vllm_shutdown import shutdown_llm  # noqa: E402


class ExportCellFromCmdTest(unittest.TestCase):
    def test_parses_export_args(self) -> None:
        cmd = [
            "python",
            "/x/export_leftover_suppress_jobs.py",
            "--model-tag",
            "qwen3_8b",
            "--seed",
            "123",
            "--datasets",
            "olympiadbench",
        ]
        self.assertEqual(
            export_cell_from_cmd(cmd), ("qwen3_8b", "olympiadbench", 123)
        )

    def test_ignores_other_commands(self) -> None:
        self.assertIsNone(
            export_cell_from_cmd(["bash", "run_matrix_prereq_cell.sh"])
        )


class ShutdownLlmTest(unittest.TestCase):
    def test_missing_engine_is_noop(self) -> None:
        shutdown_llm(None)
        shutdown_llm(SimpleNamespace())

    def test_shutdown_is_idempotent(self) -> None:
        calls = []

        def _shut(timeout=10):
            calls.append(timeout)

        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                engine_core=SimpleNamespace(shutdown=_shut)
            )
        )
        shutdown_llm(llm)
        shutdown_llm(llm)
        self.assertEqual(calls, [10])


if __name__ == "__main__":
    unittest.main()
