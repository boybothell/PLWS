from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BaselineLayoutTests(unittest.TestCase):
    def test_extra_cell_cmd_only_matches_wrappers(self) -> None:
        from plws.extra_baselines import extra_cell_method_from_cmd

        self.assertEqual(
            extra_cell_method_from_cmd(
                ["bash", "/repo/scripts/run_answer_convergence_cell.sh"]
            ),
            "answer_convergence",
        )
        self.assertEqual(
            extra_cell_method_from_cmd(
                ["bash", "/repo/baselines/dynasor/run_cell.sh"]
            ),
            "dynasor",
        )
        self.assertIsNone(
            extra_cell_method_from_cmd(
                ["python", "/repo/scripts/run_extra_baseline_queue.py"]
            )
        )
    def test_dynasor_has_runner_and_shell_entry(self) -> None:
        directory = ROOT / "baselines" / "dynasor"
        self.assertTrue((directory / "runner.py").is_file())
        self.assertTrue((directory / "run_cell.sh").is_file())

    def test_baseline_acc_uses_unified_grader(self) -> None:
        ac = (ROOT / "scripts" / "run_answer_convergence_cell.py").read_text()
        dynasor = (ROOT / "baselines" / "dynasor" / "runner.py").read_text()
        for text in (ac, dynasor):
            self.assertIn("require_grader", text)
            self.assertIn("must_grade", text)
            self.assertIn("verify_baseline_records", text)
            self.assertNotIn("check_is_correct(", text)
        for name in (
            "score_deer_github_official.py",
            "score_deer_backfill_table.py",
        ):
            text = (ROOT / "scripts" / name).read_text()
            self.assertIn("require_grader", text)
            self.assertIn("grade_many", text)
            self.assertNotIn("multiprocessing import Pool", text)
            self.assertNotIn("check_is_correct(", text)

    def test_cloud_entries_do_not_contain_local_absolute_paths(self) -> None:
        files = [
            *(ROOT / "baselines").glob("**/*.py"),
            *(ROOT / "baselines").glob("**/*.sh"),
            ROOT / "scripts" / "run_answer_convergence_cell.py",
            ROOT / "scripts" / "run_answer_convergence_cell.sh",
            ROOT / "scripts" / "run_deer_official.sh",
        ]
        for path in files:
            if not path.is_file():
                continue
            self.assertNotIn(
                "/mnt/d/lsj",
                path.read_text(encoding="utf-8"),
                path,
            )


if __name__ == "__main__":
    unittest.main()
