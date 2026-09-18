"""PUMA statistics must be verified before the fill queue reuses them."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plws.puma_grading import (  # noqa: E402
    PumaGradeVerificationError,
    puma_grades_verified,
    verify_puma_statistics,
)


def row(*, original: bool = False, compressed: bool = False) -> dict:
    return {
        "question_idx": 0,
        "ground_truth": "42",
        "original_answer": "42",
        "compressed_answer": "42",
        "original_correct": original,
        "compressed_correct": compressed,
    }


class PumaGradeVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "statistics.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, rows: list[dict]) -> None:
        self.path.write_text(json.dumps(rows), encoding="utf-8")

    @patch("plws.puma_grading.require_grader")
    @patch("plws.puma_grading.grade_many")
    def test_fix_promotes_and_stamps_statistics(self, grade_many, _require) -> None:
        self.write([row()])
        grade_many.return_value = [(True, ""), (True, "")]

        result = verify_puma_statistics(self.path, fix=True, workers=1)

        fixed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertTrue(fixed[0]["original_correct"])
        self.assertTrue(fixed[0]["compressed_correct"])
        self.assertEqual(result.promoted, 2)
        self.assertTrue(puma_grades_verified(self.path))

    @patch("plws.puma_grading.require_grader")
    @patch("plws.puma_grading.grade_many")
    def test_dry_run_refuses_stale_flags(self, grade_many, _require) -> None:
        self.write([row()])
        before = self.path.read_bytes()
        grade_many.return_value = [(True, ""), (True, "")]

        with self.assertRaises(PumaGradeVerificationError):
            verify_puma_statistics(self.path, fix=False, workers=1)

        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(puma_grades_verified(self.path))

    @patch("plws.puma_grading.require_grader")
    @patch("plws.puma_grading.grade_many")
    def test_stored_true_regraded_false_blocks(self, grade_many, _require) -> None:
        self.write([row(original=True, compressed=True)])
        grade_many.return_value = [(False, ""), (True, "")]

        with self.assertRaisesRegex(
            PumaGradeVerificationError, "stored True regraded False"
        ):
            verify_puma_statistics(self.path, fix=True, workers=1)

        self.assertFalse(puma_grades_verified(self.path))

    @patch("plws.puma_grading.require_grader")
    def test_missing_gold_blocks(self, _require) -> None:
        item = row()
        item["ground_truth"] = None
        self.write([item])

        with self.assertRaisesRegex(PumaGradeVerificationError, "missing ground_truth"):
            verify_puma_statistics(self.path, fix=True, workers=1)

    @patch("plws.puma_grading.require_grader")
    @patch("plws.puma_grading.grade_many")
    def test_marker_is_invalidated_when_statistics_change(
        self, grade_many, _require
    ) -> None:
        self.write([row(original=True, compressed=True)])
        grade_many.return_value = [(True, ""), (True, "")]
        verify_puma_statistics(self.path, fix=True, workers=1)
        self.assertTrue(puma_grades_verified(self.path))

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload[0]["original_tokens"] = 123
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(puma_grades_verified(self.path))


if __name__ == "__main__":
    unittest.main()
