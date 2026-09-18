"""Guards for the three ways a grading failure used to look like a wrong answer."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws import grading  # noqa: E402

# Equal pairs that only the sympy/LaTeX backend can settle.
LATEX_EQUAL = (
    (r"\dfrac{19}{2}", "9.5"),
    ("0.09", r"\frac{9}{100}"),
    (r"-\dfrac{n-1}{2}", r"$-(n-1) / 2$"),
    (r"256(4 - \pi)", r"$1024-256 \pi$"),
)


class HasGoldTest(unittest.TestCase):
    def test_blank_gold_is_not_usable(self) -> None:
        for value in (None, "", "   ", [], [None], ["", "  "]):
            self.assertFalse(grading.has_gold(value), value)

    def test_present_gold_is_usable(self) -> None:
        for value in ("5", r"\frac{1}{2}", 0, ["", "A"]):
            self.assertTrue(grading.has_gold(value), value)


class GradeTest(unittest.TestCase):
    def test_missing_gold_raises_instead_of_scoring_wrong(self) -> None:
        # The original bug: gold was None, check_is_correct raised, and the
        # caller recorded new_gold_ok=False for every such question.
        for gold in (None, "", "  "):
            with self.assertRaises(grading.MissingGold):
                grading.grade("42", gold)

    def test_correct_and_incorrect_answers(self) -> None:
        self.assertEqual(grading.grade("42", "42"), (True, ""))
        ok, error = grading.grade("41", "42")
        self.assertFalse(ok)
        self.assertEqual(error, "")

    def test_latex_equivalence_is_settled(self) -> None:
        # Without antlr4/latex2sympy2 every one of these grades False, which is
        # how a whole machine's Acc silently drifted low.
        for pred, gold in LATEX_EQUAL:
            ok, error = grading.grade(pred, gold)
            self.assertEqual(error, "")
            self.assertTrue(ok, f"{pred!r} vs {gold!r}")


class RequireGraderTest(unittest.TestCase):
    def test_selftest_passes_with_a_working_backend(self) -> None:
        grading.require_grader()

    def test_selftest_fails_when_latex_backend_is_gone(self) -> None:
        blocked = {"antlr4", "latex2sympy2"}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError(f"blocked {name}")
                return None

        def fake_check(pred, gold, timeout=True):
            # Stand in for the degraded grader: only exact strings match.
            return str(pred).strip() == str(gold).strip()

        original_check = grading._CHECK
        original_verified = grading._VERIFIED
        sys.meta_path.insert(0, Blocker())
        try:
            grading._CHECK = fake_check
            grading._VERIFIED = False
            with self.assertRaises(grading.GraderUnavailable):
                grading.require_grader()
        finally:
            sys.meta_path.pop(0)
            grading._CHECK = original_check
            grading._VERIFIED = original_verified


class GradeManyTest(unittest.TestCase):
    def test_parallel_grading_settles_latex_pairs(self) -> None:
        # multiprocessing.Pool workers are daemonic and the grader's timeout
        # helper spawns a child, so a Pool turned these into AssertionError and
        # then into misses. grade_many must not.
        results = grading.grade_many(list(LATEX_EQUAL), workers=4, chunksize=1)
        self.assertEqual(len(results), len(LATEX_EQUAL))
        for (pred, gold), (ok, error) in zip(LATEX_EQUAL, results):
            self.assertEqual(error, "")
            self.assertTrue(ok, f"{pred!r} vs {gold!r}")

    def test_serial_path_matches(self) -> None:
        self.assertEqual(
            grading.grade_many([("42", "42"), ("41", "42")], workers=1),
            [(True, ""), (False, "")],
        )

    def test_empty_input(self) -> None:
        self.assertEqual(grading.grade_many([]), [])


class ExportRefusesMissingGoldTest(unittest.TestCase):
    def test_dataset_without_gold_is_dropped_whole(self) -> None:
        # A half-written jobs file is worse than none: the missing questions
        # would score wrong forever.
        import export_leftover_suppress_jobs as export

        missing: dict[str, int] = {}
        jobs = export.build_jobs(
            "definitely_not_a_model",
            export.FIRSTWIN,
            seed=42,
            datasets=("math-500",),
            missing_gold=missing,
        )
        self.assertEqual(jobs, [])


class ExistingLeftoverGradeTest(unittest.TestCase):
    def test_old_row_is_regraded_and_marked_reusable(self) -> None:
        import score_leftover_suppress as score

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shard_0.jsonl"
            out.write_text(
                json.dumps(
                    {
                        "uid": "m:d:s:q0",
                        "status": "ok",
                        "new_answer": "42",
                        "new_gold_ok": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(score, "grade_many", return_value=[(True, "")]):
                changed = score.verify_existing_grades(
                    out, {"m:d:s:q0": "42"}
                )
            fixed = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(changed, 1)
            self.assertTrue(fixed["new_gold_ok"])
            self.assertEqual(fixed["gt"], "42")
            self.assertEqual(fixed["gold_error"], "")

    def test_true_to_false_review_blocks_resume(self) -> None:
        import score_leftover_suppress as score

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shard_0.jsonl"
            out.write_text(
                json.dumps(
                    {
                        "uid": "m:d:s:q0",
                        "status": "ok",
                        "new_answer": "41",
                        "new_gold_ok": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(score, "grade_many", return_value=[(False, "")]):
                with self.assertRaisesRegex(RuntimeError, "cannot be trusted"):
                    score.verify_existing_grades(out, {"m:d:s:q0": "42"})


if __name__ == "__main__":
    unittest.main()
