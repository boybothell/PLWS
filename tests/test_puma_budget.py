from __future__ import annotations

import unittest

from plws.puma_budget import audit_rows


class PumaBudgetAuditTests(unittest.TestCase):
    def test_new_rows_use_exact_generated_prefix_count(self) -> None:
        report = audit_rows(
            [
                {
                    "question_idx": 1,
                    "reasoning_prefix": "prefix",
                    "prefix_generated_tokens": 32000,
                    "count_reasoning_tokens": 32020,
                    "count_generated_tokens": 768,
                }
            ]
        )
        self.assertEqual(report["max_prefix_plus_generation"], 32768)
        self.assertEqual(report["legacy_rows"], 0)

    def test_legacy_rows_use_conservative_reasoning_count(self) -> None:
        report = audit_rows(
            [
                {
                    "question_idx": 1,
                    "reasoning_prefix": "prefix",
                    "count_reasoning_tokens": 100,
                    "count_generated_tokens": 50,
                }
            ]
        )
        self.assertEqual(report["legacy_rows"], 1)
        self.assertEqual(report["max_prefix_plus_generation"], 150)

    def test_unchanged_fullcot_rows_are_skipped(self) -> None:
        report = audit_rows(
            [
                {
                    "question_idx": 1,
                    "reasoning_prefix": "",
                    "count_reasoning_tokens": 32000,
                    "count_generated_tokens": 32000,
                }
            ]
        )
        self.assertEqual(report["regenerated_rows"], 0)

    def test_violation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "q7=32769"):
            audit_rows(
                [
                    {
                        "question_idx": 7,
                        "reasoning_prefix": "prefix",
                        "prefix_generated_tokens": 32760,
                        "count_generated_tokens": 9,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
