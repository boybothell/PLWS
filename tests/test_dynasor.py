from __future__ import annotations

import unittest

from plws.dynasor import (
    is_certain_answer,
    normalize_gpqa_answer,
    obtain_answer,
    should_early_exit,
    token_chunk_boundaries,
)


class DynasorHelpersTests(unittest.TestCase):
    def test_boundaries_exclude_natural_end(self) -> None:
        self.assertEqual(token_chunk_boundaries(64), ())
        self.assertEqual(token_chunk_boundaries(65), (64,))
        self.assertEqual(token_chunk_boundaries(192), (64, 128))

    def test_obtain_answer_handles_nested_braces(self) -> None:
        self.assertEqual(obtain_answer(r"\frac{1}{2}} trailing"), r"\frac{1}{2}")
        self.assertEqual(obtain_answer("42} trailing"), "42")
        self.assertEqual(obtain_answer("42"), "")

    def test_uncertain_word_gate_is_case_insensitive(self) -> None:
        self.assertTrue(is_certain_answer("42}"))
        self.assertFalse(is_certain_answer("Wait, 42}"))
        self.assertFalse(is_certain_answer("HMM, 42}"))

    def test_gpqa_normalization(self) -> None:
        self.assertEqual(normalize_gpqa_answer("The answer is (C)."), "C")
        self.assertEqual(normalize_gpqa_answer("not available"), "")

    def test_exit_uses_supplied_equivalence(self) -> None:
        equivalent = lambda left, right: float(left) == float(right)
        self.assertTrue(
            should_early_exit(
                ["0.5", "1/2", "0.50"],
                [True, True, True],
                equivalent=lambda left, right: (
                    float(left)
                    if "/" not in left
                    else float(left.split("/")[0]) / float(left.split("/")[1])
                )
                == (
                    float(right)
                    if "/" not in right
                    else float(right.split("/")[0]) / float(right.split("/")[1])
                ),
            )
        )
        self.assertFalse(
            should_early_exit(
                ["1", "1", "1"],
                [True, False, True],
                equivalent=equivalent,
            )
        )


if __name__ == "__main__":
    unittest.main()
