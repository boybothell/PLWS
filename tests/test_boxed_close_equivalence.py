import random
import sys
import unittest
from pathlib import Path

PUMA = Path(__file__).resolve().parents[2] / "PUMA" / "puma"
sys.path.insert(0, str(PUMA))

from prompt_utils import (  # noqa: E402
    boxed_close_keep_length,
    boxed_close_matches_full,
    first_balanced_brace_content,
)


class BoxedCloseEquivalenceTests(unittest.TestCase):
    def test_nested_formula_keeps_inner_braces(self) -> None:
        texts = ["{", "\\frac", "{", "1", "}", "{", "2", "}", "}", " extra"]
        self.assertEqual(boxed_close_keep_length(texts), 9)
        report = boxed_close_matches_full(texts)
        self.assertTrue(report["stopped_early"])
        self.assertTrue(report["answer_same"])
        self.assertTrue(report["count_same"])
        self.assertEqual(report["full_answer"], "\\frac{1}{2}")
        self.assertEqual(report["full_count"], 7)

    def test_plain_answer_ignores_text_after_the_box(self) -> None:
        texts = ["{", "27", "}", " miles", " from", " A"]
        report = boxed_close_matches_full(texts)
        self.assertEqual(report["keep"], 3)
        self.assertEqual(report["full_answer"], "27")
        self.assertEqual(report["cut_answer"], "27")
        self.assertEqual(report["full_count"], 1)
        self.assertEqual(report["cut_count"], 1)

    def test_empty_interior_does_not_stop_early(self) -> None:
        # "{" and "}" are separate tokens, so the live counter's span is empty
        # and it falls back to every generated token. Cutting here would change
        # that count.
        texts = ["{", "}", " more", " text"]
        self.assertIsNone(boxed_close_keep_length(texts))
        report = boxed_close_matches_full(texts)
        self.assertFalse(report["stopped_early"])
        self.assertTrue(report["answer_same"])
        self.assertTrue(report["count_same"])
        self.assertEqual(report["full_count"], 4)

    def test_brace_inside_one_token_does_not_stop_early(self) -> None:
        texts = ["{27}", " miles", " later"]
        self.assertIsNone(boxed_close_keep_length(texts))
        report = boxed_close_matches_full(texts)
        self.assertFalse(report["stopped_early"])
        self.assertEqual(report["full_answer"], "27")
        self.assertEqual(report["full_count"], 3)

    def test_unclosed_box_keeps_the_whole_generation(self) -> None:
        texts = ["{", "\\frac", "{", "1"]
        self.assertIsNone(boxed_close_keep_length(texts))
        self.assertIsNone(first_balanced_brace_content("".join(texts)))
        report = boxed_close_matches_full(texts)
        self.assertTrue(report["answer_same"])
        self.assertTrue(report["count_same"])

    def test_random_token_lists_match_the_full_score(self) -> None:
        rng = random.Random(0)
        pieces = [
            "{",
            "}",
            "27",
            " ",
            "\\frac",
            "1",
            "2",
            " miles",
            "{27}",
            ")",
            ".",
            "\n",
            "A",
            "\\pi",
        ]
        for _ in range(400):
            texts = [rng.choice(pieces) for _ in range(rng.randint(0, 16))]
            report = boxed_close_matches_full(texts)
            self.assertTrue(report["answer_same"], texts)
            self.assertTrue(report["count_same"], texts)


if __name__ == "__main__":
    unittest.main()
