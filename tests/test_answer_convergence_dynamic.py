import unittest

from plws.answer_convergence_dynamic import committed_prefixes, take_reasoning_chunk


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in text.split("|")]
    return [part for part in parts if part]


class DynamicAnswerConvergenceTest(unittest.TestCase):
    def test_trailing_fragment_is_not_probed(self) -> None:
        prefixes = committed_prefixes(
            "First. | Second starts",
            _sentences,
            finished=False,
        )

        self.assertEqual(prefixes, ["First."])

    def test_finished_chain_includes_the_last_fragment(self) -> None:
        prefixes = committed_prefixes(
            "First. | Second starts",
            _sentences,
            finished=True,
        )

        self.assertEqual(prefixes, ["First.", "First. | Second starts"])

    def test_sentence_punctuation_commits_the_last_sentence(self) -> None:
        prefixes = committed_prefixes("Done.", _sentences, finished=False)

        self.assertEqual(prefixes, ["Done."])

    def test_think_close_drops_the_answer_tail(self) -> None:
        reasoning, closed = take_reasoning_chunk("Step one.\n</think>\n\\boxed{A}")

        self.assertTrue(closed)
        self.assertEqual(reasoning, "Step one.\n")

    def test_open_chunk_stays_open(self) -> None:
        reasoning, closed = take_reasoning_chunk("still thinking")

        self.assertFalse(closed)
        self.assertEqual(reasoning, "still thinking")


if __name__ == "__main__":
    unittest.main()
