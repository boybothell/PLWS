import unittest

from plws.answer_convergence import (
    ConvergenceState,
    cumulative_sentence_prefixes,
    observe_answer,
)


class AnswerConvergenceTest(unittest.TestCase):
    def test_cumulative_prefixes_preserve_source_spacing(self) -> None:
        text = "First sentence.  Second sentence!\nThird?"
        sentences = ["First sentence.", "Second sentence!", "Third?"]

        self.assertEqual(
            cumulative_sentence_prefixes(text, lambda _: sentences),
            [
                "First sentence.",
                "First sentence.  Second sentence!",
                "First sentence.  Second sentence!\nThird?",
            ],
        )

    def test_convergence_occurs_on_tenth_identical_answer(self) -> None:
        state = ConvergenceState()
        stopped = []
        for _ in range(10):
            state, converged = observe_answer(state, "42", threshold=10)
            stopped.append(converged)

        self.assertEqual(stopped, [False] * 9 + [True])
        self.assertEqual(state.repeat_count, 10)

    def test_changed_answer_restarts_counter(self) -> None:
        state = ConvergenceState()
        for _ in range(9):
            state, _ = observe_answer(state, "A", threshold=10)

        state, converged = observe_answer(state, "B", threshold=10)

        self.assertFalse(converged)
        self.assertEqual(
            state, ConvergenceState(last_answer="B", repeat_count=1)
        )

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold must be positive"):
            observe_answer(ConvergenceState(), "A", threshold=0)


if __name__ == "__main__":
    unittest.main()
