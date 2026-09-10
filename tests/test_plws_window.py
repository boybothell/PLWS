from __future__ import annotations

import unittest

from plws.window import (
    EPS,
    K,
    MSS,
    TAU,
    Observation,
    WindowLevel,
    classify_window,
    first_same_answer_window,
)


def observations(
    steps: range,
    answers: list[str],
    confidences: list[float] | None = None,
) -> list[Observation[str]]:
    values = confidences or [0.5] * len(answers)
    return [
        Observation(step=step, answer=answer, confidence=confidence)
        for step, answer, confidence in zip(steps, answers, values, strict=True)
    ]


class FirstSameAnswerWindowTest(unittest.TestCase):
    def test_defaults_are_frozen(self) -> None:
        self.assertEqual(K, 4)
        self.assertEqual(MSS, 10)
        self.assertEqual(TAU, 0.995)
        self.assertEqual(EPS, 0.03)

    def test_k_is_configurable(self) -> None:
        rows = observations(range(3, 6), ["A", "A", "B"])
        window = first_same_answer_window(rows, k=2, mss=1)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual((window.start_step, window.end_step), (3, 4))

    def test_mss_applies_to_window_end(self) -> None:
        before = observations(range(6, 10), ["A"] * 4)
        self.assertIsNone(first_same_answer_window(before))

        at_boundary = observations(range(6, 11), ["A"] * 5)
        window = first_same_answer_window(at_boundary)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual((window.start_step, window.end_step), (7, 10))

    def test_returns_first_window_and_gap_breaks_run(self) -> None:
        rows = [
            Observation(7, "A", 0.5),
            Observation(8, "A", 0.5),
            Observation(10, "A", 0.5),
            Observation(11, "A", 0.5),
            Observation(12, "A", 0.5),
            Observation(13, "A", 0.5),
            Observation(14, "B", 0.5),
            Observation(15, "B", 0.5),
            Observation(16, "B", 0.5),
            Observation(17, "B", 0.5),
        ]
        window = first_same_answer_window(rows)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual((window.answer, window.start_step, window.end_step), ("A", 10, 13))

    def test_answer_comparator_is_injectable(self) -> None:
        rows = observations(range(7, 11), ["a", "A", " a ", "A"])
        window = first_same_answer_window(
            rows,
            answers_equal=lambda left, right: left.strip().casefold()
            == right.strip().casefold(),
        )
        self.assertIsNotNone(window)


class WindowLevelTest(unittest.TestCase):
    def test_high_includes_tau_and_eps_boundaries(self) -> None:
        rows = observations(
            range(7, 11),
            ["A"] * 4,
            [TAU, TAU - EPS, TAU - EPS + 0.001, TAU],
        )
        self.assertEqual(classify_window(rows), WindowLevel.H)

    def test_low_requires_every_value_strictly_below_tau(self) -> None:
        low = observations(range(7, 11), ["A"] * 4, [0.994, 0.2, 0.9, 0.9949])
        boundary = observations(range(7, 11), ["A"] * 4, [0.9, TAU, 0.9, 0.9])
        self.assertEqual(classify_window(low), WindowLevel.L)
        self.assertEqual(classify_window(boundary), WindowLevel.M)

    def test_high_failure_falls_into_mixed(self) -> None:
        rows = observations(
            range(7, 11),
            ["A"] * 4,
            [TAU, TAU - EPS - 0.001, 0.99, 0.99],
        )
        self.assertEqual(classify_window(rows), WindowLevel.M)


if __name__ == "__main__":
    unittest.main()
