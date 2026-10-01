import random
import unittest

from plws.gpqa_official import SHUFFLE_SEED, order_choices, render_question


class OfficialGpqaShuffleTest(unittest.TestCase):
    def test_gold_letter_points_at_the_correct_text(self) -> None:
        rng = random.Random(SHUFFLE_SEED)
        correct = "10^-4 eV"
        incorrect = ("10^-11 eV", "10^-8 eV", "10^-9 eV")
        ordered, letter = order_choices(correct, incorrect, rng)

        self.assertEqual(ordered["ABCD".index(letter)], correct)
        self.assertEqual(sorted(ordered), sorted((correct, *incorrect)))

    def test_same_seed_repeats_the_letter(self) -> None:
        incorrect = ("b", "c", "d")
        first, letter = order_choices("a", incorrect, random.Random(SHUFFLE_SEED))
        second, again = order_choices("a", incorrect, random.Random(SHUFFLE_SEED))

        self.assertEqual(first, second)
        self.assertEqual(letter, again)

    def test_render_uses_parenthesized_letters(self) -> None:
        text = render_question("Which energy?", ["10^-4 eV", "10^-8 eV", "10^-9 eV", "10^-11 eV"])

        self.assertIn("Choices:\n(A) 10^-4 eV\n(B) 10^-8 eV\n(C) 10^-9 eV\n(D) 10^-11 eV", text)


if __name__ == "__main__":
    unittest.main()
