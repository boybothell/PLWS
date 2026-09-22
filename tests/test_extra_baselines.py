import unittest
from types import SimpleNamespace

from plws.extra_baselines import extra_baseline_sort_key, select_extra_start


class ExtraStartTests(unittest.TestCase):
    def test_two_idle_cards_prefer_tp2(self):
        pending = [
            SimpleNamespace(model_tag="r1_14b", task_id="small"),
            SimpleNamespace(model_tag="qwq_32b", task_id="large"),
        ]
        picked = select_extra_start(pending, idle_count=2, any_loading=False)
        self.assertEqual([item.task_id for item in picked], ["large"])

    def test_one_idle_card_only_tp1(self):
        pending = [
            SimpleNamespace(model_tag="qwq_32b", task_id="large"),
            SimpleNamespace(model_tag="r1_14b", task_id="small"),
        ]
        picked = select_extra_start(pending, idle_count=1, any_loading=False)
        self.assertEqual([item.task_id for item in picked], ["small"])

    def test_loading_blocks(self):
        pending = [SimpleNamespace(model_tag="qwq_32b", task_id="large")]
        self.assertEqual(
            select_extra_start(pending, idle_count=2, any_loading=True), []
        )

    def test_sort_puts_tp2_ahead_in_same_dataset(self):
        small = extra_baseline_sort_key("answer_convergence", "r1_7b", "aime25", 42)
        large = extra_baseline_sort_key("answer_convergence", "qwq_32b", "aime25", 42)
        self.assertLess(large, small)


if __name__ == "__main__":
    unittest.main()
