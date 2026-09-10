from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plws.paths import PLWSPaths
from export_fullcot_puma_plws_feishu import COMPARE, winners
from report_fullcot_puma_plws import (
    DATASETS,
    EXTRA_MODELS,
    OVERALL_ONLY_DATASETS,
    boxed_trial_tokens_upto,
    four_seed_deer_complete,
    fmt_optional,
    leftover_suppress_dir,
    puma_token,
)


class ReportFullcotPumaPlwsTest(unittest.TestCase):
    def test_boxed_trial_stops_at_window_end(self) -> None:
        trials = [
            {"stopped_len": 8, "final_answer": "1", "count_answer_tokens": 3},
            {"stopped_len": 10, "final_answer": "1", "count_answer_tokens": 4},
            {"stopped_len": 12, "final_answer": "2", "count_answer_tokens": 9},
            {"stopped_len": 9, "final_answer": "", "count_answer_tokens": 5},
            {
                "stopped_len": 7,
                "final_answer": "1",
                "count_answer_tokens": 2,
                "skipped": True,
            },
        ]
        self.assertEqual(boxed_trial_tokens_upto(trials, 10), 7)

    def test_puma_token_adds_boxed_trial(self) -> None:
        self.assertEqual(
            puma_token({"compressed_tokens": 100, "tokens_trial_answers": 7}),
            107,
        )
        with self.assertRaises(RuntimeError):
            puma_token({"compressed_tokens": 100})

    def test_fmt_optional_blank_when_missing(self) -> None:
        self.assertEqual(fmt_optional(None), "")
        self.assertEqual(fmt_optional({"acc": 50.0, "tok": 100.4}), "50.00% / 100")

    def test_deer_four_seed_gate_matches_disk(self) -> None:
        paths = PLWSPaths.discover(ROOT)
        expect = {dataset: n for dataset, _zh, n in DATASETS}

        self.assertTrue(
            four_seed_deer_complete(paths, "qwen3_4b", "math-500", expect["math-500"])
        )
        self.assertTrue(
            four_seed_deer_complete(
                paths, "r1_14b", "olympiadbench", expect["olympiadbench"]
            )
        )
        self.assertTrue(
            four_seed_deer_complete(paths, "nemotron_8b", "math-500", expect["math-500"])
        )
        self.assertTrue(
            four_seed_deer_complete(paths, "r1_7b", "aime24", expect["aime24"])
        )
        self.assertFalse(
            four_seed_deer_complete(
                paths, "nemotron_8b", "gpqa-diamond", expect["gpqa-diamond"]
            )
        )
        self.assertTrue(
            four_seed_deer_complete(paths, "r1_7b", "math-500", expect["math-500"])
        )
        self.assertTrue(
            four_seed_deer_complete(
                paths, "r1_7b", "gpqa-diamond", expect["gpqa-diamond"]
            )
        )
        self.assertTrue(
            four_seed_deer_complete(
                paths, "r1_7b", "olympiadbench", expect["olympiadbench"]
            )
        )

    def test_amc23_is_7b_extra_row_in_overall(self) -> None:
        self.assertEqual(EXTRA_MODELS, frozenset({"r1_7b"}))
        self.assertEqual(OVERALL_ONLY_DATASETS, (("amc23", "AMC23", 160),))
        main = {dataset for dataset, _zh, _n in DATASETS}
        self.assertNotIn("amc23", main)
        self.assertNotIn("gsm8k", main)

    def test_leftover_suppress_dir_matches_archive(self) -> None:
        paths = PLWSPaths.discover(ROOT)
        low = leftover_suppress_dir(paths, "r1_7b", 42, "low")
        mix = leftover_suppress_dir(paths, "r1_7b", 0, "mix")
        self.assertEqual(low.name, "r1_7b_s42_suppress")
        self.assertEqual(mix.name, "r1_7b_s0_suppress_mix")
        self.assertTrue((low / "scores_shard0.jsonl").is_file())

    def test_feishu_winners_include_fullcot(self) -> None:
        self.assertEqual(COMPARE, ("full", "puma", "deer", "plws"))
        acc_w, tok_w = winners(
            {
                "full": {"acc": 90.0, "tok": 1000},
                "puma": {"acc": 80.0, "tok": 500},
                "plws": {"acc": 88.0, "tok": 800},
            }
        )
        self.assertEqual(acc_w, {"full"})
        self.assertEqual(tok_w, {"puma"})
        acc_w, tok_w = winners(
            {
                "full": {"acc": 90.0, "tok": 1000},
                "puma": {"acc": 90.0, "tok": 1000},
                "plws": {"acc": 88.0, "tok": 800},
            }
        )
        self.assertEqual(acc_w, {"full", "puma"})
        self.assertEqual(tok_w, {"plws"})


if __name__ == "__main__":
    unittest.main()
