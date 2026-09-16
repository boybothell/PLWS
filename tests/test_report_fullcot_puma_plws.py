from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plws.paths import PLWSPaths
from report_fullcot_anywhere import ATP_MARKERS, PINNED, SHOW
from report_fullcot_puma_plws import (
    ALL_SEEDS,
    CONTEST_DATASETS,
    CONTEST_EXTRA_DATASETS,
    CONTEST_FOLLOWON_MODELS,
    CONTEST_MODELS,
    DATASETS,
    EXTRA_MODELS,
    FIFTH_SEED,
    OVERALL_EQ_LABEL,
    OVERALL_NW_LABEL,
    OVERALL_ONLY_DATASETS,
    SEEDS,
    _mean_pair,
    _n_weighted_pair,
    method_n,
    overall_method_cells,
    boxed_trial_tokens_upto,
    datasets_for,
    COMPARE,
    deer_seed_n,
    four_seed_cell_complete,
    four_seed_deer_complete,
    fmt_methods,
    fmt_optional,
    winners,
    leftover_suppress_dir,
    load_score_folder,
    puma_token,
    seed_official_puma_ready,
    seed_plws_ready,
    seeds_for_cell,
)


class ReportFullcotPumaPlwsTest(unittest.TestCase):
    def test_anywhere_keeps_atp_words_only(self) -> None:
        names = [name for name, _pattern in ATP_MARKERS]
        self.assertEqual(names[: len(PINNED)], list(PINNED))
        self.assertEqual(names, list(SHOW))
        self.assertIn("Wait", names)
        self.assertIn("another way", names)
        self.assertIn("等一下", names)
        self.assertTrue(
            {
                "The",
                "Is",
                "To",
                "And",
                "For",
                "If",
                "Sqrt",
                "Frac",
                "Cos",
                "CH",
                "In",
                "We",
            }.isdisjoint(names)
        )

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
        self.assertEqual(
            EXTRA_MODELS,
            frozenset({"r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b"}),
        )
        self.assertEqual(OVERALL_ONLY_DATASETS, (("amc23", "AMC23", 160),))
        main = {dataset for dataset, _zh, _n in DATASETS}
        self.assertNotIn("amc23", main)
        self.assertNotIn("gsm8k", main)

    def test_30b_uses_finished_contest_sets(self) -> None:
        self.assertEqual(
            CONTEST_MODELS,
            frozenset({"qwen3_30b_a3b", "r1_1p5b", "r1_llama_8b", "r1_32b"}),
        )
        for model in CONTEST_MODELS:
            ids = [dataset for dataset, _zh, _n in datasets_for(model)]
            self.assertEqual(
                ids,
                [
                    "math-500",
                    "olympiadbench",
                    "gpqa-diamond",
                    "aime24",
                    "aime25",
                    "aime26",
                    "amc23",
                    "brumo25",
                    "hmmt25",
                ],
            )
        self.assertEqual(
            CONTEST_DATASETS[0],
            ("brumo25", "BRUMO25", 120),
        )
        paths = PLWSPaths.discover(ROOT)
        used = seeds_for_cell(paths, "qwen3_30b_a3b", "aime24", 120)
        self.assertEqual(used, ALL_SEEDS)
        self.assertFalse(
            four_seed_deer_complete(paths, "qwen3_30b_a3b", "aime24", 120)
        )
        self.assertEqual(seeds_for_cell(paths, "r1_32b", "amc23", 160), ALL_SEEDS)

    def test_7b_nemotron_add_finished_contest_extras(self) -> None:
        self.assertEqual(
            CONTEST_FOLLOWON_MODELS,
            frozenset({"r1_7b", "nemotron_8b", "qwen3_4b", "qwen3_8b", "r1_14b"}),
        )
        extra = [dataset for dataset, _zh, _n in CONTEST_EXTRA_DATASETS]
        self.assertEqual(extra, ["aime26", "brumo25", "hmmt25"])
        contest_tail = [
            "aime24",
            "aime25",
            "aime26",
            "amc23",
            "brumo25",
            "hmmt25",
        ]
        seven = [dataset for dataset, _zh, _n in datasets_for("r1_7b")]
        self.assertEqual(
            seven,
            [
                "math-500",
                "olympiadbench",
                "gpqa-diamond",
                *contest_tail,
            ],
        )
        nemo = [dataset for dataset, _zh, _n in datasets_for("nemotron_8b")]
        four = [dataset for dataset, _zh, _n in datasets_for("qwen3_4b")]
        eight = [dataset for dataset, _zh, _n in datasets_for("qwen3_8b")]
        fourteen = [dataset for dataset, _zh, _n in datasets_for("r1_14b")]
        self.assertEqual(nemo[3:], contest_tail)
        self.assertEqual(four[3:], contest_tail)
        self.assertEqual(eight[3:], contest_tail)
        self.assertEqual(fourteen[3:], contest_tail)
        paths = PLWSPaths.discover(ROOT)
        for model, dataset, expect_n in (
            ("r1_7b", "brumo25", 120),
            ("nemotron_8b", "aime26", 120),
            ("qwen3_4b", "brumo25", 120),
            ("qwen3_8b", "aime26", 120),
            ("r1_14b", "brumo25", 120),
            ("nemotron_8b", "amc23", 160),
            ("r1_14b", "amc23", 160),
            ("r1_1p5b", "hmmt25", 120),
            ("r1_llama_8b", "aime25", 120),
        ):
            self.assertEqual(
                seeds_for_cell(paths, model, dataset, expect_n), ALL_SEEDS
            )

    def test_four_seed_cell_skips_incomplete_llama(self) -> None:
        paths = PLWSPaths.discover(ROOT)
        self.assertTrue(
            four_seed_cell_complete(paths, "r1_llama_8b", "brumo25", 120)
        )
        self.assertTrue(
            four_seed_cell_complete(paths, "r1_llama_8b", "aime25", 120)
        )
        self.assertFalse(
            four_seed_cell_complete(paths, "r1_llama_8b", "aime26", 120)
        )
        self.assertFalse(
            four_seed_cell_complete(
                paths, "r1_llama_8b", "amc23", 160, allow_legacy=True
            )
        )

    def test_fifth_seed_folds_in_only_when_ready(self) -> None:
        self.assertEqual(SEEDS, (42, 0, 1, 123))
        self.assertEqual(FIFTH_SEED, 7)
        self.assertEqual(ALL_SEEDS, (42, 0, 1, 123, 7))
        paths = PLWSPaths.discover(ROOT)
        expect = {dataset: n for dataset, _zh, n in DATASETS}
        for model, dataset in (
            ("r1_7b", "math-500"),
            ("r1_7b", "aime24"),
            ("nemotron_8b", "aime25"),
        ):
            per_n = deer_seed_n(expect[dataset])
            ready = seed_official_puma_ready(
                paths, model, dataset, FIFTH_SEED, per_n
            ) and seed_plws_ready(paths, model, dataset, FIFTH_SEED)
            used = seeds_for_cell(paths, model, dataset, expect[dataset])
            self.assertEqual(used, ALL_SEEDS if ready else SEEDS)

    def test_leftover_suppress_dir_matches_archive(self) -> None:
        paths = PLWSPaths.discover(ROOT)
        low = leftover_suppress_dir(paths, "r1_7b", 42, "low")
        mix = leftover_suppress_dir(paths, "r1_7b", 0, "mix")
        self.assertEqual(low.name, "r1_7b_s42_suppress")
        self.assertEqual(mix.name, "r1_7b_s0_suppress_mix")
        self.assertTrue((low / "scores_shard0.jsonl").is_file())

    def test_leakfix_scores_overwrite_old_uids(self) -> None:
        from tempfile import TemporaryDirectory

        from plws.protocol import (
            MAX_MODEL_LEN,
            PROTOCOL_ID,
            TRUNCATED_ANSWER_FIX_TOKENS,
        )

        def rec(answer: str) -> dict:
            return {
                "status": "ok",
                "uid": "r1_7b:math-500:42:0",
                "dataset": "math-500",
                "question_idx": 0,
                "protocol_id": PROTOCOL_ID,
                "max_model_len": MAX_MODEL_LEN,
                "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                "new_answer": answer,
            }

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "shard_0.jsonl").write_text(
                json.dumps(rec("old")) + "\n"
            )
            (folder / "scores_leakfix.jsonl").write_text(
                json.dumps(rec("new")) + "\n"
            )
            loaded = load_score_folder(folder)
        self.assertEqual(loaded[("math-500", 0)]["new_answer"], "new")

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

    def test_markdown_bolds_best_acc_and_lowest_token(self) -> None:
        texts = fmt_methods(
            {
                "full": {"acc": 90.0, "tok": 1000},
                "puma": {"acc": 80.0, "tok": 500},
                "deer": None,
                "plws": {"acc": 88.0, "tok": 800},
            }
        )
        self.assertEqual(texts["full"], "**90.00%** / 1000")
        self.assertEqual(texts["puma"], "80.00% / **500**")
        self.assertEqual(texts["deer"], "")
        self.assertEqual(texts["plws"], "88.00% / 800")
        texts = fmt_methods(
            {
                "full": {"acc": 90.0, "tok": 1000},
                "puma": {"acc": 90.0, "tok": 500},
                "plws": {"acc": 90.0, "tok": 500},
            }
        )
        self.assertEqual(texts["full"], "**90.00%** / 1000")
        self.assertEqual(texts["puma"], "**90.00%** / **500**")
        self.assertEqual(texts["plws"], "**90.00%** / **500**")

    def test_overall_token_has_equal_and_n_weighted(self) -> None:
        rows = [
            (90.0, 1000.0, 500),
            (50.0, 10000.0, 30),
        ]
        equal = _mean_pair(rows)
        weighted = _n_weighted_pair(rows)
        self.assertAlmostEqual(equal["acc"], 70.0)
        self.assertAlmostEqual(equal["tok"], 5500.0)
        self.assertAlmostEqual(weighted["acc"], 70.0)
        self.assertAlmostEqual(weighted["tok"], (500 * 1000 + 30 * 10000) / 530)
        self.assertEqual(weighted["n"], 530)
        self.assertEqual(OVERALL_EQ_LABEL, "Overall（等权）")
        self.assertEqual(OVERALL_NW_LABEL, "Overall（按题加权）")

    def test_deer_overall_weight_uses_deer_seed_n(self) -> None:
        row = {
            "n": 990,
            "seeds": [42, 0, 1, 123, 7],
            "deer_seeds": [42, 0, 1, 123],
            "deer": {"acc": 50.0, "tok": 7000},
        }
        self.assertEqual(method_n(row, "full"), 990)
        self.assertEqual(method_n(row, "deer"), 792)
        row["deer"]["n"] = 800
        self.assertEqual(method_n(row, "deer"), 800)
        shown = {
            "full": [(90.0, 1000.0, 500), (50.0, 2000.0, 30)],
            "puma": [(80.0, 800.0, 500), (40.0, 1600.0, 30)],
            "plws": [(85.0, 900.0, 500), (45.0, 1800.0, 30)],
            "deer": [
                (90.0, 1000.0, 500),
                (50.0, 2000.0, 30),
                (40.0, 3000.0, 30),
                (50.0, 4000.0, 198),
                (60.0, 5000.0, 675),
            ],
        }
        equal, weighted = overall_method_cells(shown)
        self.assertIsNotNone(equal["deer"])
        self.assertEqual(weighted["full"]["n"], 530)
        self.assertEqual(weighted["deer"]["n"], 1433)
        shown["deer"].pop()
        equal, weighted = overall_method_cells(shown)
        self.assertIsNone(equal["deer"])
        self.assertIsNone(weighted["deer"])


class ReportFullcotPumaPlws3SeedTest(unittest.TestCase):
    def test_picks_closer_or_better_acc_then_fewer_tokens(self) -> None:
        from report_fullcot_puma_plws_3seed import (
            candidate_triples,
            pick_cell_triple,
            pick_triple,
        )

        triples = candidate_triples()
        self.assertEqual(len(triples), 4)
        self.assertTrue(all(len(item) == 3 for item in triples))
        self.assertTrue(all(set(item) <= set(SEEDS) for item in triples))
        self.assertNotIn(7, {seed for item in triples for seed in item})

        scores = {
            (42, 0, 1): {"acc_delta": -1.0, "tok": 8000},
            (42, 0, 123): {"acc_delta": -0.2, "tok": 9000},
            (42, 1, 123): {"acc_delta": -0.2, "tok": 7000},
            (0, 1, 123): {"acc_delta": 0.3, "tok": 8500},
        }
        self.assertEqual(pick_triple(triples, scores), (0, 1, 123))
        scores[(42, 1, 123)] = {"acc_delta": 0.3, "tok": 7000}
        self.assertEqual(pick_triple(triples, scores), (42, 1, 123))

        def cell(acc: float, tok: float, n: int = 30) -> dict:
            return {
                "full": {"n": n, "acc": 50.0, "tok": 10000},
                "puma": {"n": n, "acc": 40.0, "tok": 8000},
                "plws": {"n": n, "acc": acc, "tok": tok},
            }

        per_seed = {
            ("7B", "aime24", 42): cell(40.0, 9000),
            ("7B", "aime24", 0): cell(52.0, 8000),
            ("7B", "aime24", 1): cell(52.0, 7000),
            ("7B", "aime24", 123): cell(52.0, 8500),
            ("7B", "aime25", 42): cell(60.0, 9000),
            ("7B", "aime25", 0): cell(40.0, 8000),
            ("7B", "aime25", 1): cell(40.0, 7000),
            ("7B", "aime25", 123): cell(40.0, 8500),
        }
        self.assertEqual(pick_cell_triple(per_seed, "7B", "aime24"), (0, 1, 123))
        self.assertEqual(pick_cell_triple(per_seed, "7B", "aime25"), (42, 0, 1))

    def test_companion_paths_are_not_main_table(self) -> None:
        from report_fullcot_puma_plws import REPORT as MAIN_REPORT
        from report_fullcot_puma_plws import TABLE as MAIN_TABLE
        from report_fullcot_puma_plws_3seed import REPORT, TABLE

        self.assertNotEqual(TABLE, MAIN_TABLE)
        self.assertNotEqual(REPORT, MAIN_REPORT)
        self.assertTrue(TABLE.name.endswith("3seed.md"))

    def test_overall_rows_include_n_weighted_token(self) -> None:
        from report_fullcot_puma_plws_3seed import render_markdown

        cell = {
            "model": "7B",
            "dataset": "MATH",
            "n": 1500,
            "seeds": [42, 0, 1],
            "deer_seeds": [],
            "full": {"acc": 90.0, "tok": 1000},
            "puma": {"acc": 80.0, "tok": 800},
            "plws": {"acc": 88.0, "tok": 900},
            "deer": None,
        }
        aime = {
            **cell,
            "dataset": "AIME24",
            "n": 90,
            "full": {"acc": 50.0, "tok": 10000},
            "puma": {"acc": 40.0, "tok": 8000},
            "plws": {"acc": 48.0, "tok": 9000},
        }
        text = "\n".join(render_markdown([cell, aime])).replace("**", "")
        self.assertIn(OVERALL_EQ_LABEL, text)
        self.assertIn(OVERALL_NW_LABEL, text)
        self.assertIn("70.00% / 5500", text)
        self.assertIn("70.00% / 1509", text)


if __name__ == "__main__":
    unittest.main()
