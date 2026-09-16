from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plws.dense_reuse import merge_dense_trials, prepare_reuse_plan


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def trial(question_idx: int, stopped_len: int, answer: str) -> dict:
    return {
        "question_idx": question_idx,
        "stopped_len": stopped_len,
        "final_answer": answer,
        "confidence": 0.9,
        "success": True,
    }


class DenseReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.filtered = self.root / "filtered_steps.json"
        self.puma = self.root / "puma_trial_answers.json"
        self.shards = self.root / "shards"
        self.output = self.root / "trial_answers.json"
        write_json(
            self.filtered,
            [
                {
                    "question": "q1",
                    "reasoning_steps": ["q1s1", "q1s2", "q1s3"],
                },
                {"question": "q2", "reasoning_steps": ["q2s1", "q2s2"]},
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reuses_puma_and_existing_dense_then_generates_only_missing(self) -> None:
        write_json(
            self.puma,
            [
                trial(1, 1, "puma-11"),
                {"question_idx": 1, "stopped_len": 2, "skipped": True},
                trial(1, 3, "puma-13"),
                {"question_idx": 2, "stopped_len": 1, "skipped": True},
                {"question_idx": 2, "stopped_len": 2, "skipped": True},
            ],
        )
        write_json(
            self.shards / "filtered_steps_shard1.json",
            [{"_abs_question_idx": 2, "question": "q2", "reasoning_steps": ["a", "b"]}],
        )
        write_json(
            self.shards / "trial_answers_shard1.json",
            [trial(1, 1, "legacy-21")],
        )

        plan = prepare_reuse_plan(
            filtered_steps=self.filtered,
            puma_trials=self.puma,
            shards_dir=self.shards,
            num_shards=2,
        )
        self.assertEqual(plan["expected_steps"], 5)
        self.assertEqual(plan["puma_reused_steps"], 2)
        self.assertEqual(plan["legacy_dense_steps"], 1)
        self.assertEqual(plan["missing_steps_per_shard"], [1, 1])
        self.assertEqual(
            plan["legacy_sources"],
            [
                {
                    "questions": "filtered_steps_shard1.json",
                    "trials": "trial_answers_shard1.json",
                }
            ],
        )

        missing0 = json.loads(
            (self.shards / "missing_steps_shard0.json").read_text()
        )
        missing1 = json.loads(
            (self.shards / "missing_steps_shard1.json").read_text()
        )
        self.assertEqual(missing0[0]["should_generate_trial"], [False, True, False])
        self.assertEqual(missing1[0]["should_generate_trial"], [False, True])

        write_json(
            self.shards / "trial_answers_missing_shard0.json",
            [
                {"question_idx": 1, "stopped_len": 1, "skipped": True},
                trial(1, 2, "new-12"),
                {"question_idx": 1, "stopped_len": 3, "skipped": True},
            ],
        )
        write_json(
            self.shards / "trial_answers_missing_shard1.json",
            [
                {"question_idx": 1, "stopped_len": 1, "skipped": True},
                trial(1, 2, "new-22"),
            ],
        )

        summary = merge_dense_trials(
            filtered_steps=self.filtered,
            puma_trials=self.puma,
            shards_dir=self.shards,
            num_shards=2,
            output=self.output,
        )
        self.assertEqual(summary["steps"], 5)
        rows = json.loads(self.output.read_text())
        by_key = {
            (row["question_idx"], row["stopped_len"]): row["final_answer"]
            for row in rows
        }
        self.assertEqual(
            by_key,
            {
                (1, 1): "puma-11",
                (1, 2): "new-12",
                (1, 3): "puma-13",
                (2, 1): "legacy-21",
                (2, 2): "new-22",
            },
        )

    def test_all_puma_steps_skip_dense_generation(self) -> None:
        write_json(
            self.puma,
            [
                trial(1, 1, "11"),
                trial(1, 2, "12"),
                trial(1, 3, "13"),
                trial(2, 1, "21"),
                trial(2, 2, "22"),
            ],
        )
        plan = prepare_reuse_plan(
            filtered_steps=self.filtered,
            puma_trials=self.puma,
            shards_dir=self.shards,
            num_shards=2,
        )
        self.assertEqual(plan["active_shards"], [])
        self.assertEqual(plan["missing_steps"], 0)
        merge_dense_trials(
            filtered_steps=self.filtered,
            puma_trials=self.puma,
            shards_dir=self.shards,
            num_shards=2,
            output=self.output,
        )
        self.assertEqual(len(json.loads(self.output.read_text())), 5)

    def test_resume_plan_rejects_changed_source(self) -> None:
        write_json(self.puma, [trial(1, 1, "11")])
        prepare_reuse_plan(
            filtered_steps=self.filtered,
            puma_trials=self.puma,
            shards_dir=self.shards,
            num_shards=1,
        )
        write_json(self.puma, [trial(1, 1, "changed")])
        with self.assertRaisesRegex(RuntimeError, "does not match current inputs"):
            prepare_reuse_plan(
                filtered_steps=self.filtered,
                puma_trials=self.puma,
                shards_dir=self.shards,
                num_shards=1,
            )


if __name__ == "__main__":
    unittest.main()
