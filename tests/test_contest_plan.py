from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plws.contest import (
    CONTEST_MODELS,
    DATASETS,
    FIFTH_SEED,
    FOLLOWON_MODELS,
    MAIN_PROTOCOL,
    MODELS,
    QUEUE_MODELS,
    SEEDS,
    build_tasks,
    contest_blocked_phases,
    contest_needs_prereq,
    contest_task_ready,
    fill_contest_dispatch,
    fill_leftover_dispatch,
    is_task_log_name,
    log_loaded_after_latest_start,
    log_shows_engine_loaded,
    select_cold_starts,
    first_open_phase,
    gpu_count,
    idle_contest_gpus,
    is_eight_olympiad_leftover_cmd,
    leftover_gpus_from_proc,
    may_sample_fullcot,
    plws_task,
    protocol_for,
    prereq_task,
    sample_meta_path,
    summarize,
    task_complete,
)
from plws.matrix import (
    ALL_MODELS,
    FIRSTWIN,
    build_tasks as build_matrix_tasks,
    fill_dispatch,
    needs_prereq,
    task_ready,
)
from plws.paths import PLWSPaths
from plws.protocol import PROTOCOL_ID, TRUNCATED_ANSWER_FIX_TOKENS


class ContestPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "plws"
        self.root.mkdir()
        self.paths = PLWSPaths(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, list):
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in payload),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def _puma_ready(self, model: str, dataset: str, seed: int) -> None:
        folder = self.paths.puma_output_dir(model, dataset, seed)
        self._write(folder / "statistics.json", [{"question_idx": 0}])
        self._write(folder / "prefixed_answers.json", {})

    def _dense_ready(self, model: str, dataset: str, seed: int) -> None:
        self._write(
            self.paths.dense_trial_path(model, dataset, seed),
            [{"question_idx": 0, "stopped_len": 10}],
        )

    def _sample_ready(self, model: str, dataset: str, seed: int) -> None:
        protocol = protocol_for(model)
        self._write(
            sample_meta_path(self.paths, model, dataset, seed),
            {
                "protocol_id": protocol.protocol_id,
                "max_tokens": protocol.generation_tokens,
                "answer_fix_max_tokens": protocol.answer_fix_tokens,
                "prompt_reserve_tokens": protocol.prompt_reserve_tokens,
                "max_model_len": protocol.max_model_len,
                "prompt_version": "default",
            },
        )

    def _jobs(self, model: str, dataset: str, seed: int, count: int = 2) -> None:
        path = self.paths.jobs_path(
            model, dataset, seed, FIRSTWIN, k=4, lexicon="core"
        )
        self._write(
            path,
            [
                {
                    "uid": f"{model}:{dataset}:{seed}:firstwin:{index}",
                    "dataset": dataset,
                }
                for index in range(count)
            ],
        )

    def test_lane_is_thirty_only_five_sets_five_seeds_puma_plws_only(self) -> None:
        self.assertEqual(CONTEST_MODELS, ("qwen3_30b_a3b",))
        self.assertEqual(DATASETS, ("brumo25", "hmmt25", "aime24", "aime25", "aime26"))
        self.assertEqual(SEEDS, (42, 0, 1, 123, 7))
        self.assertEqual(FIFTH_SEED, 7)
        self.assertTrue({"qwq_32b", "qwen3_32b"}.isdisjoint(ALL_MODELS))
        tasks = build_tasks(self.paths)
        self.assertEqual(summarize(tasks)["contest_prereq"], 25)
        self.assertEqual(summarize(tasks)["contest_plws"], 25)
        self.assertEqual(summarize(tasks)["total"], 50)
        self.assertTrue(all(task.model == "qwen3_30b_a3b" for task in tasks))
        self.assertEqual(
            FOLLOWON_MODELS, ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b")
        )
        self.assertEqual(QUEUE_MODELS, CONTEST_MODELS + FOLLOWON_MODELS)
        self.assertTrue(set(FOLLOWON_MODELS).isdisjoint(CONTEST_MODELS))
        self.assertTrue({"qwen3_8b", "qwq_32b", "qwen3_32b"}.isdisjoint(QUEUE_MODELS))
        self.assertTrue(all(task.gpu_count == 2 for task in tasks))
        self.assertTrue(all(task.kind in {"prereq", "plws"} for task in tasks))
        self.assertFalse(any(task.kind == "deer" for task in tasks))
        self.assertEqual({task.phase for task in tasks}, {"contest_prereq", "contest_plws"})
        matrix = build_matrix_tasks(self.paths, include_thirty=True)
        self.assertFalse(any(task.model in {"qwq_32b", "qwen3_32b"} for task in matrix))
        self.assertFalse(
            any(task.dataset in {"brumo25", "hmmt25", "aime26"} for task in matrix)
        )

    def test_qwq_and_32b_need_prereq_unlike_main_matrix(self) -> None:
        self.assertFalse(needs_prereq(self.paths, "qwq_32b", "brumo25", 7))
        self.assertTrue(contest_needs_prereq(self.paths, "qwq_32b", "brumo25", 7))
        self.assertTrue(contest_needs_prereq(self.paths, "qwen3_32b", "aime26", 42))
        self._sample_ready("qwq_32b", "brumo25", 7)
        self._puma_ready("qwq_32b", "brumo25", 7)
        self._jobs("qwq_32b", "brumo25", 7)
        self.assertFalse(contest_needs_prereq(self.paths, "qwq_32b", "brumo25", 7))

    def test_all_models_use_32k_and_reuse_legacy_thirty_sample(self) -> None:
        for model in CONTEST_MODELS:
            protocol = protocol_for(model)
            self.assertEqual(protocol.protocol_id, PROTOCOL_ID)
            self.assertEqual(protocol.generation_tokens, 32768)
            self.assertEqual(protocol.max_model_len, 37888)
        self._write(
            sample_meta_path(self.paths, "qwen3_30b_a3b", "aime24", 42),
            {
                "max_tokens": 32768,
                "answer_fix_max_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                "prompt_version": "default",
            },
        )
        self._puma_ready("qwen3_30b_a3b", "aime24", 42)
        self._jobs("qwen3_30b_a3b", "aime24", 42)
        self.assertFalse(contest_needs_prereq(self.paths, "qwen3_30b_a3b", "aime24", 42))
        self._write(
            sample_meta_path(self.paths, "qwen3_30b_a3b", "aime24", 42),
            {
                "protocol_id": "puma-fullcot-81k-v2",
                "max_tokens": 81920,
                "answer_fix_max_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                "prompt_reserve_tokens": 3072,
                "max_model_len": 87040,
                "prompt_version": "default",
            },
        )
        self.assertTrue(contest_needs_prereq(self.paths, "qwen3_30b_a3b", "aime24", 42))

    def test_plws_waits_for_jobs_then_completes(self) -> None:
        task = plws_task("qwen3_30b_a3b", "hmmt25", 0)
        self.assertFalse(task_ready(self.paths, task))
        ok, reason = task_complete(self.paths, task)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing jobs")
        self._jobs("qwen3_30b_a3b", "hmmt25", 0, 1)
        self.assertTrue(task_ready(self.paths, task))
        score = self.paths.score_dir(
            "qwen3_30b_a3b", "hmmt25", 0, FIRSTWIN, k=4, lexicon="core"
        )
        self._write(
            score / "shard_0.jsonl",
            [
                {
                    "uid": "qwen3_30b_a3b:hmmt25:0:firstwin:0",
                    "status": "ok",
                    "gt": "42",
                    "gold_error": "",
                    "protocol_id": MAIN_PROTOCOL.protocol_id,
                    "max_model_len": MAIN_PROTOCOL.max_model_len,
                    "truncated_answer_fix_tokens": MAIN_PROTOCOL.answer_fix_tokens,
                    "fullcot_generation_tokens": MAIN_PROTOCOL.generation_tokens,
                }
            ],
        )
        ok, reason = task_complete(self.paths, task)
        self.assertTrue(ok)
        self.assertEqual(reason, "done 1/1")

    def test_prereq_before_plws_and_two_gpu_dispatch(self) -> None:
        self.assertEqual(gpu_count("qwen3_32b"), 2)
        self.assertEqual(MODELS["qwq_32b"], "/mnt/d/lsj/models/QwQ-32B")
        pending = [
            prereq_task("qwen3_32b", "aime24", 42),
            plws_task("qwen3_32b", "aime24", 42),
        ]
        self.assertEqual(first_open_phase(pending, {}), "contest_prereq")
        taken = fill_dispatch(pending, 4, self.paths)
        self.assertEqual([task.kind for task in taken], ["prereq", "prereq"][:1])
        self.assertEqual(taken[0].kind, "prereq")
        self.assertEqual(taken[0].gpu_count, 2)
        self.assertEqual(len(taken), 1)
        self._jobs("qwen3_32b", "aime24", 42)
        taken = fill_dispatch(pending, 4, self.paths)
        self.assertEqual([task.kind for task in taken], ["prereq", "plws"])

    def test_followon_waits_for_thirty_and_uses_one_gpu(self) -> None:
        self.assertEqual(gpu_count("r1_7b"), 1)
        self.assertEqual(gpu_count("qwen3_4b"), 1)
        self.assertEqual(prereq_task("r1_7b", "brumo25", 7).phase, "followon_prereq")
        self.assertEqual(plws_task("qwen3_4b", "hmmt25", 0).phase, "followon_plws")
        queue = build_tasks(self.paths, models=QUEUE_MODELS)
        counts = summarize(queue)
        self.assertEqual(counts["contest_prereq"], 25)
        self.assertEqual(counts["contest_plws"], 25)
        self.assertEqual(counts["followon_prereq"], 100)
        self.assertEqual(counts["followon_plws"], 100)
        self.assertEqual(counts["total"], 250)
        self.assertTrue(
            all(task.gpu_count == 1 for task in queue if task.model in FOLLOWON_MODELS)
        )
        pending = [
            prereq_task("qwen3_30b_a3b", "hmmt25", 7),
            plws_task("qwen3_30b_a3b", "aime24", 42),
            prereq_task("r1_7b", "brumo25", 42),
            plws_task("r1_7b", "brumo25", 42),
        ]
        self.assertEqual(first_open_phase(pending, {}), "contest_prereq")
        blocked = contest_blocked_phases(pending)
        self.assertEqual(blocked, frozenset({"followon_prereq", "followon_plws"}))
        taken = fill_contest_dispatch(pending, 8, self.paths, blocked_phases=blocked)
        self.assertEqual([task.task_id for task in taken], ["prereq__qwen3_30b_a3b__hmmt25__s7"])
        self.assertFalse(contest_task_ready(self.paths, plws_task("r1_7b", "brumo25", 42)))
        follow = [
            prereq_task("r1_7b", "brumo25", 42),
            prereq_task("qwen3_4b", "hmmt25", 0),
            plws_task("r1_7b", "brumo25", 42),
        ]
        self.assertEqual(first_open_phase(follow, {}), "followon_prereq")
        taken = fill_contest_dispatch(follow, 2, self.paths)
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__r1_7b__brumo25__s42", "prereq__qwen3_4b__hmmt25__s0"],
        )

    def test_leftover_gpu_runs_followon(self) -> None:
        pending = [
            prereq_task("qwen3_30b_a3b", "hmmt25", 7),
            prereq_task("qwen3_30b_a3b", "hmmt25", 0),
            prereq_task("r1_7b", "brumo25", 42),
            prereq_task("qwen3_4b", "hmmt25", 0),
        ]
        blocked = contest_blocked_phases(pending)
        self.assertEqual(
            [task.task_id for task in fill_contest_dispatch(pending, 1, self.paths, blocked_phases=blocked)],
            [],
        )
        self.assertEqual(
            [task.task_id for task in fill_leftover_dispatch(pending, 1, self.paths, blocked_phases=blocked)],
            ["prereq__r1_7b__brumo25__s42"],
        )
        self.assertEqual(
            [task.task_id for task in fill_leftover_dispatch(pending, 5, self.paths, blocked_phases=blocked)],
            [
                "prereq__qwen3_30b_a3b__hmmt25__s0",
                "prereq__qwen3_30b_a3b__hmmt25__s7",
                "prereq__r1_7b__brumo25__s42",
            ],
        )
        one_pair = [
            prereq_task("qwen3_30b_a3b", "hmmt25", 7),
            prereq_task("r1_7b", "brumo25", 42),
        ]
        taken = fill_leftover_dispatch(
            one_pair, 5, self.paths, blocked_phases=contest_blocked_phases(one_pair)
        )
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__qwen3_30b_a3b__hmmt25__s7", "prereq__r1_7b__brumo25__s42"],
        )
        self.assertEqual(
            [task.task_id for task in select_cold_starts(taken, any_loading=False)],
            ["prereq__qwen3_30b_a3b__hmmt25__s7"],
        )
        self.assertEqual(select_cold_starts(taken, any_loading=True), [])

    def test_frozen_models_sample_only_when_answers_missing(self) -> None:
        self.assertTrue(may_sample_fullcot(self.paths, "r1_7b", "brumo25", 42))
        self.assertTrue(may_sample_fullcot(self.paths, "qwen3_4b", "aime24", 42))
        self._write(
            self.paths.root / "samples" / "r1_7b" / "aime24" / "seed_42" / "answers.json",
            [{"question_idx": 0}],
        )
        self.assertFalse(may_sample_fullcot(self.paths, "r1_7b", "aime24", 42))
        self._write(
            self.paths.root / "samples" / "qwen3_4b" / "aime24" / "seed_42" / "answers.json",
            [{"question_idx": 0}],
        )
        self.assertTrue(may_sample_fullcot(self.paths, "qwen3_4b", "aime24", 42))

    def test_serial_load_starts_one_engine_until_loaded(self) -> None:
        self.assertTrue(log_shows_engine_loaded("Model loaded.\nTokenizer loaded\n"))
        self.assertTrue(
            log_shows_engine_loaded(
                "init engine (profile, create kv cache, warmup model) took 32.91 s"
            )
        )
        self.assertTrue(log_shows_engine_loaded("suppress shard0 official-batch step=8"))
        self.assertFalse(log_shows_engine_loaded("Loading safetensors checkpoint shards"))
        stale = (
            "# old start gpus=0\nModel loaded.\n"
            "# new start gpus=0\nLoading model from scratch...\n"
        )
        self.assertFalse(log_loaded_after_latest_start(stale))
        self.assertTrue(log_loaded_after_latest_start(stale + "Model loaded.\n"))
        pending = [
            prereq_task("qwen3_30b_a3b", "hmmt25", 7),
            plws_task("qwen3_30b_a3b", "aime24", 42),
            prereq_task("r1_7b", "brumo25", 42),
        ]
        taken = fill_contest_dispatch(pending, 8, self.paths)
        self.assertGreaterEqual(len(taken), 2)
        self.assertEqual(
            [task.task_id for task in select_cold_starts(taken, any_loading=False)],
            ["prereq__qwen3_30b_a3b__hmmt25__s7"],
        )
        self.assertEqual(select_cold_starts(taken, any_loading=True), [])
        self.assertTrue(
            is_task_log_name(
                "prereq__qwen3_30b_a3b__hmmt25__s1",
                "prereq__qwen3_30b_a3b__hmmt25__s1.attempt_1.log",
            )
        )
        self.assertFalse(
            is_task_log_name(
                "prereq__qwen3_30b_a3b__hmmt25__s1",
                "prereq__qwen3_30b_a3b__hmmt25__s123.attempt_1.log",
            )
        )

    def test_handoff_detects_8b_leftover_not_contest_queue(self) -> None:
        leftover = [
            "python",
            "/repo/scripts/score_leftover_suppress.py",
            "--mode",
            "suppress",
            "--model-tag",
            "qwen3_8b",
            "--dataset",
            "olympiadbench",
            "--seed",
            "42",
        ]
        self.assertTrue(is_eight_olympiad_leftover_cmd(leftover))
        self.assertFalse(
            is_eight_olympiad_leftover_cmd(
                leftover[:-1] + ["7"]
            )
        )
        self.assertFalse(
            is_eight_olympiad_leftover_cmd(
                [
                    "python",
                    "/repo/scripts/score_leftover_suppress.py",
                    "--model-tag",
                    "qwen3_30b_a3b",
                    "--dataset",
                    "aime24",
                    "--seed",
                    "42",
                ]
            )
        )
        self.assertEqual(
            leftover_gpus_from_proc(leftover, {"CUDA_VISIBLE_DEVICES": "3"}),
            ("3",),
        )
        self.assertEqual(
            leftover_gpus_from_proc(
                leftover,
                {"CUDA_VISIBLE_DEVICES": "4"},
            ),
            ("4",),
        )
        idle = idle_contest_gpus(
            ("3", "4", "5", "6"),
            leftover=("3", "4"),
            used_mib={"3": 41000, "4": 41000, "5": 14, "6": 14, "7": 41000},
        )
        self.assertEqual(idle, ["5", "6"])
        self.assertEqual(
            idle_contest_gpus(
                ("0", "1", "2", "3", "4", "5", "6", "7"),
                leftover=(),
                used_mib={str(i): 14 for i in range(8)},
            ),
            ["0", "1", "2", "3", "4", "5", "6", "7"],
        )

    def test_seed42_contest_dense_uses_seeded_layout(self) -> None:
        path = self.paths.dense_trial_path("qwen3_30b_a3b", "brumo25", 42)
        self.assertIn("seed_42", str(path))
        self.assertNotEqual(
            path,
            self.paths.results
            / "upstream"
            / "dense_trials"
            / "dense_G_qwen3_30b_a3b"
            / "brumo25"
            / "dense_puma"
            / "trial_answers.json",
        )


if __name__ == "__main__":
    unittest.main()
