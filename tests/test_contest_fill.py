from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plws.contest import (
    DATASETS,
    FILL_DATASETS,
    FILL_MODELS,
    FOLLOWON_MODELS,
    LOCAL_FILL_MODELS,
    MODELS,
    NO_NEW_WORK_DATASETS,
    OFFICIAL_FIRST_SEEDS,
    OFFICIAL_LATER_SEEDS,
    OFFICIAL_NEW_DATASETS,
    QUEUE_MODELS,
    build_fill_tasks,
    cells_needing_cpu_export,
    contest_blocked_phases,
    contest_reusable_score,
    engine_ready_for_next_cold_start,
    dual_lane_cap,
    contest_task_ready,
    fill_dispatch,
    fill_fullcot_complete,
    fill_needs_fullcot_sample,
    fill_sample_dataset_rank,
    fill_legacy_plws_complete,
    fill_needs_prereq,
    fill_task_complete,
    filter_fill_tasks,
    fullcot_only_tasks,
    deer_task,
    idle_contest_gpus,
    is_contest_fill_queue_cmd,
    log_loaded_after_latest_start,
    log_path_shows_gpu_released,
    log_shows_gpu_released,
    may_sample_fullcot,
    plws_task,
    protocol_for,
    prereq_task,
    sample_meta_path,
    select_cold_starts,
    should_release_gpu_lease,
)
from plws.matrix import FIRSTWIN
from plws.paths import PLWSPaths
from plws.puma_grading import write_puma_verification_marker

ROOT = Path(__file__).resolve().parents[1]


class ContestFillTest(unittest.TestCase):
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

    def _puma_ready(
        self, model: str, dataset: str, seed: int, *, verified: bool = True
    ) -> None:
        folder = self.paths.puma_output_dir(model, dataset, seed)
        self._write(folder / "statistics.json", [{"question_idx": 0}])
        self._write(folder / "prefixed_answers.json", {})
        if verified:
            write_puma_verification_marker(folder / "statistics.json")

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

    def test_official_new_run_scope(self) -> None:
        self.assertEqual(
            OFFICIAL_NEW_DATASETS,
            (
                "math-500",
                "olympiadbench",
                "gpqa-diamond",
                "aime25",
                "amc23",
            ),
        )
        self.assertEqual(OFFICIAL_FIRST_SEEDS, (42, 0, 1))
        self.assertEqual(OFFICIAL_LATER_SEEDS, (123, 7))
        self.assertEqual(NO_NEW_WORK_DATASETS, ("aime24", "aime26", "brumo25", "hmmt25"))
        self.assertIn("amc23", OFFICIAL_NEW_DATASETS)
        self.assertNotIn("amc23", NO_NEW_WORK_DATASETS)
        tasks = build_fill_tasks(
            self.paths,
            models=("r1_32b",),
            datasets=OFFICIAL_NEW_DATASETS,
            seeds=OFFICIAL_FIRST_SEEDS,
        )
        seeds = {task.seed for task in tasks}
        datasets = {task.dataset for task in tasks}
        self.assertEqual(seeds, {42, 0, 1})
        self.assertEqual(datasets, set(OFFICIAL_NEW_DATASETS))
        self.assertNotIn("aime24", datasets)
        self.assertIn("amc23", datasets)
        self.assertEqual(
            {
                task.task_id
                for task in fullcot_only_tasks(tasks)
                if task.dataset == "amc23"
            },
            {
                "prereq__r1_32b__amc23__s42",
                "prereq__r1_32b__amc23__s0",
                "prereq__r1_32b__amc23__s1",
            },
        )

    def test_cpu_export_respects_selected_seeds(self) -> None:
        model = "qwen3_30b_a3b"
        dataset = "math-500"
        for seed in (0, 123):
            self._write(
                self.paths.dense_trial_path(model, dataset, seed),
                [{"question_idx": 1, "stopped_len": 1}],
            )

        self.assertEqual(
            cells_needing_cpu_export(
                self.paths,
                (model,),
                datasets=(dataset,),
                seeds=(0,),
            ),
            [(model, dataset, 0)],
        )

    def test_fill_scope_keeps_old_contest_lane(self) -> None:
        self.assertEqual(DATASETS, ("brumo25", "hmmt25", "aime24", "aime25", "aime26"))
        self.assertEqual(
            FILL_DATASETS,
            ("brumo25", "hmmt25", "aime24", "aime25", "aime26", "amc23"),
        )
        self.assertEqual(
            FILL_MODELS,
            (
                "r1_7b",
                "nemotron_8b",
                "qwen3_4b",
                "qwen3_8b",
                "r1_14b",
                "r1_1p5b",
                "r1_llama_8b",
                "r1_32b",
                "qwen3_30b_a3b",
                "qwen3_32b",
                "qwq_32b",
            ),
        )
        self.assertNotIn("qwen3_8b", QUEUE_MODELS)
        self.assertNotIn("qwen3_8b", FOLLOWON_MODELS)
        self.assertIn("qwq_32b", FILL_MODELS)
        self.assertIn("qwen3_32b", FILL_MODELS)
        self.assertEqual(MODELS["qwen3_8b"], "/mnt/d/lsj/models/Qwen3-8B")
        self.assertEqual(prereq_task("qwen3_8b", "brumo25", 42).phase, "followon_prereq")
        self.assertEqual(plws_task("qwen3_8b", "amc23", 7).phase, "followon_plws")

    def test_filter_fill_tasks_keeps_requested_ids(self) -> None:
        pending = [
            prereq_task("r1_1p5b", "math-500", 0),
            plws_task("r1_1p5b", "math-500", 42),
            prereq_task("r1_1p5b", "gpqa-diamond", 0),
            plws_task("r1_1p5b", "gpqa-diamond", 0),
        ]
        kept = filter_fill_tasks(
            pending,
            (
                "plws__r1_1p5b__math-500__s42",
                "prereq__r1_1p5b__gpqa-diamond__s0",
                "plws__r1_1p5b__gpqa-diamond__s0",
            ),
        )
        self.assertEqual(
            [task.task_id for task in kept],
            [
                "plws__r1_1p5b__gpqa-diamond__s0",
                "plws__r1_1p5b__math-500__s42",
                "prereq__r1_1p5b__gpqa-diamond__s0",
            ],
        )
        with self.assertRaisesRegex(ValueError, "unknown fill task ids"):
            filter_fill_tasks(pending, ["plws__r1_1p5b__olympiadbench__s42"])

    def test_fullcot_only_keeps_prereq_and_needs_answers(self) -> None:
        pending = [
            prereq_task("qwq_32b", "math-500", 0),
            plws_task("qwq_32b", "math-500", 0),
            prereq_task("qwq_32b", "gpqa-diamond", 1),
        ]
        kept = fullcot_only_tasks(pending)
        self.assertEqual(
            [task.task_id for task in kept],
            [
                "prereq__qwq_32b__math-500__s0",
                "prereq__qwq_32b__gpqa-diamond__s1",
            ],
        )
        self.assertEqual(
            fill_fullcot_complete(self.paths, "qwq_32b", "math-500", 0),
            (False, "fullcot answers missing"),
        )
        self._sample_ready("qwq_32b", "math-500", 0)
        self.assertEqual(
            fill_fullcot_complete(self.paths, "qwq_32b", "math-500", 0),
            (False, "fullcot answers missing"),
        )
        self._write(
            self.paths.root / "samples/qwq_32b/math-500/seed_0/answers.json",
            [{"question_idx": 0}],
        )
        self.assertEqual(
            fill_fullcot_complete(self.paths, "qwq_32b", "math-500", 0),
            (True, "fullcot ready"),
        )

    def test_fill_adds_r1_distill_followons_after_14b(self) -> None:
        from plws.contest import gpu_count

        self.assertEqual(
            MODELS["r1_1p5b"], "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-1.5B"
        )
        self.assertEqual(
            MODELS["r1_llama_8b"], "/mnt/d/lsj/models/DeepSeek-R1-Distill-Llama-8B"
        )
        self.assertEqual(
            MODELS["r1_32b"], "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B"
        )
        self.assertEqual(gpu_count("r1_1p5b"), 1)
        self.assertEqual(gpu_count("r1_llama_8b"), 1)
        self.assertEqual(gpu_count("r1_32b"), 2)
        self.assertEqual(prereq_task("r1_1p5b", "brumo25", 42).phase, "followon_prereq")
        self.assertEqual(prereq_task("r1_llama_8b", "hmmt25", 0).phase, "followon_prereq")
        self.assertEqual(prereq_task("r1_32b", "aime24", 7).gpu_count, 2)
        self.assertLess(FILL_MODELS.index("r1_14b"), FILL_MODELS.index("r1_1p5b"))
        self.assertLess(FILL_MODELS.index("r1_32b"), FILL_MODELS.index("qwen3_30b_a3b"))

    def test_fill_dispatch_runs_single_and_dual_lanes_together(self) -> None:
        self._jobs("r1_7b", "brumo25", 42)
        pending = [
            prereq_task("qwen3_30b_a3b", "amc23", 42),
            plws_task("r1_7b", "brumo25", 42),
            prereq_task("r1_7b", "hmmt25", 0),
        ]
        self.assertEqual(
            contest_blocked_phases(pending),
            frozenset({"followon_prereq", "followon_plws"}),
        )
        taken = fill_dispatch(pending, 3, self.paths, pool_size=3)
        self.assertEqual(
            [task.task_id for task in taken],
            [
                "prereq__qwen3_30b_a3b__amc23__s42",
                "prereq__r1_7b__hmmt25__s0",
            ],
        )
        self.assertEqual(
            [task.task_id for task in select_cold_starts(taken, any_loading=False)],
            ["prereq__qwen3_30b_a3b__amc23__s42"],
        )
        self.assertEqual(select_cold_starts(taken, any_loading=True), [])
        self.assertFalse(
            engine_ready_for_next_cold_start(workers_in_d=False, log_loaded=False)
        )
        self.assertFalse(
            engine_ready_for_next_cold_start(workers_in_d=True, log_loaded=True)
        )
        self.assertTrue(
            engine_ready_for_next_cold_start(workers_in_d=False, log_loaded=True)
        )
        stale = (
            "# 2026-09-12T15:30:00 start gpus=0,1\nModel loaded.\n"
            "# 2026-09-12T18:05:20 start gpus=0,1\nLoading model from scratch...\n"
        )
        self.assertFalse(log_loaded_after_latest_start(stale))
        self.assertTrue(
            log_loaded_after_latest_start(stale + "Model loaded.\n")
        )
        taken = fill_dispatch(pending, 2, self.paths, pool_size=3)
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__qwen3_30b_a3b__amc23__s42"],
        )
        self.assertEqual(dual_lane_cap(2), 1)
        self.assertEqual(dual_lane_cap(4), 2)
        only_twos = [
            prereq_task("qwen3_30b_a3b", "amc23", 42),
            prereq_task("qwen3_30b_a3b", "aime24", 0),
        ]
        self.assertEqual(
            fill_dispatch(
                only_twos,
                2,
                self.paths,
                running_gpu_counts=(2,),
                pool_size=2,
            ),
            [],
        )
        taken = fill_dispatch(
            pending,
            1,
            self.paths,
            running_gpu_counts=(1, 1),
            pool_size=3,
        )
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__r1_7b__hmmt25__s0"],
        )
        taken = fill_dispatch(
            pending,
            1,
            self.paths,
            running_gpu_counts=(2,),
            pool_size=3,
        )
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__r1_7b__hmmt25__s0"],
        )
        singles = [
            plws_task("r1_7b", "brumo25", 42),
            prereq_task("r1_7b", "hmmt25", 0),
            prereq_task("r1_1p5b", "brumo25", 7),
        ]
        taken = fill_dispatch(singles, 3, self.paths, pool_size=3)
        self.assertEqual(
            [task.task_id for task in taken],
            [
                "prereq__r1_7b__hmmt25__s0",
                "prereq__r1_1p5b__brumo25__s7",
                "plws__r1_7b__brumo25__s42",
            ],
        )

    def test_fill_dispatch_keeps_four_cards_on_long_samples(self) -> None:
        self.assertLess(
            fill_sample_dataset_rank("olympiadbench"),
            fill_sample_dataset_rank("math-500"),
        )
        self._sample_ready("r1_7b", "math-500", 0)
        self._write(
            self.paths.root / "samples/r1_7b/math-500/seed_0/answers.json",
            [{"question_idx": 0}],
        )
        self._puma_ready("r1_7b", "math-500", 0)
        self._jobs("r1_7b", "math-500", 0)
        pending = [
            plws_task("r1_7b", "math-500", 0),
            prereq_task("r1_7b", "math-500", 0),
            prereq_task("r1_7b", "olympiadbench", 42),
            prereq_task("r1_7b", "amc23", 42),
        ]
        self.assertTrue(fill_needs_fullcot_sample(self.paths, pending[2]))
        self.assertFalse(fill_needs_fullcot_sample(self.paths, pending[1]))
        taken = fill_dispatch(pending, 1, self.paths, pool_size=4)
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__r1_7b__olympiadbench__s42"],
        )
        taken = fill_dispatch(pending, 4, self.paths, pool_size=4)
        self.assertEqual(
            [task.task_id for task in taken],
            [
                "prereq__r1_7b__olympiadbench__s42",
                "prereq__r1_7b__amc23__s42",
                "prereq__r1_7b__math-500__s0",
                "plws__r1_7b__math-500__s0",
            ],
        )
        held = fill_dispatch(
            pending,
            1,
            self.paths,
            pool_size=4,
            running_tasks=[prereq_task("r1_7b", "olympiadbench", 0)],
        )
        self.assertEqual(
            [task.task_id for task in held],
            ["prereq__r1_7b__olympiadbench__s42"],
        )
        held = fill_dispatch(
            [pending[0], pending[1]],
            1,
            self.paths,
            pool_size=4,
            running_tasks=[prereq_task("r1_7b", "olympiadbench", 0)],
        )
        self.assertEqual(
            [task.task_id for task in held],
            ["prereq__r1_7b__math-500__s0"],
        )
        other_dataset = fill_dispatch(
            [
                prereq_task("r1_llama_8b", "amc23", 42),
                prereq_task("r1_llama_8b", "math-500", 42),
            ],
            2,
            self.paths,
            pool_size=5,
            running_tasks=[prereq_task("r1_llama_8b", "olympiadbench", 42)],
        )
        self.assertEqual(
            [task.task_id for task in other_dataset],
            [
                "prereq__r1_llama_8b__math-500__s42",
                "prereq__r1_llama_8b__amc23__s42",
            ],
        )

    def test_non_olympiad_32b_plws_and_unlocker_are_top_priority(self) -> None:
        self._jobs("qwen3_32b", "aime25", 0)
        pending = [
            prereq_task("qwq_32b", "olympiadbench", 42),
            plws_task("qwen3_32b", "aime25", 0),
            prereq_task("qwen3_32b", "math-500", 42),
        ]
        taken = fill_dispatch(
            pending,
            2,
            self.paths,
            pool_size=4,
            running_tasks=[prereq_task("qwq_32b", "olympiadbench", 0)],
        )
        self.assertEqual(
            [task.task_id for task in taken],
            ["plws__qwen3_32b__aime25__s0"],
        )
        taken = fill_dispatch(
            [pending[0], pending[2]],
            2,
            self.paths,
            pool_size=4,
            running_tasks=[prereq_task("qwq_32b", "olympiadbench", 0)],
        )
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__qwen3_32b__math-500__s42"],
        )
        self._jobs("qwen3_32b", "olympiadbench", 42)
        taken = fill_dispatch(
            [
                prereq_task("qwq_32b", "olympiadbench", 42),
                plws_task("qwen3_32b", "olympiadbench", 42),
            ],
            2,
            self.paths,
            pool_size=4,
        )
        self.assertEqual(
            [task.task_id for task in taken],
            ["prereq__qwq_32b__olympiadbench__s42"],
        )

    def test_fill_skips_complete_and_published_7b_amc23(self) -> None:
        protocol = protocol_for("r1_7b")
        self._sample_ready("r1_7b", "aime24", 42)
        self._puma_ready("r1_7b", "aime24", 42)
        self._jobs("r1_7b", "aime24", 42, 1)
        score = self.paths.score_dir(
            "r1_7b", "aime24", 42, FIRSTWIN, k=4, lexicon="core"
        )
        self._write(
            score / "shard_0.jsonl",
            [
                {
                    "uid": "r1_7b:aime24:42:firstwin:0",
                    "status": "ok",
                    "gt": "42",
                    "gold_error": "",
                    "protocol_id": protocol.protocol_id,
                    "max_model_len": protocol.max_model_len,
                    "truncated_answer_fix_tokens": protocol.answer_fix_tokens,
                    "fullcot_generation_tokens": protocol.generation_tokens,
                }
            ],
        )
        legacy = (
            self.paths.results
            / "archive"
            / "legacy_layout"
            / "plws"
            / "leftover_suppress_toend"
            / "r1_7b_s42_suppress"
        )
        self._write(
            legacy / "scores_shard0.jsonl",
            [
                {
                    "uid": "r1_7b:amc23:42:0",
                    "dataset": "amc23",
                    "status": "ok",
                }
            ],
        )
        self.assertTrue(fill_legacy_plws_complete(self.paths, "r1_7b", "amc23", 42)[0])
        self.assertFalse(fill_needs_prereq(self.paths, "r1_7b", "amc23", 42))
        tasks = build_fill_tasks(
            self.paths, models=("r1_7b",), datasets=("aime24", "amc23")
        )
        open_ids = [
            task.task_id
            for task in tasks
            if not fill_task_complete(self.paths, task)[0]
        ]
        self.assertNotIn("plws__r1_7b__aime24__s42", open_ids)
        self.assertNotIn("prereq__r1_7b__aime24__s42", open_ids)
        self.assertNotIn("plws__r1_7b__amc23__s42", open_ids)
        self.assertNotIn("prereq__r1_7b__amc23__s42", open_ids)
        self.assertEqual(open_ids[0], "plws__r1_7b__aime24__s0")
        self.assertIn("plws__r1_7b__amc23__s7", open_ids)
        self.assertIn("prereq__r1_7b__amc23__s7", open_ids)
        self.assertTrue(fill_task_complete(self.paths, plws_task("r1_7b", "aime24", 42))[0])
        self.assertTrue(fill_task_complete(self.paths, plws_task("r1_7b", "amc23", 42))[0])

    def test_frozen_answers_do_not_force_prereq(self) -> None:
        self._write(
            self.paths.root / "samples" / "r1_7b" / "aime24" / "seed_42" / "answers.json",
            [{"question_idx": 0}],
        )
        self._write(
            sample_meta_path(self.paths, "r1_7b", "aime24", 42),
            {"protocol_id": "old-host", "max_tokens": 8192},
        )
        self._puma_ready("r1_7b", "aime24", 42)
        self._jobs("r1_7b", "aime24", 42)
        self.assertFalse(may_sample_fullcot(self.paths, "r1_7b", "aime24", 42))
        self.assertFalse(fill_needs_prereq(self.paths, "r1_7b", "aime24", 42))

    def test_unverified_puma_forces_prereq_even_when_jobs_exist(self) -> None:
        self._sample_ready("qwen3_32b", "math-500", 42)
        self._puma_ready("qwen3_32b", "math-500", 42, verified=False)
        self._jobs("qwen3_32b", "math-500", 42)
        self.assertTrue(
            fill_needs_prereq(self.paths, "qwen3_32b", "math-500", 42)
        )

    def test_old_leftover_row_without_grading_evidence_is_not_reusable(self) -> None:
        protocol = protocol_for("qwen3_32b")
        old = {
            "uid": "qwen3_32b:math-500:42:firstwin:0",
            "status": "ok",
            "protocol_id": protocol.protocol_id,
            "max_model_len": protocol.max_model_len,
            "truncated_answer_fix_tokens": protocol.answer_fix_tokens,
            "fullcot_generation_tokens": protocol.generation_tokens,
        }
        self.assertFalse(contest_reusable_score(old, protocol))
        old.update({"gt": "42", "gold_error": ""})
        self.assertTrue(contest_reusable_score(old, protocol))

    def test_idle_pool_stays_on_zero_to_three(self) -> None:
        idle = idle_contest_gpus(
            ["0", "1", "2", "3"],
            leftover=(),
            used_mib={"0": 41000, "1": 41000, "2": 41000, "3": 14},
        )
        self.assertEqual(idle, ["3"])
        idle = idle_contest_gpus(
            ["0", "1", "2", "3"],
            leftover=(),
            used_mib={"0": 10, "1": 10, "2": 10, "3": 10, "4": 10},
        )
        self.assertEqual(idle, ["0", "1", "2", "3"])

    def test_fill_runner_is_isolated(self) -> None:
        text = (ROOT / "scripts" / "run_contest_fill_queue.py").read_text()
        self.assertIn("contest_fill", text)
        self.assertIn("FILL_DATASETS", text)
        self.assertIn("fill_dispatch", text)
        self.assertIn("running_gpu_counts", text)
        self.assertIn("pool_size", text)
        self.assertNotIn("score_leftover_suppress.py", text)
        self.assertNotIn("contest_blocked_phases", text)
        self.assertNotIn("run_matrix_plws_cell.sh", text)
        self.assertIn("--datasets", text)
        self.assertIn("--fullcot-only", text)
        self.assertIn("run_deer_official.sh", text)
        self.assertIn("PUMA_ROOT", text)
        self.assertNotIn("FORBIDDEN_GPUS", text)
        fill = (ROOT / "scripts" / "run_contest_fill_queue.py").read_text()
        self.assertIn('DEFAULT_GPUS = ("0", "1", "2", "3")', fill)
        self.assertNotIn("adopted_with_workers=True", fill)
        self.assertNotIn("if workers:\n        item.loaded = True", fill)
        from plws.contest import is_vllm_loading_comm

        self.assertTrue(is_vllm_loading_comm("VLLM::Worker_TP"))
        self.assertTrue(is_vllm_loading_comm("VLLM::EngineCor"))
        self.assertFalse(is_vllm_loading_comm("python"))
        prereq = (ROOT / "scripts" / "run_contest_prereq_cell.sh").read_text()
        self.assertIn("plws_model_path", prereq)
        self.assertIn("plws_align_conf", prereq)
        self.assertIn("CONTEST_RUN_ROOT", prereq)
        self.assertIn("FULLCOT_ONLY", prereq)
        self.assertIn("verify_puma_grades", prereq)
        self.assertIn('VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"', prereq)
        puma = (ROOT / "scripts" / "run_puma_official.sh").read_text()
        self.assertIn("verify_statistics", puma)
        self.assertIn("-m plws.puma_grading", puma)
        self.assertIn("fullcot_barrier", puma)
        self.assertIn('rm -f "$FULLCOT_BARRIER"', puma)
        self.assertIn("release_fullcot_barrier", prereq)
        self.assertIn("run_puma_official_or_barrier", prereq)
        self.assertIn("stop_after_fullcot", prereq)
        self.assertIn("plant_fullcot_barriers", fill)
        self.assertIn("running_tasks", fill)
        self.assertIn("--repair-existing-grades", fill)
        self.assertIn("repair_existing_plws_grades", fill)
        dense = (ROOT / "scripts" / "run_dense_trials_model.sh").read_text()
        self.assertIn(
            'VLLM_GPU_MEMORY_UTILIZATION="${DENSE_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"',
            dense,
        )
        leftover = (ROOT / "scripts" / "score_leftover_suppress.py").read_text()
        self.assertIn("DENSE_VLLM_GPU_MEMORY_UTILIZATION", fill)
        self.assertIn(
            'env["DENSE_VLLM_GPU_MEMORY_UTILIZATION"] = env.get(',
            fill,
        )
        self.assertIn('os.environ.get("VLLM_MAX_NUM_SEQS")', leftover)
        self.assertIn('os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS")', leftover)
        puma_patch = (ROOT / "vendor" / "patches" / "puma-fullcot-32k-v2.patch").read_text()
        self.assertIn('os.environ.get("VLLM_MAX_NUM_SEQS")', puma_patch)
        self.assertIn('os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS")', puma_patch)
        self.assertTrue(
            is_contest_fill_queue_cmd(
                ["python", "/repo/scripts/run_contest_fill_queue.py", "--gpus", "0,1,2,3"]
            )
        )
        self.assertFalse(
            is_contest_fill_queue_cmd(
                ["python", "/repo/scripts/run_contest_fill_queue.py", "--dry-run"]
            )
        )

    def test_fill_deer_waits_for_leftover_and_uses_official_ids(self) -> None:
        self.assertEqual(
            LOCAL_FILL_MODELS,
            (
                "r1_7b",
                "nemotron_8b",
                "qwen3_4b",
                "qwen3_8b",
                "r1_14b",
                "r1_1p5b",
                "r1_llama_8b",
            ),
        )
        self.assertTrue(
            {"r1_32b", "qwen3_30b_a3b", "qwen3_32b", "qwq_32b"}.isdisjoint(
                LOCAL_FILL_MODELS
            )
        )
        deer = deer_task("qwen3_8b", "hmmt25", 42)
        self.assertEqual(deer.task_id, "deer__qwen3_8b__hmmt25__s42")
        self.assertEqual(deer.phase, "followon_deer")
        self.assertEqual(deer.kind, "deer")
        self.assertFalse(contest_task_ready(self.paths, deer))
        ok, reason = fill_task_complete(self.paths, deer)
        self.assertFalse(ok)
        self.assertIn("waiting leftover", reason)

        self._jobs("qwen3_8b", "hmmt25", 42, count=1)
        protocol = protocol_for("qwen3_8b")
        leftover = self.paths.score_dir(
            "qwen3_8b", "hmmt25", 42, FIRSTWIN, k=4, lexicon="core"
        )
        self._write(
            leftover / "shard_0.jsonl",
            [
                {
                    "uid": "qwen3_8b:hmmt25:42:firstwin:0",
                    "dataset": "hmmt25",
                    "status": "ok",
                    "gt": "42",
                    "gold_error": "",
                    "protocol_id": protocol.protocol_id,
                    "max_model_len": protocol.max_model_len,
                    "truncated_answer_fix_tokens": protocol.answer_fix_tokens,
                    "fullcot_generation_tokens": protocol.generation_tokens,
                }
            ],
        )
        self.assertFalse(fill_needs_prereq(self.paths, "qwen3_8b", "hmmt25", 42))
        self.assertTrue(contest_task_ready(self.paths, deer))
        ok, reason = fill_task_complete(self.paths, deer)
        self.assertFalse(ok)
        self.assertNotIn("waiting leftover", reason)

        leftover_task = plws_task("r1_llama_8b", "amc23", 42)
        self._jobs("r1_llama_8b", "amc23", 42, count=1)
        taken = fill_dispatch([deer, leftover_task], 1, self.paths)
        self.assertEqual([task.task_id for task in taken], [leftover_task.task_id])
        sample = prereq_task("r1_llama_8b", "amc23", 1)
        taken = fill_dispatch([deer, sample], 1, self.paths)
        self.assertTrue(fill_needs_fullcot_sample(self.paths, sample))
        self.assertEqual([task.task_id for task in taken], [sample.task_id])

        tasks = build_fill_tasks(
            self.paths,
            models=("qwen3_8b",),
            datasets=("hmmt25",),
            seeds=(42,),
        )
        self.assertEqual(
            [task.task_id for task in tasks],
            [
                "plws__qwen3_8b__hmmt25__s42",
                "deer__qwen3_8b__hmmt25__s42",
            ],
        )


class GpuLeaseReleaseTest(unittest.TestCase):
    def test_puma_trial_shutdown_does_not_free_cards(self) -> None:
        text = (
            "# 2026-09-15 start gpus=3,4\n"
            "Model loaded.\n"
            "vLLM generation complete, releasing engine before CPU postprocess\n"
        )
        self.assertFalse(log_shows_gpu_released(text))
        self.assertFalse(
            should_release_gpu_lease(
                ("3", "4"),
                log_text=text,
                has_workers=False,
                used_mib={"3": 14, "4": 14},
            )
        )

    def test_dense_start_without_its_own_shutdown_keeps_lease(self) -> None:
        text = (
            "# 2026-09-15 start gpus=3,4\n"
            "vLLM generation complete, releasing engine before CPU postprocess\n"
            "[dense-model] GPUS=3,4 shard=0 TP=2\n"
            "Running generation...\n"
        )
        self.assertFalse(log_shows_gpu_released(text))

    def test_dense_cpu_postprocess_releases_empty_cards(self) -> None:
        text = (
            "# 2026-09-15 start gpus=3,4\n"
            "[dense-model] GPUS=3,4 shard=0 TP=2\n"
            "vLLM generation complete, releasing engine before CPU postprocess\n"
        )
        self.assertTrue(log_shows_gpu_released(text))
        self.assertTrue(
            should_release_gpu_lease(
                ("3", "4"),
                log_text=text,
                has_workers=False,
                used_mib={"3": 14, "4": 14},
            )
        )
        self.assertFalse(
            should_release_gpu_lease(
                ("3", "4"),
                log_text=text,
                has_workers=True,
                used_mib={"3": 14, "4": 14},
            )
        )
        self.assertFalse(
            should_release_gpu_lease(
                ("3", "4"),
                log_text=text,
                has_workers=False,
                used_mib={"3": 41000, "4": 14},
            )
        )

    def test_old_attempt_release_is_ignored(self) -> None:
        text = (
            "# 2026-09-14 start gpus=3,4\n"
            "[dense-model] GPU phase done\n"
            "# 2026-09-15 start gpus=3,4\n"
            "Running generation...\n"
        )
        self.assertFalse(log_shows_gpu_released(text))

    def test_long_dense_log_still_releases(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dense.attempt_1.log"
            path.write_text(
                "# 2026-09-15 start gpus=3,4\n"
                "[dense-model] GPUS=3,4 shard=0 TP=2\n"
                + ("processed prompt\n" * 20000)
                + "vLLM generation complete, releasing engine before CPU postprocess\n",
                encoding="utf-8",
            )
            self.assertTrue(log_path_shows_gpu_released(path))
            self.assertTrue(
                should_release_gpu_lease(
                    ("3", "4"),
                    log_path=path,
                    has_workers=False,
                    used_mib={"3": 14, "4": 14},
                )
            )

if __name__ == "__main__":
    unittest.main()
