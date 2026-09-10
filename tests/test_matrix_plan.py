from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plws.matrix import (
    DEER_DEFERRED,
    DEER_EXCLUDED,
    EIGHT_MODEL,
    FIRSTWIN,
    FIVE_MODELS,
    FROZEN_FULLCOT,
    TIER_KINDS,
    THIRTY_MODEL,
    blockers_idle,
    build_tasks,
    deer_complete,
    deer_expected_policy,
    dense_complete,
    ensure_firstwin_jobs,
    first_open_phase,
    gpu_count,
    deer_task,
    dispatch_blocked_phases,
    fill_dispatch,
    next_dispatchable,
    plws_task,
    task_ready,
    import_aime4s_core,
    jobs_present,
    cells_needing_cpu_export,
    needs_cpu_export,
    needs_prereq,
    plws_complete,
    prereq_task,
    shard_count,
    summarize,
    task_complete,
)
from plws.paths import PLWSPaths
from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)


class MatrixPlanTest(unittest.TestCase):
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

    def _jobs(self, model: str, dataset: str, seed: int, counts: dict[str, int]) -> None:
        rows = []
        for kind, count in counts.items():
            rows.extend(
                {
                    "uid": f"{model}:{dataset}:{seed}:{kind}:{index}",
                    "dataset": dataset,
                    "kind": kind,
                }
                for index in range(count)
            )
        self._write(
            self.paths.jobs_path(
                model, dataset, seed, FIRSTWIN, k=4, lexicon="core"
            ),
            rows,
        )

    def test_frozen_models_never_need_gpu_prereq(self) -> None:
        for model in FROZEN_FULLCOT:
            self.assertFalse(needs_prereq(self.paths, model, "math-500", 42))

    def test_qwen_missing_dense_needs_prereq(self) -> None:
        self.assertTrue(needs_prereq(self.paths, "qwen3_8b", "olympiadbench", 0))
        self._puma_ready("qwen3_4b", "olympiadbench", 0)
        self.assertTrue(needs_prereq(self.paths, "qwen3_4b", "olympiadbench", 0))
        self._dense_ready("qwen3_4b", "olympiadbench", 0)
        self.assertTrue(dense_complete(self.paths, "qwen3_4b", "olympiadbench", 0))
        self.assertFalse(needs_prereq(self.paths, "qwen3_4b", "olympiadbench", 0))

    def test_dense_complete_does_not_need_g(self) -> None:
        self._dense_ready("qwen3_4b", "olympiadbench", 0)
        self.assertTrue(dense_complete(self.paths, "qwen3_4b", "olympiadbench", 0))
        self.assertFalse(self.paths.dense_g_path("qwen3_4b", "olympiadbench", 0).is_file())

    def test_existing_jobs_skip_prereq_only_if_puma_ready(self) -> None:
        self._jobs("qwen3_8b", "math-500", 42, {"low": 1, "mix": 0, "high": 0})
        self.assertTrue(jobs_present(self.paths, "qwen3_8b", "math-500", 42))
        self.assertTrue(needs_prereq(self.paths, "qwen3_8b", "math-500", 42))
        self._puma_ready("qwen3_8b", "math-500", 42)
        self.assertFalse(needs_prereq(self.paths, "qwen3_8b", "math-500", 42))

    def test_tier_files_merge_into_firstwin(self) -> None:
        for kind in TIER_KINDS:
            self._write(
                self.paths.jobs_path(
                    "r1_7b", "aime24", 42, kind, k=4, lexicon="core"
                ),
                [
                    {
                        "uid": f"r1_7b:aime24:42:{kind}:0",
                        "dataset": "aime24",
                        "kind": kind,
                    }
                ],
            )
        self.assertTrue(jobs_present(self.paths, "r1_7b", "aime24", 42))
        self.assertTrue(ensure_firstwin_jobs(self.paths, "r1_7b", "aime24", 42))
        self.assertTrue(
            self.paths.jobs_path(
                "r1_7b", "aime24", 42, FIRSTWIN, k=4, lexicon="core"
            ).is_file()
        )
        self.assertFalse(ensure_firstwin_jobs(self.paths, "r1_7b", "aime24", 42))

    def test_nemotron_gpqa_deer_after_thirty(self) -> None:
        tasks = build_tasks(self.paths, include_thirty=True)
        late = [
            task
            for task in tasks
            if task.model == "nemotron_8b"
            and task.dataset == "gpqa-diamond"
            and task.kind == "deer"
        ]
        self.assertEqual(len(late), 4)
        self.assertTrue(all(task.phase == "late_deer" for task in late))
        self.assertTrue(all(task.gpu_count == 1 for task in late))
        self.assertIn(("nemotron_8b", "gpqa-diamond"), DEER_DEFERRED)
        self.assertEqual(DEER_EXCLUDED, frozenset())
        thirty_last = max(
            task.priority for task in tasks if task.phase.startswith("thirty_")
        )
        self.assertGreater(late[0].priority, thirty_last)
        blocked = dispatch_blocked_phases(
            [prereq_task("qwen3_30b_a3b", "math-500", 0)], ()
        )
        self.assertEqual(blocked, frozenset({"late_deer"}))
        self.assertEqual(
            fill_dispatch(late, 2, self.paths, blocked_phases=blocked),
            [],
        )
        self.assertEqual(
            len(fill_dispatch(late, 2, self.paths)),
            2,
        )

    def test_thirty_only_plan_is_puma_plws_deer(self) -> None:
        tasks = build_tasks(self.paths, models=(THIRTY_MODEL,))
        self.assertTrue(tasks)
        self.assertTrue(all(task.model == THIRTY_MODEL for task in tasks))
        self.assertTrue(all(task.gpu_count == 2 for task in tasks))
        self.assertEqual(
            {task.phase for task in tasks},
            {"thirty_prereq", "thirty_plws", "thirty_deer"},
        )
        self.assertTrue(any(task.kind == "prereq" for task in tasks))
        self.assertTrue(any(task.kind == "plws" for task in tasks))
        self.assertTrue(any(task.kind == "deer" for task in tasks))
        self.assertFalse(any(task.phase.startswith("eight_") for task in tasks))

    def test_thirty_is_after_five_and_uses_two_gpus(self) -> None:
        self.assertEqual(gpu_count(THIRTY_MODEL), 2)
        self.assertEqual(gpu_count("r1_14b"), 1)
        tasks = build_tasks(self.paths, include_thirty=True)
        five_last = max(
            (
                task.priority
                for task in tasks
                if task.phase.startswith("five_")
            ),
            default=-1,
        )
        eight_first = min(
            task.priority for task in tasks if task.phase.startswith("eight_")
        )
        thirty_first = min(
            task.priority for task in tasks if task.phase.startswith("thirty_")
        )
        self.assertLess(five_last, eight_first)
        self.assertLess(eight_first, thirty_first)
        self.assertTrue(all(task.phase.startswith("eight_") for task in tasks if task.model == EIGHT_MODEL))
        self.assertTrue(
            all(task.gpu_count == 2 for task in tasks if task.model == THIRTY_MODEL)
        )

    def test_large_datasets_are_two_shards(self) -> None:
        self.assertEqual(shard_count("olympiadbench"), 2)
        self.assertEqual(shard_count("aime24"), 1)
        tasks = build_tasks(self.paths, models=("r1_7b",))
        oly = [
            task
            for task in tasks
            if task.kind == "plws"
            and task.dataset == "olympiadbench"
            and task.seed == 42
        ]
        self.assertEqual(len(oly), 2)
        self.assertEqual(oly[0].run_kind, FIRSTWIN)
        self.assertTrue(all("__low" not in task.task_id for task in oly))

    def test_phase_barrier(self) -> None:
        tasks = [
            prereq_task("qwen3_4b", "math-500", 0),
            prereq_task("qwen3_8b", "math-500", 0),
            prereq_task("qwen3_30b_a3b", "math-500", 0),
        ]
        self.assertEqual(first_open_phase(tasks, {}), "five_prereq")
        self.assertEqual(
            first_open_phase(tasks, {tasks[0].task_id: "done"}),
            "eight_prereq",
        )
        self.assertEqual(
            first_open_phase(
                tasks, {tasks[0].task_id: "done", tasks[1].task_id: "done"}
            ),
            "thirty_prereq",
        )
        self.assertIsNone(
            first_open_phase(tasks, {task.task_id: "done" for task in tasks})
        )

    def test_dispatch_skips_busy_phase(self) -> None:
        eight = prereq_task("qwen3_8b", "math-500", 0)
        thirty = prereq_task("qwen3_30b_a3b", "math-500", 0)
        blocked = plws_task("qwen3_8b", "math-500", 0, 0, 2)
        self.assertFalse(task_ready(self.paths, blocked))
        self.assertEqual(
            next_dispatchable([blocked, thirty, eight], 1, self.paths),
            eight,
        )
        self.assertIsNone(next_dispatchable([blocked, thirty], 1, self.paths))
        self.assertEqual(
            next_dispatchable([blocked, thirty], 2, self.paths),
            thirty,
        )

    def test_dispatch_ignores_running_earlier_phase(self) -> None:
        pending = [
            plws_task("qwen3_8b", "math-500", 0, 0, 2),
            prereq_task("qwen3_8b", "math-500", 0),
            deer_task("qwen3_8b", "aime24", 0),
        ]
        taken = fill_dispatch(pending, 1, self.paths)
        self.assertEqual([task.task_id for task in taken], [pending[1].task_id])

    def test_dispatch_fills_four_gpus_with_ready_prereq(self) -> None:
        pending = [
            prereq_task("qwen3_8b", "math-500", seed)
            for seed in (0, 1, 42, 123)
        ]
        pending.extend(
            [
                plws_task("qwen3_8b", "math-500", 0, 0, 2),
                prereq_task("qwen3_30b_a3b", "math-500", 0),
            ]
        )
        taken = fill_dispatch(pending, 4, self.paths)
        self.assertEqual([task.seed for task in taken], [0, 1, 123, 42])
        self.assertTrue(all(task.kind == "prereq" for task in taken))

    def test_dispatch_uses_ready_plws_not_missing_jobs(self) -> None:
        self._jobs("qwen3_8b", "aime24", 1, {FIRSTWIN: 3})
        blocked = plws_task("qwen3_8b", "math-500", 0, 0, 2)
        ready = plws_task("qwen3_8b", "aime24", 1, 0, 1)
        deer = deer_task("qwen3_8b", "math-500", 0)
        taken = fill_dispatch([blocked, ready, deer], 1, self.paths)
        self.assertEqual([task.task_id for task in taken], [ready.task_id])

    def test_dispatch_waits_for_two_gpus_on_thirty(self) -> None:
        thirty = prereq_task("qwen3_30b_a3b", "math-500", 0)
        self.assertEqual(fill_dispatch([thirty], 1, self.paths), [])
        self.assertEqual(fill_dispatch([thirty], 2, self.paths), [thirty])

    def test_frozen_missing_jobs_is_fullcot_fallback(self) -> None:
        ok, reason = plws_complete(self.paths, "r1_7b", "math-500", 42)
        self.assertTrue(ok)
        self.assertIn("Full-CoT", reason)
        ok, reason = plws_complete(self.paths, "qwen3_8b", "math-500", 42)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing jobs")

    def test_empty_window_shard_is_complete(self) -> None:
        self._jobs("qwen3_4b", "aime24", 0, {"low": 0, "mix": 0, "high": 0})
        ok, reason = plws_complete(self.paths, "qwen3_4b", "aime24", 0)
        self.assertTrue(ok)
        self.assertEqual(reason, "empty window shard")

    def test_plws_scores_must_cover_jobs(self) -> None:
        self._jobs("qwen3_4b", "aime24", 0, {"low": 2, "mix": 0, "high": 0})
        ok, _ = plws_complete(self.paths, "qwen3_4b", "aime24", 0)
        self.assertFalse(ok)
        self._write(
            self.paths.score_dir(
                "qwen3_4b", "aime24", 0, FIRSTWIN, k=4, lexicon="core"
            )
            / "scores.jsonl",
            [
                {
                    "uid": "qwen3_4b:aime24:0:low:0",
                    "status": "ok",
                    "protocol_id": PROTOCOL_ID,
                    "max_model_len": MAX_MODEL_LEN,
                    "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                },
                {
                    "uid": "qwen3_4b:aime24:0:low:1",
                    "status": "too_long",
                    "protocol_id": PROTOCOL_ID,
                    "max_model_len": MAX_MODEL_LEN,
                    "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                },
            ],
        )
        ok, reason = plws_complete(self.paths, "qwen3_4b", "aime24", 0)
        self.assertTrue(ok)
        self.assertEqual(reason, "done 2/2")

    def test_old_protocol_scores_do_not_complete_plws(self) -> None:
        self._jobs("nemotron_8b", "aime24", 0, {"low": 1, "mix": 0, "high": 0})
        self._write(
            self.paths.score_dir(
                "nemotron_8b", "aime24", 0, FIRSTWIN, k=4, lexicon="core"
            )
            / "shard_0.jsonl",
            [
                {
                    "uid": "nemotron_8b:aime24:0:low:0",
                    "status": "ok",
                    "protocol_id": "old-leftover",
                    "max_model_len": 35840,
                    "truncated_answer_fix_tokens": 1024,
                }
            ],
        )
        ok, reason = plws_complete(self.paths, "nemotron_8b", "aime24", 0)
        self.assertFalse(ok)
        self.assertEqual(reason, "done 0/1")

    def test_deer_family_policy(self) -> None:
        self.assertEqual(
            deer_expected_policy("r1_14b")["think_ratio"],
            0.6,
        )
        self.assertEqual(deer_expected_policy("qwen3_8b")["confidence_policy"], "avg2")
        self.assertTrue(deer_expected_policy("qwen3_8b")["require_probe_think_close"])

    def test_deer_complete_gate(self) -> None:
        data = (
            self.root.parent
            / "PUMA"
            / "data"
            / "aime24_test.jsonl"
        )
        self._write(data, [{"q": 1}, {"q": 2}])
        output = (
            self.paths.results
            / "baselines"
            / "deer"
            / "puma_fullcot_32k_v2"
            / "qwen3_4b"
            / "aime24"
            / "seed_42"
        )
        rows = [
            {
                "protocol_id": PROTOCOL_ID,
                "generated_text": "ok",
                "deer_family_policy": "qwen3",
            }
        ]
        self._write(output / "deer.jsonl", rows)
        self._write(
            output / "manifest.json",
            {
                "method": "deer",
                "protocol_id": PROTOCOL_ID,
                "model_tag": "qwen3_4b",
                "dataset": "aime24",
                "seed": 42,
                "fullcot_generation_tokens": FULLCOT_GENERATION_TOKENS,
                "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                "max_model_len": MAX_MODEL_LEN,
                "deer": deer_expected_policy("qwen3_4b"),
            },
        )
        ok, _ = deer_complete(self.paths, "qwen3_4b", "aime24", 42)
        self.assertFalse(ok)
        rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "generated_text": "ok2",
                "deer_family_policy": "qwen3",
            }
        )
        self._write(output / "deer.jsonl", rows)
        ok, reason = deer_complete(self.paths, "qwen3_4b", "aime24", 42)
        self.assertTrue(ok)
        self.assertEqual(reason, "2 records")

    def test_deer_complete_accepts_seed_zero(self) -> None:
        data = (
            self.root.parent
            / "PUMA"
            / "data"
            / "aime24_test.jsonl"
        )
        self._write(data, [{"q": 1}, {"q": 2}])
        output = (
            self.paths.results
            / "baselines"
            / "deer"
            / "puma_fullcot_32k_v2"
            / "qwen3_4b"
            / "aime24"
            / "seed_0"
        )
        rows = [
            {
                "protocol_id": PROTOCOL_ID,
                "generated_text": "ok",
                "deer_family_policy": "qwen3",
            },
            {
                "protocol_id": PROTOCOL_ID,
                "generated_text": "ok2",
                "deer_family_policy": "qwen3",
            },
        ]
        self._write(output / "deer.jsonl", rows)
        manifest = {
            "method": "deer",
            "protocol_id": PROTOCOL_ID,
            "model_tag": "qwen3_4b",
            "dataset": "aime24",
            "seed": 0,
            "fullcot_generation_tokens": FULLCOT_GENERATION_TOKENS,
            "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "deer": deer_expected_policy("qwen3_4b"),
        }
        self._write(output / "manifest.json", manifest)
        ok, reason = deer_complete(self.paths, "qwen3_4b", "aime24", 0)
        self.assertTrue(ok)
        self.assertEqual(reason, "2 records")
        del manifest["seed"]
        self._write(output / "manifest.json", manifest)
        ok, _ = deer_complete(self.paths, "qwen3_4b", "aime24", 0)
        self.assertFalse(ok)

    def test_aime4s_import_copies_only_v2_core(self) -> None:
        self._jobs("qwen3_4b", "aime24", 0, {"low": 2, "mix": 0, "high": 0})
        folder = (
            self.paths.results
            / "experiments"
            / "lexicon_ablation"
            / "aime4s"
            / "qwen3_4b"
            / "s0"
            / "core"
        )
        self._write(
            folder / "scores.jsonl",
            [
                {
                    "uid": "qwen3_4b:aime24:0:low:0",
                    "status": "ok",
                    "protocol_id": PROTOCOL_ID,
                    "max_model_len": MAX_MODEL_LEN,
                    "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                },
                {
                    "uid": "qwen3_4b:aime24:0:low:1",
                    "status": "ok",
                    "protocol_id": "old",
                    "max_model_len": MAX_MODEL_LEN,
                    "truncated_answer_fix_tokens": TRUNCATED_ANSWER_FIX_TOKENS,
                },
            ],
        )
        copied = import_aime4s_core(self.paths, "qwen3_4b", 0)
        self.assertEqual(copied["aime24:firstwin"], 1)
        self.assertTrue(
            (
                self.paths.score_dir(
                    "qwen3_4b", "aime24", 0, FIRSTWIN, k=4, lexicon="core"
                )
                / "shard_0.jsonl"
            ).is_file()
        )
        ok, reason = plws_complete(self.paths, "qwen3_4b", "aime24", 0)
        self.assertFalse(ok)
        self.assertEqual(reason, "done 1/2")

    def test_blockers_idle_without_status(self) -> None:
        idle, reason = blockers_idle(self.paths)
        self.assertTrue(idle)
        self.assertIn("idle", reason)
        status = (
            self.root
            / "results/experiments/lexicon_ablation/pre/queue/status.json"
        )
        self._write(
            status,
            {"counts": {"running": 1, "pending": 0}, "running": [], "pending": ["x"]},
        )
        idle, reason = blockers_idle(self.paths)
        self.assertFalse(idle)
        self.assertIn("pending=1", reason)

    def test_prereq_task_completes_after_jobs_appear(self) -> None:
        task = prereq_task("qwen3_8b", "math-500", 0)
        ok, _ = task_complete(self.paths, task)
        self.assertFalse(ok)
        self._jobs("qwen3_8b", "math-500", 0, {"low": 0})
        ok, reason = task_complete(self.paths, task)
        self.assertFalse(ok)
        self._puma_ready("qwen3_8b", "math-500", 0)
        ok, reason = task_complete(self.paths, task)
        self.assertTrue(ok)
        self.assertEqual(reason, "jobs ready")

    def test_prereq_releases_gpu_after_dense_without_jobs(self) -> None:
        task = prereq_task("qwen3_8b", "olympiadbench", 0)
        self._puma_ready("qwen3_8b", "olympiadbench", 0)
        self._dense_ready("qwen3_8b", "olympiadbench", 0)
        ok, reason = task_complete(self.paths, task)
        self.assertTrue(ok)
        self.assertIn(reason, {"prereq no longer required", "jobs ready"})
        self.assertFalse(jobs_present(self.paths, "qwen3_8b", "olympiadbench", 0))
        self.assertTrue(needs_cpu_export(self.paths, "qwen3_8b", "olympiadbench", 0))
        self.assertEqual(
            cells_needing_cpu_export(self.paths, ["qwen3_8b"]),
            [("qwen3_8b", "olympiadbench", 0)],
        )
        plws = plws_task("qwen3_8b", "olympiadbench", 0, 0, 2)
        self.assertFalse(task_ready(self.paths, plws))
        self._jobs("qwen3_8b", "olympiadbench", 0, {"low": 1})
        self.assertTrue(task_ready(self.paths, plws))
        self.assertEqual(cells_needing_cpu_export(self.paths, ["qwen3_8b"]), [])

    def test_summarize_counts(self) -> None:
        tasks = build_tasks(self.paths, include_thirty=False)
        counts = summarize(tasks)
        self.assertEqual(counts["total"], len(tasks))
        self.assertGreater(counts["five_plws"], 0)
        self.assertGreater(counts["five_deer"], 0)
        self.assertEqual(counts["thirty_deer"], 0)


if __name__ == "__main__":
    unittest.main()
