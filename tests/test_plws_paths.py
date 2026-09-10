from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plws.artifacts import atomic_write_json, atomic_write_jsonl, done_uids
from plws.paths import PLWSPaths, datasets_for_jobs, repository_root


class RepositoryRootTest(unittest.TestCase):
    def test_environment_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "configured"
            with patch.dict(os.environ, {"PLWS_ROOT": str(configured)}):
                self.assertEqual(repository_root("/does/not/matter"), configured.resolve())

    def test_discovers_root_from_script(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src" / "plws").mkdir(parents=True)
            script = root / "scripts" / "task.py"
            script.parent.mkdir()
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(repository_root(script), root.resolve())


class PLWSPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = PLWSPaths(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_canonical_window_paths(self) -> None:
        jobs = self.paths.jobs_path(
            "r1_7b", "math-500", 42, "high", k=4, lexicon="core"
        )
        scores = self.paths.score_dir(
            "r1_7b", "math-500", 42, "low", k=4, lexicon="core"
        )
        prefix = (
            self.root.resolve()
            / "results/runs/plws/window_first/k_4/lexicon_core"
        )
        cell = prefix / "r1_7b/math-500/seed_42"
        self.assertEqual(jobs, cell / "jobs/high.jsonl")
        self.assertEqual(scores, cell / "scores/low")
        self.assertEqual(
            self.paths.score_path(
                "r1_7b",
                "math-500",
                42,
                "suppress",
                "low",
                3,
                k=4,
            ),
            cell / "scores/low/shard_3.jsonl",
        )

    def test_jobs_fall_back_to_legacy_but_prefer_canonical(self) -> None:
        legacy = self.paths.legacy_jobs_path("r1_7b", 42, "low")
        legacy.parent.mkdir(parents=True)
        legacy.write_text("{}\n")
        self.assertEqual(
            self.paths.resolve_jobs_path(
                "r1_7b", "math-500", 42, "low", k=4
            ),
            legacy,
        )

        canonical = self.paths.jobs_path(
            "r1_7b", "math-500", 42, "low", k=4
        )
        canonical.parent.mkdir(parents=True)
        canonical.write_text("{}\n")
        self.assertEqual(
            self.paths.resolve_jobs_path(
                "r1_7b", "math-500", 42, "low", k=4
            ),
            canonical,
        )

    def test_k_ablation_reads_old_partial_scores(self) -> None:
        legacy = self.paths.legacy_score_dirs(
            "r1_7b",
            42,
            "suppress",
            "high",
            k=2,
            lexicon="core",
        )
        legacy = legacy[0]
        atomic_write_jsonl(
            legacy / "scores_shard0.jsonl",
            [{"uid": "done", "status": "ok"}, {"uid": "retry", "status": "error"}],
        )
        migrated = (
            self.root
            / "results/runs/plws/window_first/k_2/lexicon_core/"
            "r1_7b/math-500/seed_42/scores/high"
        )
        atomic_write_jsonl(
            migrated / "shard_7.jsonl",
            [{"uid": "migrated", "status": "ok"}],
        )
        directories = self.paths.score_read_dirs(
            "r1_7b",
            "math-500",
            42,
            "suppress",
            "high",
            k=2,
            lexicon="core",
        )
        self.assertEqual(done_uids(directories), {"done", "migrated"})
        self.assertEqual(
            directories[0],
            self.paths.score_dir(
                "r1_7b",
                "math-500",
                42,
                "high",
                k=2,
                lexicon="core",
            ),
        )

    def test_safe_lexicon_does_not_reuse_core_legacy_scores(self) -> None:
        self.assertEqual(
            self.paths.legacy_score_dirs(
                "r1_7b", 42, "suppress", "low", k=4, lexicon="safe"
            ),
            (),
        )

    def test_atomic_json_accepts_manifest_and_list_artifact(self) -> None:
        manifest = self.root / "manifest.json"
        artifact = self.root / "rows.json"
        atomic_write_json(manifest, {"state": "succeeded"})
        atomic_write_json(artifact, [{"uid": "a"}])
        self.assertEqual(json.loads(manifest.read_text())["state"], "succeeded")
        self.assertEqual(json.loads(artifact.read_text()), [{"uid": "a"}])

    def test_mixed_jobs_require_explicit_output(self) -> None:
        rows = [{"dataset": "math-500"}, {"dataset": "aime24"}]
        with self.assertRaisesRegex(ValueError, "mixed-dataset"):
            datasets_for_jobs(rows)
        self.assertEqual(
            datasets_for_jobs(rows, allow_mixed=True),
            ("aime24", "math-500"),
        )
        self.assertEqual(
            datasets_for_jobs(rows, requested="aime24"),
            ("aime24",),
        )

    def test_seed42_flat_dense_is_not_reused_for_other_seeds(self) -> None:
        flat = (
            self.root
            / "results/upstream/dense_trials/dense_G_qwen3_8b/math-500"
            / "dense_puma/trial_answers.json"
        )
        flat.parent.mkdir(parents=True)
        flat.write_text("[]\n")
        self.assertEqual(self.paths.dense_trial_path("qwen3_8b", "math-500", 42), flat)
        other = self.paths.dense_trial_path("qwen3_8b", "math-500", 0)
        self.assertNotEqual(other, flat)
        self.assertFalse(other.is_file())
        self.assertEqual(other.name, "trial_answers.json")
        self.assertIn("seed_0", str(other))


if __name__ == "__main__":
    unittest.main()
