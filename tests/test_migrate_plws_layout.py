from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_plws_layout.py"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MigratePlwsLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.results = Path(self.temporary.name) / "results"
        self.results.mkdir()

        puma = self.results / "puma_offline_r1_7b"
        puma.mkdir()
        (puma / "answers.json").write_text('{"ok": true}\n', encoding="utf-8")
        deer = self.results / "math500_official" / "deer"
        deer.mkdir(parents=True)
        (deer / "scores.json").write_text("[]\n", encoding="utf-8")
        dense = self.results / "dense_G_r1_7b"
        dense.mkdir()
        (dense / "steps.json").write_text("[]\n", encoding="utf-8")

        base = {
            "model": "r1_7b",
            "dataset": "math-500",
            "seed": 42,
        }
        write_jsonl(
            self.results / "leftover_jump" / "r1_7b_s42" / "jobs.jsonl",
            [{"uid": "r1_7b:math-500:42:1", "kind": "low", **base}],
        )
        sidecar = self.results / "leftover_jump" / "r1_7b_s42" / "jobs_amcgsm.jsonl"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {"kind": "low", **base, "uid": "r1_7b:math-500:42:1"},
                sort_keys=False,
                separators=(", ", ": "),
            )
            + "\n",
            encoding="utf-8",
        )
        write_jsonl(
            self.results / "leftover_jump" / "r1_7b_s42" / "jobs_k2_high.jsonl",
            [{"uid": "r1_7b:math-500:42:2", "kind": "high", "k": 2, **base}],
        )
        write_jsonl(
            self.results
            / "leftover_suppress_toend"
            / "r1_7b_s42_suppress"
            / "scores_shard0.jsonl",
            [
                {"status": "ok", "uid": "r1_7b:math-500:42:1", "kind": "low"},
                {
                    "status": "retry",
                    "uid": "r1_7b:math-500:42:3",
                    "kind": "low",
                    "message": "status rows are data",
                },
            ],
        )
        write_jsonl(
            self.results
            / "leftover_suppress_kablate"
            / "k2"
            / "r1_7b_s42_suppress_high"
            / "scores_shard0.jsonl",
            [{"status": "ok", "uid": "r1_7b:math-500:42:2", "kind": "high"}],
        )
        write_jsonl(
            self.results
            / "leftover_suppress_toend"
            / "r1_7b_s42_free"
            / "scores_shard0.jsonl",
            [
                {
                    "status": "ok",
                    "mode": "free",
                    "uid": "r1_7b:math-500:42:4",
                    "kind": "low",
                }
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--results-root", str(self.results), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_is_default_and_writes_nothing(self) -> None:
        before = {
            path: digest(path)
            for path in self.results.rglob("*")
            if path.is_file()
        }
        completed = self.run_script()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["mode"], "dry-run")
        accounting = report["row_accounting"]
        self.assertEqual(accounting["input"], {"jobs": 3, "scores": 4, "total": 7})
        self.assertEqual(
            accounting["unique_output"], {"jobs": 2, "scores": 4, "total": 6}
        )
        self.assertEqual(
            accounting["identical_duplicates"],
            {"jobs": 1, "scores": 0, "total": 1},
        )
        self.assertTrue(accounting["matches"])
        self.assertEqual(report["duplicate_rows"]["jobs"], 1)
        self.assertEqual(len(report["duplicate_sources"]), 1)
        self.assertTrue(
            report["duplicate_sources"][0]["duplicate_source"].endswith(
                "jobs_amcgsm.jsonl"
            )
        )
        self.assertFalse((self.results / "runs").exists())
        self.assertFalse((self.results / "upstream").exists())
        self.assertFalse((self.results / "migration_report.json").exists())
        after = {path: digest(path) for path in before}
        self.assertEqual(before, after)

    def test_apply_splits_rows_preserves_status_and_is_idempotent(self) -> None:
        legacy = [
            path
            for path in self.results.rglob("*")
            if path.is_file()
        ]
        before = {path: digest(path) for path in legacy}

        first = self.run_script("--apply")
        self.assertEqual(first.returncode, 0, first.stderr)
        k4 = (
            self.results
            / "runs/plws/window_first/k_4/lexicon_core/r1_7b/math-500/seed_42"
        )
        k2 = (
            self.results
            / "runs/plws/window_first/k_2/lexicon_core/r1_7b/math-500/seed_42"
        )
        score_rows = [
            json.loads(line)
            for line in (k4 / "scores/low/shard_0.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([row["status"] for row in score_rows], ["ok", "retry"])
        job_rows = (k4 / "jobs/low.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(job_rows), 1)
        self.assertEqual(
            len(
                (k4 / "scores/low/shard_free_0.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            1,
        )
        self.assertEqual(
            len((k2 / "scores/high/shard_0.jsonl").read_text(encoding="utf-8").splitlines()),
            1,
        )
        self.assertTrue((k4 / "jobs/high.jsonl").is_file())
        self.assertEqual((k4 / "jobs/high.jsonl").stat().st_size, 0)
        self.assertTrue((k4 / "manifest.json").is_file())
        self.assertTrue((k4 / "status.json").is_file())
        self.assertTrue((k4 / "metrics").is_dir())

        puma_link = self.results / "upstream/puma_official/puma_offline_r1_7b"
        self.assertTrue(puma_link.is_symlink())
        puma_manifest = json.loads(
            (self.results / "upstream/puma_official/manifest.json").read_text(encoding="utf-8")
        )
        deer_manifest = json.loads(
            (self.results / "upstream/deer/manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(puma_manifest["protected"])
        self.assertTrue(deer_manifest["protected"])
        self.assertIn("sha256", puma_manifest["legacy_sources"][0]["files"][0])
        migration_report = json.loads(
            (self.results / "migration_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(migration_report["duplicate_rows"]["jobs"], 1)
        self.assertEqual(len(migration_report["duplicate_sources"]), 1)
        self.assertEqual(before, {path: digest(path) for path in legacy})

        second = self.run_script("--apply")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout)["actions"]["create"], 0)

    def test_conflicting_sidecar_duplicate_uid_is_a_hard_error(self) -> None:
        write_jsonl(
            self.results / "leftover_jump" / "r1_7b_s42" / "jobs_oly.jsonl",
            [
                {
                    "uid": "r1_7b:math-500:42:1",
                    "kind": "low",
                    "model": "r1_7b",
                    "dataset": "math-500",
                    "seed": 42,
                    "question": "different semantic content",
                }
            ],
        )
        completed = self.run_script()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("conflicting jobs rows for uid", completed.stderr)
        self.assertFalse((self.results / "runs").exists())

    def test_duplicate_score_uid_in_one_target_is_a_hard_error(self) -> None:
        write_jsonl(
            self.results
            / "leftover_suppress_toend"
            / "r1_7b_s42_suppress"
            / "scores_shard0.jsonl",
            [
                {"status": "ok", "uid": "r1_7b:math-500:42:1", "kind": "low"},
                {"status": "ok", "uid": "r1_7b:math-500:42:1", "kind": "low"},
            ],
        )
        completed = self.run_script()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicate score uid", completed.stderr)
        self.assertFalse((self.results / "runs").exists())

    def test_refuses_inconsistent_existing_output(self) -> None:
        target = (
            self.results
            / "runs/plws/window_first/k_4/lexicon_core/r1_7b/math-500/seed_42/jobs"
        )
        target.mkdir(parents=True)
        (target / "low.jsonl").write_text('{"wrong": true}\n', encoding="utf-8")
        completed = self.run_script("--apply")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("refusing to overwrite inconsistent", completed.stderr)
        self.assertFalse((self.results / "migration_report.json").exists())

    def test_unknown_dataset_status_row_is_a_hard_error(self) -> None:
        write_jsonl(
            self.results
            / "leftover_suppress_toend"
            / "r1_7b_s42_suppress"
            / "scores_shard9.jsonl",
            [{"status": "failed", "message": "dataset unavailable"}],
        )
        completed = self.run_script()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("cannot determine dataset", completed.stderr)
        self.assertFalse((self.results / "runs").exists())


if __name__ == "__main__":
    unittest.main()
