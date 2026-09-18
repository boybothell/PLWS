from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from plws.deploy import (
    configured_tensor_parallel,
    load_deployment_profile,
    profile_name,
)

ROOT = Path(__file__).resolve().parents[1]


class DeploymentProfileTests(unittest.TestCase):
    def test_local_profile_covers_all_models(self) -> None:
        profile = load_deployment_profile("local_48g", root=ROOT)
        self.assertEqual(profile.tp_for("r1_14b"), 1)
        self.assertEqual(profile.tp_for("r1_32b"), 2)
        self.assertEqual(len(profile.allowed_models), 11)

    def test_a800_profile_only_allows_four_large_models(self) -> None:
        profile = load_deployment_profile("a800_80g", root=ROOT)
        self.assertEqual(len(profile.allowed_models), 4)
        self.assertEqual(profile.tp_for("qwen3_30b_a3b"), 1)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            profile.tp_for("r1_14b")

    def test_environment_selects_profile(self) -> None:
        env = {
            "PLWS_DEPLOY_PROFILE": "a800_80g",
            "PLWS_LARGE_TP": "",
            "PLWS_TP": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(profile_name(), "a800_80g")
            self.assertEqual(
                configured_tensor_parallel("qwq_32b", root=ROOT),
                1,
            )

    def test_explicit_tp_override_is_validated(self) -> None:
        env = {
            "PLWS_DEPLOY_PROFILE": "a800_80g",
            "PLWS_LARGE_TP": "",
            "PLWS_TP": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                configured_tensor_parallel("r1_32b", root=ROOT),
                2,
            )
            with self.assertRaisesRegex(ValueError, "not allowed"):
                configured_tensor_parallel("r1_14b", root=ROOT)


if __name__ == "__main__":
    unittest.main()
