import sys
import unittest
from pathlib import Path

PUMA = Path(__file__).resolve().parents[2] / "PUMA"
sys.path.insert(0, str(PUMA))

from baselines.deer.canonical_protocol import (  # noqa: E402
    CANONICAL_MAX_MODEL_LEN,
    PROTOCOL_ID,
    answer_budget,
    policy_for_model,
    probe_gate_passed,
)


class DeerProtocolTest(unittest.TestCase):
    def test_protocol_identity(self) -> None:
        self.assertEqual(PROTOCOL_ID, "puma-fullcot-32k-v2")
        self.assertEqual(CANONICAL_MAX_MODEL_LEN, 37888)

    def test_natural_or_conf_exit_uses_main_remainder(self) -> None:
        self.assertEqual(
            answer_budget(
                delivered_thinking_tokens=10_000,
                host_answer_fix=False,
                context_room=30_000,
            ),
            (22_768, 22_768),
        )

    def test_truncation_uses_separate_answer_fix(self) -> None:
        self.assertEqual(
            answer_budget(
                delivered_thinking_tokens=32_772,
                host_answer_fix=True,
                context_room=2_048,
            ),
            (2_048, 2_048),
        )

    def test_context_room_caps_fix_without_going_negative(self) -> None:
        self.assertEqual(
            answer_budget(
                delivered_thinking_tokens=32_768,
                host_answer_fix=True,
                context_room=300,
            ),
            (2_048, 300),
        )
        self.assertEqual(
            answer_budget(
                delivered_thinking_tokens=32_768,
                host_answer_fix=True,
                context_room=-1,
            ),
            (2_048, 0),
        )

    def test_standard_family_policy(self) -> None:
        policy = policy_for_model("DeepSeek-R1-Distill-Qwen-14B")
        self.assertEqual(policy.name, "standard")
        self.assertEqual(policy.think_ratio, 0.6)
        self.assertEqual(policy.confidence_policy, "avg1")
        self.assertFalse(policy.require_probe_think_close)

    def test_qwen3_family_policy(self) -> None:
        policy = policy_for_model("Qwen3-8B")
        self.assertEqual(policy.name, "qwen3")
        self.assertEqual(policy.think_ratio, 0.8)
        self.assertEqual(policy.confidence_policy, "avg2")
        self.assertTrue(policy.require_probe_think_close)
        self.assertFalse(
            probe_gate_passed(policy, probe_think_closed=False)
        )
        self.assertTrue(
            probe_gate_passed(policy, probe_think_closed=True)
        )

    def test_method_limit_keeps_main_answer_remainder(self) -> None:
        self.assertEqual(
            answer_budget(
                delivered_thinking_tokens=19_660,
                host_answer_fix=False,
                context_room=20_000,
            ),
            (13_108, 13_108),
        )

    def test_deepseek_qwen3_architecture_uses_qwen3_policy(self) -> None:
        policy = policy_for_model("DeepSeek-R1-0528-Qwen3-8B")
        self.assertEqual(policy.name, "qwen3")


if __name__ == "__main__":
    unittest.main()
