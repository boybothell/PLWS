import json
import tempfile
import unittest
from pathlib import Path

from plws.host_protocol import validate_fullcot_sample_meta
from plws.protocol import (
    MAX_MODEL_LEN,
    PROTOCOL_ID,
    puma_final_regeneration_tokens,
)


class HostProtocolTest(unittest.TestCase):
    def canonical_meta(self) -> dict[str, object]:
        return {
            "protocol_id": PROTOCOL_ID,
            "model_tag": "r1_7b",
            "dataset": "math-500",
            "seed": 42,
            "max_tokens": 32768,
            "answer_fix_max_tokens": 2048,
            "prompt_reserve_tokens": 3072,
            "max_model_len": MAX_MODEL_LEN,
            "prompt_version": "default",
        }

    def write_meta(self, payload: dict[str, object]) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "sample_meta.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(lambda: directory.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_accepts_canonical_v2_metadata(self) -> None:
        path = self.write_meta(self.canonical_meta())
        meta = validate_fullcot_sample_meta(
            path,
            model_tag="r1_7b",
            dataset="math-500",
            seed=42,
        )
        self.assertEqual(meta["max_model_len"], 37888)

    def test_rejects_legacy_context(self) -> None:
        payload = self.canonical_meta()
        payload["max_model_len"] = 35840
        path = self.write_meta(payload)
        with self.assertRaisesRegex(ValueError, "max_model_len=35840"):
            validate_fullcot_sample_meta(
                path,
                model_tag="r1_7b",
                dataset="math-500",
                seed=42,
            )

    def test_rejects_wrong_cell_identity(self) -> None:
        path = self.write_meta(self.canonical_meta())
        with self.assertRaisesRegex(ValueError, "dataset='math-500'"):
            validate_fullcot_sample_meta(
                path,
                model_tag="r1_7b",
                dataset="gpqa-diamond",
                seed=42,
            )

    def test_puma_regeneration_is_capped_by_remaining_host_budget(self) -> None:
        self.assertEqual(puma_final_regeneration_tokens(100), 4096)
        self.assertEqual(puma_final_regeneration_tokens(28672), 4096)
        self.assertEqual(puma_final_regeneration_tokens(32000), 768)
        self.assertEqual(puma_final_regeneration_tokens(32768), 0)

    def test_puma_regeneration_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefix_tokens"):
            puma_final_regeneration_tokens(-1)
        with self.assertRaisesRegex(ValueError, "requested_cap"):
            puma_final_regeneration_tokens(1, requested_cap=0)


if __name__ == "__main__":
    unittest.main()
