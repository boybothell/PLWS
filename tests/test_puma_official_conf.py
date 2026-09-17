from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plws.puma_official_conf import (
    conf_usable,
    pick_official_conf,
    render_local_conf,
    require_embedding_model,
)


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class PumaOfficialConfTests(unittest.TestCase):
    def test_pick_uses_align_conf_on_clean_rental_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ae = Path(raw) / "plws"
            puma = Path(raw) / "PUMA"
            official = _write(
                puma / "configs" / "DS-32B.conf",
                'SIMILARITY_THRESHOLD=0.35\nEMBEDDING_MODEL="hf/id"\n',
            )
            picked = pick_official_conf(
                ae, puma, "qwq_32b", "hmmt25", "DS-32B.conf"
            )
            self.assertEqual(picked, official)

    def test_pick_prefers_same_model_seed42_template(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ae = Path(raw) / "plws"
            puma = Path(raw) / "PUMA"
            _write(
                puma / "configs" / "DS-32B.conf",
                "SIMILARITY_THRESHOLD=0.35\n",
            )
            existing = _write(
                ae
                / "results/baselines/puma/puma_offline_qwq_32b/hmmt25/_local.conf",
                "SIMILARITY_THRESHOLD=0.35\nSEED=42\n",
            )
            picked = pick_official_conf(
                ae, puma, "qwq_32b", "hmmt25", "DS-32B.conf"
            )
            self.assertEqual(picked, existing)

    def test_pick_errors_instead_of_missing_r1_7b_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ae = Path(raw) / "plws"
            puma = Path(raw) / "PUMA"
            with self.assertRaises(FileNotFoundError) as ctx:
                pick_official_conf(
                    ae, puma, "qwq_32b", "aime25", "DS-32B.conf"
                )
            self.assertIn("ALIGN_CONF=DS-32B.conf", str(ctx.exception))
            self.assertFalse(
                (
                    ae
                    / "results/baselines/puma/puma_offline_r1_7b"
                    / "gpqa-diamond/_DS-7B.local.conf"
                ).exists()
            )

    def test_conf_usable_requires_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "empty.conf"
            path.write_text("SEED=42\n", encoding="utf-8")
            self.assertFalse(conf_usable(path))
            path.write_text("SIMILARITY_THRESHOLD=0.35\n", encoding="utf-8")
            self.assertTrue(conf_usable(path))

    def test_render_rewrites_embedding_and_appends_seed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = _write(
                Path(raw) / "DS-32B.conf",
                'SIMILARITY_THRESHOLD=0.35\nEMBEDDING_MODEL="hf/id"\nSEED=42\n',
            )
            embed = Path(raw) / "embed"
            _write(embed / "config.json", "{}")
            text = render_local_conf(src, seed=0, embed_path=embed)
            self.assertIn(f'EMBEDDING_MODEL="{embed}"', text)
            self.assertTrue(text.endswith("SEED=0\n"))
            self.assertNotIn("hf/id", text)

    def test_require_embedding_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaises(FileNotFoundError):
                require_embedding_model(root)
            _write(root / "qwen3-embedding-redundancy-detector-0.6B" / "config.json", "{}")
            self.assertTrue(
                (require_embedding_model(root) / "config.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
