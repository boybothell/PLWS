from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plws.piece_text import (
    has_tokenizer_pieces,
    row_has_tokenizer_pieces,
    sanitize_row,
    sanitize_tokenizer_pieces,
)

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from repair_piece_encoded_samples import (  # noqa: E402
    quarantine_cell,
    rewrite_sample,
    scan_samples,
)
from plws.paths import PLWSPaths  # noqa: E402


class PieceTextTest(unittest.TestCase):
    def test_sanitize_restores_newlines_and_spaces(self) -> None:
        raw = "ĊOkay,ĠsoĠWait.ĊĊTheĠanswerĠis"
        self.assertTrue(has_tokenizer_pieces(raw))
        fixed = sanitize_tokenizer_pieces(raw)
        self.assertEqual(fixed, "\nOkay, so Wait.\n\nThe answer is")
        self.assertFalse(has_tokenizer_pieces(fixed))

    def test_clean_text_untouched(self) -> None:
        text = "Okay, so Wait.\n\nThe answer is"
        self.assertFalse(has_tokenizer_pieces(text))
        self.assertEqual(sanitize_tokenizer_pieces(text), text)

    def test_rewrite_sample_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "plws"
            paths = PLWSPaths(root)
            sample = (
                root / "samples" / "r1_llama_8b" / "aime24" / "seed_42" / "answers.json"
            )
            sample.parent.mkdir(parents=True)
            sample.write_text(
                json.dumps(
                    [
                        {
                            "generated_text": "ĊOkay,Ġboxed",
                            "reasoning": "ĊOkay",
                            "raw_response": "Ġ204",
                            "reasoning_steps": ["ĊOkay"],
                        }
                    ]
                )
                + "\n"
            )
            puma = paths.puma_output_dir("r1_llama_8b", "aime24", 42)
            puma.mkdir(parents=True)
            (puma / "statistics.json").write_text("[]\n")
            jobs = paths.jobs_path(
                "r1_llama_8b", "aime24", 42, "firstwin", k=4, lexicon="core"
            )
            jobs.parent.mkdir(parents=True)
            jobs.write_text("\n")
            hits = scan_samples(root)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["model"], "r1_llama_8b")
            self.assertTrue(rewrite_sample(sample))
            rows = json.loads(sample.read_text())
            self.assertFalse(row_has_tokenizer_pieces(rows[0]))
            self.assertIn("Okay, boxed", rows[0]["generated_text"])
            self.assertTrue(sample.with_name("answers.piece.json").is_file())
            quarantine_cell(paths, "r1_llama_8b", "aime24", 42)
            self.assertFalse(puma.exists())
            self.assertFalse(jobs.exists())

if __name__ == "__main__":
    unittest.main()
