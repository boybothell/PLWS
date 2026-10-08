import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from plws.flashinfer_guard import apply_flashinfer_guard, isolate_flashinfer_cache


class FlashinferGuardTest(unittest.TestCase):
    def test_disable_autotune_when_env_is_off(self) -> None:
        kwargs = {"model": "x"}
        with patch.dict(os.environ, {"VLLM_ENABLE_FLASHINFER_AUTOTUNE": "0"}, clear=False):
            os.environ.pop("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", None)
            apply_flashinfer_guard(kwargs)

        self.assertFalse(kwargs["enable_flashinfer_autotune"])

    def test_isolate_cache_under_provided_root(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": tmp,
                    "VLLM_ENABLE_FLASHINFER_AUTOTUNE": "0",
                },
                clear=False,
            ):
                kwargs: dict = {}
                apply_flashinfer_guard(kwargs)
                isolated = os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"]
                self.assertTrue(isolated.startswith(tmp))
                self.assertTrue(Path(isolated).is_dir())
                self.assertNotEqual(isolated, tmp)


if __name__ == "__main__":
    unittest.main()
