"""Tests for portable dataset configuration."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dasgauge.config import load_dataset_config


class DatasetConfigTests(unittest.TestCase):
    def _config(self, directory, recording_dir):
        path = Path(directory) / "dataset.yaml"
        path.write_text(
            "dataset_id: test\n"
            f"recording_dir: '{recording_dir}'\n"
            "vendor: optasense\n",
            encoding="utf-8",
        )
        return path

    def test_expands_recording_directory_environment_variable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._config(temporary, "${DASGAUGE_TEST_RECORDING}")
            with mock.patch.dict(
                "os.environ", {"DASGAUGE_TEST_RECORDING": "/data/recording"}
            ):
                config = load_dataset_config(path)
        self.assertEqual(config["recording_dir"], "/data/recording")

    def test_rejects_unresolved_recording_directory_variable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._config(temporary, "${DASGAUGE_MISSING_RECORDING}")
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "unresolved recording_dir"):
                    load_dataset_config(path)


if __name__ == "__main__":
    unittest.main()
