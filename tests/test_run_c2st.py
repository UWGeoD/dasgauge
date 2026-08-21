"""Tests for the single-directory C2ST command."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_c2st


class RunC2STTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sample_dir = Path(self.temp.name)
        (self.sample_dir / "samples.h5").touch()
        (self.sample_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
        feature_dir = self.sample_dir / "features"
        feature_dir.mkdir()
        rng = np.random.default_rng(4)
        coupling = rng.normal(size=(20, 5))
        uncoupling = rng.normal(loc=0.5, size=(20, 5))
        np.savez_compressed(
            feature_dir / "summary_v1.npz", coupling=coupling, uncoupling=uncoupling
        )

    def test_explicit_methods_write_one_compatible_report_each(self):
        methods = ["original_binomial_c2st", "paired_mcnemar"]
        exit_code = run_c2st.main(
            [
                str(self.sample_dir),
                "--representation", "summary_v1",
                "--methods", *methods,
                "--repeats", "2",
                "--k", "1",
            ]
        )
        self.assertEqual(exit_code, 0)
        reports = []
        for method in methods:
            paths = list((self.sample_dir / "results").glob(f"*__{method}__*.json"))
            self.assertEqual(len(paths), 1)
            reports.append(json.loads(paths[0].read_text(encoding="utf-8")))
        self.assertEqual([report["method"] for report in reports], methods)
        self.assertTrue(all(report["summary"]["repeat_count"] == 2 for report in reports))
        self.assertEqual(
            reports[0]["trials"][0]["assignment_seed"],
            reports[1]["trials"][0]["assignment_seed"],
        )

    def test_legacy_calibration_still_uses_legacy_filename(self):
        run_c2st.main(
            [
                str(self.sample_dir),
                "--representation", "summary_v1",
                "--calibration", "binomial",
                "--repeats", "1",
                "--k", "1",
            ]
        )
        paths = list((self.sample_dir / "results").glob("*__binomial__*.json"))
        self.assertEqual(len(paths), 1)
        report = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(report["method"], "pair_preserving_binomial")


if __name__ == "__main__":
    unittest.main()
