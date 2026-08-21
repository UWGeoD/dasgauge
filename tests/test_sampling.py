"""Tests for sample strategies and HDF5 materialization."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

from dasgauge.io import DASRecording
from dasgauge.sampling import hold_skip_windows, split_recording


def _make_part(path, data, start, fs=4.0):
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("Acquisition")
        raw = acquisition.create_group("Raw[0]")
        dataset = raw.create_dataset("RawData", data=np.asarray(data))
        acquisition.attrs["SpatialSamplingInterval"] = 1.0
        acquisition.attrs["MeasurementStartTime"] = start.isoformat()
        raw.attrs["OutputDataRate"] = fs
        dataset.attrs["PartStartTime"] = start.isoformat()


class StrategyTests(unittest.TestCase):
    def test_hold_skip_windows(self):
        windows = hold_skip_windows(43, fs=4, s1=2, s2=1)
        self.assertEqual(
            [(item.start_sample, item.stop_sample) for item in windows],
            [(0, 8), (12, 20), (24, 32)],
        )

    def test_zero_skip_is_allowed(self):
        windows = hold_skip_windows(12, fs=2, s1=2, s2=0)
        self.assertEqual(len(windows), 3)


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        first = np.arange(5 * 6, dtype=np.int16).reshape(5, 6)
        second = 100 + np.arange(10 * 6, dtype=np.int16).reshape(10, 6)
        first_path = self.root / "first.h5"
        second_path = self.root / "second.h5"
        _make_part(first_path, first, start)
        _make_part(second_path, second, start + timedelta(seconds=5 / 4))
        self.full = np.concatenate((first, second), axis=0)
        self.recording = DASRecording([second_path, first_path])

    def test_writes_paired_sample_major_hdf5(self):
        output = self.root / "run"
        manifest = split_recording(
            self.recording,
            output,
            dataset_id="synthetic",
            strategy="hold_skip",
            parameters={"s1": 1.0, "s2": 0.5},
            coupling_channels=(0, 2),
            uncoupling_channels=(4, 6),
        )

        self.assertEqual(manifest["samples"]["count"], 2)
        with h5py.File(output / "samples.h5", "r") as handle:
            coupling = handle["samples/coupling"]
            uncoupling = handle["samples/uncoupling"]
            self.assertEqual(coupling.shape, (2, 2, 4))
            self.assertEqual(coupling.dtype, np.dtype("int16"))
            np.testing.assert_array_equal(coupling[0], self.full[0:4, 0:2].T)
            # The first window crosses the 5-sample source boundary only when
            # read with a wider range; the second stored window is in part two.
            np.testing.assert_array_equal(uncoupling[1], self.full[6:10, 4:6].T)
            self.assertEqual(coupling.compression, "lzf")
        self.assertTrue((output / "index.csv").is_file())
        loaded = json.loads((output / "manifest.json").read_text())
        self.assertEqual(loaded["channels"]["coupling"], [0, 2])

    def test_applies_configured_preprocessing(self):
        output = self.root / "processed"
        manifest = split_recording(
            self.recording,
            output,
            dataset_id="synthetic",
            strategy="hold_skip",
            parameters={"s1": 1.0, "s2": 0.5},
            coupling_channels=(0, 2),
            uncoupling_channels=(4, 6),
            preprocess_steps=[("downsample", {"target_fs": 2.0})],
        )
        with h5py.File(output / "samples.h5", "r") as handle:
            self.assertEqual(handle["samples/coupling"].shape, (2, 2, 2))
        self.assertEqual(manifest["samples"]["output_fs_hz"], 2.0)

    def test_refuses_overwrite(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            split_recording(
                self.recording,
                output,
                dataset_id="synthetic",
                strategy="hold_skip",
                parameters={"s1": 1.0, "s2": 0.5},
                coupling_channels=(0, 2),
                uncoupling_channels=(4, 6),
            )

    def test_overwrite_replaces_existing_run(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "old-marker.txt").write_text("old")

        manifest = split_recording(
            self.recording,
            output,
            dataset_id="synthetic",
            strategy="hold_skip",
            parameters={"s1": 1.0, "s2": 0.5},
            coupling_channels=(0, 2),
            uncoupling_channels=(4, 6),
            overwrite=True,
        )

        self.assertEqual(manifest["samples"]["count"], 2)
        self.assertFalse((output / "old-marker.txt").exists())
        self.assertTrue((output / "samples.h5").is_file())
        self.assertEqual(list(self.root.glob(".existing.backup-*")), [])


if __name__ == "__main__":
    unittest.main()
