"""Tests for lazy, continuous multi-file DAS reads."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

from dasgauge.io import DASRecording


def _make_part(path, data, start, fs=4.0, dx=1.0):
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("Acquisition")
        raw = acquisition.create_group("Raw[0]")
        dataset = raw.create_dataset("RawData", data=np.asarray(data))
        acquisition.attrs["SpatialSamplingInterval"] = dx
        acquisition.attrs["MeasurementStartTime"] = start.isoformat()
        raw.attrs["OutputDataRate"] = fs
        dataset.attrs["PartStartTime"] = start.isoformat()
        dataset.attrs["PartEndTime"] = (
            start + timedelta(seconds=(len(data) - 1) / fs)
        ).isoformat()


class DASRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def test_orders_by_metadata_and_reads_across_boundary(self):
        first = np.arange(5 * 4, dtype=np.int16).reshape(5, 4)
        second = 100 + np.arange(6 * 4, dtype=np.int16).reshape(6, 4)
        late = self.root / "a_late.h5"
        early = self.root / "z_early.h5"
        _make_part(early, first, self.start)
        _make_part(late, second, self.start + timedelta(seconds=5 / 4))

        recording = DASRecording([late, early])

        self.assertEqual([part.path.name for part in recording.parts], [early.name, late.name])
        self.assertEqual(recording.n_samples, 11)
        expected = np.concatenate((first, second), axis=0)[3:9, 1:4].T
        np.testing.assert_array_equal(recording.read(3, 9, 1, 4), expected)
        self.assertEqual(len(recording.source_spans(3, 9)), 2)

    def test_rejects_temporal_gap(self):
        data = np.zeros((4, 3), dtype=np.int16)
        first = self.root / "first.h5"
        second = self.root / "second.h5"
        _make_part(first, data, self.start)
        _make_part(second, data, self.start + timedelta(seconds=2))

        with self.assertRaisesRegex(ValueError, "gap"):
            DASRecording([first, second])

    def test_validates_channel_and_time_bounds(self):
        path = self.root / "only.h5"
        _make_part(path, np.zeros((5, 3), dtype=np.int16), self.start)
        recording = DASRecording([path])
        with self.assertRaisesRegex(ValueError, "channel_start"):
            recording.read(0, 2, 2, 4)
        with self.assertRaisesRegex(ValueError, "start < stop"):
            recording.read(2, 2, 0, 2)


if __name__ == "__main__":
    unittest.main()
