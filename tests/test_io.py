"""Tests for dasgauge.io HDF5 readers using synthetic files."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dasgauge.io import DAS, _decode_attr, _parse_start_time_attr


def _make_optasense(path, n_ch=8, n_t=64, fs=1000.0, dx=1.02):
    data_time_loci = np.arange(n_ch * n_t, dtype=np.float32).reshape(n_t, n_ch)
    with h5py.File(path, "w") as f:
        acq = f.create_group("Acquisition")
        raw0 = acq.create_group("Raw[0]")
        ds = raw0.create_dataset("RawData", data=data_time_loci)
        acq.attrs["SpatialSamplingInterval"] = dx
        acq.attrs["MeasurementStartTime"] = b"2024-05-01T12:00:00.123456789Z"
        acq.attrs["VendorCode"] = b"OptaSense"
        acq.attrs["GaugeLength"] = 10.0
        raw0.attrs["OutputDataRate"] = fs
        raw0.attrs["RawDataUnit"] = b"strain-rate"
        raw0.attrs["RawDescription"] = b"synthetic"
        ds.attrs["PartStartTime"] = b"2024-05-01T12:00:00Z"
        ds.attrs["PartEndTime"] = b"2024-05-01T12:00:01Z"
    return data_time_loci


def _make_silixa(path, n_ch=8, n_t=64, fs=1000.0, dx=1.021):
    data_time_loci = np.arange(n_ch * n_t, dtype=np.float32).reshape(n_t, n_ch)
    with h5py.File(path, "w") as f:
        acq = f.create_group("DasMetadata/Interrogator/Acquisition")
        # Silixa stores these as strings; the reader must coerce them.
        acq.attrs["AcquisitionSampleRate"] = str(int(fs))
        acq.attrs["SpatialSamplingInterval"] = str(dx)
        acq.attrs["GaugeLength"] = "10.0"
        acq.attrs["AcquisitionStartTime"] = b"2024-05-01T12:00:00Z"
        f["DasMetadata/Interrogator"].attrs["InterrogatorManufacturer"] = b"Silixa"
        raw = f.create_group("DasRawData")
        raw.create_dataset("RawData", data=data_time_loci)
        ta = raw.create_dataset("DasTimeArray", data=np.arange(n_t, dtype=np.float64))
        ta.attrs["StartTime"] = b"2024-05-01T12:00:00Z"
        ta.attrs["EndTime"] = b"2024-05-01T12:00:01Z"
    return data_time_loci


def _make_raw(path, key="strainrate", n_ch=8, n_t=64):
    data = np.arange(n_ch * n_t, dtype=np.float32).reshape(n_ch, n_t)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=data)
    return data


class ReaderTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class OptaSenseReaderTests(ReaderTestCase):
    def test_optasense_reader(self):
        p = self.tmp_path / "opta.h5"
        raw = _make_optasense(p)

        das = DAS(str(p), vendor="OptaSense")

        # stored (time, loci) must be transposed to (channels, time)
        np.testing.assert_array_equal(das.data, raw.T)
        self.assertEqual(das.data.shape, (8, 64))
        self.assertEqual(das.meta["fs"], 1000.0)
        self.assertAlmostEqual(das.meta["dx"], 1.02)
        self.assertAlmostEqual(das.meta["dt"], 1e-3)
        self.assertEqual(das.meta["vendor"], "OptaSense")
        self.assertEqual(das.meta["gauge_length_m"], 10.0)
        self.assertEqual(das.meta["raw_path"], "Acquisition/Raw[0]/RawData")
        self.assertEqual(das.meta["raw_shape"], (64, 8))
        # >6-digit fractional seconds and trailing Z must parse
        self.assertEqual(das.meta["start_time_dt"].year, 2024)
        self.assertTrue(das.meta["start_time_iso"].startswith("2024-05-01T12:00:00.123456"))

    def test_optasense_channel_selection(self):
        p = self.tmp_path / "opta.h5"
        raw = _make_optasense(p)

        das = DAS(str(p), select_channels=[0, 2, 4], vendor="OptaSense")

        self.assertEqual(das.data.shape, (3, 64))
        np.testing.assert_array_equal(das.data, raw.T[[0, 2, 4], :])


class SilixaReaderTests(ReaderTestCase):
    def test_silixa_reader(self):
        p = self.tmp_path / "silixa.h5"
        raw = _make_silixa(p)

        das = DAS(str(p), vendor="Silixa")

        np.testing.assert_array_equal(das.data, raw.T)
        # string-valued attributes must be coerced to float
        self.assertEqual(das.meta["fs"], 1000.0)
        self.assertAlmostEqual(das.meta["dx"], 1.021)
        self.assertEqual(das.meta["vendor"], "Silixa")
        self.assertEqual(das.meta["raw_path"], "DasRawData/RawData")
        self.assertIsNotNone(das.meta["part_start_time_dt"])
        self.assertIsNotNone(das.meta["part_end_time_dt"])


class RawHdf5ReaderTests(ReaderTestCase):
    def test_raw_hdf5_reader(self):
        p = self.tmp_path / "raw.h5"
        data = _make_raw(p)

        das = DAS(str(p), vendor="raw_hdf5", fs=500.0, dx=2.0)

        np.testing.assert_array_equal(das.data, data)
        self.assertEqual(das.meta["fs"], 500.0)
        self.assertEqual(das.meta["dx"], 2.0)
        self.assertAlmostEqual(das.meta["dt"], 1 / 500.0)
        self.assertEqual(das.meta["vendor"], "raw_hdf5")
        # no timestamps are available in a bare HDF5 file
        self.assertIsNone(das.meta["start_time_dt"])
        self.assertIsNone(das.meta["start_time_iso"])

    def test_raw_hdf5_custom_key_and_missing_key(self):
        p = self.tmp_path / "raw.h5"
        _make_raw(p, key="data")

        das = DAS(str(p), vendor="raw_hdf5", fs=100.0, dx=1.0, dataset_key="data")
        self.assertEqual(das.data.shape, (8, 64))

        with self.assertRaisesRegex(KeyError, "strainrate"):
            DAS(str(p), vendor="raw_hdf5", fs=100.0, dx=1.0)

    def test_raw_hdf5_rejects_non_2d(self):
        p = self.tmp_path / "raw3d.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("strainrate", data=np.zeros((2, 3, 4), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "2-D"):
            DAS(str(p), vendor="raw_hdf5", fs=100.0, dx=1.0)

    def test_unknown_vendor(self):
        p = self.tmp_path / "raw.h5"
        _make_raw(p)
        with self.assertRaisesRegex(ValueError, "Unknown vendor"):
            DAS(str(p), vendor="NotAVendor")


class AttributeHelperTests(unittest.TestCase):
    def test_decode_attr(self):
        self.assertEqual(_decode_attr(b"abc"), "abc")
        self.assertEqual(_decode_attr("abc"), "abc")
        self.assertEqual(_decode_attr(3.0), 3.0)

    def test_parse_start_time_variants(self):
        for raw in (
            b"2024-05-01T12:00:00Z",
            "2024-05-01T12:00:00+00:00",
            "2024-05-01T12:00:00.123456789Z",
            np.datetime64("2024-05-01T12:00:00"),
            1714564800.0,          # seconds
            1714564800_000.0,      # milliseconds
            1714564800_000000.0,   # microseconds
        ):
            with self.subTest(raw=raw):
                dt = _parse_start_time_attr(raw)
                self.assertIsNotNone(dt)
                self.assertIsNotNone(dt.tzinfo)
                self.assertEqual((dt.year, dt.month, dt.day), (2024, 5, 1))

    def test_parse_start_time_none_and_garbage(self):
        self.assertIsNone(_parse_start_time_attr(None))
        self.assertIsNone(_parse_start_time_attr(b"not-a-time"))


if __name__ == "__main__":
    unittest.main()
