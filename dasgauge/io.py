"""
Multi-vendor DAS HDF5 readers.

Selected functionality adapted from UWGeoD/DAS_Preprocessing,
source commit 896e00005b68c7878ae80d50833bd9eeabe7ebc7, original file `DAS.py`
(helpers at lines 7-94, class ``DAS`` at lines 98-266).

The code may have been reorganized for dasgauge. See PROVENANCE.md for exact
source ranges, imported symbols, licensing, and modifications.

Original work Copyright (c) 2025 GeoD Research Group, MIT License.
"""

import numpy as np
import h5py
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = ["DAS", "DASRecording", "RecordingPart"]


def _decode_attr(val):
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    return val


def _parse_start_time_attr(raw):
    """
    Parse common DAS time forms into tz-aware UTC datetime:
      - ISO 8601 strings/bytes, possibly with 'Z' and >6-digit fractional seconds
      - numeric epochs (s/ms/us/ns)
      - numpy.datetime64
    Returns datetime|None (UTC).
    """
    if raw is None:
        return None

    # numpy.datetime64?
    if isinstance(raw, (np.datetime64,)):
        try:
            ns = raw.astype('datetime64[ns]').astype('int64')
            sec, nsec = divmod(ns, 1_000_000_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=nsec // 1000)
        except Exception:
            pass

    # bytes/str -> ISO handling with long fractional seconds
    val = _decode_attr(raw)
    if isinstance(val, str):
        s = val.strip()
        try:
            # normalize trailing Z
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            # if there is a fractional second with >6 digits, truncate
            # find the time part up to timezone sign (+/-) after date 'YYYY-MM-DD'
            if 'T' in s:
                # split off timezone offset if present (keeps sign)
                main, tz = s, ''
                plus = s.rfind('+')
                minus = s.rfind('-')
                cut = max(plus, minus if minus > 9 else -1)  # ignore date hyphens
                if cut > 9:
                    main, tz = s[:cut], s[cut:]
                # trim fractional seconds in main
                if '.' in main:
                    head, frac = main.split('.', 1)
                    frac_digits = ''.join(ch for ch in frac if ch.isdigit())
                    main = head + ('.' + frac_digits[:6] if frac_digits else '')
                s_for_dt = main + tz
            else:
                s_for_dt = s
            dt = datetime.fromisoformat(s_for_dt)
            # ensure tz-aware UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            pass

    # numeric epoch?
    if isinstance(val, (int, float, np.integer, np.floating)):
        x = float(val)
        # heuristics by magnitude
        if x > 1e17:    # ns
            sec, nsec = divmod(int(x), 1_000_000_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=nsec // 1000)
        elif x > 1e14:  # us
            sec, usec = divmod(int(x), 1_000_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=usec)
        elif x > 1e11:  # ms
            sec, msec = divmod(int(x), 1_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=msec * 1000)
        else:           # s
            return datetime.fromtimestamp(x, tz=timezone.utc)

    # last try: numpy conversion from stringy inputs
    try:
        dt64 = np.datetime64(val)
        ns = dt64.astype('datetime64[ns]').astype('int64')
        sec, nsec = divmod(ns, 1_000_000_000)
        return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=nsec // 1000)
    except Exception:
        return None


class DAS:
    """
    Multi-vendor DAS reader.
    Provides:
      - self.data: ndarray [channels, time]
      - self.meta: dict with 'fs','dt','dx','start_time_dt','start_time_iso', and light extras

    vendor="raw_hdf5": reads a bare HDF5 dataset (e.g. Yuan et al. 2024 DAS_data.h5).
      Required kwargs: fs (sample rate Hz), dx (channel spacing m).
      Optional kwargs: dataset_key (default "strainrate").
      No timestamp metadata is available; start_time_dt will be None.
    """
    def __init__(self, file, select_channels=None, vendor="OptaSense", **kwargs):
        self.file = file
        self.vendor = vendor
        self.select_channels = select_channels  # None => read all channels
        self.data = None
        self.meta = {}

        if str(vendor).lower() == "optasense":
            self._read_optasense()
        elif str(vendor).lower() == "silixa":
            self._read_silixa()
        elif str(vendor).lower() == "raw_hdf5":
            self._read_raw_hdf5(**kwargs)
        else:
            raise ValueError(f"Unknown vendor '{vendor}'. Use 'OptaSense', 'Silixa', or 'raw_hdf5'.")

    # ----------------------
    # OptaSense reader
    # ----------------------
    def _read_optasense(self):
        RAW_PATH = "Acquisition/Raw[0]/RawData"
        RAW0_GRP = "Acquisition/Raw[0]"
        ACQ_GRP  = "Acquisition"

        with h5py.File(self.file, "r") as f:
            raw_ds = f[RAW_PATH]              # (time, loci)
            arr = raw_ds[...].T               # -> (channels, time)
            if self.select_channels is not None:
                arr = arr[self.select_channels, :]
            self.data = arr

            dx = float(f[ACQ_GRP].attrs["SpatialSamplingInterval"])
            fs = float(f[RAW0_GRP].attrs["OutputDataRate"])
            dt = 1.0 / fs

            start_dt = _parse_start_time_attr(f[ACQ_GRP].attrs.get("MeasurementStartTime"))
            if start_dt is None and "Acquisition/Raw[0]/RawDataTime" in f:
                start_dt = _parse_start_time_attr(f["Acquisition/Raw[0]/RawDataTime"].attrs.get("StartTime"))

            part_start_dt = _parse_start_time_attr(raw_ds.attrs.get("PartStartTime"))
            part_end_dt   = _parse_start_time_attr(raw_ds.attrs.get("PartEndTime"))

            self.meta.update({
                "dx": dx,
                "fs": fs,
                "dt": dt,
                "start_time_dt": start_dt,
                "start_time_iso": None if start_dt is None else start_dt.isoformat(),
                "part_start_time_dt": part_start_dt,
                "part_end_time_dt":   part_end_dt,
                "vendor": _decode_attr(f[ACQ_GRP].attrs.get("VendorCode")),
                "gauge_length_m": float(f[ACQ_GRP].attrs.get("GaugeLength", np.nan)),
                "raw_unit": _decode_attr(f[RAW0_GRP].attrs.get("RawDataUnit")),
                "raw_description": _decode_attr(f[RAW0_GRP].attrs.get("RawDescription")),
                "raw_path": RAW_PATH,
                "raw_dtype": str(raw_ds.dtype),
                "raw_shape": tuple(raw_ds.shape),  # original (time, loci)
            })

    # ----------------------
    # Silixa (DAS-RCN style) reader
    # ----------------------
    def _read_silixa(self):
        META_ROOT   = "DasMetadata"
        META_ACQ    = "DasMetadata/Interrogator/Acquisition"
        RAW_PATH    = "DasRawData/RawData"        # (time step, locus)
        TIME_ARRAY  = "DasRawData/DasTimeArray"   # (time,), attrs StartTime/EndTime

        with h5py.File(self.file, "r") as f:
            raw_ds = f[RAW_PATH]                  # (time, locus)
            arr = raw_ds[...].T                   # -> (channels, time)
            if self.select_channels is not None:
                arr = arr[self.select_channels, :]
            self.data = arr

            # Core meta
            # AcquisitionSampleRate can be a string; SpatialSamplingInterval too.
            fs_raw = f[META_ACQ].attrs.get("AcquisitionSampleRate")
            dx_raw = f[META_ACQ].attrs.get("SpatialSamplingInterval")
            fs = float(_decode_attr(fs_raw))      # e.g., '1000' -> 1000.0
            dx = float(_decode_attr(dx_raw))      # e.g., '1.021' -> 1.021
            dt = 1.0 / fs

            # Start times (prefer explicit StartTime from DasTimeArray; else AcquisitionStartTime)
            part_start_dt = _parse_start_time_attr(f[TIME_ARRAY].attrs.get("StartTime"))
            part_end_dt   = _parse_start_time_attr(f[TIME_ARRAY].attrs.get("EndTime"))

            start_dt = _parse_start_time_attr(f[META_ACQ].attrs.get("AcquisitionStartTime"))
            if start_dt is None:
                start_dt = part_start_dt

            # Light extras
            vendor_name = _decode_attr(f.get(f"{META_ROOT}/Interrogator", {}).attrs.get("InterrogatorManufacturer")) \
                          if f"{META_ROOT}/Interrogator" in f else "Silixa"

            self.meta.update({
                "dx": dx,
                "fs": fs,
                "dt": dt,
                "start_time_dt": start_dt,
                "start_time_iso": None if start_dt is None else start_dt.isoformat(),
                "part_start_time_dt": part_start_dt,
                "part_end_time_dt":   part_end_dt,
                "vendor": vendor_name,
                "gauge_length_m": float(_decode_attr(f[META_ACQ].attrs.get("GaugeLength", "nan"))) if META_ACQ in f else np.nan,
                "raw_unit": None,  # not standardized here; can be added if present
                "raw_description": None,
                "raw_path": RAW_PATH,
                "raw_dtype": str(raw_ds.dtype),
                "raw_shape": tuple(raw_ds.shape),  # original (time, locus)
            })

    # ----------------------
    # Raw HDF5 reader (Yuan et al. 2024 style: bare dataset, no metadata)
    # ----------------------
    def _read_raw_hdf5(self, fs, dx, dataset_key="strainrate"):
        """
        Read a plain HDF5 file that contains one dataset of shape (channels, time).
        fs and dx must be provided explicitly (not stored in the file).
        """
        with h5py.File(self.file, "r") as f:
            if dataset_key not in f:
                available = list(f.keys())
                raise KeyError(
                    f"Dataset '{dataset_key}' not found in {self.file}. "
                    f"Available keys: {available}"
                )
            ds = f[dataset_key]
            arr = ds[...].astype(np.float32)   # (channels, time)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2-D dataset (channels, time), got shape {arr.shape}"
                )
            if self.select_channels is not None:
                arr = arr[self.select_channels, :]
            self.data = arr

        fs = float(fs)
        self.meta.update({
            "dx": float(dx),
            "fs": fs,
            "dt": 1.0 / fs,
            "start_time_dt": None,
            "start_time_iso": None,
            "part_start_time_dt": None,
            "part_end_time_dt": None,
            "vendor": "raw_hdf5",
            "gauge_length_m": float(dx),
            "raw_unit": None,
            "raw_description": None,
            "raw_path": dataset_key,
            "raw_dtype": str(self.data.dtype),
            "raw_shape": tuple(self.data.shape),
        })


@dataclass(frozen=True)
class RecordingPart:
    """Metadata and global sample bounds for one file in a recording."""

    path: Path
    start_time: datetime
    stop_time: datetime
    start_sample: int
    stop_sample: int
    n_time: int
    n_channels: int
    fs: float
    dx: float
    dtype: str
    raw_path: str


class DASRecording:
    """A timestamp-ordered, lazy view over consecutive OptaSense files.

    Unlike :class:`DAS`, this class reads only the requested channel and time
    slices. Global time ranges may cross any number of source-file boundaries.
    Files are ordered by ``PartStartTime`` and continuity is validated before
    signal data are read.
    """

    def __init__(self, files, *, vendor="OptaSense", continuity_tolerance_samples=0.5):
        paths = [Path(path) for path in files]
        if not paths:
            raise ValueError("DASRecording requires at least one source file")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Source files do not exist: {missing}")

        vendor_name = str(vendor).lower()
        if vendor_name != "optasense":
            raise NotImplementedError(
                "DASRecording currently supports timestamped OptaSense files only"
            )

        inspected = [self._inspect_optasense(path) for path in paths]
        inspected.sort(key=lambda part: part["start_time"])
        self.vendor = "OptaSense"
        self._validate_common_metadata(inspected)

        tolerance_samples = float(continuity_tolerance_samples)
        if tolerance_samples < 0:
            raise ValueError("continuity_tolerance_samples must be >= 0")

        parts = []
        global_start = 0
        for index, info in enumerate(inspected):
            if index:
                previous = inspected[index - 1]
                expected = previous["start_time"] + timedelta(
                    seconds=previous["n_time"] / previous["fs"]
                )
                error_seconds = (info["start_time"] - expected).total_seconds()
                tolerance_seconds = tolerance_samples / info["fs"]
                if abs(error_seconds) > tolerance_seconds:
                    relation = "gap" if error_seconds > 0 else "overlap"
                    raise ValueError(
                        f"Recording has a {relation} of {abs(error_seconds):.9f} s "
                        f"between {previous['path'].name} and {info['path'].name}"
                    )

            global_stop = global_start + info["n_time"]
            parts.append(
                RecordingPart(
                    path=info["path"],
                    start_time=info["start_time"],
                    stop_time=info["start_time"]
                    + timedelta(seconds=info["n_time"] / info["fs"]),
                    start_sample=global_start,
                    stop_sample=global_stop,
                    n_time=info["n_time"],
                    n_channels=info["n_channels"],
                    fs=info["fs"],
                    dx=info["dx"],
                    dtype=info["dtype"],
                    raw_path=info["raw_path"],
                )
            )
            global_start = global_stop

        self.parts = tuple(parts)
        first = self.parts[0]
        self.fs = first.fs
        self.dx = first.dx
        self.n_channels = first.n_channels
        self.dtype = np.dtype(first.dtype)
        self.n_samples = self.parts[-1].stop_sample
        self.start_time = first.start_time
        self.stop_time = self.parts[-1].stop_time

    @staticmethod
    def _inspect_optasense(path):
        raw_path = "Acquisition/Raw[0]/RawData"
        with h5py.File(path, "r") as handle:
            dataset = handle[raw_path]
            if dataset.ndim != 2:
                raise ValueError(
                    f"Expected 2-D OptaSense data in {path}, got {dataset.shape}"
                )
            n_time, n_channels = map(int, dataset.shape)
            fs = float(handle["Acquisition/Raw[0]"].attrs["OutputDataRate"])
            dx = float(handle["Acquisition"].attrs["SpatialSamplingInterval"])
            start_time = _parse_start_time_attr(dataset.attrs.get("PartStartTime"))
            if start_time is None:
                start_time = _parse_start_time_attr(
                    handle["Acquisition"].attrs.get("MeasurementStartTime")
                )
            if start_time is None:
                raise ValueError(f"No usable start timestamp in {path}")
            return {
                "path": path,
                "start_time": start_time,
                "n_time": n_time,
                "n_channels": n_channels,
                "fs": fs,
                "dx": dx,
                "dtype": str(dataset.dtype),
                "raw_path": raw_path,
            }

    @staticmethod
    def _validate_common_metadata(parts):
        first = parts[0]
        fields = ("n_channels", "fs", "dx", "dtype", "raw_path")
        for part in parts[1:]:
            for field in fields:
                if part[field] != first[field]:
                    raise ValueError(
                        f"Inconsistent {field}: {first['path'].name} has "
                        f"{first[field]!r}, but {part['path'].name} has {part[field]!r}"
                    )

    @property
    def duration_seconds(self):
        return self.n_samples / self.fs

    def _validate_bounds(self, start, stop, channel_start, channel_stop):
        values = (start, stop, channel_start, channel_stop)
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise TypeError("Sample and channel bounds must be integers")
        start, stop, channel_start, channel_stop = map(int, values)
        if not 0 <= start < stop <= self.n_samples:
            raise ValueError(
                f"Require 0 <= start < stop <= {self.n_samples}; got {start}:{stop}"
            )
        if not 0 <= channel_start < channel_stop <= self.n_channels:
            raise ValueError(
                f"Require 0 <= channel_start < channel_stop <= {self.n_channels}; "
                f"got {channel_start}:{channel_stop}"
            )
        return start, stop, channel_start, channel_stop

    def source_spans(self, start, stop):
        """Return the source-file slices contributing to a global time range."""
        start, stop, _, _ = self._validate_bounds(
            start, stop, 0, self.n_channels
        )
        spans = []
        for part in self.parts:
            overlap_start = max(start, part.start_sample)
            overlap_stop = min(stop, part.stop_sample)
            if overlap_start >= overlap_stop:
                continue
            spans.append(
                {
                    "file": part.path.name,
                    "local_start": overlap_start - part.start_sample,
                    "local_stop": overlap_stop - part.start_sample,
                }
            )
        return spans

    def read(self, start, stop, channel_start=0, channel_stop=None):
        """Read ``[channel_start:channel_stop, start:stop]`` lazily."""
        if channel_stop is None:
            channel_stop = self.n_channels
        start, stop, channel_start, channel_stop = self._validate_bounds(
            start, stop, channel_start, channel_stop
        )

        pieces = []
        for part in self.parts:
            overlap_start = max(start, part.start_sample)
            overlap_stop = min(stop, part.stop_sample)
            if overlap_start >= overlap_stop:
                continue
            local_start = overlap_start - part.start_sample
            local_stop = overlap_stop - part.start_sample
            with h5py.File(part.path, "r") as handle:
                # OptaSense stores (time, channel); transpose only the small slice.
                piece = handle[part.raw_path][
                    local_start:local_stop, channel_start:channel_stop
                ].T
            pieces.append(piece)

        if not pieces:
            raise RuntimeError("No recording parts intersect the requested range")
        result = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
        expected = (channel_stop - channel_start, stop - start)
        if result.shape != expected:
            raise RuntimeError(f"Read produced shape {result.shape}; expected {expected}")
        return result
