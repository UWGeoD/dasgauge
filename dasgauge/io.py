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
from datetime import datetime, timezone

__all__ = ["DAS"]


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
