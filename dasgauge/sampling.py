"""Strategies and storage for splitting continuous DAS recordings into samples."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

from .preprocessing import make_preprocess

__all__ = [
    "SampleWindow",
    "hold_skip_windows",
    "make_run_name",
    "sampling_preview",
    "split_recording",
]


@dataclass(frozen=True)
class SampleWindow:
    sample_id: str
    start_sample: int
    stop_sample: int


def _seconds_to_samples(seconds, fs, name, *, allow_zero=False):
    seconds = float(seconds)
    if seconds < 0 or (seconds == 0 and not allow_zero):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}, got {seconds}")
    exact = seconds * float(fs)
    rounded = int(round(exact))
    if abs(exact - rounded) > 1e-6:
        raise ValueError(f"{name}={seconds} s is not an integer number of samples at {fs} Hz")
    if rounded == 0 and not allow_zero:
        raise ValueError(f"{name} is shorter than one sample at {fs} Hz")
    return rounded


def hold_skip_windows(n_samples, fs, s1, s2):
    """Return complete ``s1`` windows separated by ``s2`` skipped seconds."""
    n_samples = int(n_samples)
    hold = _seconds_to_samples(s1, fs, "s1")
    skip = _seconds_to_samples(s2, fs, "s2", allow_zero=True)
    stride = hold + skip
    windows = []
    start = 0
    while start + hold <= n_samples:
        windows.append(
            SampleWindow(
                sample_id=f"sample_{len(windows):05d}",
                start_sample=start,
                stop_sample=start + hold,
            )
        )
        start += stride
    return windows


def _pipeline_output_fs(steps, fs):
    current = float(fs)
    for name, params in steps:
        if name == "downsample":
            target = float(params.get("target_fs", 50.0))
            factor = max(1, int(round(current / target)))
            current /= factor
    return current


def _preprocess_json(steps):
    return [{"name": name, "params": params} for name, params in steps]


def _slug_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def make_run_name(strategy, parameters, coupling_channels, uncoupling_channels, steps):
    if strategy != "hold_skip":
        raise ValueError(f"Unknown splitting strategy: {strategy}")
    canonical = json.dumps(_preprocess_json(steps), sort_keys=True, separators=(",", ":"))
    preprocess_id = "none" if not steps else hashlib.sha256(canonical.encode()).hexdigest()[:10]
    return (
        f"hold_skip__s1-{_slug_number(parameters['s1'])}"
        f"__s2-{_slug_number(parameters['s2'])}"
        f"__c-{coupling_channels[0]}-{coupling_channels[1]}"
        f"__u-{uncoupling_channels[0]}-{uncoupling_channels[1]}"
        f"__pp-{preprocess_id}"
    )


def sampling_preview(recording, *, strategy, parameters, coupling_channels, uncoupling_channels):
    if strategy != "hold_skip":
        raise ValueError(f"Unknown splitting strategy: {strategy}")
    windows = hold_skip_windows(
        recording.n_samples, recording.fs, parameters["s1"], parameters["s2"]
    )
    for name, bounds in (
        ("coupling_channels", coupling_channels),
        ("uncoupling_channels", uncoupling_channels),
    ):
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= recording.n_channels:
            raise ValueError(
                f"{name} must be a half-open range within 0:{recording.n_channels}, got {bounds}"
            )
    hold_samples = windows[0].stop_sample - windows[0].start_sample if windows else 0
    total_values = len(windows) * hold_samples * (
        coupling_channels[1] - coupling_channels[0]
        + uncoupling_channels[1] - uncoupling_channels[0]
    )
    tail_start = windows[-1].stop_sample if windows else 0
    return {
        "n_samples": len(windows),
        "input_fs_hz": recording.fs,
        "recording_duration_seconds": recording.duration_seconds,
        "hold_samples": hold_samples,
        "tail_after_last_sample_seconds": (recording.n_samples - tail_start) / recording.fs,
        "uncompressed_input_bytes": total_values * recording.dtype.itemsize,
        "windows": windows,
    }


def _chunk_shape(shape, dtype):
    _, n_channels, n_time = shape
    target_bytes = 1024 * 1024
    chunk_time = max(1, min(n_time, target_bytes // (n_channels * np.dtype(dtype).itemsize)))
    return (1, n_channels, chunk_time)


def _apply_pipeline(data, pipeline, fs):
    return np.asarray(data if pipeline is None else pipeline(data, fs))


def split_recording(
    recording,
    output_dir,
    *,
    dataset_id,
    strategy,
    parameters,
    coupling_channels,
    uncoupling_channels,
    preprocess_steps=(),
    config_path=None,
    overwrite=False,
):
    """Materialize paired coupling/uncoupling samples in a chunked HDF5 store."""
    preview = sampling_preview(
        recording,
        strategy=strategy,
        parameters=parameters,
        coupling_channels=coupling_channels,
        uncoupling_channels=uncoupling_channels,
    )
    windows = preview.pop("windows")
    if not windows:
        raise ValueError("The strategy produced no complete sample windows")

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing sample run: {output_dir}")

    steps = [(name, dict(params)) for name, params in preprocess_steps]
    pipeline = make_preprocess(steps, dx=recording.dx) if steps else None
    output_fs = _pipeline_output_fs(steps, recording.fs)

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent) as temp:
        temp_dir = Path(temp)
        data_path = temp_dir / "samples.h5"
        index_path = temp_dir / "index.csv"

        first_window = windows[0]
        first_coupling = _apply_pipeline(
            recording.read(
                first_window.start_sample,
                first_window.stop_sample,
                *coupling_channels,
            ),
            pipeline,
            recording.fs,
        )
        first_uncoupling = _apply_pipeline(
            recording.read(
                first_window.start_sample,
                first_window.stop_sample,
                *uncoupling_channels,
            ),
            pipeline,
            recording.fs,
        )
        if first_coupling.ndim != 2 or first_uncoupling.ndim != 2:
            raise ValueError("Preprocessing must return 2-D [channel, time] arrays")
        if first_coupling.shape[1] != first_uncoupling.shape[1]:
            raise ValueError("Coupling and uncoupling preprocessing produced different time lengths")

        coupling_shape = (len(windows),) + first_coupling.shape
        uncoupling_shape = (len(windows),) + first_uncoupling.shape
        coupling_dtype = first_coupling.dtype
        uncoupling_dtype = first_uncoupling.dtype
        rows = []

        with h5py.File(data_path, "w") as handle:
            handle.attrs.update(
                {
                    "schema_version": 1,
                    "dataset_id": dataset_id,
                    "strategy": strategy,
                    "input_fs_hz": recording.fs,
                    "output_fs_hz": output_fs,
                    "dx_m": recording.dx,
                    "preprocess_json": json.dumps(_preprocess_json(steps), sort_keys=True),
                }
            )
            group = handle.create_group("samples")
            coupling_ds = group.create_dataset(
                "coupling",
                shape=coupling_shape,
                dtype=coupling_dtype,
                chunks=_chunk_shape(coupling_shape, coupling_dtype),
                compression="lzf",
                shuffle=True,
            )
            uncoupling_ds = group.create_dataset(
                "uncoupling",
                shape=uncoupling_shape,
                dtype=uncoupling_dtype,
                chunks=_chunk_shape(uncoupling_shape, uncoupling_dtype),
                compression="lzf",
                shuffle=True,
            )
            coupling_ds.attrs["channel_start"] = coupling_channels[0]
            coupling_ds.attrs["channel_stop"] = coupling_channels[1]
            uncoupling_ds.attrs["channel_start"] = uncoupling_channels[0]
            uncoupling_ds.attrs["channel_stop"] = uncoupling_channels[1]

            for index, window in enumerate(windows):
                if index == 0:
                    coupling = first_coupling
                    uncoupling = first_uncoupling
                else:
                    coupling = _apply_pipeline(
                        recording.read(
                            window.start_sample, window.stop_sample, *coupling_channels
                        ),
                        pipeline,
                        recording.fs,
                    )
                    uncoupling = _apply_pipeline(
                        recording.read(
                            window.start_sample, window.stop_sample, *uncoupling_channels
                        ),
                        pipeline,
                        recording.fs,
                    )
                if coupling.shape != first_coupling.shape:
                    raise ValueError(f"Inconsistent coupling shape at {window.sample_id}")
                if uncoupling.shape != first_uncoupling.shape:
                    raise ValueError(f"Inconsistent uncoupling shape at {window.sample_id}")
                coupling_ds[index] = coupling
                uncoupling_ds[index] = uncoupling

                start_time = recording.start_time + timedelta(
                    seconds=window.start_sample / recording.fs
                )
                stop_time = recording.start_time + timedelta(
                    seconds=window.stop_sample / recording.fs
                )
                rows.append(
                    {
                        "sample_id": window.sample_id,
                        "start_sample": window.start_sample,
                        "stop_sample": window.stop_sample,
                        "start_seconds": f"{window.start_sample / recording.fs:.9f}",
                        "stop_seconds": f"{window.stop_sample / recording.fs:.9f}",
                        "start_time_utc": start_time.astimezone(timezone.utc).isoformat(),
                        "stop_time_utc": stop_time.astimezone(timezone.utc).isoformat(),
                        "source_spans_json": json.dumps(
                            recording.source_spans(window.start_sample, window.stop_sample),
                            separators=(",", ":"),
                        ),
                    }
                )

        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "config_path": None if config_path is None else str(config_path),
            "strategy": {"name": strategy, "parameters": dict(parameters)},
            "channels": {
                "convention": "zero-based half-open [start, stop)",
                "coupling": list(coupling_channels),
                "uncoupling": list(uncoupling_channels),
            },
            "preprocess": _preprocess_json(steps),
            "recording": {
                "vendor": recording.vendor,
                "source_files": [str(part.path) for part in recording.parts],
                "start_time_utc": recording.start_time.astimezone(timezone.utc).isoformat(),
                "stop_time_utc": recording.stop_time.astimezone(timezone.utc).isoformat(),
                "n_input_samples": recording.n_samples,
                "input_fs_hz": recording.fs,
                "dx_m": recording.dx,
            },
            "samples": {
                "count": len(windows),
                "paired_by": "sample_id and time interval",
                "coupling_shape": list(coupling_shape),
                "uncoupling_shape": list(uncoupling_shape),
                "coupling_dtype": str(coupling_dtype),
                "uncoupling_dtype": str(uncoupling_dtype),
                "output_fs_hz": output_fs,
            },
            "storage": {
                "format": "HDF5",
                "file": "samples.h5",
                "datasets": ["/samples/coupling", "/samples/uncoupling"],
                "compression": "lzf",
                "access_pattern": "sample-major chunking",
            },
            "preview": preview,
        }
        with (temp_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        backup_dir = None
        if output_dir.exists():
            backup_dir = output_dir.with_name(
                f".{output_dir.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(output_dir, backup_dir)
        try:
            os.replace(temp_dir, output_dir)
        except BaseException:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                os.replace(backup_dir, output_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    return manifest
