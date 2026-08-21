# Provenance

`dasgauge` contains selected code **derived from** the repository
[`UWGeoD/DAS_Preprocessing`](https://github.com/UWGeoD/DAS_Preprocessing).
This file is the authoritative human-readable record of what was imported,
from where, and how it was changed. A machine-readable copy lives at
[`provenance/DAS_Preprocessing.json`](provenance/DAS_Preprocessing.json).

## Source

| Field | Value |
| --- | --- |
| Source repository (SSH) | `git@github.com:UWGeoD/DAS_Preprocessing.git` |
| Source repository (HTTPS) | https://github.com/UWGeoD/DAS_Preprocessing |
| Source branch | `main` |
| Source commit | `896e00005b68c7878ae80d50833bd9eeabe7ebc7` |
| Source commit date | 2026-07-24T23:09:58-05:00 |
| Extraction date | 2026-07-31 |
| Source license | MIT — Copyright (c) 2025 GeoD Research Group |

All line numbers, symbol names and permalinks below refer to that exact commit.
The extraction was performed from a throwaway clone made outside this
repository; that clone has been deleted and `DAS_Preprocessing` is **not** a
submodule, subtree, remote, or runtime dependency of `dasgauge`.

### Source file hashes (SHA-256, at the source commit)

| File | SHA-256 |
| --- | --- |
| `DAS.py` | `6cfc09148542347cbf3bc47c6007ec063911b1eb62f89295aaf051a5e41c835d` |
| `preprocessing.py` | `ae9e5b4bff4747c3071f61a26b2f23473f0bf8de2ed4bd248d27193a08d4638a` |
| `Utilities.py` | `67fc708bb4b6457ebb222338de23ede6364e9d22e2ae37c826d7ed14459be2bb` |

## What was imported

### 1. `DAS.py` → [`dasgauge/io.py`](dasgauge/io.py)

HDF5 readers for OptaSense, Silixa and generic/bare HDF5 files.

| Original symbol | Original lines | Permalink | Destination symbol |
| --- | --- | --- | --- |
| `_decode_attr` | 7–13 | [L7-L13](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/DAS.py#L7-L13) | `_decode_attr` |
| `_parse_start_time_attr` | 15–94 | [L15-L94](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/DAS.py#L15-L94) | `_parse_start_time_attr` |
| `DAS.__init__` (+ class docstring) | 98–124 | [L98-L124](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/DAS.py#L98-L124) | `DAS.__init__` |
| `DAS._read_optasense` | 129–167 | [L129-L167](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/DAS.py#L129-L167) | `DAS._read_optasense` |
| `DAS._read_silixa` | 172–222 | [L172-L222](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/DAS.py#L172-L222) | `DAS._read_silixa` |
| `DAS._read_raw_hdf5` | 228–266 | [L228-L266](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/DAS.py#L228-L266) | `DAS._read_raw_hdf5` |

**Not imported:** `DAS.plot` (268–282), `DAS.plot_single` (284–287), `MulDAS` (293–348).

**Changes made during extraction**

- Removed `import Utilities` — a Project A module-namespace import.
- Removed `DAS.plot` / `DAS.plot_single`: thin convenience wrappers that imported
  Project A's `Utilities` and `preprocessing` top-level modules. The underlying
  plotting functions are imported separately and are available as
  `dasgauge.plotting.plot_das_data` / `plot_single`.
- Removed `MulDAS` (multi-file sort-and-concatenate) as outside the requested scope.
- Removed the unused local `RAW_GRP` in `_read_silixa` and the author-specific
  comment `# Paths from your example`.
- Added a module docstring carrying source attribution, and `__all__ = ["DAS"]`.
- No function was renamed. Reader algorithms, metadata keys, vendor dispatch and
  error messages are otherwise unchanged.

### 2. `preprocessing.py` → [`dasgauge/preprocessing.py`](dasgauge/preprocessing.py)

Detrending, bandpass filtering, f-k filtering, downsampling, integration,
Hilbert-transform processing and curvelet processing, plus the pipeline builder
that composes them.

| Original symbol | Original lines | Permalink | Destination symbol |
| --- | --- | --- | --- |
| `Array2D` | 32 | [L32](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L32) | `Array2D` |
| `_as_2d_float` | 38–45 | [L38-L45](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L38-L45) | `_as_2d_float` |
| `_require_positive` | 48–52 | [L48-L52](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L48-L52) | `_require_positive` |
| `detrend_linear` | 58–61 | [L58-L61](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L58-L61) | `detrend_linear` |
| `bandpass_sos` | 64–97 | [L64-L97](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L64-L97) | `bandpass_sos` |
| `detrend_then_bandpass` | 100–119 | [L100-L119](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L100-L119) | `detrend_then_bandpass` |
| `fk_filter` | 122–216 | [L122-L216](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L122-L216) | `fk_filter` |
| `integrate_to_strain` | 219–238 | [L219-L238](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L219-L238) | `integrate_to_strain` |
| `integrate_strain_rate_fd` | 241–289 | [L241-L289](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L241-L289) | `integrate_strain_rate_fd` |
| `downsample_time` | 292–315 | [L292-L315](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L292-L315) | `downsample_time` |
| `hilbert_transform` | 318–345 | [L318-L345](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L318-L345) | `hilbert_transform` |
| `_meyer_nu` | 348–351 | [L348-L351](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L348-L351) | `_meyer_nu` |
| `_meyer_W` | 354–367 | [L354-L367](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L354-L367) | `_meyer_W` |
| `_meyer_V` | 370–381 | [L370-L381](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L370-L381) | `_meyer_V` |
| `_curvelet_soft_threshold` | 384–397 | [L384-L397](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L384-L397) | `_curvelet_soft_threshold` |
| `curvelet_denoise` | 400–508 | [L400-L508](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L400-L508) | `curvelet_denoise` |
| `Step` | 518–523 | [L518-L523](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L518-L523) | `Step` |
| `_normalize_spec` | 526–542 | [L526-L542](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L526-L542) | `_normalize_spec` |
| `_STEP_REGISTRY` | 545–553 | [L545-L553](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L545-L553) | `_STEP_REGISTRY` |
| `make_preprocess` | 556–667 | [L556-L667](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/preprocessing.py#L556-L667) | `make_preprocess` |

**Not imported:** `curvelet_like_denoise` (512), `load_and_preprocess` (670–689).

**Changes made during extraction**

- Removed the alias `curvelet_like_denoise = curvelet_denoise` — a
  backward-compatibility shim for Project A's notebooks.
- Removed `load_and_preprocess`, which loads `sample_XXXXX.npy` files following
  Project A's dataset layout.
- Dropped the unused `Iterable` name from the `typing` import (source line 25).
- `integrate_strain_rate_fd` docstring: the sentence describing the sign
  convention originally referenced Project A's "SpatialDAE training" and model
  collapsing to `pred=0`; it was reworded project-neutrally to describe the same
  consequence. Executable code is unchanged.
- `curvelet_denoise` docstring: added one line stating that the implementation is
  pure NumPy FFT and requires no external curvelet library. Code unchanged.
- Added source attribution to the head of the module docstring and an `__all__`.
- No function was renamed. Algorithms, defaults, validation rules and the
  scientific references (Candès et al. 2006; van den Ende et al. 2023; Zhang et
  al. 2025) are preserved.

### 3. `Utilities.py` → [`dasgauge/plotting.py`](dasgauge/plotting.py)

DAS heatmap plotting and single-channel plotting.

| Original symbol | Original lines | Permalink | Destination symbol |
| --- | --- | --- | --- |
| `epsilon` | 9 | [L9](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/Utilities.py#L9) | `epsilon` |
| `normalize` | 10 | [L10](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/Utilities.py#L10) | `normalize` |
| `plot_das_data` | 201–249 | [L201-L249](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/Utilities.py#L201-L249) | `plot_das_data` |
| `plot_single` | 252–296 | [L252-L296](https://github.com/UWGeoD/DAS_Preprocessing/blob/896e00005b68c7878ae80d50833bd9eeabe7ebc7/Utilities.py#L252-L296) | `plot_single` |

`epsilon` and `normalize` are the only helpers the two plotting functions
actually require.

**Not imported:** `_PUB_COLORS` (12–16), `_STEP_DISPLAY` (18–25), `_fmt_step`
(28–41), `plot_das_comparison` (43–198), `downsample_data` (299–305),
`draw_signal_rects` (308–337), `compute_snr` (340–407), `time_intervals_to_mask`
(410–461), `snr_from_time_mask` (464–510).

**Changes made during extraction**

- Replaced the module-level `import matplotlib.pyplot as plt` with a lazy
  `_pyplot()` helper (**new code, not derived from Project A**) so that
  matplotlib is an optional dependency: importing `dasgauge` never requires it,
  and only calling a plotting function without it raises an `ImportError` naming
  the `dasgauge[plotting]` extra.
- Dropped imports the original module carried but these functions do not use:
  `scipy.signal`, `datetime`, `collections.OrderedDict`, `re`.
- Removed two commented-out alternative `cmap`/`data` lines inside
  `plot_das_data`'s `imshow` call (source lines 226 and 228).
- Added a one-line summary to `plot_das_data`'s docstring; the original text is kept.
- Excluded `plot_das_comparison` and its helpers: a raw/preprocessed/UNet-denoised
  SNR comparison figure specific to Project A's experiments.
- Excluded `draw_signal_rects`, `compute_snr`, `time_intervals_to_mask` and
  `snr_from_time_mask`: metrics coupled to Project A's video-frame and
  `labels.csv` conventions, outside the requested plotting scope.
- Excluded `downsample_data`: superseded by `preprocessing.downsample_time`,
  which is the requested downsampling implementation.
- No function was renamed. Plot geometry, normalization, colormap, axis labels
  and return values are unchanged.

## Removed coupling and hard-coded paths

The extracted code contains **no** hard-coded paths. Project A's `DAS.py`,
`preprocessing.py` and `Utilities.py` did not themselves define hard-coded local
paths (those live in Project A's `config.example.py` and `sweep_paths.py`, which
were not imported). The coupling that *was* removed is:

- `import Utilities` and `from preprocessing import make_preprocess` in `DAS.py`
  — top-level imports resolvable only from Project A's repository root.
- The `curvelet_like_denoise` alias kept for Project A notebooks.
- `load_and_preprocess`, tied to Project A's `.npy` sample layout.
- Project A configuration, CLI behaviour, experiment/sweep code and global state
  were not imported at all.

Inside `dasgauge` all imports are relative to the `dasgauge` package
(`from .io import DAS`, etc.). `tests/test_independence.py` enforces this
mechanically.

## Dependencies

| Scope | Packages |
| --- | --- |
| Required | `numpy>=1.24`, `scipy>=1.10`, `h5py>=3.8`, `PyYAML>=6.0` |
| Optional extra `plotting` | `matplotlib>=3.7` |
| Optional extra `c2st` | `torch==2.5.1`, `torchvision==0.20.1` |

The test suite uses only the standard library's `unittest` (plus matplotlib for
the plotting tests), so no test dependency is declared.

Curvelet processing needs **no** specialized package — `curvelet_denoise` is
implemented directly on `numpy.fft` (Meyer windows and shear geometry are
computed in-module), so no CurveLab / PyCurvelab / curvelops extra exists.
Matplotlib and PyTorch/TorchVision are optional runtime dependencies and are
imported lazily with actionable error messages.

Project A's `requirements.txt` additionally listed `tzlocal`, `jupyterlab`,
`pandas` and `pyyaml`; none are used by the imported upstream code. This
project's independently written dataset-config module does use PyYAML.

## License

Project A (`UWGeoD/DAS_Preprocessing`) is MIT licensed,
Copyright (c) 2025 GeoD Research Group. The MIT license permits this reuse,
including modification and redistribution, provided the copyright and permission
notice are retained.

`dasgauge` is likewise MIT licensed. Its pre-existing `LICENSE` file
(`Copyright (c) 2026 GeoD Research Group` — same copyright holder, different
year) is left **unchanged** by this import. The MIT condition to retain the
original notice is satisfied by carrying the upstream
`Copyright (c) 2025 GeoD Research Group, MIT License` line in the module
docstring of every destination file containing derived code
(`dasgauge/io.py`, `dasgauge/preprocessing.py`, `dasgauge/plotting.py`).

## Independence

`dasgauge` and `DAS_Preprocessing` are **maintained independently** after this
import:

- The import is a one-time copy of source text, recorded in a single commit and
  an annotated tag.
- Project A's history was not merged; no subtree, submodule, or remote linking
  the two repositories exists.
- Project A is not a build-time or runtime dependency of `dasgauge`.
- Later changes in Project A do **not** propagate to `dasgauge`, and changes in
  `dasgauge` do not propagate to Project A.
- Nothing in Project A was modified by this import.

Any future re-sync must be done deliberately and recorded by updating this file
and `provenance/DAS_Preprocessing.json`.
