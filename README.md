# dasgauge

Reading, preprocessing and plotting utilities for Distributed Acoustic Sensing (DAS) data.

## Contents

| Module | Purpose |
| --- | --- |
| `dasgauge.io` | HDF5 readers for OptaSense, Silixa (DAS-RCN style), and generic/bare HDF5 files |
| `dasgauge.preprocessing` | Detrend, bandpass, f-k filter, downsample, integration, Hilbert transform, curvelet denoising, plus a composable pipeline builder |
| `dasgauge.plotting` | DAS heatmap, all-channel waterfall, and single-channel waveform plots |
| `dasgauge.sampling` | Continuous multi-file hold/skip sampling and chunked HDF5 storage |
| `dasgauge.c2st` | ResNet-34/summary features, KNN, original Binomial C2ST, paired McNemar, and full-refit swap inference |

## Install

Create and activate an isolated virtual environment, then install the complete
project from the conventional requirements file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. The
audited full environment contains NumPy, SciPy, h5py, PyYAML, Matplotlib,
PyTorch, and TorchVision. Matplotlib and PyTorch/TorchVision are optional to the
library itself but included by `requirements.txt` because the plots, manuscript
figures, and ResNet-34 C2ST workflow use them. KNN, exact tests, and curvelet
denoising are implemented in this repository; no scikit-learn, pandas, or
external curvelet package is needed.

## Quick start

```python
import numpy as np
from dasgauge import DAS, make_preprocess, plot_das_data, plot_waterfall

das = DAS("record.h5", vendor="OptaSense")

pp = make_preprocess([
    ("detrend", {}),
    ("bandpass", {"f_lo": 1.0, "f_hi": 15.0, "order": 5}),
    ("downsample", {"target_fs": 50.0}),
], dx=das.meta["dx"])

clean = pp(das.data, das.meta["fs"])
plot_das_data(clean, np.arange(clean.shape[0]), das.meta["dx"], 1 / 50.0, show=False)
plot_waterfall(clean, np.arange(clean.shape[0]), 1 / 50.0, show=False)
```

For side-by-side waterfall panels, compute or choose one gain and pass that
same value to every panel whose trace amplitudes should be directly compared.

## Split samples and run C2ST

Dataset YAML files define the recording, optional sample-local preprocessing,
channel defaults, and strategy defaults. Channel ranges are zero-based and
half-open. For the included Newville configuration, point the environment
variable at the directory containing the recording's HDF5 parts:

```bash
export DASGAUGE_NEWVILLE_RECORDING=/path/to/Newville_20251113
```

```yaml
preprocess: []

sampling_defaults:
  coupling_channels: [35, 66]
  uncoupling_channels: [72, 103]
  strategies:
    hold_skip: {s1: 10, s2: 50}
```

Preview the temporally ordered recording and planned storage without writing:

```bash
python split_samples.py --strategy hold_skip --dataset newville_nov --dry-run
```

Materialize one sample-major, LZF-compressed HDF5 store. Command-line values
override the YAML defaults:

```bash
python split_samples.py \
  --strategy hold_skip --s1 10 --s2 50 \
  --coupling-channels 35 66 --uncoupling-channels 72 103
```

An identical parameter set maps to the same run directory. Replace that entire
run—including any cached features and results beneath it—with a staged write:

```bash
python split_samples.py --strategy hold_skip --overwrite
```

Run the paper-style pipeline: frozen ImageNet ResNet-34 features, KNN with
`k=floor(sqrt(n_train))`, held-out accuracy, and any of the named inference
methods. This example runs the three methods used in the manuscript on one
sample directory:

```bash
python run_c2st.py samples/newville_nov/hold_skip/<run-name> \
  --comparison s-p \
  --representation resnet34 --classifier knn \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 1 --permutations 999 --seed 0
```

Each DAS space-time array is robustly mapped to `[0,1]`, resized to 224 by 224,
repeated over three channels, ImageNet-normalized, and passed through ResNet-34.
The final classifier layer is removed, yielding one 512-dimensional feature
vector per DAS sample. Feature arrays are cached beside the sample store.

Each method writes its own timestamped JSON report to the existing
`<run-name>/results/` directory. The original Binomial C2ST splits individual
observations and uses the classical accuracy-Binomial reference. McNemar keeps
the held-out pairs together and calibrates their discordant correctness
outcomes. The swap test keeps pairs together, swaps labels only within each
matched time pair, and refits KNN for every randomized labeling. Use
`--permutation-unit block --block-size N` when consecutive pairs must remain a
single temporal unit.

For backward compatibility, omitting `--methods` retains the old behavior:
`--calibration permutation` runs the paired swap test and `--calibration
binomial` runs the pair-preserving Binomial diagnostic. The latter is not the
original C2ST; request `--methods original_binomial_c2st` explicitly for that
procedure.

Here `S` denotes coupling and `P` denotes uncoupling. The same-region controls
make two disjoint groups by pairing consecutive windows and randomly assigning
one window per pair to each group. Each repeat gets a fresh grouping, train/test
split, KNN fit, and permutation null:

```bash
python run_c2st.py samples/newville_nov/hold_skip/<run-name> \
  --comparison s-s \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999 --seed 0

python run_c2st.py samples/newville_nov/hold_skip/<run-name> \
  --comparison p-p \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999 --seed 0
```

For a sample-size-matched comparison, `s-p-matched` selects one matched S/P time
from each adjacent temporal block, giving the same `floor(n/2)` observations
per class as the within-region controls:

```bash
python run_c2st.py samples/newville_nov/hold_skip/<run-name> \
  --comparison s-p-matched \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999 --seed 0
```

Neither the original Binomial test nor McNemar mathematically requires
repetition: use `--repeats 1` for one prespecified split. Repeats are used here
to measure split/grouping sensitivity and to form the control rejection counts.
They reuse the same finite recording, so their rejection rate is a diagnostic
rather than an estimate from 100 independent recordings. Failure to reject does
not prove equality.

## Sweep a sampling grid

`sweep_c2st.py` runs the whole `(s1, s2)` grid through sampling, feature
extraction and every comparison, then aggregates the results. It shells out to
the two scripts above with the same arguments you would type by hand, so the
per-run directories, feature caches and result JSONs are exactly what a manual
run produces — no YAML editing required, because `--s1`/`--s2` already override
the config defaults.

Reproduce the checked-in `newville_nov` summary from an existing sample/result
store with the following command. Samples and feature caches stay under the
specified samples root, while aggregate tables are written to
`sweep/newville_nov/` in this repository:

```bash
python sweep_c2st.py \
  --dataset newville_nov \
  --output-root /path/to/samples \
  --sweep-dir sweep/newville_nov \
  --s1 5 10 20 --s2 5 10 20 \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999 --jobs 4
```

This runs all four comparisons (`s-p`, `s-p-matched`, `s-s`, and `p-p`) for
the nine `(L, G)` cells and three methods. To preview the planned cells, sample
counts, and pending work without writing anything, append `--dry-run`:

```bash
python sweep_c2st.py \
  --dataset newville_nov \
  --output-root /path/to/samples \
  --sweep-dir sweep/newville_nov \
  --s1 5 10 20 --s2 5 10 20 \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999 --jobs 4 \
  --dry-run
```

The supplementary limiting-case diagnostic removes the inter-block gap while
retaining the three block durations. Its compact results are archived under
`sweep/newville_nov_no_gap/` and can be reproduced with:

```bash
python sweep_c2st.py \
  --dataset newville_nov \
  --output-root /path/to/samples \
  --sweep-dir sweep/newville_nov_no_gap \
  --s1 5 10 20 --s2 0 \
  --comparisons s-p s-p-matched s-s p-p \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999 --jobs 4
```

To rebuild only the CSV, JSON, and Markdown summaries from existing result
files, without sampling, extracting features, or running a C2ST again:

```bash
python sweep_c2st.py \
  --dataset newville_nov \
  --output-root /path/to/samples \
  --sweep-dir sweep/newville_nov \
  --s1 5 10 20 --s2 5 10 20 \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999 \
  --collect-only
```

Open the generated report in VS Code, then press `Ctrl+Shift+V` (or run
**Markdown: Open Preview** from the Command Palette) for the rendered tables:

```bash
code sweep/newville_nov/summary.md
```

For a plain-text terminal preview instead, run:

```bash
less sweep/newville_nov/summary.md
```

Replace `/path/to/samples` with your samples root. If samples are stored in this
repository's default `samples/` directory, omit `--output-root`.

Every stage resumes. A cell whose run directory exists is not re-sampled
(`--overwrite-samples` forces it), and a comparison that already has a result
file for the same representation, method, repeat count and seed is not
recomputed (`--force-c2st` forces it). ResNet-34 features are extracted once per
cell before the comparisons fan out, so the four comparisons share one cache.

Aggregated tables land in `sweep/<dataset_id>/` as `summary.csv`,
`summary.json` and a `summary.md` of per-comparison `L x G` grids holding
rejection rate, mean held-out accuracy and median p-value.

## Inference methods and calibration study

The explicit public inference functions are:

- `run_paired_mcnemar_c2st`: pair-preserving split and exact directional
  McNemar reference for `H0_dist`, assuming i.i.d. held-out pairs;
- `run_original_binomial_c2st`: the original C2ST observation split and simple
  Binomial accuracy reference, whose validity assumes independent held-out
  observations;
- `run_paired_swap_permutation_c2st`: pair-preserving split and full refitting
  for every within-pair transformation, targeting `H0_swap`;
- `run_pair_preserving_binomial_c2st`: the diagnostic pair-split-plus-Binomial
  variant.

Use the same `sweep_c2st.py` interface for one cell or the full `(L,G)` grid.
Existing samples, ResNet caches, and matching legacy swap results are reused:

```bash
python sweep_c2st.py \
  --output-root /path/to/samples \
  --dataset newville_nov --s1 5 --s2 5 \
  --methods original_binomial_c2st paired_mcnemar paired_swap_permutation \
  --repeats 100 --permutations 999
```

Per-method reports stay beside each sample run, while the sweep-level CSV, JSON,
and Markdown grids stay in `sweep/<dataset_id>/`. Repeat index zero is the
deterministic reference split for each cell. Other repeats are sensitivity
analyses: their p-values are not combined and they are not independent
replications.

Run the reproducible null-calibration and power study:

```bash
python simulate_c2st_calibration.py \
  --trials 1000 --sample-sizes 40 80 --permutations 199 \
  --jobs 4 --include-diagnostic
```

The six regimes cover an independent null, negatively dependent exchangeable
pairs, exact copies, the equal-marginal/nonexchangeable three-state cycle,
record-wide serial dependence, and a mean-shift alternative. Every summary
reports the rejection rate, Monte Carlo standard error, and Wilson interval,
and labels whether the result is valid Type-I calibration, out-of-assumption
null behavior, sensitivity to a swap-null violation, or power.

The default outputs are the compact CSV, JSON, and Markdown summaries used by
the supplementary material. Add `--save-trials` only when a compressed
per-trial audit trail is needed; raw trial dumps are reproducible intermediates
and are not version-controlled.

## Tests

The suite is standard-library `unittest` — no extra test dependency:

```bash
python -m unittest discover -s tests -v
```

## Provenance

Parts of this package are derived from
[UWGeoD/DAS_Preprocessing](https://github.com/UWGeoD/DAS_Preprocessing) at commit
`896e00005b68c7878ae80d50833bd9eeabe7ebc7`. See [PROVENANCE.md](PROVENANCE.md) and
[provenance/DAS_Preprocessing.json](provenance/DAS_Preprocessing.json) for exact
source files, line ranges, symbols, modifications and licensing. The two
repositories are maintained independently after that import.

## License

MIT — see [LICENSE](LICENSE).
