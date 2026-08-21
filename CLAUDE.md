# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt         # full editable environment

python -m unittest discover -s tests -v            # full suite
python -m unittest tests.test_preprocessing        # one module
python -m unittest tests.test_preprocessing.PipelineTests.test_runs_full_chain
```

There is no linter, formatter, or CI configured.

## Architecture

The package also includes `sampling` for continuous multi-file sample stores,
`config` for dataset YAML, and `c2st` for frozen representations, KNN, and inference.
The original `io`/`preprocessing`/`plotting` boundaries remain unchanged.

**Array convention.** Every function in `preprocessing` and `plotting` takes and returns 2-D
`(n_channels, n_time)` float arrays, time on `axis=1`. `dasgauge/io.py` transposes vendor HDF5
layouts (which are stored `(time, channels)`) at read time so the rest of the package never
deals with the other orientation. `_as_2d_float` in [preprocessing.py](dasgauge/preprocessing.py)
enforces this at every entry point.

**`DAS` reader.** [io.py](dasgauge/io.py) dispatches on the `vendor` kwarg to
`_read_optasense` / `_read_silixa` / `_read_raw_hdf5`, each of which fills the same
`self.data` + `self.meta` contract. `meta` always carries the keys `dx`, `fs`, `dt`,
`start_time_dt`, `start_time_iso`, `vendor`, `raw_path`, `raw_shape` — readers fill missing
values with `None`/`nan` rather than omitting keys, so downstream code can index blindly.
Adding a vendor means adding a `_read_*` method that populates that full key set plus a
dispatch branch. `raw_hdf5` has no timestamp metadata and requires explicit `fs`/`dx` kwargs.

**Continuous recordings.** `DASRecording` inspects OptaSense metadata without loading signal
arrays, sorts parts by `PartStartTime`, validates sample-level continuity and common metadata,
then reads only requested time/channel slices. Global half-open ranges may cross files.

**Samples and C2ST.** `sampling.split_recording` writes paired coupling/uncoupling arrays to one
sample-major, LZF-compressed HDF5 file plus CSV/JSON provenance. `c2st` lazily imports optional
PyTorch, removes ResNet-34's classifier to produce 512-D features, and runs KNN with paired or
block label-swap calibration. Train/test splitting occurs at the pair/block level.

**Preprocessing pipeline.** The primitives (`detrend_linear`, `bandpass_sos`, `fk_filter`,
`curvelet_denoise`, `hilbert_transform`, `downsample_time`, `integrate_strain_rate_fd`) are
stateless pure functions usable on their own. `make_preprocess(steps, dx=..., dt=...)` composes
them into a closure `preprocess(x, fs, **ctx) -> y`. Steps are named strings resolved through
`_STEP_REGISTRY`; a new step needs a registry entry *and* a dispatch branch inside the
`preprocess` closure, because each step has a different argument-injection rule (`bandpass`
gets `fs`, `fk_filter` gets `dx`/`dt`, `detrend` gets neither).

The closure tracks `current_fs` across steps: `downsample` divides it by the decimation factor
and recomputes `dt_`, so any `bandpass` or `integrate_fd` placed after a `downsample` correctly
sees the reduced rate. Step order is therefore semantically significant, not just cosmetic.

**Domain constraints that are easy to get wrong:**
- `downsample_time` is bare decimation with *no* anti-alias filter. It must follow a `bandpass`
  whose `f_hi < target_fs / 2`.
- `integrate_strain_rate_fd` uses the negative sign convention `-Y/(2jπf)` from van den Ende
  et al. (2023). Flipping it inverts every strain value. Prefer it over `integrate_to_strain`
  (cumulative trapezoid), which accumulates low-frequency drift.
- `curvelet_denoise` is a from-scratch NumPy/FFT implementation of the Candès et al. (2006)
  curvelet frame — deliberately *no* CurveLab/PyCurvelab dependency. The Meyer window helpers
  (`_meyer_W`, `_meyer_V`) and the centrosymmetric cone pairing exist to satisfy the tight-frame
  partition of unity; changing them breaks reconstruction, which
  `test_zero_threshold_approximately_reconstructs` guards.

**Optional matplotlib.** [plotting.py](dasgauge/plotting.py) imports matplotlib lazily via the
`_pyplot()` helper so `import dasgauge` works without it and only calling a plot function fails,
with an install message. Keep any new plotting code behind `_pyplot()`.

## Provenance obligations

Parts of this package are derived from `UWGeoD/DAS_Preprocessing` at commit
`896e00005b68c7878ae80d50833bd9eeabe7ebc7`. Two invariants are enforced by
[tests/test_independence.py](tests/test_independence.py) and will fail the suite if broken:

1. Every derived module's docstring must name that exact source commit.
2. `dasgauge/` may import only declared dependencies (`numpy`, `scipy`, `h5py`, `matplotlib`,
   `yaml`, and optional `torch`/`torchvision`) and the standard library —
   no imports from the source project, no `sys.path` tampering, no absolute path constants.

When adding or moving derived code, update both [PROVENANCE.md](PROVENANCE.md) and
[provenance/DAS_Preprocessing.json](provenance/DAS_Preprocessing.json) with the source symbol,
line range and modifications.

---

# Behavioral guidelines

Adapted from [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) —
guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require
constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to
overcomplication, and clarifying questions come before implementation rather than after mistakes.
