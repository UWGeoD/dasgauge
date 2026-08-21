#!/usr/bin/env python3
"""Sweep a hold/skip (s1, s2) grid through sampling, features, and every C2ST comparison.

Wraps the existing split_samples.py and run_c2st.py command lines; it adds no new
sampling or inference behaviour. Every stage resumes: an existing run directory is
not re-sampled and an existing matching result file is not recomputed.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_c2st
from dasgauge.config import find_dataset_config, preprocess_steps
from dasgauge.sampling import make_run_name

REPO_DIR = Path(__file__).resolve().parent
STRATEGY = "hold_skip"
COMPARISONS = ("s-p", "s-p-matched", "s-s", "p-p")
METHOD_LABELS = {
    "original_binomial_c2st": "Original Binomial C2ST",
    "paired_mcnemar": "Paired McNemar",
    "paired_swap_permutation": "Paired swap permutation",
    "pair_preserving_binomial": "Pair-preserving Binomial diagnostic",
}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="newville_nov", help="dataset_id or YAML path")
    parser.add_argument("--s1", nargs="+", type=float, default=[5, 10, 20],
                        help="sample lengths L in seconds")
    parser.add_argument("--s2", nargs="+", type=float, default=[5, 10, 20],
                        help="skipped gaps G in seconds")
    parser.add_argument("--comparisons", nargs="+", choices=COMPARISONS, default=list(COMPARISONS))
    parser.add_argument("--coupling-channels", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--uncoupling-channels", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--output-root", type=Path, default=REPO_DIR / "samples")
    parser.add_argument("--sweep-dir", type=Path, help="default: <repo>/sweep/<dataset_id>")

    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--representation", choices=("resnet34", "summary_v1"), default="resnet34")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=run_c2st.METHODS + tuple(run_c2st.METHOD_ALIASES),
        help=(
            "inference methods per grid cell; default preserves --calibration behavior "
            "(permutation=paired swap, binomial=pair-preserving diagnostic)"
        ),
    )
    parser.add_argument(
        "--calibration", choices=("permutation", "binomial"), default="permutation",
        help="legacy single-method selector used only when --methods is omitted",
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--permutation-unit", choices=("pair", "block"), default="pair")
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)

    parser.add_argument("--jobs", type=int, default=4, help="concurrent run_c2st.py processes")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--collect-only", action="store_true", help="re-aggregate existing results")
    parser.add_argument("--overwrite-samples", action="store_true",
                        help="re-sample cells whose run directory already exists")
    parser.add_argument("--force-c2st", action="store_true",
                        help="rerun comparisons that already have a matching result file")
    return parser


def _setting(cli_value, defaults, key):
    value = cli_value if cli_value is not None else defaults.get(key)
    if value is None:
        raise ValueError(f"Set --{key.replace('_', '-')} or define sampling_defaults.{key}")
    return tuple(value)


def _plan(args):
    """Resolve the grid to run directories using the same naming as split_samples.py."""
    config = find_dataset_config(args.dataset, REPO_DIR / "configs")
    defaults = config.get("sampling_defaults", {})
    coupling = _setting(args.coupling_channels, defaults, "coupling_channels")
    uncoupling = _setting(args.uncoupling_channels, defaults, "uncoupling_channels")
    steps = preprocess_steps(config)

    cells = []
    for s1 in args.s1:
        for s2 in args.s2:
            parameters = {"s1": s1, "s2": s2}
            run_name = make_run_name(STRATEGY, parameters, coupling, uncoupling, steps)
            cells.append(
                SimpleNamespace(
                    s1=s1,
                    s2=s2,
                    run_dir=args.output_root / config["dataset_id"] / STRATEGY / run_name,
                )
            )
    return config, coupling, uncoupling, cells


def _planned_sample_counts(config, cells):
    """Best-effort pair counts, read from the recording metadata only."""
    from dasgauge.io import DASRecording
    from dasgauge.sampling import hold_skip_windows

    root = Path(config["recording_dir"])
    recording = DASRecording(
        list(root.glob(config.get("file_pattern", "*.h5"))), vendor=config["vendor"]
    )
    counts = {
        (cell.s1, cell.s2): len(hold_skip_windows(recording.n_samples, recording.fs, cell.s1, cell.s2))
        for cell in cells
    }
    return counts, recording.duration_seconds


def _selected_methods(args):
    return run_c2st._selected_methods(args)


def _result_prefix(args, comparison, method=None):
    result_tag = args.calibration if method is None else method
    return (
        f"c2st__{comparison}__{args.representation}__knn__{result_tag}"
        f"__repeats-{args.repeats}__seed-{args.seed}__"
    )


def _existing_result(run_dir, args, comparison, method=None):
    """Newest result file matching this exact configuration, if any."""
    prefixes = [_result_prefix(args, comparison, method)]
    if method == "paired_swap_permutation":
        prefixes.append(_result_prefix(args, comparison, None).replace(
            f"__{args.calibration}__", "__permutation__"
        ))
    elif method == "pair_preserving_binomial":
        prefixes.append(_result_prefix(args, comparison, None).replace(
            f"__{args.calibration}__", "__binomial__"
        ))
    matches = sorted({
        path
        for prefix in prefixes
        for path in (run_dir / "results").glob(prefix + "*.json")
    })
    return matches[-1] if matches else None


def _feature_args(args, run_dir):
    return SimpleNamespace(
        sample_dir=run_dir,
        representation=args.representation,
        batch_size=args.batch_size,
        device=args.device,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        no_feature_cache=False,
    )


def _log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def stage_sample(args, cell):
    if cell.run_dir.is_dir() and not args.overwrite_samples:
        _log(f"sample  L={cell.s1:g} G={cell.s2:g}  reuse {cell.run_dir.name}")
        return
    command = [
        sys.executable, str(REPO_DIR / "split_samples.py"),
        "--strategy", STRATEGY,
        "--dataset", args.dataset,
        "--s1", f"{cell.s1:g}",
        "--s2", f"{cell.s2:g}",
        "--output-root", str(args.output_root),
    ]
    if args.coupling_channels:
        command += ["--coupling-channels", *map(str, args.coupling_channels)]
    if args.uncoupling_channels:
        command += ["--uncoupling-channels", *map(str, args.uncoupling_channels)]
    if args.overwrite_samples:
        command.append("--overwrite")

    started = time.time()
    _log(f"sample  L={cell.s1:g} G={cell.s2:g}  writing {cell.run_dir.name}")
    subprocess.run(command, cwd=REPO_DIR, check=True, stdout=subprocess.DEVNULL)
    _log(f"sample  L={cell.s1:g} G={cell.s2:g}  done in {time.time() - started:.0f}s")


def stage_features(args, cell):
    """Warm the feature cache once per cell so parallel comparisons all hit it."""
    started = time.time()
    _, _, cache_path, cache_hit = run_c2st._features(
        _feature_args(args, cell.run_dir), cell.run_dir / "samples.h5"
    )
    verb = "cached" if cache_hit else f"extracted in {time.time() - started:.0f}s"
    _log(f"feature L={cell.s1:g} G={cell.s2:g}  {verb} -> {cache_path.name}")


def stage_c2st(args, cell, comparison, method, env):
    existing = _existing_result(cell.run_dir, args, comparison, method)
    if existing is not None and not args.force_c2st:
        _log(
            f"c2st    L={cell.s1:g} G={cell.s2:g} {comparison:12s} "
            f"{method:24s} reuse {existing.name}"
        )
        return
    command = [
        sys.executable, str(REPO_DIR / "run_c2st.py"), str(cell.run_dir),
        "--comparison", comparison,
        "--representation", args.representation,
        "--methods", method,
        "--permutations", str(args.permutations),
        "--permutation-unit", args.permutation_unit,
        "--block-size", str(args.block_size),
        "--test-fraction", str(args.test_fraction),
        "--alpha", str(args.alpha),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--lower-quantile", str(args.lower_quantile),
        "--upper-quantile", str(args.upper_quantile),
    ]
    if args.k is not None:
        command += ["--k", str(args.k)]

    started = time.time()
    _log(f"c2st    L={cell.s1:g} G={cell.s2:g} {comparison:12s} {method:24s} started")
    subprocess.run(command, cwd=REPO_DIR, check=True, stdout=subprocess.DEVNULL, env=env)
    _log(
        f"c2st    L={cell.s1:g} G={cell.s2:g} {comparison:12s} {method:24s} "
        f"done in {time.time() - started:.0f}s"
    )


CSV_FIELDS = [
    "s1", "s2", "comparison", "method", "target_null", "n_samples", "n_pairs",
    "repeats", "k", "n_test_examples",
    "accuracy_mean", "accuracy_median", "accuracy_minimum", "accuracy_maximum",
    "p_value_mean", "p_value_median", "p_value_minimum", "p_value_maximum",
    "rejection_count", "rejection_rate", "alpha", "result_file",
]


def _collect_row(args, cell, comparison, method=None):
    result_file = _existing_result(cell.run_dir, args, comparison, method)
    if result_file is None:
        return None
    with result_file.open(encoding="utf-8") as handle:
        report = json.load(handle)
    summary = report["summary"]
    first = report["trials"][0]
    manifest_file = cell.run_dir / "manifest.json"
    n_samples = ""
    if manifest_file.is_file():
        with manifest_file.open(encoding="utf-8") as handle:
            n_samples = json.load(handle)["samples"]["count"]
    return {
        "s1": f"{cell.s1:g}",
        "s2": f"{cell.s2:g}",
        "comparison": comparison,
        "method": report.get("method") or method or _selected_methods(args)[0],
        "target_null": report.get("target_null") or first["c2st"].get("target_null", ""),
        "n_samples": n_samples,
        "n_pairs": len(first["grouping"]["group_a_sample_indices"]),
        "repeats": summary["repeat_count"],
        "k": first["c2st"]["k"],
        "n_test_examples": first["c2st"]["n_test_examples"],
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_median": summary["accuracy_median"],
        "accuracy_minimum": summary["accuracy_minimum"],
        "accuracy_maximum": summary["accuracy_maximum"],
        "p_value_mean": summary["p_value_mean"],
        "p_value_median": summary["p_value_median"],
        "p_value_minimum": summary["p_value_minimum"],
        "p_value_maximum": summary["p_value_maximum"],
        "rejection_count": summary["rejection_count"],
        "rejection_rate": summary["rejection_rate"],
        "alpha": summary["alpha"],
        "result_file": result_file.relative_to(args.output_root).as_posix(),
    }


def _grid_table(rows, args, comparison, key, fmt, method=None):
    lookup = {
        (row["s1"], row["s2"]): row
        for row in rows
        if row["comparison"] == comparison and (method is None or row["method"] == method)
    }
    header = " | ".join([f"L \\ G"] + [f"{s2:g} s" for s2 in args.s2])
    lines = [f"| {header} |", "| " + " | ".join(["---"] * (len(args.s2) + 1)) + " |"]
    for s1 in args.s1:
        cells = []
        for s2 in args.s2:
            row = lookup.get((f"{s1:g}", f"{s2:g}"))
            value = None if row is None else row[key]
            cells.append("—" if value in (None, "") else format(value, fmt))
        lines.append("| " + " | ".join([f"**{s1:g} s**"] + cells) + " |")
    return "\n".join(lines)


def stage_collect(args, cells, sweep_dir):
    methods = _selected_methods(args)
    rows = [
        row
        for cell in cells
        for comparison in args.comparisons
        for method in methods
        if (row := _collect_row(args, cell, comparison, method)) is not None
    ]
    if not rows:
        _log("collect no matching result files found")
        return rows

    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    json_path = sweep_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "strategy": STRATEGY,
                "grid": {"s1": args.s1, "s2": args.s2},
                "comparisons": args.comparisons,
                "methods": methods,
                "repeats": args.repeats,
                "representation": args.representation,
                "legacy_calibration": args.calibration if args.methods is None else None,
                "permutations": args.permutations,
                "test_fraction": args.test_fraction,
                "alpha": args.alpha,
                "seed": args.seed,
                "rows": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    blocks = [
        f"# C2ST sweep — {args.dataset}",
        "",
        f"L = `s1` (sample length), G = `s2` (skipped gap). "
        f"{args.repeats} repeats, {args.representation} features, KNN, "
        f"methods = {', '.join(methods)}, alpha = {args.alpha:g}.",
        "",
        "## Samples per cell",
        "",
        _grid_table(rows, args, args.comparisons[0], "n_samples", "d", methods[0]),
    ]
    for comparison in args.comparisons:
        blocks += ["", f"## {comparison}"]
        for method in methods:
            blocks += [
                "",
                f"### {METHOD_LABELS[method]}",
                "",
                "Rejection rate over repeats:",
                "",
                _grid_table(rows, args, comparison, "rejection_rate", ".2f", method),
                "",
                "Mean held-out accuracy:",
                "",
                _grid_table(rows, args, comparison, "accuracy_mean", ".3f", method),
                "",
                "Median p-value:",
                "",
                _grid_table(rows, args, comparison, "p_value_median", ".3f", method),
            ]
    markdown_path = sweep_dir / "summary.md"
    markdown_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")

    _log(f"collect {len(rows)} rows -> {csv_path}, {json_path}, {markdown_path}")
    return rows


def main(argv=None):
    args = _parser().parse_args(argv)
    args.output_root = args.output_root.resolve()
    config, coupling, uncoupling, cells = _plan(args)
    methods = _selected_methods(args)
    sweep_dir = args.sweep_dir or REPO_DIR / "sweep" / config["dataset_id"]

    if args.collect_only:
        stage_collect(args, cells, sweep_dir)
        return 0

    if args.dry_run:
        try:
            counts, duration = _planned_sample_counts(config, cells)
        except Exception as error:  # recording may be unmounted
            counts, duration = {}, None
            _log(f"recording unavailable, sample counts unknown ({error})")
        plan = {
            "dataset_id": config["dataset_id"],
            "recording_duration_seconds": duration,
            "channels": {"coupling": list(coupling), "uncoupling": list(uncoupling)},
            "preprocess": [{"name": name, "params": params} for name, params in preprocess_steps(config)],
            "comparisons": args.comparisons,
            "methods": methods,
            "repeats": args.repeats,
            "sweep_dir": str(sweep_dir),
            "cells": [
                {
                    "s1": cell.s1,
                    "s2": cell.s2,
                    "planned_samples": counts.get((cell.s1, cell.s2)),
                    "run_dir": str(cell.run_dir),
                    "sampled": cell.run_dir.is_dir(),
                    "pending_runs": [
                        {"comparison": comparison, "method": method}
                        for comparison in args.comparisons
                        for method in methods
                        if args.force_c2st
                        or _existing_result(cell.run_dir, args, comparison, method) is None
                    ],
                }
                for cell in cells
            ],
        }
        print(json.dumps(plan, indent=2))
        return 0

    started = time.time()
    for cell in cells:
        stage_sample(args, cell)
    for cell in cells:
        stage_features(args, cell)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = [
        (cell, comparison, method)
        for cell in cells
        for comparison in args.comparisons
        for method in methods
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(stage_c2st, args, cell, comparison, method, env)
            for cell, comparison, method in jobs
        ]
        for future in futures:
            future.result()

    rows = stage_collect(args, cells, sweep_dir)
    _log(f"sweep complete: {len(cells)} cells x {len(args.comparisons)} comparisons "
         f"x {len(methods)} methods "
         f"({len(rows)} rows) in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
