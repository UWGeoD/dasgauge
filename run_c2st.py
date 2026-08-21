#!/usr/bin/env python3
"""Extract DAS features and run a paired classifier two-sample test."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dasgauge.c2st import (
    extract_resnet34_features,
    extract_summary_features,
    make_comparison_groups,
    run_original_binomial_c2st,
    run_pair_preserving_binomial_c2st,
    run_paired_mcnemar_c2st,
    run_paired_swap_permutation_c2st,
)


METHODS = (
    "original_binomial_c2st",
    "paired_mcnemar",
    "paired_swap_permutation",
    "pair_preserving_binomial",
)
METHOD_ALIASES = {"independent_binomial": "original_binomial_c2st"}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument(
        "--comparison",
        choices=("s-p", "s-p-matched", "s-s", "p-p"),
        default="s-p",
        help="S=coupling, P=uncoupling; same-region modes are negative controls",
    )
    parser.add_argument(
        "--representation", choices=("resnet34", "summary_v1"), default="resnet34"
    )
    parser.add_argument("--classifier", choices=("knn",), default="knn")
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS + tuple(METHOD_ALIASES),
        help=(
            "inference methods to run; default preserves --calibration behavior "
            "(permutation=paired swap, binomial=pair-preserving diagnostic)"
        ),
    )
    parser.add_argument(
        "--calibration",
        choices=("permutation", "binomial"),
        default="permutation",
        help="legacy single-method selector used only when --methods is omitted",
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--permutation-unit", choices=("pair", "block"), default="pair")
    parser.add_argument("--block-size", type=int, default=1, help="consecutive pairs per block")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeats",
        "--control-repeats",
        dest="repeats",
        type=int,
        help="independent group/split seeds (default: 1 for s-p, 100 for controls)",
    )
    parser.add_argument("--k", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--no-feature-cache", action="store_true")
    return parser


def _feature_cache_name(args):
    if args.representation == "summary_v1":
        return "summary_v1.npz"
    lower = f"{args.lower_quantile:g}".replace(".", "p")
    upper = f"{args.upper_quantile:g}".replace(".", "p")
    return f"resnet34_imagenet1k_v1__resize-224__q-{lower}-{upper}.npz"


def _features(args, sample_file):
    cache_dir = args.sample_dir / "features"
    cache_path = cache_dir / _feature_cache_name(args)
    if cache_path.is_file() and not args.no_feature_cache:
        with np.load(cache_path) as cached:
            return cached["coupling"], cached["uncoupling"], cache_path, True

    if args.representation == "resnet34":
        coupling, uncoupling = extract_resnet34_features(
            sample_file,
            batch_size=args.batch_size,
            device=args.device,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
        )
    else:
        coupling, uncoupling = extract_summary_features(sample_file)

    if not args.no_feature_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, coupling=coupling, uncoupling=uncoupling)
    return coupling, uncoupling, cache_path, False


def _trial_seeds(seed, repeats, comparison):
    if repeats == 1 and comparison == "s-p":
        return [(int(seed), int(seed))]
    rng = np.random.default_rng(seed)
    maximum = np.iinfo(np.uint32).max
    return [
        (
            int(rng.integers(0, maximum, dtype=np.uint32)),
            int(rng.integers(0, maximum, dtype=np.uint32)),
        )
        for _ in range(repeats)
    ]


def _compact_null_statistics(result):
    values = np.asarray(result.pop("null_statistics"), dtype=float)
    if not len(values):
        return
    result["null_statistic_summary"] = {
        "minimum": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _selected_methods(args):
    if args.methods is None:
        return [
            "paired_swap_permutation"
            if args.calibration == "permutation"
            else "pair_preserving_binomial"
        ]
    methods = []
    for method in args.methods:
        canonical = METHOD_ALIASES.get(method, method)
        if canonical not in methods:
            methods.append(canonical)
    return methods


def _run_method(args, method, group_a, group_b, seed):
    common = {
        "test_fraction": args.test_fraction,
        "alpha": args.alpha,
        "seed": seed,
        "k": args.k,
    }
    if method == "original_binomial_c2st":
        return run_original_binomial_c2st(group_a, group_b, **common)
    if method == "paired_mcnemar":
        return run_paired_mcnemar_c2st(
            group_a,
            group_b,
            split_unit=args.permutation_unit,
            block_size=args.block_size,
            **common,
        )
    if method == "paired_swap_permutation":
        return run_paired_swap_permutation_c2st(
            group_a,
            group_b,
            permutations=args.permutations,
            permutation_unit=args.permutation_unit,
            block_size=args.block_size,
            **common,
        )
    if method == "pair_preserving_binomial":
        return run_pair_preserving_binomial_c2st(
            group_a,
            group_b,
            split_unit=args.permutation_unit,
            block_size=args.block_size,
            **common,
        )
    raise ValueError(f"Unknown method: {method}")


def _summary(trials, alpha):
    statistics = np.asarray([trial["c2st"]["statistic"] for trial in trials])
    p_values = np.asarray([trial["c2st"]["p_value"] for trial in trials])
    rejected = np.asarray([trial["c2st"]["reject_null"] for trial in trials])
    return {
        "repeat_count": len(trials),
        "alpha": float(alpha),
        "rejection_count": int(rejected.sum()),
        "rejection_rate": float(rejected.mean()),
        "accuracy_mean": float(statistics.mean()),
        "accuracy_median": float(np.median(statistics)),
        "accuracy_minimum": float(statistics.min()),
        "accuracy_maximum": float(statistics.max()),
        "p_value_mean": float(p_values.mean()),
        "p_value_median": float(np.median(p_values)),
        "p_value_minimum": float(p_values.min()),
        "p_value_maximum": float(p_values.max()),
    }


def main(argv=None):
    args = _parser().parse_args(argv)
    args.sample_dir = args.sample_dir.resolve()
    sample_file = args.sample_dir / "samples.h5"
    manifest_file = args.sample_dir / "manifest.json"
    if not sample_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(f"Not a sample run directory: {args.sample_dir}")

    coupling, uncoupling, cache_path, cache_hit = _features(args, sample_file)
    repeats = args.repeats
    if repeats is None:
        repeats = 1 if args.comparison == "s-p" else 100
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    methods = _selected_methods(args)
    trial_seeds = _trial_seeds(args.seed, repeats, args.comparison)
    result_dir = args.sample_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for method in methods:
        trials = []
        for repeat_index, (assignment_seed, c2st_seed) in enumerate(trial_seeds):
            group_a, group_b, grouping = make_comparison_groups(
                coupling,
                uncoupling,
                args.comparison,
                seed=assignment_seed,
            )
            c2st = _run_method(args, method, group_a, group_b, c2st_seed)
            source_pairs = grouping["source_index_pairs"]
            grouping["train_source_index_pairs"] = [
                source_pairs[index] for index in c2st["train_pair_indices"]
            ]
            grouping["test_source_index_pairs"] = [
                source_pairs[index] for index in c2st["test_pair_indices"]
            ]
            if repeats > 1:
                _compact_null_statistics(c2st)
            trials.append(
                {
                    "repeat_index": repeat_index,
                    "assignment_seed": grouping["assignment_seed"],
                    "c2st_seed": c2st_seed,
                    "grouping": grouping,
                    "c2st": c2st,
                }
            )

        report = {
            "sample_dir": str(args.sample_dir),
            "comparison": args.comparison,
            "distribution_names": {"S": "coupling", "P": "uncoupling"},
            "representation": args.representation,
            "classifier": args.classifier,
            "method": method,
            "target_null": trials[0]["c2st"]["target_null"],
            "classifier_training": {
                "fresh_fit_per_repeat": True,
                "fresh_fit_per_permutation": method == "paired_swap_permutation",
                "test_examples_excluded_from_training": True,
            },
            "repeated_control_interpretation": (
                "Repeated trials reuse the finite recording and are calibration diagnostics, "
                "not independent replications."
                if repeats > 1
                else None
            ),
            "feature_dimension": int(coupling.shape[1]),
            "feature_cache": str(cache_path),
            "feature_cache_hit": cache_hit,
            "master_seed": args.seed,
            "summary": _summary(trials, args.alpha),
            "trials": trials,
            "resnet34": None
            if args.representation != "resnet34"
            else {
                "role": "shared frozen feature extractor; not refit for C2ST comparisons",
                "weights": "IMAGENET1K_V1",
                "output_dimension": 512,
                "resize": [224, 224],
                "input_mapping": "per-sample quantile clip, [0,1], grayscale repeated to RGB",
                "lower_quantile": args.lower_quantile,
                "upper_quantile": args.upper_quantile,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result_tag = method if args.methods is not None else args.calibration
        result_file = result_dir / (
            f"c2st__{args.comparison}__{args.representation}__knn__{result_tag}"
            f"__repeats-{repeats}__seed-{args.seed}__{timestamp}.json"
        )
        with result_file.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        outputs.append(
            {
                "method": method,
                "target_null": report["target_null"],
                "summary": report["summary"],
                "result_file": str(result_file),
            }
        )

    payload = {"comparison": args.comparison, "results": outputs}
    if len(outputs) == 1:
        payload.update(outputs[0])
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
