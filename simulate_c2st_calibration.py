#!/usr/bin/env python3
"""Monte Carlo calibration and power study for the three C2ST procedures."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dasgauge.c2st import (
    run_original_binomial_c2st,
    run_pair_preserving_binomial_c2st,
    run_paired_mcnemar_c2st,
    run_paired_swap_permutation_c2st,
)


REPO_DIR = Path(__file__).resolve().parent
HEADLINE_METHODS = (
    "paired_mcnemar",
    "original_binomial_c2st",
    "paired_swap_permutation",
)
DIAGNOSTIC_METHOD = "pair_preserving_binomial"
REGIMES = {
    "iid_null": {
        "label": "A. Independent i.i.d. null",
        "description": "Mutually independent Gaussian samples with identical marginals.",
        "scientific_null_holds": True,
        "swap_null_holds": True,
        "iid_pairs": True,
        "independent_individual_observations": True,
    },
    "negative_exchangeable": {
        "label": "B. I.i.d. exchangeable pairs with strong negative dependence",
        "description": (
            "Each pair has a shared continuous key and complementary Bernoulli coordinates, "
            "so (X,Y) and (Y,X) have the same law."
        ),
        "scientific_null_holds": True,
        "swap_null_holds": True,
        "iid_pairs": True,
        "independent_individual_observations": False,
    },
    "exact_copy": {
        "label": "C. Exact-copy pairs",
        "description": "Y_i = X_i with i.i.d. Gaussian pair values.",
        "scientific_null_holds": True,
        "swap_null_holds": True,
        "iid_pairs": True,
        "independent_individual_observations": False,
    },
    "cyclic_nonexchangeable": {
        "label": "D. Equal marginals, nonexchangeable i.i.d. pairs",
        "description": (
            "V is uniform on three states, X=x_V, and Y=x_(V+1 mod 3): "
            "H0_dist holds but H0_swap does not."
        ),
        "scientific_null_holds": True,
        "swap_null_holds": False,
        "iid_pairs": True,
        "independent_individual_observations": False,
    },
    "serial_common_state": {
        "label": "E. Across-pair dependence",
        "description": (
            "A common latent sign persists through the record; unconditional region "
            "marginals are equal, but held-out pairs are not independent."
        ),
        "scientific_null_holds": True,
        "swap_null_holds": False,
        "iid_pairs": False,
        "independent_individual_observations": False,
    },
    "mean_shift": {
        "label": "F. Mean-shift alternative",
        "description": "Independent Gaussian samples with a prespecified shift in one coordinate.",
        "scientific_null_holds": False,
        "swap_null_holds": False,
        "iid_pairs": True,
        "independent_individual_observations": True,
    },
}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "experiments" / "three_method_comparison",
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[40, 80])
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument(
        "--methods", nargs="+", choices=HEADLINE_METHODS, default=list(HEADLINE_METHODS)
    )
    parser.add_argument("--include-diagnostic", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--master-seed", type=int, default=20260821)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--mean-shift", type=float, default=0.8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--save-trials",
        action="store_true",
        help="also save the compressed per-trial JSONL audit trail",
    )
    return parser


def generate_regime(name, n_pairs, rng, *, dimension=8, mean_shift=0.8):
    """Generate one paired feature sample from a prespecified study regime."""
    if name not in REGIMES:
        raise ValueError(f"Unknown regime: {name}")
    n_pairs = int(n_pairs)
    dimension = int(dimension)
    if n_pairs < 2 or dimension < 2:
        raise ValueError("Require at least two pairs and feature dimension >= 2")

    if name == "iid_null":
        return (
            rng.normal(size=(n_pairs, dimension)),
            rng.normal(size=(n_pairs, dimension)),
        )

    if name == "negative_exchangeable":
        shared_key = rng.normal(size=(n_pairs, dimension - 1))
        bit = rng.integers(0, 2, size=n_pairs)
        group_a = np.column_stack((shared_key, bit))
        group_b = np.column_stack((shared_key, 1 - bit))
        return group_a, group_b

    if name == "exact_copy":
        group_a = rng.normal(size=(n_pairs, dimension))
        return group_a, group_a.copy()

    if name == "cyclic_nonexchangeable":
        centers = np.zeros((3, dimension))
        centers[:, :2] = np.array(
            [[1.0, 0.0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]]
        )
        state = rng.integers(0, 3, size=n_pairs)
        return centers[state], centers[(state + 1) % 3]

    if name == "serial_common_state":
        common_sign = 1 if rng.integers(0, 2) else -1
        direction = np.zeros(dimension)
        direction[0] = 1.0
        group_a = rng.normal(scale=0.45, size=(n_pairs, dimension))
        group_b = rng.normal(scale=0.45, size=(n_pairs, dimension))
        group_a += common_sign * mean_shift * direction
        group_b -= common_sign * mean_shift * direction
        return group_a, group_b

    group_a = rng.normal(size=(n_pairs, dimension))
    group_b = rng.normal(size=(n_pairs, dimension))
    group_b[:, 0] += mean_shift
    return group_a, group_b


def _reference_valid(regime, method):
    metadata = REGIMES[regime]
    if regime == "mean_shift":
        return None
    if method == "paired_mcnemar":
        return bool(metadata["scientific_null_holds"] and metadata["iid_pairs"])
    if method in {"original_binomial_c2st", "pair_preserving_binomial"}:
        return bool(
            metadata["scientific_null_holds"]
            and metadata["independent_individual_observations"]
        )
    if method == "paired_swap_permutation":
        return bool(metadata["swap_null_holds"])
    raise ValueError(f"Unknown method: {method}")


def _interpretation(regime, method):
    if regime == "mean_shift":
        return "power"
    if _reference_valid(regime, method):
        return "type_i_calibration"
    metadata = REGIMES[regime]
    target_holds = (
        metadata["swap_null_holds"]
        if method == "paired_swap_permutation"
        else metadata["scientific_null_holds"]
    )
    if target_holds:
        return "null_behavior_outside_method_assumptions"
    return "sensitivity_to_swap_null_violation_under_H0_dist"


def _run_method(method, group_a, group_b, settings, split_seed):
    common = {
        "test_fraction": settings["test_fraction"],
        "alpha": settings["alpha"],
        "seed": split_seed,
        "k": settings["k"],
    }
    if method == "paired_mcnemar":
        return run_paired_mcnemar_c2st(group_a, group_b, **common)
    if method == "original_binomial_c2st":
        return run_original_binomial_c2st(group_a, group_b, **common)
    if method == "pair_preserving_binomial":
        return run_pair_preserving_binomial_c2st(group_a, group_b, **common)
    if method == "paired_swap_permutation":
        return run_paired_swap_permutation_c2st(
            group_a,
            group_b,
            permutations=settings["permutations"],
            **common,
        )
    raise ValueError(f"Unknown method: {method}")


def _run_trial(task):
    regime, regime_index, n_pairs, trial_index, methods, settings = task
    master_seed = settings["master_seed"]
    data_sequence = np.random.SeedSequence(
        [master_seed, regime_index, n_pairs, trial_index, 0]
    )
    split_sequence = np.random.SeedSequence(
        [master_seed, regime_index, n_pairs, trial_index, 1]
    )
    rng = np.random.default_rng(data_sequence)
    split_seed = int(split_sequence.generate_state(1, dtype=np.uint32)[0])
    group_a, group_b = generate_regime(
        regime,
        n_pairs,
        rng,
        dimension=settings["dimension"],
        mean_shift=settings["mean_shift"],
    )
    records = []
    for method in methods:
        result = _run_method(method, group_a, group_b, settings, split_seed)
        records.append(
            {
                "regime": regime,
                "regime_label": REGIMES[regime]["label"],
                "n_pairs": int(n_pairs),
                "trial_index": int(trial_index),
                "data_seed_entropy": [master_seed, regime_index, n_pairs, trial_index, 0],
                "split_seed": split_seed,
                "method": method,
                "target_null": result["target_null"],
                "reference_valid_in_regime": _reference_valid(regime, method),
                "interpretation": _interpretation(regime, method),
                "accuracy": result["accuracy"],
                "p_value": result["p_value"],
                "reject_null": result["reject_null"],
                "N_plus": result.get("N_plus"),
                "N_minus": result.get("N_minus"),
                "N_disc": result.get("N_disc"),
                "n_held_out_pairs": result["n_held_out_pairs"],
                "n_held_out_individual_observations": result[
                    "n_held_out_individual_observations"
                ],
                "permutations": result["permutations"],
                "k": result["k"],
            }
        )
    return records


def run_calibration(args):
    if args.trials < 1:
        raise ValueError("trials must be >= 1")
    if any(n < 2 for n in args.sample_sizes):
        raise ValueError("sample sizes must be >= 2")
    methods = list(args.methods)
    if args.include_diagnostic and DIAGNOSTIC_METHOD not in methods:
        methods.append(DIAGNOSTIC_METHOD)
    settings = {
        "test_fraction": args.test_fraction,
        "alpha": args.alpha,
        "permutations": args.permutations,
        "master_seed": args.master_seed,
        "dimension": args.dimension,
        "k": args.k,
        "mean_shift": args.mean_shift,
    }
    regime_positions = {name: list(REGIMES).index(name) for name in args.regimes}
    tasks = [
        (regime, regime_positions[regime], n_pairs, trial, methods, settings)
        for regime in args.regimes
        for n_pairs in args.sample_sizes
        for trial in range(args.trials)
    ]
    records = []
    if args.jobs == 1:
        iterator = map(_run_trial, tasks)
        for index, trial_records in enumerate(iterator, start=1):
            records.extend(trial_records)
            if index % max(1, min(100, len(tasks))) == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)} simulated datasets", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            for index, trial_records in enumerate(
                pool.map(_run_trial, tasks, chunksize=max(1, len(tasks) // (args.jobs * 20))),
                start=1,
            ):
                records.extend(trial_records)
                if index % max(1, min(100, len(tasks))) == 0 or index == len(tasks):
                    print(f"completed {index}/{len(tasks)} simulated datasets", flush=True)
    return records, methods


def _wilson_interval(successes, trials, z=1.959963984540054):
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize(records):
    keys = sorted({(r["regime"], r["n_pairs"], r["method"]) for r in records})
    rows = []
    for regime, n_pairs, method in keys:
        selected = [
            r
            for r in records
            if (r["regime"], r["n_pairs"], r["method"])
            == (regime, n_pairs, method)
        ]
        rejected = int(sum(r["reject_null"] for r in selected))
        trial_count = len(selected)
        rate = rejected / trial_count
        ci_low, ci_high = _wilson_interval(rejected, trial_count)
        rows.append(
            {
                "regime": regime,
                "regime_label": REGIMES[regime]["label"],
                "n_pairs": n_pairs,
                "method": method,
                "target_null": selected[0]["target_null"],
                "reference_valid_in_regime": selected[0]["reference_valid_in_regime"],
                "interpretation": selected[0]["interpretation"],
                "trial_count": trial_count,
                "rejection_count": rejected,
                "rejection_rate": rate,
                "monte_carlo_standard_error": math.sqrt(rate * (1 - rate) / trial_count),
                "wilson_95_ci_low": ci_low,
                "wilson_95_ci_high": ci_high,
                "accuracy_mean": float(np.mean([r["accuracy"] for r in selected])),
                "p_value_median": float(np.median([r["p_value"] for r in selected])),
            }
        )
    return rows


def _write_outputs(args, records, rows, methods):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    if args.save_trials:
        trials_path = args.output_dir / "calibration_trials.jsonl.gz"
        with gzip.open(trials_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        output_paths.append(trials_path)

    csv_path = args.output_dir / "calibration_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "study": "C2ST null calibration and power",
        "alpha": args.alpha,
        "master_seed": args.master_seed,
        "trials_per_regime_and_sample_size": args.trials,
        "sample_sizes": args.sample_sizes,
        "methods": methods,
        "permutations": args.permutations,
        "k": args.k,
        "regimes": {name: REGIMES[name] for name in args.regimes},
        "rows": rows,
    }
    json_path = args.output_dir / "calibration_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    def rates(regime, selected_methods):
        return [
            row["rejection_rate"]
            for row in rows
            if row["regime"] == regime and row["method"] in selected_methods
        ]

    findings = []
    iid_rates = rates("iid_null", HEADLINE_METHODS)
    if iid_rates:
        findings.append(
            "- Under the independent i.i.d. null, all three headline references were "
            f"valid and rejection rates ranged from {min(iid_rates):.3f} to "
            f"{max(iid_rates):.3f}."
        )
    paired_rates = rates(
        "negative_exchangeable", ("paired_mcnemar", "paired_swap_permutation")
    )
    diagnostic_rates = rates("negative_exchangeable", (DIAGNOSTIC_METHOD,))
    if paired_rates:
        sentence = (
            "- Under strongly dependent but exchangeable pairs, the valid McNemar and "
            f"swap rates ranged from {min(paired_rates):.3f} to {max(paired_rates):.3f}."
        )
        if diagnostic_rates:
            sentence += (
                " The pair-split Binomial diagnostic rejected at rates "
                + "/".join(f"{value:.3f}" for value in diagnostic_rates)
                + ", showing anti-conservative individual-correctness calibration."
            )
        findings.append(sentence)
    cyclic_rates = rates("cyclic_nonexchangeable", ("paired_swap_permutation",))
    if cyclic_rates:
        findings.append(
            "- The cyclic equal-marginal construction violated the swap null but the "
            f"chosen accuracy statistic rejected at rates {min(cyclic_rates):.3f}--"
            f"{max(cyclic_rates):.3f}; this is lack of sensitivity, not validation of "
            "swap invariance."
        )
    serial_rates = rates("serial_common_state", HEADLINE_METHODS)
    if serial_rates:
        findings.append(
            "- With record-wide dependence, all headline references were outside their "
            f"stated assumptions and rejection rates were {min(serial_rates):.3f}--"
            f"{max(serial_rates):.3f}."
        )
    power_rates = rates("mean_shift", HEADLINE_METHODS)
    if power_rates:
        findings.append(
            "- Under the mean-shift alternative, headline power ranged from "
            f"{min(power_rates):.3f} to {max(power_rates):.3f} across the two sample sizes."
        )

    lines = [
        "# C2ST calibration and power study",
        "",
        "Rates labeled `type_i_calibration` estimate Type-I error for a method under its",
        "own target null and assumptions. Other null rows are deliberately outside those",
        "assumptions or, for the swap method in regimes D/E, outside `H0_swap`.",
        "",
        "## Key findings",
        "",
        *findings,
        "",
        "## Complete results",
        "",
        "| regime | n pairs | method | interpretation | reject rate | MCSE | 95% Wilson CI |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime_label']} | {row['n_pairs']} | `{row['method']}` | "
            f"{row['interpretation']} | {row['rejection_rate']:.3f} | "
            f"{row['monte_carlo_standard_error']:.3f} | "
            f"[{row['wilson_95_ci_low']:.3f}, {row['wilson_95_ci_high']:.3f}] |"
        )
    markdown_path = args.output_dir / "CALIBRATION_RESULTS.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tuple(output_paths + [csv_path, json_path, markdown_path])


def main(argv=None):
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    records, methods = run_calibration(args)
    rows = summarize(records)
    paths = _write_outputs(args, records, rows, methods)
    print(
        json.dumps(
            {
                "simulated_dataset_count": len(records) // len(methods),
                "method_result_count": len(records),
                "summary_row_count": len(rows),
                "outputs": [str(path) for path in paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
