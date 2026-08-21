"""Smoke tests for the C2ST null-calibration experiment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import simulate_c2st_calibration as calibration


class RegimeTests(unittest.TestCase):
    def test_constructed_pair_nulls_have_the_claimed_structure(self):
        rng = np.random.default_rng(4)
        negative_a, negative_b = calibration.generate_regime(
            "negative_exchangeable", 20, rng, dimension=6
        )
        np.testing.assert_array_equal(negative_a[:, :-1], negative_b[:, :-1])
        np.testing.assert_array_equal(negative_a[:, -1] + negative_b[:, -1], 1)

        copy_a, copy_b = calibration.generate_regime(
            "exact_copy", 20, rng, dimension=6
        )
        np.testing.assert_array_equal(copy_a, copy_b)

        cyclic_a, cyclic_b = calibration.generate_regime(
            "cyclic_nonexchangeable", 30, rng, dimension=6
        )
        self.assertEqual(len(np.unique(cyclic_a, axis=0)), 3)
        self.assertEqual(len(np.unique(cyclic_b, axis=0)), 3)
        self.assertTrue(calibration.REGIMES["cyclic_nonexchangeable"]["scientific_null_holds"])
        self.assertFalse(calibration.REGIMES["cyclic_nonexchangeable"]["swap_null_holds"])

    def test_small_calibration_trial_is_reproducible_and_labeled(self):
        settings = {
            "test_fraction": 0.5,
            "alpha": 0.05,
            "permutations": 5,
            "master_seed": 91,
            "dimension": 4,
            "k": 1,
            "mean_shift": 0.8,
        }
        methods = list(calibration.HEADLINE_METHODS)
        task = ("iid_null", 0, 12, 0, methods, settings)
        first = calibration._run_trial(task)
        second = calibration._run_trial(task)
        self.assertEqual(first, second)
        self.assertEqual({row["method"] for row in first}, set(methods))
        self.assertTrue(all(row["reference_valid_in_regime"] for row in first))
        self.assertTrue(all(row["interpretation"] == "type_i_calibration" for row in first))

    def test_cyclic_regime_separates_method_targets(self):
        self.assertTrue(calibration._reference_valid("cyclic_nonexchangeable", "paired_mcnemar"))
        self.assertFalse(
            calibration._reference_valid(
                "cyclic_nonexchangeable", "paired_swap_permutation"
            )
        )
        self.assertEqual(
            calibration._interpretation(
                "cyclic_nonexchangeable", "paired_swap_permutation"
            ),
            "sensitivity_to_swap_null_violation_under_H0_dist",
        )

    def test_summary_reports_monte_carlo_uncertainty(self):
        records = []
        for trial, rejected in enumerate((False, False, True, False)):
            records.append(
                {
                    "regime": "iid_null",
                    "regime_label": calibration.REGIMES["iid_null"]["label"],
                    "n_pairs": 12,
                    "method": "paired_mcnemar",
                    "target_null": "H0_dist: P_A = P_B",
                    "reference_valid_in_regime": True,
                    "interpretation": "type_i_calibration",
                    "trial_index": trial,
                    "reject_null": rejected,
                    "accuracy": 0.5,
                    "p_value": 0.5,
                }
            )
        row = calibration.summarize(records)[0]
        self.assertEqual(row["rejection_rate"], 0.25)
        self.assertGreater(row["monte_carlo_standard_error"], 0)
        self.assertLess(row["wilson_95_ci_low"], 0.25)
        self.assertGreater(row["wilson_95_ci_high"], 0.25)

    def test_raw_trial_output_is_opt_in(self):
        row = {
            "regime": "iid_null",
            "regime_label": calibration.REGIMES["iid_null"]["label"],
            "n_pairs": 12,
            "method": "paired_mcnemar",
            "target_null": "H0_dist: P_A = P_B",
            "reference_valid_in_regime": True,
            "interpretation": "type_i_calibration",
            "trial_count": 1,
            "rejection_count": 0,
            "rejection_rate": 0.0,
            "monte_carlo_standard_error": 0.0,
            "wilson_95_ci_low": 0.0,
            "wilson_95_ci_high": 0.8,
            "accuracy_mean": 0.5,
            "p_value_median": 0.5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            args = calibration._parser().parse_args(["--output-dir", temporary])
            args.output_dir = Path(temporary)
            paths = calibration._write_outputs(
                args, [{"trial_index": 0}], [row], ["paired_mcnemar"]
            )
            self.assertEqual(len(paths), 3)
            self.assertFalse((args.output_dir / "calibration_trials.jsonl.gz").exists())

            args.save_trials = True
            paths = calibration._write_outputs(
                args, [{"trial_index": 0}], [row], ["paired_mcnemar"]
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue((args.output_dir / "calibration_trials.jsonl.gz").is_file())


if __name__ == "__main__":
    unittest.main()
