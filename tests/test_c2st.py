"""Tests for C2ST features, KNN, and paired inference."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from dasgauge.c2st import (
    extract_summary_features,
    knn_predict,
    make_comparison_groups,
    paired_mcnemar_counts,
    run_independent_binomial_c2st,
    run_original_binomial_c2st,
    run_pair_preserving_binomial_c2st,
    run_paired_c2st,
    run_paired_mcnemar_c2st,
    run_paired_swap_permutation_c2st,
    split_individual_indices,
    split_paired_indices,
)


class KNNTests(unittest.TestCase):
    def test_separates_simple_clusters(self):
        x_train = np.array([[0.0], [0.1], [10.0], [10.1]])
        y_train = np.array([0, 0, 1, 1])
        predictions, probabilities, k = knn_predict(
            x_train, y_train, np.array([[0.2], [9.9]]), k=1
        )
        np.testing.assert_array_equal(predictions, [0, 1])
        np.testing.assert_array_equal(probabilities, [0.0, 1.0])
        self.assertEqual(k, 1)


class FeatureTests(unittest.TestCase):
    def test_summary_features_have_fixed_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("samples")
                group.create_dataset("coupling", data=np.ones((3, 2, 8)))
                group.create_dataset("uncoupling", data=np.ones((3, 4, 8)) * 2)
            coupling, uncoupling = extract_summary_features(path)
        self.assertEqual(coupling.shape, (3, 44))
        self.assertEqual(uncoupling.shape, (3, 44))
        self.assertTrue(np.isfinite(coupling).all())


class PairedC2STTests(unittest.TestCase):
    def test_strong_difference_is_reproducible(self):
        coupling = np.column_stack((np.zeros(20), np.arange(20)))
        uncoupling = np.column_stack((np.full(20, 10.0), np.arange(20)))
        first = run_paired_c2st(coupling, uncoupling, permutations=199, seed=7, k=1)
        second = run_paired_c2st(coupling, uncoupling, permutations=199, seed=7, k=1)
        self.assertEqual(first, second)
        self.assertEqual(first["statistic"], 1.0)
        self.assertLessEqual(first["p_value"], 0.05)
        self.assertTrue(set(first["train_pair_indices"]).isdisjoint(first["test_pair_indices"]))

    def test_block_split_keeps_blocks_together(self):
        coupling = np.arange(24, dtype=float).reshape(12, 2)
        uncoupling = coupling + 1
        result = run_paired_c2st(
            coupling,
            uncoupling,
            permutations=9,
            seed=1,
            permutation_unit="block",
            block_size=3,
        )
        train_blocks = {index // 3 for index in result["train_pair_indices"]}
        test_blocks = {index // 3 for index in result["test_pair_indices"]}
        self.assertTrue(train_blocks.isdisjoint(test_blocks))

    def test_binomial_mode_reports_original_paper_reference(self):
        coupling = np.zeros((8, 2))
        uncoupling = np.ones((8, 2))
        result = run_paired_c2st(
            coupling, uncoupling, calibration="binomial", seed=2, k=1
        )
        self.assertEqual(result["p_value"], result["binomial_p_value_reference"])
        self.assertEqual(result["permutations"], 0)

    def test_exact_copy_is_only_a_chance_level_smoke_test(self):
        features = np.arange(24, dtype=float).reshape(12, 2)
        result = run_paired_c2st(features, features.copy(), permutations=19, seed=3)
        self.assertEqual(result["statistic"], 0.5)
        self.assertFalse(result["reject_null"])

    def test_permutation_refits_classifier_for_every_labeling(self):
        group_a = np.zeros((8, 2))
        group_b = np.ones((8, 2))
        from dasgauge import c2st as c2st_module

        original = c2st_module.knn_predict
        with mock.patch.object(c2st_module, "knn_predict", wraps=original) as fitted:
            run_paired_c2st(group_a, group_b, permutations=7, seed=4, k=1)
        self.assertEqual(fitted.call_count, 8)  # observed fit + one fit per permutation


class ThreeMethodInferenceTests(unittest.TestCase):
    def setUp(self):
        self.group_a = np.column_stack((np.zeros(20), np.arange(20)))
        self.group_b = np.column_stack((np.full(20, 10.0), np.arange(20)))

    def test_mcnemar_count_construction(self):
        pred_a = np.array([0, 1, 0, 1, 0, 1])
        pred_b = np.array([1, 0, 0, 1, 1, 1])
        self.assertEqual(paired_mcnemar_counts(pred_a, pred_b), (2, 1, 3))

    def test_mcnemar_accuracy_identity_and_perfect_p_value(self):
        result = run_paired_mcnemar_c2st(
            self.group_a, self.group_b, seed=7, k=1
        )
        m = result["n_held_out_pairs"]
        expected = 0.5 + (result["N_plus"] - result["N_minus"]) / (2 * m)
        self.assertEqual(result["accuracy"], expected)
        self.assertEqual(result["N_plus"], m)
        self.assertEqual(result["N_minus"], 0)
        self.assertEqual(result["p_value"], 2.0 ** (-m))

    def test_all_tied_predictions_return_p_one(self):
        features = np.arange(40, dtype=float).reshape(20, 2)
        result = run_paired_mcnemar_c2st(features, features.copy(), seed=3, k=1)
        self.assertEqual(result["N_disc"], 0)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["p_value"], 1.0)

    def test_reversed_perfect_predictions_have_large_one_sided_p(self):
        def reversed_predictions(_x_train, _y_train, x_test, *, k=None):
            m = len(x_test) // 2
            predictions = np.r_[np.ones(m, dtype=np.int8), np.zeros(m, dtype=np.int8)]
            return predictions, predictions.astype(float), 1

        from dasgauge import c2st as c2st_module

        with mock.patch.object(c2st_module, "knn_predict", side_effect=reversed_predictions):
            result = run_paired_mcnemar_c2st(
                self.group_a, self.group_b, seed=4, k=1
            )
        self.assertEqual(result["N_plus"], 0)
        self.assertEqual(result["N_minus"], result["n_held_out_pairs"])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["p_value"], 1.0)

    def test_pair_split_preserves_pairs(self):
        train, test = split_paired_indices(25, test_fraction=0.4, seed=11)
        self.assertTrue(set(train).isdisjoint(test))
        self.assertEqual(set(train) | set(test), set(range(25)))
        result = run_paired_mcnemar_c2st(
            self.group_a, self.group_b, seed=11, k=1
        )
        self.assertEqual(result["train_indices"]["group_a"], result["train_indices"]["group_b"])
        self.assertEqual(result["test_indices"]["group_a"], result["test_indices"]["group_b"])

    def test_individual_split_ignores_pair_membership(self):
        indices = split_individual_indices(20, test_fraction=0.5, seed=5)
        test_a = set(indices["test_group_a_indices"])
        test_b = set(indices["test_group_b_indices"])
        self.assertNotEqual(test_a, test_b)
        self.assertTrue(test_a ^ test_b)
        result = run_original_binomial_c2st(
            self.group_a, self.group_b, seed=5, k=1
        )
        self.assertEqual(result["split_unit"], "individual_observation")
        self.assertTrue(result["separated_pair_indices"])

    def test_methods_are_reproducible_and_have_explicit_schemas(self):
        calls = [
            lambda: run_paired_mcnemar_c2st(self.group_a, self.group_b, seed=2, k=1),
            lambda: run_original_binomial_c2st(self.group_a, self.group_b, seed=2, k=1),
            lambda: run_paired_swap_permutation_c2st(
                self.group_a, self.group_b, seed=2, permutations=9, k=1
            ),
            lambda: run_pair_preserving_binomial_c2st(
                self.group_a, self.group_b, seed=2, k=1
            ),
        ]
        expected_methods = [
            "paired_mcnemar",
            "original_binomial_c2st",
            "paired_swap_permutation",
            "pair_preserving_binomial",
        ]
        required = {
            "method", "target_null", "assumptions", "split_seed", "train_indices",
            "test_indices", "n_held_out_pairs", "n_held_out_individual_observations",
            "accuracy", "N_plus", "N_minus", "N_disc", "p_value", "reject_null",
            "permutations",
        }
        for call, method in zip(calls, expected_methods):
            first = call()
            second = call()
            self.assertEqual(first, second)
            self.assertEqual(first["method"], method)
            self.assertTrue(required.issubset(first))

    def test_old_independent_name_is_a_compatibility_alias(self):
        old = run_independent_binomial_c2st(self.group_a, self.group_b, seed=2, k=1)
        new = run_original_binomial_c2st(self.group_a, self.group_b, seed=2, k=1)
        self.assertEqual(old, new)
        self.assertEqual(old["method"], "original_binomial_c2st")

    def test_named_swap_method_refits_once_per_transformation(self):
        from dasgauge import c2st as c2st_module

        original = c2st_module.knn_predict
        with mock.patch.object(c2st_module, "knn_predict", wraps=original) as fitted:
            run_paired_swap_permutation_c2st(
                self.group_a[:8], self.group_b[:8], permutations=5, seed=8, k=1
            )
        self.assertEqual(fitted.call_count, 6)


class ComparisonConstructionTests(unittest.TestCase):
    def setUp(self):
        self.s = np.arange(36 * 2, dtype=float).reshape(36, 2)
        self.p = 1000 + self.s

    def test_s_s_control_is_disjoint_and_balanced(self):
        group_a, group_b, metadata = make_comparison_groups(
            self.s, self.p, "s-s", seed=9
        )
        a_indices = metadata["group_a_sample_indices"]
        b_indices = metadata["group_b_sample_indices"]
        self.assertEqual(group_a.shape, (18, 2))
        self.assertEqual(group_b.shape, (18, 2))
        self.assertTrue(set(a_indices).isdisjoint(b_indices))
        self.assertEqual(set(a_indices) | set(b_indices), set(range(36)))
        for a_index, b_index in metadata["source_index_pairs"]:
            self.assertEqual(a_index // 2, b_index // 2)
        np.testing.assert_array_equal(group_a, self.s[a_indices])
        np.testing.assert_array_equal(group_b, self.s[b_indices])

    def test_p_p_control_uses_only_p(self):
        group_a, group_b, metadata = make_comparison_groups(
            self.s, self.p, "p-p", seed=2
        )
        np.testing.assert_array_equal(
            group_a, self.p[metadata["group_a_sample_indices"]]
        )
        np.testing.assert_array_equal(
            group_b, self.p[metadata["group_b_sample_indices"]]
        )

    def test_power_matched_s_p_has_control_sample_size(self):
        group_a, group_b, metadata = make_comparison_groups(
            self.s, self.p, "s-p-matched", seed=5
        )
        selected = metadata["group_a_sample_indices"]
        self.assertEqual(group_a.shape, (18, 2))
        self.assertEqual(selected, metadata["group_b_sample_indices"])
        np.testing.assert_array_equal(group_a, self.s[selected])
        np.testing.assert_array_equal(group_b, self.p[selected])

    def test_odd_final_observation_is_recorded_as_dropped(self):
        _, _, metadata = make_comparison_groups(
            self.s[:35], self.p[:35], "s-s", seed=0
        )
        self.assertEqual(metadata["dropped_sample_indices"], [34])


if __name__ == "__main__":
    unittest.main()
