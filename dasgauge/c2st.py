"""Feature extraction, classifiers, and inference for DAS C2ST experiments."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import binomtest

__all__ = [
    "extract_resnet34_features",
    "extract_summary_features",
    "knn_predict",
    "make_comparison_groups",
    "paired_mcnemar_counts",
    "run_original_binomial_c2st",
    "run_independent_binomial_c2st",
    "run_pair_preserving_binomial_c2st",
    "run_paired_mcnemar_c2st",
    "run_paired_swap_permutation_c2st",
    "run_paired_c2st",
    "split_individual_indices",
    "split_paired_indices",
]


H0_DIST = "H0_dist: P_A = P_B"
H0_SWAP = (
    "H0_swap: tau_epsilon Z =_d Z for every permitted within-pair swap vector"
)


def _summary_one(sample):
    x = np.asarray(sample, dtype=np.float64)
    means = x.mean(axis=1)
    stds = x.std(axis=1)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    medians = np.median(x, axis=1)
    mad = np.median(np.abs(x - medians[:, None]), axis=1)
    q05, q25, q75, q95 = np.quantile(x, [0.05, 0.25, 0.75, 0.95], axis=1)
    peaks = np.max(np.abs(x), axis=1)
    crest = peaks / np.maximum(rms, np.finfo(float).eps)
    channel_metrics = (means, stds, rms, medians, mad, q05, q25, q75, q95, crest)
    features = []
    for values in channel_metrics:
        features.extend((values.mean(), values.std(), values.min(), values.max()))

    if x.shape[0] > 1:
        centered = x - means[:, None]
        numerator = np.sum(centered[:-1] * centered[1:], axis=1)
        denominator = np.sqrt(
            np.sum(centered[:-1] ** 2, axis=1) * np.sum(centered[1:] ** 2, axis=1)
        )
        adjacent = numerator / np.maximum(denominator, np.finfo(float).eps)
        features.extend((adjacent.mean(), adjacent.std(), adjacent.min(), adjacent.max()))
    else:
        features.extend((0.0, 0.0, 0.0, 0.0))
    return np.asarray(features, dtype=np.float32)


def extract_summary_features(sample_file):
    """Extract a small, fixed DAS summary vector from every paired sample."""
    output = []
    with h5py.File(sample_file, "r") as handle:
        for region in ("coupling", "uncoupling"):
            dataset = handle[f"samples/{region}"]
            output.append(np.stack([_summary_one(dataset[index]) for index in range(len(dataset))]))
    return tuple(output)


def _import_torchvision():
    try:
        import torch
        from torchvision.models import ResNet34_Weights, resnet34
    except ImportError as exc:
        raise ImportError(
            "ResNet-34 features require the c2st extra: pip install -e '.[c2st]'"
        ) from exc
    return torch, ResNet34_Weights, resnet34


def _das_batch_to_resnet(batch, torch, device, lower_quantile, upper_quantile):
    images = []
    for sample in np.asarray(batch):
        sample = np.asarray(sample, dtype=np.float32)
        lower, upper = np.quantile(sample, [lower_quantile, upper_quantile])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("DAS sample contains non-finite values")
        if upper <= lower:
            scaled = np.zeros_like(sample, dtype=np.float32)
        else:
            scaled = np.clip((sample - lower) / (upper - lower), 0.0, 1.0)
        images.append(scaled)

    tensor = torch.from_numpy(np.stack(images)[:, None, :, :]).to(device)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(224, 224), mode="bilinear", align_corners=False
    )
    tensor = tensor.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def extract_resnet34_features(
    sample_file,
    *,
    batch_size=8,
    device="auto",
    lower_quantile=0.01,
    upper_quantile=0.99,
):
    """Map each 2-D DAS sample to the pretrained ResNet-34's 512-D feature."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Require 0 <= lower_quantile < upper_quantile <= 1")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")

    torch, weights_type, resnet34 = _import_torchvision()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    weights = weights_type.DEFAULT
    model = resnet34(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    output = []
    with h5py.File(sample_file, "r") as handle, torch.inference_mode():
        for region in ("coupling", "uncoupling"):
            dataset = handle[f"samples/{region}"]
            region_features = []
            for start in range(0, len(dataset), int(batch_size)):
                batch = dataset[start : start + int(batch_size)]
                tensor = _das_batch_to_resnet(
                    batch,
                    torch,
                    device,
                    lower_quantile,
                    upper_quantile,
                )
                region_features.append(model(tensor).cpu().numpy().astype(np.float32))
            output.append(np.concatenate(region_features, axis=0))
    return tuple(output)


def _standardize_from_train(x_train, x_test):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return (x_train - mean) / scale, (x_test - mean) / scale


def knn_predict(x_train, y_train, x_test, *, k=None):
    """Paper-style Euclidean KNN with ``k=floor(sqrt(n_train))`` by default."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int8)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train and test features must be 2-D with equal feature dimension")
    if len(x_train) != len(y_train) or len(x_train) == 0:
        raise ValueError("Training features and labels must be non-empty and aligned")
    if not np.all(np.isin(y_train, (0, 1))):
        raise ValueError("KNN labels must be binary 0/1")

    if k is None:
        k = max(1, int(math.sqrt(len(x_train))))
    k = int(k)
    if not 1 <= k <= len(x_train):
        raise ValueError(f"k must be within 1:{len(x_train)}")

    scaled_train, scaled_test = _standardize_from_train(x_train, x_test)
    distances = np.sum(
        (scaled_test[:, None, :] - scaled_train[None, :, :]) ** 2, axis=2
    )
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    probabilities = y_train[nearest].mean(axis=1)
    predictions = (probabilities > 0.5).astype(np.int8)
    return predictions, probabilities, k


def _adjacent_index_pairs(n_observations):
    """Pair consecutive observations, dropping one final observation if needed."""
    n_observations = int(n_observations)
    usable = n_observations - n_observations % 2
    if usable < 4:
        raise ValueError("Within-region controls require at least four observations")
    pairs = np.arange(usable, dtype=int).reshape(-1, 2)
    dropped = list(range(usable, n_observations))
    return pairs, dropped


def make_comparison_groups(s_features, p_features, comparison, *, seed=0):
    """Construct paired feature groups for main and negative-control tests.

    ``S`` is the coupling region and ``P`` is the uncoupling region. For the
    within-region controls, consecutive observations form temporal pairs and a
    seeded coin flip assigns one observation to each artificial source group.
    ``s-p-matched`` selects one time index from each adjacent pair so its class
    size and temporal coverage match the within-region controls.
    """
    s_features = np.asarray(s_features)
    p_features = np.asarray(p_features)
    if s_features.ndim != 2 or p_features.ndim != 2:
        raise ValueError("S and P features must both be 2-D")
    if s_features.shape != p_features.shape:
        raise ValueError("S and P features must have identical shapes")
    if comparison not in {"s-p", "s-p-matched", "s-s", "p-p"}:
        raise ValueError(f"Unknown comparison: {comparison}")

    n_observations = len(s_features)
    if comparison == "s-p":
        indices = np.arange(n_observations, dtype=int)
        return s_features, p_features, {
            "comparison": comparison,
            "group_a": "S",
            "group_b": "P",
            "group_a_sample_indices": indices.tolist(),
            "group_b_sample_indices": indices.tolist(),
            "source_index_pairs": np.column_stack((indices, indices)).tolist(),
            "dropped_sample_indices": [],
            "assignment_seed": None,
            "design": "time-matched coupling versus uncoupling",
            "inferential": True,
        }

    adjacent_pairs, dropped = _adjacent_index_pairs(n_observations)
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=len(adjacent_pairs), dtype=np.int8)

    if comparison == "s-p-matched":
        selected = adjacent_pairs[np.arange(len(adjacent_pairs)), flips]
        return s_features[selected], p_features[selected], {
            "comparison": comparison,
            "group_a": "S",
            "group_b": "P",
            "group_a_sample_indices": selected.tolist(),
            "group_b_sample_indices": selected.tolist(),
            "source_index_pairs": np.column_stack((selected, selected)).tolist(),
            "adjacent_selection_blocks": adjacent_pairs.tolist(),
            "dropped_sample_indices": dropped,
            "assignment_seed": int(seed),
            "design": "one time-matched S/P observation selected from each adjacent temporal block",
            "inferential": True,
        }

    source = s_features if comparison == "s-s" else p_features
    group_a_indices = adjacent_pairs[np.arange(len(adjacent_pairs)), flips]
    group_b_indices = adjacent_pairs[np.arange(len(adjacent_pairs)), 1 - flips]
    region = "S" if comparison == "s-s" else "P"
    return source[group_a_indices], source[group_b_indices], {
        "comparison": comparison,
        "group_a": f"{region}_A",
        "group_b": f"{region}_B",
        "group_a_sample_indices": group_a_indices.tolist(),
        "group_b_sample_indices": group_b_indices.tolist(),
        "source_index_pairs": np.column_stack(
            (group_a_indices, group_b_indices)
        ).tolist(),
        "adjacent_temporal_pairs": adjacent_pairs.tolist(),
        "dropped_sample_indices": dropped,
        "assignment_seed": int(seed),
        "design": f"disjoint within-{region} groups, one observation per adjacent temporal pair",
        "inferential": True,
    }


def _block_ids(n_pairs, permutation_unit, block_size):
    if permutation_unit == "pair":
        return np.arange(n_pairs)
    if permutation_unit != "block":
        raise ValueError("permutation_unit must be 'pair' or 'block'")
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    return np.arange(n_pairs) // block_size


def _split_pairs(block_ids, test_fraction, rng):
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 2:
        raise ValueError("At least two independent split blocks are required")
    shuffled = rng.permutation(unique_blocks)
    n_test_blocks = int(round(len(unique_blocks) * float(test_fraction)))
    n_test_blocks = min(max(1, n_test_blocks), len(unique_blocks) - 1)
    test_blocks = shuffled[:n_test_blocks]
    is_test = np.isin(block_ids, test_blocks)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def _validate_c2st_inputs(group_a_features, group_b_features, test_fraction, alpha):
    group_a = np.asarray(group_a_features)
    group_b = np.asarray(group_b_features)
    if group_a.ndim != 2 or group_b.ndim != 2 or group_a.shape != group_b.shape:
        raise ValueError("The two feature groups must have identical 2-D shapes")
    if len(group_a) < 2:
        raise ValueError("At least two samples per group are required")
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be between 0 and 1")
    return group_a, group_b


def split_paired_indices(
    n_pairs,
    *,
    test_fraction=0.5,
    seed=0,
    split_unit="pair",
    block_size=1,
):
    """Split pair indices without ever separating the two pair members.

    ``split_unit='block'`` keeps consecutive blocks of ``block_size`` pairs in
    one partition. This changes only the split; it does not by itself make the
    held-out pairs independent.
    """
    n_pairs = int(n_pairs)
    if n_pairs < 2:
        raise ValueError("At least two paired samples are required")
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    block_ids = _block_ids(n_pairs, split_unit, block_size)
    train, test = _split_pairs(
        block_ids, test_fraction, np.random.default_rng(seed)
    )
    return train, test


def split_individual_indices(n_pairs, *, test_fraction=0.5, seed=0):
    """Split the two source samples independently, deliberately ignoring pairs.

    Splitting each source separately retains balanced train and test classes,
    as in the simple independent-sample C2ST reference, while allowing one
    member of a physical pair to appear in training and its mate in testing.
    """
    n_pairs = int(n_pairs)
    if n_pairs < 2:
        raise ValueError("At least two samples per group are required")
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    n_test = int(round(n_pairs * float(test_fraction)))
    n_test = min(max(1, n_test), n_pairs - 1)
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(n_pairs) for _ in range(2)]
    names = ("group_a", "group_b")
    result = {}
    for name, order in zip(names, permutations):
        result[f"train_{name}_indices"] = np.sort(order[n_test:])
        result[f"test_{name}_indices"] = np.sort(order[:n_test])
    return result


def paired_mcnemar_counts(group_a_predictions, group_b_predictions):
    """Return the directional discordance counts for held-out pair predictions."""
    pred_a = np.asarray(group_a_predictions, dtype=np.int8)
    pred_b = np.asarray(group_b_predictions, dtype=np.int8)
    if pred_a.ndim != 1 or pred_b.ndim != 1 or pred_a.shape != pred_b.shape:
        raise ValueError("Paired predictions must be aligned one-dimensional arrays")
    if not np.all(np.isin(pred_a, (0, 1))) or not np.all(np.isin(pred_b, (0, 1))):
        raise ValueError("Paired predictions must be binary 0/1")
    n_plus = int(np.sum((pred_a == 0) & (pred_b == 1)))
    n_minus = int(np.sum((pred_a == 1) & (pred_b == 0)))
    return n_plus, n_minus, n_plus + n_minus


def _fit_original_labels(group_a, group_b, train_pairs, test_pairs, k):
    x_train = np.concatenate((group_a[train_pairs], group_b[train_pairs]), axis=0)
    y_train = np.concatenate(
        (np.zeros(len(train_pairs), dtype=np.int8), np.ones(len(train_pairs), dtype=np.int8))
    )
    x_test = np.concatenate((group_a[test_pairs], group_b[test_pairs]), axis=0)
    predictions, probabilities, used_k = knn_predict(x_train, y_train, x_test, k=k)
    m = len(test_pairs)
    return (
        predictions[:m],
        predictions[m:],
        probabilities[:m],
        probabilities[m:],
        used_k,
    )


def _paired_observed_fields(
    pred_a,
    pred_b,
    prob_a,
    prob_b,
    train_pairs,
    test_pairs,
    used_k,
):
    n_plus, n_minus, n_disc = paired_mcnemar_counts(pred_a, pred_b)
    m = len(test_pairs)
    correct = m + n_plus - n_minus
    accuracy = correct / (2 * m)
    identity_accuracy = 0.5 + (n_plus - n_minus) / (2 * m)
    if not np.isclose(accuracy, identity_accuracy):
        raise RuntimeError("Paired accuracy identity failed")
    return {
        "statistic": float(accuracy),
        "accuracy": float(accuracy),
        "correct": int(correct),
        "n_test_examples": int(2 * m),
        "n_held_out_pairs": int(m),
        "n_held_out_individual_observations": int(2 * m),
        "N_plus": n_plus,
        "N_minus": n_minus,
        "N_disc": n_disc,
        "accuracy_from_mcnemar_counts": float(identity_accuracy),
        "k": int(used_k),
        "train_pair_indices": train_pairs.tolist(),
        "test_pair_indices": test_pairs.tolist(),
        "train_indices": {
            "group_a": train_pairs.tolist(),
            "group_b": train_pairs.tolist(),
        },
        "test_indices": {
            "group_a": test_pairs.tolist(),
            "group_b": test_pairs.tolist(),
        },
        "test_group_a_predictions": pred_a.tolist(),
        "test_group_b_predictions": pred_b.tolist(),
        "test_group_a_probabilities": prob_a.tolist(),
        "test_group_b_probabilities": prob_b.tolist(),
        "test_predictions": np.concatenate((pred_a, pred_b)).tolist(),
        "test_probabilities": np.concatenate((prob_a, prob_b)).tolist(),
        "test_labels": ([0] * m) + ([1] * m),
    }


def run_paired_mcnemar_c2st(
    group_a_features,
    group_b_features,
    *,
    test_fraction=0.5,
    alpha=0.05,
    seed=0,
    split_unit="pair",
    block_size=1,
    k=None,
):
    """Run a pair-split, held-out McNemar C2ST for ``H0_dist``.

    The classifier and all train-estimated standardization are fixed without
    using held-out pairs. The exact one-sided McNemar calculation allows
    arbitrary dependence between the two members of a held-out pair. Its
    Binomial discordance reference requires independent, identically
    distributed held-out pairs (or a separately justified dependence-aware
    extension).
    """
    group_a, group_b = _validate_c2st_inputs(
        group_a_features, group_b_features, test_fraction, alpha
    )
    train_pairs, test_pairs = split_paired_indices(
        len(group_a),
        test_fraction=test_fraction,
        seed=seed,
        split_unit=split_unit,
        block_size=block_size,
    )
    pred_a, pred_b, prob_a, prob_b, used_k = _fit_original_labels(
        group_a, group_b, train_pairs, test_pairs, k
    )
    result = _paired_observed_fields(
        pred_a, pred_b, prob_a, prob_b, train_pairs, test_pairs, used_k
    )
    if result["N_disc"] == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                result["N_plus"], result["N_disc"], 0.5, alternative="greater"
            ).pvalue
        )
    result.update(
        {
            "method": "paired_mcnemar",
            "target_null": H0_DIST,
            "assumptions": [
                "classifier, preprocessing estimation, and tuning are fixed independently of held-out pairs",
                "held-out pairs are independent and identically distributed",
                "dependence within each held-out pair may be arbitrary",
            ],
            "split_unit": split_unit,
            "block_size": int(block_size),
            "calibration": "exact_one_sided_mcnemar",
            "p_value": p_value,
            "alpha": float(alpha),
            "reject_null": bool(p_value <= alpha),
            "permutations": 0,
            "split_seed": int(seed),
            "seed": int(seed),
            "null_statistics": [],
        }
    )
    return result


def run_pair_preserving_binomial_c2st(
    group_a_features,
    group_b_features,
    *,
    test_fraction=0.5,
    alpha=0.05,
    seed=0,
    split_unit="pair",
    block_size=1,
    k=None,
):
    """Diagnostic: pair-preserving split with an individual accuracy Binomial.

    This retains pairs during splitting but deliberately uses the simple
    independent-correctness reference. It isolates calibration effects from
    leakage due to observation-level splitting and is not generally valid for
    paired held-out observations.
    """
    result = run_paired_mcnemar_c2st(
        group_a_features,
        group_b_features,
        test_fraction=test_fraction,
        alpha=alpha,
        seed=seed,
        split_unit=split_unit,
        block_size=block_size,
        k=k,
    )
    p_value = float(
        binomtest(
            result["correct"],
            result["n_held_out_individual_observations"],
            0.5,
            alternative="greater",
        ).pvalue
    )
    result.update(
        {
            "method": "pair_preserving_binomial",
            "target_null": H0_DIST,
            "assumptions": [
                "classifier is fixed independently of held-out observations",
                "individual held-out correctness outcomes follow the independent Bernoulli(1/2) reference",
                "the second assumption is generally false for paired data; this method is diagnostic only",
            ],
            "calibration": "exact_binomial_accuracy_diagnostic",
            "p_value": p_value,
            "reject_null": bool(p_value <= alpha),
            "binomial_p_value_reference": p_value,
        }
    )
    return result


def run_original_binomial_c2st(
    group_a_features,
    group_b_features,
    *,
    test_fraction=0.5,
    alpha=0.05,
    seed=0,
    k=None,
):
    """Run the original accuracy-Binomial C2ST while ignoring pairing.

    Each source sample is split independently to keep train and test classes
    balanced. The exact one-sided Binomial accuracy reference is appropriate
    for independent held-out observations under the classical C2ST setup; the
    method is intentionally a naive benchmark for paired or serial data.
    """
    group_a, group_b = _validate_c2st_inputs(
        group_a_features, group_b_features, test_fraction, alpha
    )
    indices = split_individual_indices(
        len(group_a), test_fraction=test_fraction, seed=seed
    )
    train_a = indices["train_group_a_indices"]
    train_b = indices["train_group_b_indices"]
    test_a = indices["test_group_a_indices"]
    test_b = indices["test_group_b_indices"]
    x_train = np.concatenate((group_a[train_a], group_b[train_b]), axis=0)
    y_train = np.concatenate(
        (np.zeros(len(train_a), dtype=np.int8), np.ones(len(train_b), dtype=np.int8))
    )
    x_test = np.concatenate((group_a[test_a], group_b[test_b]), axis=0)
    y_test = np.concatenate(
        (np.zeros(len(test_a), dtype=np.int8), np.ones(len(test_b), dtype=np.int8))
    )
    predictions, probabilities, used_k = knn_predict(x_train, y_train, x_test, k=k)
    correct = int(np.sum(predictions == y_test))
    n_test = len(y_test)
    accuracy = correct / n_test
    p_value = float(binomtest(correct, n_test, 0.5, alternative="greater").pvalue)

    test_a_set = set(test_a.tolist())
    test_b_set = set(test_b.tolist())
    both_test = sorted(test_a_set & test_b_set)
    separated = sorted(test_a_set ^ test_b_set)
    both_train = sorted(set(train_a.tolist()) & set(train_b.tolist()))
    return {
        "method": "original_binomial_c2st",
        "target_null": H0_DIST,
        "assumptions": [
            "individual observations are independent within and across source samples",
            "classifier, preprocessing estimation, and tuning are fixed independently of held-out observations",
            "balanced held-out correctness outcomes follow the Bernoulli(1/2) reference under the null",
            "these assumptions are not asserted for paired or serial DAS observations",
        ],
        "split_unit": "individual_observation",
        "block_size": 1,
        "calibration": "exact_binomial_accuracy",
        "statistic": float(accuracy),
        "accuracy": float(accuracy),
        "correct": correct,
        "n_test_examples": int(n_test),
        "n_held_out_pairs": len(both_test),
        "n_held_out_individual_observations": int(n_test),
        "N_plus": None,
        "N_minus": None,
        "N_disc": None,
        "p_value": p_value,
        "alpha": float(alpha),
        "reject_null": bool(p_value <= alpha),
        "permutations": 0,
        "k": int(used_k),
        "split_seed": int(seed),
        "seed": int(seed),
        "train_pair_indices": both_train,
        "test_pair_indices": both_test,
        "separated_pair_indices": separated,
        "train_indices": {
            "group_a": train_a.tolist(),
            "group_b": train_b.tolist(),
        },
        "test_indices": {
            "group_a": test_a.tolist(),
            "group_b": test_b.tolist(),
        },
        "test_labels": y_test.tolist(),
        "test_predictions": predictions.tolist(),
        "test_probabilities": probabilities.tolist(),
        "binomial_p_value_reference": p_value,
        "null_statistics": [],
    }


def run_independent_binomial_c2st(*args, **kwargs):
    """Compatibility alias for :func:`run_original_binomial_c2st`.

    ``independent_binomial`` described an assumption rather than the established
    C2ST procedure. New code and serialized results use
    ``original_binomial_c2st``.
    """
    return run_original_binomial_c2st(*args, **kwargs)


def run_paired_swap_permutation_c2st(
    group_a_features,
    group_b_features,
    *,
    test_fraction=0.5,
    permutations=999,
    alpha=0.05,
    seed=0,
    permutation_unit="pair",
    block_size=1,
    k=None,
):
    """Run the full-refit within-pair swap test for ``H0_swap``.

    The pair-preserving split is held fixed. Every sampled transformation is
    applied to training and test pairs, after which training standardization
    and KNN are refitted. Exactness targets swap invariance, not marginal
    distributional equality by itself.
    """
    group_a, group_b = _validate_c2st_inputs(
        group_a_features, group_b_features, test_fraction, alpha
    )
    n_pairs = len(group_a)
    permutations = int(permutations)
    if permutations < 1:
        raise ValueError("permutations must be >= 1")

    rng = np.random.default_rng(seed)
    block_ids = _block_ids(n_pairs, permutation_unit, block_size)
    train_pairs, test_pairs = _split_pairs(block_ids, test_fraction, rng)

    def evaluate(swaps):
        x_train = np.concatenate((group_a[train_pairs], group_b[train_pairs]), axis=0)
        y_train = np.concatenate(
            (swaps[train_pairs].astype(np.int8), (~swaps[train_pairs]).astype(np.int8))
        )
        x_test = np.concatenate((group_a[test_pairs], group_b[test_pairs]), axis=0)
        y_test = np.concatenate(
            (swaps[test_pairs].astype(np.int8), (~swaps[test_pairs]).astype(np.int8))
        )
        predictions, probabilities, used_k = knn_predict(x_train, y_train, x_test, k=k)
        return float(np.mean(predictions == y_test)), predictions, probabilities, y_test, used_k

    observed_swaps = np.zeros(n_pairs, dtype=bool)
    observed, predictions, probabilities, test_labels, used_k = evaluate(observed_swaps)
    m = len(test_pairs)
    pred_a, pred_b = predictions[:m], predictions[m:]
    prob_a, prob_b = probabilities[:m], probabilities[m:]
    result = _paired_observed_fields(
        pred_a, pred_b, prob_a, prob_b, train_pairs, test_pairs, used_k
    )
    binomial_p = float(
        binomtest(result["correct"], 2 * m, 0.5, alternative="greater").pvalue
    )

    null_statistics = []
    unique_blocks = np.unique(block_ids)
    for _ in range(permutations):
        block_swaps = rng.integers(
            0, 2, size=len(unique_blocks), dtype=np.int8
        ).astype(bool)
        swaps = block_swaps[block_ids]
        statistic, _, _, _, _ = evaluate(swaps)
        null_statistics.append(statistic)
    exceedances = int(np.sum(np.asarray(null_statistics) >= observed))
    p_value = (1 + exceedances) / (permutations + 1)
    result.update(
        {
            "method": "paired_swap_permutation",
            "target_null": H0_SWAP,
            "assumptions": [
                "the complete retained paired vector is invariant under every permitted swap vector",
                "the pair-preserving train/test split is fixed across transformations",
                "training standardization and the classifier are refitted for every transformation",
            ],
            "split_unit": "pair" if permutation_unit == "pair" else "pair_block",
            "calibration": "permutation",
            "p_value": float(p_value),
            "alpha": float(alpha),
            "reject_null": bool(p_value <= alpha),
            "binomial_p_value_reference": binomial_p,
            "permutations": permutations,
            "permutation_unit": permutation_unit,
            "block_size": int(block_size),
            "split_seed": int(seed),
            "seed": int(seed),
            "test_labels": test_labels.tolist(),
            "null_statistics": null_statistics,
        }
    )
    return result


def run_paired_c2st(
    group_a_features,
    group_b_features,
    *,
    test_fraction=0.5,
    permutations=999,
    alpha=0.05,
    seed=0,
    permutation_unit="pair",
    block_size=1,
    calibration="permutation",
    k=None,
):
    """Compatibility wrapper for the original paired C2ST API.

    New code should call :func:`run_paired_swap_permutation_c2st` or
    :func:`run_pair_preserving_binomial_c2st` so the method and target null are
    explicit.
    """
    if calibration == "permutation":
        return run_paired_swap_permutation_c2st(
            group_a_features,
            group_b_features,
            test_fraction=test_fraction,
            permutations=permutations,
            alpha=alpha,
            seed=seed,
            permutation_unit=permutation_unit,
            block_size=block_size,
            k=k,
        )
    if calibration == "binomial":
        result = run_pair_preserving_binomial_c2st(
            group_a_features,
            group_b_features,
            test_fraction=test_fraction,
            alpha=alpha,
            seed=seed,
            split_unit=permutation_unit,
            block_size=block_size,
            k=k,
        )
        result["permutation_unit"] = permutation_unit
        return result
    raise ValueError("calibration must be 'permutation' or 'binomial'")
