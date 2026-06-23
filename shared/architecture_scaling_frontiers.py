#!/usr/bin/env python3
"""Architecture scaling frontiers for FI-2010 NoAuction Z-score.

This script is for direct architecture scaling questions: as inference compute
increases within a family, how does out-of-sample log-loss shrink, and where do
the family curves intersect?
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from compute_scaling_cf1 import (
    CLASS_ORDER,
    FULL_FEATURE_COUNT,
    HORIZONS,
    RANDOM_SEED,
    find_data_root,
    load_no_auction_zscore_cf,
    load_or_compute_feature_importance,
    make_feature_specs,
    parse_int_list,
)

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BUNDLE_ROOT / "outputs"


@dataclass(frozen=True)
class ModelSpec:
    family: str
    architecture: str
    name: str
    compute_proxy_ops: float
    compute_proxy_kind: str
    n_parameters: float
    fit_fn: Callable[[], object]


def parse_hidden_layers(raw: str) -> list[tuple[int, ...]]:
    specs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        specs.append(tuple(int(piece) for piece in part.replace("x", "-").split("-") if piece))
    return specs


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def safe_filename(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return safe.strip("_") or "run"


def maybe_limit_rows(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_rows <= 0 or max_rows >= len(x_train):
        return x_train, y_train
    return x_train.iloc[:max_rows].reset_index(drop=True), y_train.iloc[:max_rows].reset_index(drop=True)


def chronological_train_validation_split(
    x: pd.DataFrame,
    y: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_fraction <= 0:
        return x, y, x.iloc[0:0].copy(), y.iloc[0:0].copy()
    if validation_fraction >= 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    n_val = max(1, int(round(len(x) * validation_fraction)))
    n_fit = len(x) - n_val
    if n_fit <= 0:
        raise ValueError("validation_fraction leaves no fitting rows")
    return (
        x.iloc[:n_fit].reset_index(drop=True),
        y.iloc[:n_fit].reset_index(drop=True),
        x.iloc[n_fit:].reset_index(drop=True),
        y.iloc[n_fit:].reset_index(drop=True),
    )


def make_lagged_snapshot_dataset(
    x: pd.DataFrame,
    y: pd.DataFrame,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return x.reset_index(drop=True), y.reset_index(drop=True)
    if window > len(x):
        raise ValueError(f"window={window} exceeds dataset length={len(x)}")

    values = x.to_numpy()
    blocks = []
    columns = []
    for lag in range(window - 1, -1, -1):
        blocks.append(values[window - 1 - lag : len(values) - lag])
        columns.extend([f"{col}_lag{lag}" for col in x.columns])
    lagged = pd.DataFrame(np.concatenate(blocks, axis=1), columns=columns)
    aligned_y = y.iloc[window - 1 :].reset_index(drop=True)
    return lagged, aligned_y


def proba_frame(model, x: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(x)
    proba = np.nan_to_num(np.asarray(proba, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    bad_rows = row_sums.squeeze(axis=1) <= 0
    if np.any(bad_rows):
        proba[bad_rows, :] = 1.0 / proba.shape[1]
        row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / row_sums
    return pd.DataFrame(proba, columns=model.classes_).reindex(columns=CLASS_ORDER, fill_value=0.0)


def score_probabilities(proba: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    pred = proba.idxmax(axis=1).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=CLASS_ORDER),
        "macro_f1": f1_score(y_true, pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, labels=CLASS_ORDER, average="weighted", zero_division=0),
        "weighted_precision": precision_score(y_true, pred, labels=CLASS_ORDER, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, pred, labels=CLASS_ORDER, average="weighted", zero_division=0),
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def class_balance_metrics(y: pd.Series, prefix: str) -> dict[str, float]:
    counts = y.value_counts().reindex(CLASS_ORDER, fill_value=0).astype(int)
    total = int(counts.sum())
    metrics: dict[str, float] = {f"{prefix}_rows": total}
    for label, stem in [(-1, "down"), (0, "stationary"), (1, "up")]:
        count = int(counts.loc[label])
        metrics[f"{prefix}_class_count_{stem}"] = count
        metrics[f"{prefix}_class_fraction_{stem}"] = count / total if total else np.nan
    return metrics


def prior_proba_from_train(y_train: pd.Series, n_rows: int) -> pd.DataFrame:
    counts = y_train.value_counts().reindex(CLASS_ORDER, fill_value=0).astype(float)
    if counts.sum() <= 0:
        probs = np.full(len(CLASS_ORDER), 1.0 / len(CLASS_ORDER))
    else:
        probs = counts.to_numpy() / counts.sum()
    return pd.DataFrame(np.tile(probs, (n_rows, 1)), columns=CLASS_ORDER)


def prior_log_loss_from_train(y_train: pd.Series, y_eval: pd.Series) -> float:
    if not len(y_eval):
        return np.nan
    return float(log_loss(y_eval, prior_proba_from_train(y_train, len(y_eval)), labels=CLASS_ORDER))


def one_observation_slice(x: pd.DataFrame | np.ndarray, idx: int):
    if hasattr(x, "iloc"):
        return x.iloc[[idx]]
    return x[idx : idx + 1]


def time_predict_proba_batch(model, x: pd.DataFrame, repeats: int) -> tuple[float, float]:
    warmup = model.predict_proba(x)
    if len(warmup) != len(x):
        raise ValueError("predict_proba returned the wrong number of observations")

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = model.predict_proba(x)
        elapsed = time.perf_counter() - start
        if len(out) != len(x):
            raise ValueError("predict_proba returned the wrong number of observations")
        timings.append(elapsed)
    median_s = float(np.median(timings))
    return median_s, median_s * 1_000_000 / len(x)


def time_predict_proba_single_observation(
    model,
    x: pd.DataFrame | np.ndarray,
    repeats: int,
    samples: int,
) -> tuple[float, float]:
    if len(x) == 0:
        return np.nan, np.nan
    n_samples = min(len(x), max(1, int(samples)))
    indices = np.linspace(0, len(x) - 1, n_samples, dtype=int)
    warmup = model.predict_proba(one_observation_slice(x, int(indices[0])))
    if len(warmup) != 1:
        raise ValueError("single-observation predict_proba returned the wrong number of observations")

    timings = []
    for _ in range(repeats):
        for idx in indices:
            x_one = one_observation_slice(x, int(idx))
            start = time.perf_counter()
            out = model.predict_proba(x_one)
            elapsed = time.perf_counter() - start
            if len(out) != 1:
                raise ValueError("single-observation predict_proba returned the wrong number of observations")
            timings.append(elapsed)
    median_s = float(np.median(timings))
    return median_s, median_s * 1_000_000


def time_predict_proba(
    model,
    x: pd.DataFrame | np.ndarray,
    repeats: int,
    timing_mode: str = "batch",
    single_samples: int = 256,
) -> tuple[float, float]:
    if timing_mode == "single":
        return time_predict_proba_single_observation(model, x, repeats, single_samples)
    if timing_mode == "batch":
        return time_predict_proba_batch(model, x, repeats)
    raise ValueError(f"unknown timing_mode: {timing_mode}")


def time_predict_proba_with_torch_timing_device(
    model,
    x: pd.DataFrame,
    repeats: int,
    torch_timing_device: str = "same",
    timing_mode: str = "batch",
    single_samples: int = 256,
) -> tuple[float, float, str]:
    if torch_timing_device in {"", "same"} or not (hasattr(model, "model") and hasattr(model, "device")):
        timing_s, latency_us = time_predict_proba(model, x, repeats, timing_mode, single_samples)
        device_name = str(getattr(model, "device", "default"))
        return timing_s, latency_us, device_name

    old_device = model.device
    timing_device = resolve_torch_device(torch_timing_device)
    try:
        model.model.to(timing_device)
        model.device = timing_device
        timing_s, latency_us = time_predict_proba(model, x, repeats, timing_mode, single_samples)
        return timing_s, latency_us, str(timing_device)
    finally:
        model.model.to(old_device)
        model.device = old_device


def tree_average_depth(tree: DecisionTreeClassifier, x: pd.DataFrame) -> float:
    # Standalone trees keep feature names; ensemble base estimators are fitted
    # on ndarray views. Match that to avoid repeated sklearn warnings.
    if hasattr(tree, "feature_names_in_"):
        path_input = x
    else:
        path_input = x.to_numpy() if hasattr(x, "to_numpy") else x
    path = tree.decision_path(path_input)
    return float(np.mean(path.getnnz(axis=1) - 1))


def forest_average_depth(forest, x: pd.DataFrame) -> float:
    return float(sum(tree_average_depth(estimator, x) for estimator in forest.estimators_))


def tree_count_for_model(model) -> int:
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return 1
    return int(len(estimators))


def make_forward_work_audit(
    *,
    units: float,
    method: str,
    confidence: str,
    split_comparisons: float = 0.0,
    multiply_adds: float = 0.0,
    table_lookups: float = 0.0,
    additions: float = 0.0,
    reductions: float = 0.0,
    activation_ops: float = 0.0,
    notes: str = "",
) -> dict[str, float | str]:
    """Canonical paper-facing work accounting plus legacy audit aliases."""
    units = float(units)
    split_comparisons = float(split_comparisons)
    multiply_adds = float(multiply_adds)
    table_lookups = float(table_lookups)
    additions = float(additions)
    reductions = float(reductions)
    activation_ops = float(activation_ops)
    legacy_confidence = {
        "exact": "high",
        "analytic": "medium_high",
        "sensitivity_only": "medium",
        "proxy": "low_medium",
    }.get(confidence, confidence)
    return {
        "forward_work_units": units,
        "forward_work_method": method,
        "forward_work_confidence": confidence,
        "forward_split_comparisons": split_comparisons,
        "forward_multiply_adds": multiply_adds,
        "forward_table_lookups": table_lookups,
        "forward_additions": additions,
        "forward_reductions": reductions,
        "forward_activation_ops": activation_ops,
        "forward_work_notes": notes,
        # Backward-compatible aliases consumed by existing analysis scripts.
        "audited_forward_ops": units,
        "audited_total_ops": units,
        "audited_critical_path_ops": units,
        "audited_multiply_adds": multiply_adds if multiply_adds else np.nan,
        "audited_comparisons": split_comparisons if split_comparisons else np.nan,
        "audited_table_lookups": table_lookups if table_lookups else np.nan,
        "audited_additions": additions if additions else np.nan,
        "compute_audit_method": method,
        "compute_audit_confidence": legacy_confidence,
        "compute_audit_notes": notes,
    }


def tree_forward_work(
    *,
    split_comparisons: float,
    tree_count: int,
    additions_per_tree: float,
    method: str,
    confidence: str = "exact",
    avg_tree_path_depth: float | None = None,
    notes: str = "",
) -> dict[str, float | str]:
    table_lookups = float(tree_count)
    additions = float(tree_count) * float(additions_per_tree)
    units = float(split_comparisons) + table_lookups + additions
    audit = make_forward_work_audit(
        units=units,
        method=method,
        confidence=confidence,
        split_comparisons=split_comparisons,
        table_lookups=table_lookups,
        additions=additions,
        notes=notes,
    )
    if avg_tree_path_depth is not None and np.isfinite(avg_tree_path_depth):
        audit["audited_avg_tree_path_depth"] = float(avg_tree_path_depth)
        audit["forward_avg_tree_path_depth"] = float(avg_tree_path_depth)
    audit["forward_tree_count"] = int(tree_count)
    return audit


def parse_tree_node_id(raw) -> int | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw)
    if "-" in text:
        text = text.rsplit("-", 1)[-1]
    try:
        return int(float(text))
    except ValueError:
        return None


def hgb_tree_path_depths(predictor, x_values: np.ndarray) -> np.ndarray:
    nodes = predictor.nodes
    node_idx = np.zeros(x_values.shape[0], dtype=np.uint32)
    depths = np.zeros(x_values.shape[0], dtype=np.float64)
    while True:
        current = nodes[node_idx]
        active = ~current["is_leaf"].astype(bool)
        if not np.any(active):
            break
        active_idx = np.flatnonzero(active)
        active_nodes = current[active]
        feature_idx = active_nodes["feature_idx"].astype(np.int64)
        values = x_values[active_idx, feature_idx]
        go_left = values <= active_nodes["num_threshold"]
        missing = np.isnan(values)
        if np.any(missing):
            go_left[missing] = active_nodes["missing_go_to_left"].astype(bool)[missing]
        node_idx[active_idx] = np.where(go_left, active_nodes["left"], active_nodes["right"]).astype(np.uint32)
        depths[active_idx] += 1.0
    return depths


def hgb_empirical_path_depth(model, x: pd.DataFrame) -> tuple[float, float, int]:
    try:
        x_values = x.to_numpy(dtype=np.float64) if hasattr(x, "to_numpy") else np.asarray(x, dtype=np.float64)
        predictors = getattr(model, "_predictors", [])
        effective_iterations = int(getattr(model, "n_iter_", len(predictors)))
        total_depths = np.zeros(x_values.shape[0], dtype=np.float64)
        tree_count = 0
        for iteration in predictors[:effective_iterations]:
            for predictor in iteration:
                total_depths += hgb_tree_path_depths(predictor, x_values)
                tree_count += 1
        if tree_count == 0:
            return np.nan, np.nan, 0
        return float(np.mean(total_depths)), float(np.mean(total_depths) / tree_count), int(tree_count)
    except Exception:
        return np.nan, np.nan, 0


def xgboost_average_path_depth(model, x: pd.DataFrame, effective_estimators: int | None = None) -> tuple[float, float, int]:
    try:
        import xgboost as xgb

        booster = model.get_booster()
        trees = booster.trees_to_dataframe()
        depth_maps: dict[int, dict[int, int]] = {}
        for tree_id, group in trees.groupby("Tree"):
            children: dict[int, list[int]] = {}
            leaves: set[int] = set()
            for _, row in group.iterrows():
                node = parse_tree_node_id(row.get("Node"))
                if node is None:
                    continue
                if row.get("Feature") == "Leaf":
                    leaves.add(node)
                    continue
                yes = parse_tree_node_id(row.get("Yes"))
                no = parse_tree_node_id(row.get("No"))
                children[node] = [child for child in (yes, no) if child is not None]
            depths = {0: 0}
            stack = [0]
            while stack:
                node = stack.pop()
                for child in children.get(node, []):
                    depths[child] = depths[node] + 1
                    stack.append(child)
            depth_maps[int(tree_id)] = {leaf: depths.get(leaf, 0) for leaf in leaves}

        dmatrix = xgb.DMatrix(x)
        kwargs = {"pred_leaf": True}
        if effective_estimators is not None and effective_estimators > 0:
            kwargs["iteration_range"] = (0, int(effective_estimators))
        leaves = booster.predict(dmatrix, **kwargs)
        leaves = np.asarray(leaves)
        if leaves.ndim == 1:
            leaves = leaves[:, None]
        leaves = leaves.astype(int, copy=False)
        tree_ids = list(range(leaves.shape[1]))
        path_depths = np.zeros_like(leaves, dtype=float)
        for col, tree_id in enumerate(tree_ids):
            depth_map = depth_maps.get(tree_id, {})
            path_depths[:, col] = [depth_map.get(int(leaf), 0) for leaf in leaves[:, col]]
        total_per_observation = path_depths.sum(axis=1)
        return float(np.mean(total_per_observation)), float(np.mean(path_depths)), int(leaves.shape[1])
    except Exception:
        return np.nan, np.nan, 0


def lightgbm_average_path_depth(model, x: pd.DataFrame, effective_estimators: int | None = None) -> tuple[float, float, int]:
    try:
        booster = model.booster_
        dump = booster.dump_model(num_iteration=effective_estimators)
        depth_maps: list[dict[int, int]] = []

        def visit(node: dict, depth: int, out: dict[int, int]) -> None:
            if "leaf_index" in node:
                out[int(node["leaf_index"])] = depth
                return
            visit(node["left_child"], depth + 1, out)
            visit(node["right_child"], depth + 1, out)

        for tree_info in dump.get("tree_info", []):
            mapping: dict[int, int] = {}
            visit(tree_info["tree_structure"], 0, mapping)
            depth_maps.append(mapping)

        leaves = np.asarray(model.predict(x, pred_leaf=True, num_iteration=effective_estimators))
        if leaves.ndim == 1:
            leaves = leaves[:, None]
        leaves = leaves.astype(int, copy=False)
        n_cols = min(leaves.shape[1], len(depth_maps))
        path_depths = np.zeros((leaves.shape[0], n_cols), dtype=float)
        for col in range(n_cols):
            depth_map = depth_maps[col]
            path_depths[:, col] = [depth_map.get(int(leaf), 0) for leaf in leaves[:, col]]
        total_per_observation = path_depths.sum(axis=1)
        return float(np.mean(total_per_observation)), float(np.mean(path_depths)), int(n_cols)
    except Exception:
        return np.nan, np.nan, 0


def catboost_total_path_depth(model) -> tuple[float, float, int]:
    try:
        leaf_counts = np.asarray(model.get_tree_leaf_counts(), dtype=float)
        if leaf_counts.size == 0:
            return np.nan, np.nan, 0
        depths = np.log2(leaf_counts)
        return float(np.sum(depths)), float(np.mean(depths)), int(len(depths))
    except Exception:
        return np.nan, np.nan, 0


def pipeline_step_by_attr(model, attr: str):
    for _, step in getattr(model, "steps", []):
        if hasattr(step, attr):
            return step
    return None


def random_feature_forward_work(model, spec: ModelSpec) -> dict[str, float | str]:
    transformer = pipeline_step_by_attr(model, "n_features_out_")
    if transformer is None:
        return make_forward_work_audit(
            units=spec.compute_proxy_ops,
            method=f"analytic_{spec.compute_proxy_kind}",
            confidence="analytic",
            multiply_adds=spec.compute_proxy_ops,
            notes="random-feature transformer internals unavailable; using stored analytic proxy",
        )

    feature_count = int(getattr(transformer, "n_features_out_", 0))
    if hasattr(transformer, "lengths_") and hasattr(transformer, "window"):
        window = int(getattr(transformer, "window"))
        lengths = [int(value) for value in getattr(transformer, "lengths_")]
        response_counts = [max(1, window - length + 1) for length in lengths]
        conv_multiply_adds = float(sum(count * length for count, length in zip(response_counts, lengths)))
        max_comparisons = float(sum(max(0, count - 1) for count in response_counts))
        mean_additions = float(sum(max(0, count - 1) for count in response_counts))
        positive_comparisons = float(sum(response_counts))
        positive_additions = float(sum(max(0, count - 1) for count in response_counts))
        pooling_comparisons = max_comparisons + positive_comparisons
        pooling_additions = mean_additions + positive_additions
        standardization_multiply_adds = float(2 * feature_count)
        head_multiply_adds = float(feature_count * len(CLASS_ORDER))
        multiply_adds = conv_multiply_adds + standardization_multiply_adds + head_multiply_adds
        units = multiply_adds + pooling_comparisons + pooling_additions
        return make_forward_work_audit(
            units=units,
            method="analytic_random_conv_dot_products_pooling_standardization_plus_linear_logits",
            confidence="analytic",
            split_comparisons=pooling_comparisons,
            multiply_adds=multiply_adds,
            additions=pooling_additions,
            reductions=pooling_comparisons + pooling_additions,
        )

    return make_forward_work_audit(
        units=spec.compute_proxy_ops,
        method=f"analytic_{spec.compute_proxy_kind}",
        confidence="analytic",
        multiply_adds=spec.compute_proxy_ops,
        notes="non-convolutional random-feature transformer; using stored analytic work count",
    )


def sklearn_mlp_activation_ops(model) -> tuple[float, str]:
    classifier = getattr(model, "classifier", model)
    if not isinstance(classifier, MLPClassifier):
        return 0.0, ""
    hidden = classifier.hidden_layer_sizes
    if isinstance(hidden, int):
        hidden_sizes = (hidden,)
    else:
        hidden_sizes = tuple(int(width) for width in hidden)
    activation_ops = float(sum(hidden_sizes))
    if activation_ops <= 0:
        return 0.0, ""
    return activation_ops, f"plus {classifier.activation} hidden activation applications"


def output_numel(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (tuple, list)):
        return int(sum(output_numel(item) for item in value))
    return 0


def torch_module_activation_ops(model, x_probe: pd.DataFrame) -> tuple[float, str]:
    torch_model = getattr(model, "model", None)
    if torch_model is None or not hasattr(model, "predict_logits"):
        return 0.0, ""
    try:
        import torch
        from torch import nn

        activation_types = (
            nn.ELU,
            nn.GELU,
            nn.LeakyReLU,
            nn.LogSigmoid,
            nn.LogSoftmax,
            nn.ReLU,
            nn.SELU,
            nn.SiLU,
            nn.Sigmoid,
            nn.Softmax,
            nn.Tanh,
        )
        total = 0

        def hook(_module, _inputs, output):
            nonlocal total
            total += output_numel(output)

        hooks = [module.register_forward_hook(hook) for module in torch_model.modules() if isinstance(module, activation_types)]
        was_training = bool(torch_model.training)
        sample = x_probe.iloc[:1] if hasattr(x_probe, "iloc") else x_probe[:1]
        try:
            with torch.no_grad():
                model.predict_logits(sample)
        finally:
            for handle in hooks:
                handle.remove()
            torch_model.train(was_training)
        if total <= 0:
            return 0.0, ""
        return float(total), "plus explicit torch activation-module applications counted on one probe row"
    except Exception as exc:
        return 0.0, f"torch activation count unavailable: {exc}"


def neural_activation_ops(model, x_probe: pd.DataFrame) -> tuple[float, str]:
    activation_ops, notes = sklearn_mlp_activation_ops(model)
    if activation_ops > 0 or notes:
        return activation_ops, notes
    return torch_module_activation_ops(model, x_probe)


def audit_forward_compute(model, spec: ModelSpec, x_probe: pd.DataFrame) -> dict[str, float | str]:
    """Estimate per-observation forward work with explicit confidence labels."""
    total_ops = float(spec.compute_proxy_ops)
    audit = make_forward_work_audit(
        units=total_ops,
        method=spec.compute_proxy_kind,
        confidence="proxy",
        notes="legacy compute proxy carried forward; not a paper-facing exact work count",
    )
    architecture = spec.architecture
    classifier = getattr(model, "classifier", model)

    if architecture == "decision_tree":
        audit = tree_forward_work(
            split_comparisons=total_ops,
            tree_count=1,
            additions_per_tree=0.0,
            method="empirical_test_set_tree_path_depth_plus_leaf_lookup",
            avg_tree_path_depth=total_ops,
        )
    elif architecture in {"random_forest", "extra_trees"}:
        tree_count = tree_count_for_model(classifier)
        audit = tree_forward_work(
            split_comparisons=total_ops,
            tree_count=tree_count,
            additions_per_tree=len(CLASS_ORDER),
            method="empirical_test_set_forest_path_depth_plus_leaf_lookup_and_vote_accumulation",
            avg_tree_path_depth=total_ops / tree_count if tree_count else np.nan,
        )
    elif architecture == "xgboost":
        effective_estimators = getattr(model, "best_iteration", None)
        effective_estimators = int(effective_estimators + 1) if effective_estimators is not None else None
        path_sum, path_avg, tree_count = xgboost_average_path_depth(model, x_probe, effective_estimators)
        if np.isfinite(path_sum):
            audit = tree_forward_work(
                split_comparisons=path_sum,
                tree_count=tree_count,
                additions_per_tree=1.0,
                method="xgboost_empirical_pred_leaf_path_depth_plus_leaf_lookup_and_logit_add",
                avg_tree_path_depth=path_avg,
            )
    elif architecture == "lightgbm":
        effective_estimators = getattr(classifier, "best_iteration_", None)
        effective_estimators = int(effective_estimators) if effective_estimators else None
        path_sum, path_avg, tree_count = lightgbm_average_path_depth(classifier, x_probe, effective_estimators)
        if np.isfinite(path_sum):
            audit = tree_forward_work(
                split_comparisons=path_sum,
                tree_count=tree_count,
                additions_per_tree=1.0,
                method="lightgbm_empirical_pred_leaf_path_depth_plus_leaf_lookup_and_logit_add",
                avg_tree_path_depth=path_avg,
            )
    elif architecture == "catboost_symmetric_trees":
        path_sum, path_avg, tree_count = catboost_total_path_depth(classifier)
        if np.isfinite(path_sum):
            audit = tree_forward_work(
                split_comparisons=path_sum,
                tree_count=tree_count,
                additions_per_tree=1.0,
                method="catboost_symmetric_tree_depth_plus_leaf_lookup_and_logit_add",
                avg_tree_path_depth=path_avg,
            )
    elif "multiply_adds" in spec.compute_proxy_kind or architecture in {
        "mlp_classifier",
        "torch_mlp",
        "torch_temporal_cnn",
        "torch_temporal_tcn",
        "torch_cnn_attention",
        "torch_gru",
        "torch_lstm",
        "torch_deeplob_lite",
        "torch_deeplob_full",
        "torch_factorization_machine",
        "torch_tabular_transformer",
    }:
        activation_ops, activation_notes = neural_activation_ops(model, x_probe)
        units = total_ops + activation_ops
        method = f"analytic_{spec.compute_proxy_kind}"
        if activation_ops > 0:
            method = f"{method}_plus_activation_applications"
        audit = make_forward_work_audit(
            units=units,
            method=method,
            confidence="analytic",
            multiply_adds=total_ops,
            activation_ops=activation_ops,
            notes=activation_notes,
        )
    elif architecture in {"nystroem_rbf_logistic", "rbf_random_features_sgd", "random_conv_sgd", "random_conv_logistic"}:
        audit = random_feature_forward_work(model, spec)
    elif architecture == "explainable_boosting_machine":
        audit = make_forward_work_audit(
            units=total_ops * 2,
            method="ebm_terms_table_lookup_count_plus_term_additions",
            confidence="analytic",
            table_lookups=total_ops,
            additions=total_ops,
        )
    elif architecture == "hist_gradient_boosting":
        path_sum, path_avg, tree_count = hgb_empirical_path_depth(classifier, x_probe)
        if np.isfinite(path_sum):
            audit = tree_forward_work(
                split_comparisons=path_sum,
                tree_count=tree_count,
                additions_per_tree=1.0,
                method="sklearn_hgb_empirical_tree_path_depth_plus_leaf_lookup_and_logit_add",
                avg_tree_path_depth=path_avg,
            )
        else:
            tree_count = max(1, int(round(total_ops / max(1.0, np.log2(max(2.0, spec.n_parameters or 2.0))))))
            audit = tree_forward_work(
                split_comparisons=total_ops,
                tree_count=tree_count,
                additions_per_tree=1.0,
                method="hgb_configured_iterations_times_log2_leaves_proxy_plus_leaf_lookup_and_logit_add",
                confidence="proxy",
                notes="fallback proxy; exact sklearn HGB traversal failed",
            )
    elif "parameter_count_proxy" in spec.compute_proxy_kind:
        audit = make_forward_work_audit(
            units=total_ops,
            method=spec.compute_proxy_kind,
            confidence="proxy",
            notes="parameter count proxy is not a forward-work count",
        )
    return audit


def mlp_parameter_count(d: int, hidden: tuple[int, ...], n_classes: int) -> int:
    widths = (d,) + hidden + (n_classes,)
    return int(sum((left + 1) * right for left, right in zip(widths[:-1], widths[1:])))


def mlp_compute_ops(d: int, hidden: tuple[int, ...], n_classes: int) -> int:
    widths = (d,) + hidden + (n_classes,)
    return int(sum(left * right for left, right in zip(widths[:-1], widths[1:])))


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_torch_device(raw: str):
    import torch

    if raw != "auto":
        return torch.device(raw)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_label_array(y: pd.Series) -> np.ndarray:
    return y.map({-1: 0, 0: 1, 1: 2}).to_numpy(dtype=np.int64)


def torch_proba_frame(prob: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(prob, columns=CLASS_ORDER)


def make_torch_arrays(x: pd.DataFrame, y: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    x_arr = x.to_numpy(dtype=np.float32)
    if y is None:
        return x_arr, None
    return x_arr, torch_label_array(y)


def standardize_arrays(
    x_fit: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_fit.mean(axis=0, keepdims=True)
    std = x_fit.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x_fit - mean) / std, (x_validation - mean) / std, (x_test - mean) / std, mean, std


def reshape_window_array(x: np.ndarray, window: int, base_d: int) -> np.ndarray:
    return x.reshape(len(x), window, base_d)


def is_sequence_torch_input(input_kind: str) -> bool:
    return input_kind in {
        "temporal_cnn",
        "temporal_inception",
        "temporal_tcn",
        "temporal_cnn_attention",
        "temporal_recurrent",
        "deeplob_lite",
        "mlplob",
        "tlob",
    }


class RandomConvFeatureTransformer:
    """Fixed random temporal convolutions with simple ROCKET-style summaries."""

    def __init__(
        self,
        *,
        window: int,
        base_d: int,
        n_kernels: int,
        kernel_sizes: tuple[int, ...],
        random_state: int,
    ):
        self.window = int(window)
        self.base_d = int(base_d)
        self.n_kernels = int(n_kernels)
        self.kernel_sizes = tuple(int(size) for size in kernel_sizes)
        self.random_state = int(random_state)

    def fit(self, x, y=None):
        if self.window < 2:
            raise ValueError("random temporal convolution features require window >= 2")
        sizes = [size for size in self.kernel_sizes if 1 <= size <= self.window]
        if not sizes:
            raise ValueError(f"no random convolution kernel sizes fit window={self.window}")
        rng = np.random.default_rng(self.random_state + self.n_kernels * 997 + self.window)
        self.channels_ = rng.integers(0, self.base_d, size=self.n_kernels)
        self.lengths_ = rng.choice(np.array(sizes, dtype=np.int64), size=self.n_kernels, replace=True)
        self.biases_ = rng.normal(0.0, 0.5, size=self.n_kernels).astype(np.float32)
        weights = []
        for length in self.lengths_:
            kernel = rng.normal(0.0, 1.0, size=int(length)).astype(np.float32)
            kernel -= kernel.mean()
            norm = np.linalg.norm(kernel)
            if norm > 0:
                kernel /= norm
            weights.append(kernel)
        self.weights_ = weights
        self.n_features_out_ = self.n_kernels * 3
        self.compute_proxy_ops_ = float(
            sum(max(1, self.window - int(length) + 1) * int(length) for length in self.lengths_)
            + self.n_features_out_
        )
        return self

    def transform(self, x):
        arr = x.to_numpy(dtype=np.float32) if hasattr(x, "to_numpy") else np.asarray(x, dtype=np.float32)
        arr = reshape_window_array(arr, self.window, self.base_d)
        out = np.empty((len(arr), self.n_features_out_), dtype=np.float32)
        for idx, (channel, length, bias, weights) in enumerate(
            zip(self.channels_, self.lengths_, self.biases_, self.weights_)
        ):
            series = arr[:, :, int(channel)]
            windows = np.lib.stride_tricks.sliding_window_view(series, int(length), axis=1)
            response = np.einsum("ntk,k->nt", windows, weights, optimize=True) + bias
            offset = idx * 3
            out[:, offset] = response.max(axis=1)
            out[:, offset + 1] = response.mean(axis=1)
            out[:, offset + 2] = (response > 0).mean(axis=1)
        return out

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


class TemporalSummaryTransformer:
    """Causal multiscale summaries over the flattened raw LOB window."""

    def __init__(self, *, window: int, base_d: int, summary_windows: tuple[int, ...]):
        self.window = int(window)
        self.base_d = int(base_d)
        self.summary_windows = tuple(int(size) for size in summary_windows)

    def fit(self, x, y=None):
        windows = sorted({size for size in self.summary_windows if 1 <= size <= self.window})
        if not windows:
            windows = [self.window]
        self.active_windows_ = tuple(windows)
        # Current value plus mean/std/min/max/delta for each active trailing window.
        self.n_features_out_ = self.base_d * (1 + 5 * len(self.active_windows_))
        self.compute_proxy_ops_ = float(self.base_d * sum(5 * size for size in self.active_windows_))
        return self

    def transform(self, x):
        arr = x.to_numpy(dtype=np.float32) if hasattr(x, "to_numpy") else np.asarray(x, dtype=np.float32)
        arr = reshape_window_array(arr, self.window, self.base_d)
        features = [arr[:, -1, :]]
        for size in self.active_windows_:
            tail = arr[:, -size:, :]
            features.extend(
                [
                    tail.mean(axis=1),
                    tail.std(axis=1),
                    tail.min(axis=1),
                    tail.max(axis=1),
                    tail[:, -1, :] - tail[:, 0, :],
                ]
            )
        return np.concatenate(features, axis=1)

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


class TransformedClassifier:
    def __init__(self, transformer, classifier, class_map: dict[int, int] | None = None):
        self.transformer = transformer
        self.classifier = classifier
        self.class_map = class_map or {}
        self.classes_ = np.array(CLASS_ORDER)

    def predict_proba(self, x):
        raw = self.classifier.predict_proba(self.transformer.transform(x))
        raw_classes = getattr(self.classifier, "classes_", self.classes_)
        mapped_classes = [self.class_map.get(int(label), int(label)) for label in raw_classes]
        frame = pd.DataFrame(raw, columns=mapped_classes)
        return frame.reindex(columns=CLASS_ORDER, fill_value=0.0).to_numpy()


class ClassMappedClassifier:
    def __init__(self, classifier, class_map: dict[int, int] | None = None, metadata: dict | None = None):
        self.classifier = classifier
        self.class_map = class_map or {}
        self.metadata = metadata or {}
        self.classes_ = np.array(CLASS_ORDER)

    def predict_proba(self, x):
        raw = self.classifier.predict_proba(x)
        raw_classes = getattr(self.classifier, "classes_", self.classes_)
        mapped_classes = [self.class_map.get(int(label), int(label)) for label in raw_classes]
        frame = pd.DataFrame(raw, columns=mapped_classes)
        return frame.reindex(columns=CLASS_ORDER, fill_value=0.0).to_numpy()


class TorchProbClassifier:
    def __init__(
        self,
        model,
        device,
        input_kind: str,
        window: int,
        base_d: int,
        mean: np.ndarray,
        std: np.ndarray,
        temperature: float = 1.0,
        predict_batch_size: int = 8192,
        metadata: dict | None = None,
    ):
        self.model = model
        self.device = device
        self.input_kind = input_kind
        self.window = window
        self.base_d = base_d
        self.mean = mean
        self.std = std
        self.temperature = max(float(temperature), 1e-6)
        self.predict_batch_size = max(1, int(predict_batch_size))
        self.metadata = metadata or {}
        self.classes_ = np.array(CLASS_ORDER)

    def _prepare(self, x: pd.DataFrame | np.ndarray):
        import torch

        arr = x.to_numpy(dtype=np.float32) if hasattr(x, "to_numpy") else np.asarray(x, dtype=np.float32)
        arr = (arr - self.mean) / self.std
        if self.input_kind == "deeplob_full":
            arr = reshape_window_array(arr, self.window, self.base_d)
            arr = arr[:, None, :, :]
        elif is_sequence_torch_input(self.input_kind):
            arr = reshape_window_array(arr, self.window, self.base_d)
            arr = np.transpose(arr, (0, 2, 1))
        return torch.from_numpy(arr)

    def predict_logits(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        x_tensor = self._prepare(x)
        logits_out = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), self.predict_batch_size):
                logits = self.model(x_tensor[start : start + self.predict_batch_size].to(self.device))
                logits_out.append(logits.cpu().numpy())
        return np.concatenate(logits_out, axis=0)

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        import torch

        logits = self.predict_logits(x)
        logits_tensor = torch.from_numpy(logits / self.temperature)
        return torch.softmax(logits_tensor, dim=1).numpy()

    def predict_proba_uncalibrated(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        import torch

        logits = self.predict_logits(x)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()


def build_torch_mlp(input_dim: int, hidden: tuple[int, ...], residual: bool):
    import torch
    from torch import nn

    if not residual:
        layers = []
        prev = input_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        layers.append(nn.Linear(prev, len(CLASS_ORDER)))
        return nn.Sequential(*layers)

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            if not hidden:
                raise ValueError("Residual MLP needs at least one hidden width")
            self.input = nn.Linear(input_dim, hidden[0])
            self.blocks = nn.ModuleList()
            for width in hidden:
                self.blocks.append(
                    nn.Sequential(
                        nn.LayerNorm(width),
                        nn.Linear(width, width),
                        nn.ReLU(),
                        nn.Linear(width, width),
                    )
                )
            self.output = nn.Linear(hidden[0], len(CLASS_ORDER))

        def forward(self, x):
            z = torch.relu(self.input(x))
            for block in self.blocks:
                z = torch.relu(z + block(z))
            return self.output(z)

    if len(set(hidden)) != 1:
        raise ValueError("Residual MLP hidden widths must all match, e.g. 128x128x128")
    return ResidualMLP()


def build_torch_factorization_machine(input_dim: int, rank: int):
    import torch
    from torch import nn

    class FactorizationMachine(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_dim, len(CLASS_ORDER))
            self.factors = nn.Parameter(torch.empty(len(CLASS_ORDER), input_dim, rank))
            nn.init.xavier_uniform_(self.factors)

        def forward(self, x):
            linear_logits = self.linear(x)
            xv = torch.einsum("bd,cdr->bcr", x, self.factors)
            x2v2 = torch.einsum("bd,cdr->bcr", x * x, self.factors * self.factors)
            interactions = 0.5 * (xv * xv - x2v2).sum(dim=2)
            return linear_logits + interactions

    return FactorizationMachine()


def build_torch_tabular_transformer(input_dim: int, token_dim: int, layers: int, heads: int, ff_multiplier: int = 2):
    import torch
    from torch import nn

    class DenseFeatureTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            if token_dim % heads != 0:
                raise ValueError("token_dim must be divisible by heads")
            self.feature_weight = nn.Parameter(torch.empty(input_dim, token_dim))
            self.feature_bias = nn.Parameter(torch.zeros(input_dim, token_dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            nn.init.xavier_uniform_(self.feature_weight)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=heads,
                dim_feedforward=token_dim * ff_multiplier,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output = nn.Linear(token_dim, len(CLASS_ORDER))

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            z = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.output(z[:, 0, :])

    return DenseFeatureTransformer()


def build_torch_mlplob(
    base_d: int,
    window: int,
    hidden_dim: int,
    layers: int,
    expansion: int,
    dropout: float,
    use_bin: bool,
    bin_eps: float,
):
    import torch
    from torch import nn

    if hidden_dim <= 0 or layers <= 0 or expansion <= 0:
        raise ValueError("MLPLOB hidden_dim, layers, and expansion must be positive")

    class SimpleBiN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(1, base_d, 1))
            self.beta = nn.Parameter(torch.zeros(1, base_d, 1))
            self.mix_logit = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            time_norm = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True, unbiased=False).clamp_min(bin_eps)
            feature_norm = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True, unbiased=False).clamp_min(bin_eps)
            mix = torch.sigmoid(self.mix_logit)
            return (mix * time_norm + (1.0 - mix) * feature_norm) * self.gamma + self.beta

    class MixerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_norm = nn.LayerNorm(hidden_dim)
            self.feature_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * expansion, hidden_dim),
                nn.Dropout(dropout),
            )
            self.temporal_norm = nn.LayerNorm(hidden_dim)
            self.temporal_mlp = nn.Sequential(
                nn.Linear(window, window * expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(window * expansion, window),
                nn.Dropout(dropout),
            )

        def forward(self, z):
            z = z + self.feature_mlp(self.feature_norm(z))
            temporal = self.temporal_norm(z).transpose(1, 2)
            temporal = self.temporal_mlp(temporal).transpose(1, 2)
            return z + temporal

    class MLPLOB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bin = SimpleBiN() if use_bin else nn.Identity()
            self.feature_embed = nn.Linear(base_d, hidden_dim)
            self.time_embed = nn.Parameter(torch.zeros(1, window, hidden_dim))
            self.blocks = nn.ModuleList([MixerBlock() for _ in range(layers)])
            self.norm = nn.LayerNorm(hidden_dim)
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(CLASS_ORDER)),
            )

        def forward(self, x):
            z = self.bin(x).transpose(1, 2)
            z = self.feature_embed(z) + self.time_embed
            for block in self.blocks:
                z = block(z)
            z = self.norm(z).mean(dim=1)
            return self.output(z)

    return MLPLOB()


def build_torch_tlob(
    base_d: int,
    window: int,
    hidden_dim: int,
    layers: int,
    heads: int,
    ff_multiplier: int,
    dropout: float,
    use_bin: bool,
    bin_eps: float,
):
    import torch
    from torch import nn

    if hidden_dim <= 0 or layers <= 0 or heads <= 0 or ff_multiplier <= 0:
        raise ValueError("TLOB hidden_dim, layers, heads, and ff_multiplier must be positive")
    if hidden_dim % heads != 0:
        raise ValueError("TLOB hidden_dim must be divisible by heads")

    class SimpleBiN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(1, base_d, 1))
            self.beta = nn.Parameter(torch.zeros(1, base_d, 1))
            self.mix_logit = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            time_norm = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True, unbiased=False).clamp_min(bin_eps)
            feature_norm = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True, unbiased=False).clamp_min(bin_eps)
            mix = torch.sigmoid(self.mix_logit)
            return (mix * time_norm + (1.0 - mix) * feature_norm) * self.gamma + self.beta

    class TLOBBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_norm = nn.LayerNorm(hidden_dim)
            self.temporal_attention = nn.MultiheadAttention(
                hidden_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            self.spatial_norm = nn.LayerNorm(hidden_dim)
            self.spatial_attention = nn.MultiheadAttention(
                hidden_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
                nn.Dropout(dropout),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, z):
            batch, time_steps, features, width = z.shape
            temporal = z.permute(0, 2, 1, 3).reshape(batch * features, time_steps, width)
            temporal_norm = self.temporal_norm(temporal)
            temporal_update, _ = self.temporal_attention(
                temporal_norm,
                temporal_norm,
                temporal_norm,
                need_weights=False,
            )
            temporal = temporal + self.dropout(temporal_update)
            z = temporal.reshape(batch, features, time_steps, width).permute(0, 2, 1, 3)

            spatial = z.reshape(batch * time_steps, features, width)
            spatial_norm = self.spatial_norm(spatial)
            spatial_update, _ = self.spatial_attention(
                spatial_norm,
                spatial_norm,
                spatial_norm,
                need_weights=False,
            )
            spatial = spatial + self.dropout(spatial_update)
            z = spatial.reshape(batch, time_steps, features, width)
            return z + self.ffn(self.ffn_norm(z))

    class TLOB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bin = SimpleBiN() if use_bin else nn.Identity()
            self.value_embed = nn.Linear(1, hidden_dim)
            self.time_embed = nn.Parameter(torch.zeros(1, window, 1, hidden_dim))
            self.feature_embed = nn.Parameter(torch.zeros(1, 1, base_d, hidden_dim))
            self.blocks = nn.ModuleList([TLOBBlock() for _ in range(layers)])
            self.norm = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, len(CLASS_ORDER))

        def forward(self, x):
            z = self.bin(x).transpose(1, 2).unsqueeze(-1)
            z = self.value_embed(z) + self.time_embed + self.feature_embed
            for block in self.blocks:
                z = block(z)
            z = self.norm(z).mean(dim=(1, 2))
            return self.output(z)

    return TLOB()


def build_torch_temporal_cnn(base_d: int, channels: tuple[int, ...], kernel_size: int):
    import torch
    from torch import nn

    layers = []
    in_channels = base_d
    padding = kernel_size // 2
    for out_channels in channels:
        layers.extend(
            [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
            ]
        )
        in_channels = out_channels

    class TemporalCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*layers)
            self.output = nn.Linear(in_channels, len(CLASS_ORDER))

        def forward(self, x):
            z = self.body(x)
            z = z.mean(dim=-1)
            return self.output(z)

    return TemporalCNN()


def build_torch_inception_time(
    base_d: int,
    block_channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    bottleneck_channels: int,
):
    import torch
    from torch import nn

    if not block_channels:
        raise ValueError("InceptionTime needs at least one block channel width")
    if any(kernel % 2 == 0 for kernel in kernel_sizes):
        raise ValueError("InceptionTime kernel sizes must be odd to preserve sequence length")
    if not kernel_sizes:
        raise ValueError("InceptionTime needs at least one kernel size")

    class InceptionBlock(nn.Module):
        def __init__(self, in_channels: int, branch_channels: int):
            super().__init__()
            bottleneck_out = min(bottleneck_channels, in_channels) if bottleneck_channels > 0 else in_channels
            self.bottleneck = (
                nn.Sequential(nn.Conv1d(in_channels, bottleneck_out, kernel_size=1), nn.ReLU())
                if bottleneck_out != in_channels
                else nn.Identity()
            )
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(bottleneck_out, branch_channels, kernel_size=kernel, padding=kernel // 2),
                        nn.GroupNorm(1, branch_channels),
                        nn.ReLU(),
                    )
                    for kernel in kernel_sizes
                ]
            )
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels, branch_channels, kernel_size=1),
                nn.GroupNorm(1, branch_channels),
                nn.ReLU(),
            )
            out_channels = branch_channels * (len(kernel_sizes) + 1)
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            z = self.bottleneck(x)
            out = torch.cat([branch(z) for branch in self.branches] + [self.pool_branch(x)], dim=1)
            return torch.relu(out + self.skip(x))

    class InceptionTime(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = base_d
            for branch_channels in block_channels:
                blocks.append(InceptionBlock(in_channels, branch_channels))
                in_channels = branch_channels * (len(kernel_sizes) + 1)
            self.body = nn.Sequential(*blocks)
            self.output = nn.Linear(in_channels, len(CLASS_ORDER))

        def forward(self, x):
            z = self.body(x)
            z = z.mean(dim=-1)
            return self.output(z)

    return InceptionTime()


# TODO: Add the remaining higher-compute neural baselines after DeepLOB-full and
# TCN are evaluated: small transformer encoders, LOB-level 2D ResNets,
# CNN-attention pooling heads, and calibrated tree/NN probability ensembles.


def build_torch_tcn(base_d: int, channels: tuple[int, ...], kernel_size: int):
    import torch
    from torch import nn

    if kernel_size % 2 == 0:
        raise ValueError("TCN kernel size must be odd to preserve sequence length")

    class TCNBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int):
            super().__init__()
            padding = dilation * (kernel_size // 2)
            self.body = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.GroupNorm(1, out_channels),
            )
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return torch.relu(self.body(x) + self.skip(x))

    class TemporalTCN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = base_d
            for idx, out_channels in enumerate(channels):
                blocks.append(TCNBlock(in_channels, out_channels, dilation=2**idx))
                in_channels = out_channels
            self.body = nn.Sequential(*blocks)
            self.output = nn.Linear(in_channels, len(CLASS_ORDER))

        def forward(self, x):
            z = self.body(x)
            z = z.mean(dim=-1)
            return self.output(z)

    if not channels:
        raise ValueError("TCN needs at least one channel width")
    return TemporalTCN()


def build_torch_cnn_attention(base_d: int, channels: tuple[int, ...], kernel_size: int):
    import torch
    from torch import nn

    if kernel_size % 2 == 0:
        raise ValueError("CNN-attention kernel size must be odd to preserve sequence length")
    if not channels:
        raise ValueError("CNN-attention needs at least one channel width")

    class TemporalCNNAttention(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_channels = base_d
            for out_channels in channels:
                layers.extend(
                    [
                        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                        nn.GroupNorm(1, out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = out_channels
            self.body = nn.Sequential(*layers)
            self.attention = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.Tanh(),
                nn.Linear(in_channels, 1),
            )
            self.output = nn.Linear(in_channels, len(CLASS_ORDER))

        def forward(self, x):
            z = self.body(x).transpose(1, 2)
            weights = torch.softmax(self.attention(z).squeeze(-1), dim=1)
            context = (z * weights.unsqueeze(-1)).sum(dim=1)
            return self.output(context)

    return TemporalCNNAttention()


def build_torch_recurrent(base_d: int, hidden: int, layers: int, rnn_type: str):
    import torch
    from torch import nn

    if hidden <= 0 or layers <= 0:
        raise ValueError("Recurrent hidden size and layer count must be positive")
    if rnn_type not in {"gru", "lstm"}:
        raise ValueError("Recurrent type must be gru or lstm")

    class TemporalRecurrent(nn.Module):
        def __init__(self):
            super().__init__()
            recurrent_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
            self.recurrent = recurrent_cls(input_size=base_d, hidden_size=hidden, num_layers=layers, batch_first=True)
            self.output = nn.Linear(hidden, len(CLASS_ORDER))

        def forward(self, x):
            z = x.transpose(1, 2)
            _, hidden_state = self.recurrent(z)
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]
            return self.output(hidden_state[-1])

    return TemporalRecurrent()


def build_torch_deeplob_lite(
    base_d: int,
    stem_channels: int,
    inception_channels: int,
    recurrent_hidden: int,
    kernel_sizes: tuple[int, ...],
):
    import torch
    from torch import nn

    if any(kernel % 2 == 0 for kernel in kernel_sizes):
        raise ValueError("DeepLOB-lite inception kernel sizes must be odd to preserve sequence length")

    class DeepLOBLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(base_d, stem_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, stem_channels),
                nn.ReLU(),
                nn.Conv1d(stem_channels, stem_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, stem_channels),
                nn.ReLU(),
            )
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(stem_channels, inception_channels, kernel_size=kernel, padding=kernel // 2),
                        nn.GroupNorm(1, inception_channels),
                        nn.ReLU(),
                    )
                    for kernel in kernel_sizes
                ]
            )
            gru_input = inception_channels * len(kernel_sizes)
            self.recurrent = nn.GRU(input_size=gru_input, hidden_size=recurrent_hidden, batch_first=True)
            self.output = nn.Linear(recurrent_hidden, len(CLASS_ORDER))

        def forward(self, x):
            z = self.stem(x)
            z = torch.cat([branch(z) for branch in self.branches], dim=1)
            z = z.transpose(1, 2)
            _, hidden = self.recurrent(z)
            return self.output(hidden[-1])

    if not kernel_sizes:
        raise ValueError("DeepLOB-lite needs at least one inception kernel")
    return DeepLOBLite()


def build_torch_deeplob_full(
    base_d: int,
    conv_filters: int,
    inception_filters: int,
    lstm_hidden: int,
    dropout: float,
    norm: str,
    conv2_activation: str,
):
    import torch
    from torch import nn

    if base_d != 40:
        raise ValueError("Full DeepLOB expects the raw 40 LOB features")
    if norm not in {"batchnorm", "groupnorm", "none"}:
        raise ValueError("DeepLOB-full norm must be batchnorm, groupnorm, or none")
    if conv2_activation not in {"leaky_relu", "tanh"}:
        raise ValueError("DeepLOB-full conv2 activation must be leaky_relu or tanh")

    def same_time_conv2d(in_channels: int, out_channels: int, kernel_time: int):
        top = (kernel_time - 1) // 2
        bottom = kernel_time - 1 - top
        return nn.Sequential(nn.ZeroPad2d((0, 0, top, bottom)), nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_time, 1)))

    def activation(kind: str = "leaky_relu"):
        if kind == "tanh":
            return nn.Tanh()
        return nn.LeakyReLU(negative_slope=0.01)

    def norm_layer(channels: int):
        if norm == "batchnorm":
            return nn.BatchNorm2d(channels)
        if norm == "groupnorm":
            return nn.GroupNorm(1, channels)
        return nn.Identity()

    def conv_then_norm_then_activation(module, channels: int, act_kind: str = "leaky_relu"):
        return [module, norm_layer(channels), activation(act_kind)]

    class DeepLOBFull(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                *conv_then_norm_then_activation(nn.Conv2d(1, conv_filters, kernel_size=(1, 2), stride=(1, 2)), conv_filters),
                *conv_then_norm_then_activation(same_time_conv2d(conv_filters, conv_filters, 4), conv_filters),
                *conv_then_norm_then_activation(same_time_conv2d(conv_filters, conv_filters, 4), conv_filters),
                *conv_then_norm_then_activation(
                    nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 2), stride=(1, 2)),
                    conv_filters,
                    conv2_activation,
                ),
                *conv_then_norm_then_activation(same_time_conv2d(conv_filters, conv_filters, 4), conv_filters, conv2_activation),
                *conv_then_norm_then_activation(same_time_conv2d(conv_filters, conv_filters, 4), conv_filters, conv2_activation),
                *conv_then_norm_then_activation(nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 10)), conv_filters),
                *conv_then_norm_then_activation(same_time_conv2d(conv_filters, conv_filters, 4), conv_filters),
                *conv_then_norm_then_activation(same_time_conv2d(conv_filters, conv_filters, 4), conv_filters),
            )
            self.branch_3 = nn.Sequential(
                *conv_then_norm_then_activation(nn.Conv2d(conv_filters, inception_filters, kernel_size=(1, 1)), inception_filters),
                *conv_then_norm_then_activation(same_time_conv2d(inception_filters, inception_filters, 3), inception_filters),
            )
            self.branch_5 = nn.Sequential(
                *conv_then_norm_then_activation(nn.Conv2d(conv_filters, inception_filters, kernel_size=(1, 1)), inception_filters),
                *conv_then_norm_then_activation(same_time_conv2d(inception_filters, inception_filters, 5), inception_filters),
            )
            self.branch_pool = nn.Sequential(
                nn.ZeroPad2d((0, 0, 1, 1)),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1)),
                *conv_then_norm_then_activation(nn.Conv2d(conv_filters, inception_filters, kernel_size=(1, 1)), inception_filters),
            )
            self.dropout = nn.Dropout(p=dropout)
            self.lstm = nn.LSTM(input_size=3 * inception_filters, hidden_size=lstm_hidden, batch_first=True)
            self.output = nn.Linear(lstm_hidden, len(CLASS_ORDER))

        def forward(self, x):
            z = self.conv(x)
            z = torch.cat([self.branch_3(z), self.branch_5(z), self.branch_pool(z)], dim=1)
            z = self.dropout(z)
            z = z.squeeze(-1).transpose(1, 2)
            _, hidden = self.lstm(z)
            return self.output(hidden[0][-1])

    return DeepLOBFull()


def torch_parameter_count(model) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def torch_temporal_cnn_compute_ops(window: int, base_d: int, channels: tuple[int, ...], kernel_size: int) -> int:
    ops = 0
    in_channels = base_d
    for out_channels in channels:
        ops += window * in_channels * out_channels * kernel_size
        in_channels = out_channels
    ops += in_channels * len(CLASS_ORDER)
    return int(ops)


def torch_factorization_machine_compute_ops(d: int, rank: int, n_classes: int) -> int:
    return int(n_classes * (2 * d * rank + d))


def torch_tabular_transformer_compute_ops(d: int, token_dim: int, layers: int, heads: int, ff_multiplier: int) -> int:
    del heads
    tokens = d + 1
    tokenizer_ops = d * token_dim
    attention_ops = tokens * tokens * token_dim
    projection_ops = 4 * tokens * token_dim * token_dim
    ff_ops = 2 * tokens * token_dim * token_dim * ff_multiplier
    output_ops = token_dim * len(CLASS_ORDER)
    return int(tokenizer_ops + layers * (attention_ops + projection_ops + ff_ops) + output_ops)


def torch_mlplob_compute_ops(window: int, base_d: int, hidden_dim: int, layers: int, expansion: int) -> int:
    embed_ops = window * base_d * hidden_dim
    feature_mlp_ops = 2 * window * hidden_dim * hidden_dim * expansion
    temporal_mlp_ops = 2 * hidden_dim * window * window * expansion
    head_ops = hidden_dim * hidden_dim + hidden_dim * len(CLASS_ORDER)
    return int(embed_ops + layers * (feature_mlp_ops + temporal_mlp_ops) + head_ops)


def torch_tlob_compute_ops(
    window: int,
    base_d: int,
    hidden_dim: int,
    layers: int,
    heads: int,
    ff_multiplier: int,
) -> int:
    del heads
    embed_ops = window * base_d * hidden_dim
    temporal_attention_ops = base_d * (window * window * hidden_dim + 4 * window * hidden_dim * hidden_dim)
    spatial_attention_ops = window * (base_d * base_d * hidden_dim + 4 * base_d * hidden_dim * hidden_dim)
    ffn_ops = window * base_d * 2 * hidden_dim * hidden_dim * ff_multiplier
    output_ops = hidden_dim * len(CLASS_ORDER)
    return int(embed_ops + layers * (temporal_attention_ops + spatial_attention_ops + ffn_ops) + output_ops)


def torch_inception_time_compute_ops(
    window: int,
    base_d: int,
    block_channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    bottleneck_channels: int,
) -> int:
    ops = 0
    in_channels = base_d
    for branch_channels in block_channels:
        bottleneck_out = min(bottleneck_channels, in_channels) if bottleneck_channels > 0 else in_channels
        if bottleneck_out != in_channels:
            ops += window * in_channels * bottleneck_out
        ops += sum(window * bottleneck_out * branch_channels * kernel for kernel in kernel_sizes)
        ops += window * in_channels * branch_channels
        out_channels = branch_channels * (len(kernel_sizes) + 1)
        if in_channels != out_channels:
            ops += window * in_channels * out_channels
        in_channels = out_channels
    ops += in_channels * len(CLASS_ORDER)
    return int(ops)


def torch_tcn_compute_ops(window: int, base_d: int, channels: tuple[int, ...], kernel_size: int) -> int:
    ops = 0
    in_channels = base_d
    for out_channels in channels:
        ops += window * in_channels * out_channels * kernel_size
        ops += window * out_channels * out_channels * kernel_size
        if in_channels != out_channels:
            ops += window * in_channels * out_channels
        in_channels = out_channels
    ops += in_channels * len(CLASS_ORDER)
    return int(ops)


def torch_cnn_attention_compute_ops(window: int, base_d: int, channels: tuple[int, ...], kernel_size: int) -> int:
    ops = torch_temporal_cnn_compute_ops(window, base_d, channels, kernel_size)
    last_channels = channels[-1]
    ops += window * last_channels * last_channels
    ops += window * last_channels
    return int(ops)


def torch_recurrent_compute_ops(window: int, base_d: int, hidden: int, layers: int, rnn_type: str) -> int:
    gates = 3 if rnn_type == "gru" else 4
    ops = 0
    in_dim = base_d
    for _ in range(layers):
        ops += window * gates * (in_dim * hidden + hidden * hidden)
        in_dim = hidden
    ops += hidden * len(CLASS_ORDER)
    return int(ops)


def torch_deeplob_lite_compute_ops(
    window: int,
    base_d: int,
    stem_channels: int,
    inception_channels: int,
    recurrent_hidden: int,
    kernel_sizes: tuple[int, ...],
) -> int:
    ops = 0
    ops += window * base_d * stem_channels * 3
    ops += window * stem_channels * stem_channels * 3
    ops += sum(window * stem_channels * inception_channels * kernel for kernel in kernel_sizes)
    gru_input = inception_channels * len(kernel_sizes)
    ops += window * 3 * (gru_input * recurrent_hidden + recurrent_hidden * recurrent_hidden)
    ops += recurrent_hidden * len(CLASS_ORDER)
    return int(ops)


def torch_deeplob_full_compute_ops(
    window: int,
    base_d: int,
    conv_filters: int,
    inception_filters: int,
    lstm_hidden: int,
) -> int:
    width = base_d
    ops = 0
    width = (width - 2) // 2 + 1
    ops += window * width * conv_filters * 2
    ops += 2 * window * width * conv_filters * conv_filters * 4
    width = (width - 2) // 2 + 1
    ops += window * width * conv_filters * conv_filters * 2
    ops += 2 * window * width * conv_filters * conv_filters * 4
    width = width - 10 + 1
    ops += window * width * conv_filters * conv_filters * 10
    ops += 2 * window * width * conv_filters * conv_filters * 4
    ops += 2 * window * width * conv_filters * inception_filters
    ops += window * width * conv_filters * inception_filters
    ops += window * width * inception_filters * inception_filters * 3
    ops += window * width * inception_filters * inception_filters * 5
    lstm_input = 3 * inception_filters
    ops += window * 4 * (lstm_input * lstm_hidden + lstm_hidden * lstm_hidden)
    ops += lstm_hidden * len(CLASS_ORDER)
    return int(ops)


def torch_eval_loss_accuracy(model, x_tensor, y_tensor, loss_fn, device, batch_size: int = 8192) -> dict[str, float]:
    import torch

    if x_tensor is None or y_tensor is None or len(x_tensor) == 0:
        return {"loss": np.nan, "accuracy": np.nan}
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            batch_x = x_tensor[start : start + batch_size].to(device)
            batch_y = y_tensor[start : start + batch_size].to(device)
            logits = model(batch_x)
            losses.append(float(loss_fn(logits, batch_y).detach().cpu().item()) * len(batch_y))
            correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu().item())
            total += int(len(batch_y))
    return {
        "loss": float(sum(losses) / total) if total else np.nan,
        "accuracy": float(correct / total) if total else np.nan,
    }


def torch_clear_device_cache(device) -> None:
    try:
        import torch

        if getattr(device, "type", "") == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif getattr(device, "type", "") == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def torch_collect_logits(model, x_tensor, device, batch_size: int = 8192) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            logits = model(x_tensor[start : start + batch_size].to(device))
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(CLASS_ORDER)), dtype=np.float32)


def fit_temperature_from_logits(logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    import torch
    from torch import nn

    if len(logits) == 0:
        return 1.0, np.nan
    logits_tensor = torch.from_numpy(logits.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.int64))
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = loss_fn(logits_tensor / temperature, y_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        temperature = float(torch.exp(log_temperature).clamp(0.05, 20.0).item())
        loss = float(loss_fn(logits_tensor / temperature, y_tensor).item())
    return temperature, loss


def fit_torch_classifier(
    *,
    model,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    args,
    input_kind: str,
    window: int,
    base_d: int,
    mean: np.ndarray,
    std: np.ndarray,
    run_label: str = "torch_model",
) -> TorchProbClassifier:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    device = resolve_torch_device(args.torch_device)
    model = model.to(device)
    if args.torch_optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.torch_lr,
            weight_decay=args.torch_weight_decay,
            eps=args.torch_adam_eps,
        )
    elif args.torch_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.torch_lr,
            weight_decay=args.torch_weight_decay,
            eps=args.torch_adam_eps,
        )
    else:
        raise ValueError(f"Unknown torch optimizer: {args.torch_optimizer}")
    loss_fn = nn.CrossEntropyLoss()
    if input_kind == "deeplob_full":
        x_fit = reshape_window_array(x_fit, window, base_d)[:, None, :, :].copy()
        x_validation = reshape_window_array(x_validation, window, base_d)[:, None, :, :].copy() if len(x_validation) else x_validation
    elif is_sequence_torch_input(input_kind):
        x_fit = np.transpose(reshape_window_array(x_fit, window, base_d), (0, 2, 1)).copy()
        x_validation = np.transpose(reshape_window_array(x_validation, window, base_d), (0, 2, 1)).copy() if len(x_validation) else x_validation
    train_x = torch.from_numpy(x_fit)
    train_y = torch.from_numpy(y_fit)
    generator = torch.Generator()
    generator.manual_seed(args.torch_seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.torch_batch_size, shuffle=True, generator=generator)

    validation_x = torch.from_numpy(x_validation) if len(x_validation) else None
    validation_y = torch.from_numpy(y_validation) if len(y_validation) else None
    best_state = None
    best_metric_value = np.inf if args.torch_early_stop_metric == "loss" else -np.inf
    running_best_validation_loss = np.inf
    running_best_validation_accuracy = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history_rows = []
    stopped_early = False

    for epoch in range(args.torch_epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        if validation_x is None:
            train_loss = float(np.mean(train_losses)) if train_losses else np.nan
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": np.nan,
                    "validation_accuracy": np.nan,
                    "early_stop_metric": args.torch_early_stop_metric,
                    "early_stop_value": np.nan,
                    "best_early_stop_value": np.nan,
                    "best_validation_loss": np.nan,
                    "best_validation_accuracy": np.nan,
                    "bad_epochs": bad_epochs,
                }
            )
            continue
        model.eval()
        validation_metrics = torch_eval_loss_accuracy(
            model,
            validation_x,
            validation_y,
            loss_fn,
            device,
            batch_size=args.torch_eval_batch_size,
        )
        val_loss = validation_metrics["loss"]
        val_accuracy = validation_metrics["accuracy"]
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        early_stop_value = val_loss if args.torch_early_stop_metric == "loss" else val_accuracy
        running_best_validation_loss = min(running_best_validation_loss, val_loss)
        running_best_validation_accuracy = max(running_best_validation_accuracy, val_accuracy)
        if args.torch_early_stop_metric == "loss":
            improved = val_loss < best_metric_value - args.torch_min_delta
        else:
            improved = val_accuracy > best_metric_value + args.torch_min_delta
        if improved:
            best_metric_value = early_stop_value
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "early_stop_metric": args.torch_early_stop_metric,
                "early_stop_value": early_stop_value,
                "best_early_stop_value": best_metric_value,
                "best_validation_loss": running_best_validation_loss,
                "best_validation_accuracy": running_best_validation_accuracy,
                "bad_epochs": bad_epochs,
            }
        )
        if args.torch_log_epochs:
            print(
                f"torch epoch {epoch + 1}/{args.torch_epochs}: "
                f"train_loss={train_loss:.6f} validation_loss={val_loss:.6f} "
                f"validation_accuracy={val_accuracy:.6f} "
                f"early_stop_{args.torch_early_stop_metric}={early_stop_value:.6f} best={best_metric_value:.6f}",
                flush=True,
            )
        if not improved and bad_epochs >= args.torch_patience:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif history_rows:
        best_epoch = len(history_rows)

    final_train = torch_eval_loss_accuracy(model, train_x, train_y, loss_fn, device, batch_size=args.torch_eval_batch_size)
    final_validation = (
        torch_eval_loss_accuracy(model, validation_x, validation_y, loss_fn, device, batch_size=args.torch_eval_batch_size)
        if validation_x is not None
        else {
            "loss": np.nan,
            "accuracy": np.nan,
        }
    )
    temperature = 1.0
    temperature_validation_loss = np.nan
    if args.torch_temperature_scaling and validation_x is not None:
        validation_logits = torch_collect_logits(model, validation_x, device)
        temperature, temperature_validation_loss = fit_temperature_from_logits(
            validation_logits,
            y_validation,
            max_iter=args.torch_temperature_max_iter,
        )

    history_path = ""
    if args.torch_history_output_dir:
        history_dir = Path(args.torch_history_output_dir)
        history_dir.mkdir(parents=True, exist_ok=True)
        history_stem = (
            f"{run_label}_seed{args.torch_seed}_opt{args.torch_optimizer}_lr{args.torch_lr:g}"
            f"_wd{args.torch_weight_decay:g}_bs{args.torch_batch_size}"
            f"_do{args.torch_deeplob_full_dropout:g}_norm{args.torch_deeplob_full_norm}"
            f"_act{args.torch_deeplob_full_conv2_activation}"
        )
        history_path = str(history_dir / f"{safe_filename(history_stem)}_history.csv")
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

    metadata = {
        "torch_seed": args.torch_seed,
        "torch_optimizer": args.torch_optimizer,
        "torch_lr": args.torch_lr,
        "torch_weight_decay": args.torch_weight_decay,
        "torch_batch_size": args.torch_batch_size,
        "torch_eval_batch_size": args.torch_eval_batch_size,
        "torch_epochs": args.torch_epochs,
        "torch_patience": args.torch_patience,
        "torch_min_delta": args.torch_min_delta,
        "torch_early_stop_metric": args.torch_early_stop_metric,
        "torch_adam_eps": args.torch_adam_eps,
        "torch_input_kind": input_kind,
        "torch_input_window": window,
        "torch_epochs_ran": len(history_rows),
        "torch_best_epoch": best_epoch,
        "torch_stopped_early": int(stopped_early),
        "torch_final_train_loss": final_train["loss"],
        "torch_final_train_accuracy": final_train["accuracy"],
        "torch_final_validation_loss": final_validation["loss"],
        "torch_final_validation_accuracy": final_validation["accuracy"],
        "torch_temperature_scaling": int(args.torch_temperature_scaling),
        "torch_temperature": temperature,
        "torch_temperature_validation_loss": temperature_validation_loss,
        "torch_history_path": history_path,
    }
    return TorchProbClassifier(
        model,
        device,
        input_kind,
        window,
        base_d,
        mean,
        std,
        temperature,
        args.torch_eval_batch_size,
        metadata,
    )


def add_result_row(
    rows: list[dict],
    *,
    base: dict,
    spec: ModelSpec,
    model,
    x_validation: pd.DataFrame,
    y_validation_target: pd.Series,
    x_test: pd.DataFrame,
    y_test_target: pd.Series,
    timing_repeats: int,
    checkpoint_path: Path | None = None,
) -> None:
    validation_proba = None
    if len(x_validation):
        validation_proba = proba_frame(model, x_validation)
    proba = proba_frame(model, x_test)
    torch_timing_device = str(base.get("torch_timing_device", "same"))
    timing_mode = str(base.get("timing_mode", "batch"))
    single_timing_samples = int(base.get("single_timing_samples", 256))
    timing_s, latency_us, latency_timing_device = time_predict_proba_with_torch_timing_device(
        model,
        x_test,
        timing_repeats,
        torch_timing_device,
        timing_mode,
        single_timing_samples,
    )

    row = dict(base)
    row.update(
        {
            "model_family": spec.family,
            "architecture": spec.architecture,
            "model_name": spec.name,
            "compute_proxy_ops": spec.compute_proxy_ops,
            "compute_proxy_kind": spec.compute_proxy_kind,
            "n_parameters": spec.n_parameters,
            "prediction_time_s_median": timing_s,
            "latency_us_per_obs_median": latency_us,
            "latency_timing_mode": timing_mode,
            "latency_single_observation_samples": single_timing_samples if timing_mode == "single" else 0,
            "latency_timing_device": latency_timing_device,
            "critical_path_latency_us_per_obs": latency_us,
            "serial_latency_us_per_obs": latency_us,
            "thread_parallel_latency_us_per_obs": latency_us,
            "ideal_parallel_latency_us_per_obs": latency_us,
            "total_compute_proxy_ops": spec.compute_proxy_ops,
            "critical_path_compute_proxy_ops": spec.compute_proxy_ops,
            "parallel_width": 1,
        }
    )
    row.update(audit_forward_compute(model, spec, x_test))
    if hasattr(model, "device"):
        row["torch_device"] = str(model.device)
    if hasattr(model, "metadata"):
        row.update(model.metadata)
    if hasattr(model, "predict_proba_uncalibrated"):
        if len(x_validation):
            validation_raw_proba = torch_proba_frame(model.predict_proba_uncalibrated(x_validation))
            row.update(prefix_metrics(score_probabilities(validation_raw_proba, y_validation_target), "validation_uncalibrated"))
        test_raw_proba = torch_proba_frame(model.predict_proba_uncalibrated(x_test))
        row.update(prefix_metrics(score_probabilities(test_raw_proba, y_test_target), "test_uncalibrated"))
    if validation_proba is not None:
        row.update(prefix_metrics(score_probabilities(validation_proba, y_validation_target), "validation"))
    row.update(prefix_metrics(score_probabilities(proba, y_test_target), "test"))
    # Backward-compatible aliases for existing plotting/analysis code.
    row["accuracy"] = row["test_accuracy"]
    row["balanced_accuracy"] = row["test_balanced_accuracy"]
    row["log_loss"] = row["test_log_loss"]
    rows.append(row)
    if checkpoint_path is not None:
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)


def build_tree_specs(args, d: int, x_probe: pd.DataFrame) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for leaves in args.tree_leaf_values:
        name = f"tree_L{leaves}"

        def fit_fn(leaves=leaves):
            return DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=RANDOM_SEED)

        # The final empirical path depth is measured after fitting. This value is
        # an upfront proxy used only for labels before the fitted correction.
        specs.append(
            ModelSpec(
                family="tree",
                architecture="decision_tree",
                name=name,
                compute_proxy_ops=float(np.log2(leaves)),
                compute_proxy_kind="empirical_tree_path_depth",
                n_parameters=float(leaves),
                fit_fn=fit_fn,
            )
        )
    return specs


def fit_tree_spec(spec: ModelSpec, x_train, y_train, x_test) -> tuple[object, ModelSpec]:
    model = spec.fit_fn()
    model.fit(x_train, y_train)
    avg_depth = tree_average_depth(model, x_test)
    n_nodes = float(model.tree_.node_count)
    corrected = ModelSpec(
        family=spec.family,
        architecture=spec.architecture,
        name=spec.name,
        compute_proxy_ops=avg_depth,
        compute_proxy_kind=spec.compute_proxy_kind,
        n_parameters=n_nodes,
        fit_fn=spec.fit_fn,
    )
    return model, corrected


def fit_forest_model(model, x_train, y_train, x_test, base_spec: ModelSpec) -> tuple[object, ModelSpec]:
    model.fit(x_train, y_train)
    depth_sum = forest_average_depth(model, x_test)
    n_nodes = float(sum(est.tree_.node_count for est in model.estimators_))
    corrected = ModelSpec(
        family=base_spec.family,
        architecture=base_spec.architecture,
        name=base_spec.name,
        compute_proxy_ops=depth_sum,
        compute_proxy_kind=base_spec.compute_proxy_kind,
        n_parameters=n_nodes,
        fit_fn=base_spec.fit_fn,
    )
    return model, corrected


def lower_frontier(df: pd.DataFrame, x_col: str, y_col: str = "log_loss") -> pd.DataFrame:
    work = df.dropna(subset=[x_col, y_col]).sort_values([x_col, y_col])
    rows = []
    best = np.inf
    for _, row in work.iterrows():
        if row[y_col] < best:
            rows.append(row)
            best = row[y_col]
    return pd.DataFrame(rows)


def family_frontiers(df: pd.DataFrame, x_col: str, y_col: str = "log_loss") -> pd.DataFrame:
    rows = []
    for _, group in df.groupby("model_family"):
        frontier = lower_frontier(group, x_col, y_col)
        if not frontier.empty:
            rows.append(frontier)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_frontiers(
    df: pd.DataFrame,
    output_dir: Path,
    x_col: str,
    xlabel: str,
    stem: str,
    y_col: str = "log_loss",
    ylabel: str = "test log-loss",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if y_col not in df.columns:
        return
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return

    global_frontier = lower_frontier(plot_df, x_col, y_col)
    fam_frontiers = family_frontiers(plot_df, x_col, y_col)
    global_frontier.to_csv(output_dir / f"{stem}_global_frontier.csv", index=False)
    fam_frontiers.to_csv(output_dir / f"{stem}_family_frontiers.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    families = sorted(plot_df["model_family"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {family: cmap(idx % 10) for idx, family in enumerate(families)}

    for family, group in plot_df.groupby("model_family"):
        ax.scatter(group[x_col], group[y_col], s=18, alpha=0.18, color=colors[family])
        frontier = lower_frontier(group, x_col, y_col)
        if not frontier.empty:
            ax.plot(
                frontier[x_col],
                frontier[y_col],
                marker="o",
                linewidth=1.6,
                markersize=4,
                color=colors[family],
                label=family,
            )

    ax.plot(
        global_frontier[x_col],
        global_frontier[y_col],
        marker="o",
        linewidth=3.0,
        markersize=5,
        color="black",
        alpha=0.35,
        label="global frontier",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_plots(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    dataset_col = "dataset_variant" if "dataset_variant" in results.columns else "feature_set"
    has_validation = "validation_log_loss" in results.columns and results["validation_log_loss"].notna().any()
    for (target, dataset_variant), group in results.groupby(["score_target", dataset_col]):
        target_dir = output_dir / target / str(dataset_variant)
        plot_frontiers(
            group,
            target_dir,
            "compute_proxy_ops",
            "inference compute proxy ops",
            "test_compute_frontier",
            y_col="test_log_loss" if "test_log_loss" in group.columns else "log_loss",
            ylabel="test log-loss",
        )
        if "audited_forward_ops" in group.columns and group["audited_forward_ops"].notna().any():
            plot_frontiers(
                group,
                target_dir,
                "audited_forward_ops",
                "audited forward operations per observation",
                "test_audited_forward_ops_frontier",
                y_col="test_log_loss" if "test_log_loss" in group.columns else "log_loss",
                ylabel="test log-loss",
            )
        if "forward_work_units" in group.columns and group["forward_work_units"].notna().any():
            plot_frontiers(
                group,
                target_dir,
                "forward_work_units",
                "forward work units per observation",
                "test_forward_work_frontier",
                y_col="test_log_loss" if "test_log_loss" in group.columns else "log_loss",
                ylabel="test log-loss",
            )
        plot_frontiers(
            group,
            target_dir,
            "latency_us_per_obs_median",
            "median predict_proba latency, microseconds/observation",
            "test_latency_frontier",
            y_col="test_log_loss" if "test_log_loss" in group.columns else "log_loss",
            ylabel="test log-loss",
        )
        # Backward-compatible filenames for existing notebooks/scripts.
        plot_frontiers(group, target_dir, "compute_proxy_ops", "inference compute proxy ops", "compute_frontier")
        plot_frontiers(
            group,
            target_dir,
            "latency_us_per_obs_median",
            "median predict_proba latency, microseconds/observation",
            "latency_frontier",
        )
        if has_validation:
            plot_frontiers(
                group,
                target_dir,
                "compute_proxy_ops",
                "inference compute proxy ops",
                "validation_compute_frontier",
                y_col="validation_log_loss",
                ylabel="validation log-loss",
            )
            if "audited_forward_ops" in group.columns and group["audited_forward_ops"].notna().any():
                plot_frontiers(
                    group,
                    target_dir,
                    "audited_forward_ops",
                    "audited forward operations per observation",
                    "validation_audited_forward_ops_frontier",
                    y_col="validation_log_loss",
                    ylabel="validation log-loss",
                )
            if "forward_work_units" in group.columns and group["forward_work_units"].notna().any():
                plot_frontiers(
                    group,
                    target_dir,
                    "forward_work_units",
                    "forward work units per observation",
                    "validation_forward_work_frontier",
                    y_col="validation_log_loss",
                    ylabel="validation log-loss",
                )
            plot_frontiers(
                group,
                target_dir,
                "latency_us_per_obs_median",
                "median predict_proba latency, microseconds/observation",
                "validation_latency_frontier",
                y_col="validation_log_loss",
                ylabel="validation log-loss",
            )

    summary_rows = []
    selection_col = "validation_log_loss" if has_validation else "log_loss"
    for (target, dataset_variant, family), group in results.groupby(["score_target", dataset_col, "model_family"]):
        sortable = group.dropna(subset=[selection_col])
        sort_col = selection_col
        if sortable.empty:
            sortable = group.dropna(subset=["log_loss"])
            sort_col = "log_loss"
        if sortable.empty:
            continue
        best = sortable.sort_values(sort_col).iloc[0]
        summary_rows.append(
            {
                "score_target": target,
                "dataset_variant": dataset_variant,
                "model_family": family,
                "selection_metric": selection_col,
                "best_model_name": best["model_name"],
                "best_architecture": best["architecture"],
                "best_validation_log_loss": best.get("validation_log_loss", np.nan),
                "best_test_log_loss": best.get("test_log_loss", best["log_loss"]),
                "best_log_loss": best["log_loss"],
                "best_compute_proxy_ops": best["compute_proxy_ops"],
                "best_latency_us_per_obs_median": best["latency_us_per_obs_median"],
            }
        )
    best_sort_col = "best_validation_log_loss" if has_validation else "best_log_loss"
    pd.DataFrame(summary_rows).sort_values(["score_target", "dataset_variant", best_sort_col]).to_csv(
        output_dir / "best_by_family.csv",
        index=False,
    )


def run_feature_spec(spec, x_train_all, x_test_all, y_train_all, y_test, args) -> list[dict]:
    rows: list[dict] = []

    for window in args.window_sizes:
        if window > 1 and spec.selection_mode != "raw_first_40" and not args.allow_engineered_windows:
            raise ValueError(
                "Windowed experiments on engineered/supervised FI-2010 features are disabled by default "
                f"because they can leak label-aligned information. Got feature_set={spec.name}, window={window}. "
                "Use --feature-modes lob40 for causal raw LOB windows, or pass --allow-engineered-windows "
                "only for an explicit leakage audit."
            )
        x_train_full, y_train_full = make_lagged_snapshot_dataset(x_train_all[spec.columns], y_train_all, window)
        x_test, y_test_windowed = make_lagged_snapshot_dataset(x_test_all[spec.columns], y_test, window)
        x_test, y_test_windowed = maybe_limit_rows(x_test, y_test_windowed, args.max_test_rows)
        x_train, y_train = maybe_limit_rows(x_train_full, y_train_full, args.max_train_rows)
        x_fit, y_fit, x_validation, y_validation = chronological_train_validation_split(
            x_train,
            y_train,
            args.validation_fraction,
        )
        d = x_train.shape[1]
        dataset_variant = f"{spec.name}_W{window}"
        input_kind = "snapshot" if window == 1 else "flattened_lag_window"

        for target in args.horizons:
            base = {
                "cf": args.cf,
                "feature_set": spec.name,
                "dataset_variant": dataset_variant,
                "representation_kind": "raw_lob",
                "input_kind": input_kind,
                "input_window": window,
                "feature_selection": spec.selection_mode,
                "source_feature_count": spec.source_feature_count,
                "n_feature_rows_available": FULL_FEATURE_COUNT,
                "d": d,
                "base_d": len(spec.columns),
                "train_rows": len(x_train),
                "fit_rows": len(x_fit),
                "validation_rows": len(x_validation),
                "validation_fraction": args.validation_fraction,
                "test_rows": len(x_test),
                "score_target": target,
                "timing_mode": args.timing_mode,
                "single_timing_samples": args.single_timing_samples,
                "torch_timing_device": args.torch_timing_device,
            }
            y_target = y_fit[target]
            y_validation_target = y_validation[target] if len(y_validation) else pd.Series(dtype=y_target.dtype)
            base.update(class_balance_metrics(y_target, "fit"))
            base.update(class_balance_metrics(y_validation_target, "validation"))
            base.update(class_balance_metrics(y_test_windowed[target], "test"))
            base["naive_fit_prior_validation_log_loss"] = prior_log_loss_from_train(y_target, y_validation_target)
            base["naive_fit_prior_test_log_loss"] = prior_log_loss_from_train(y_target, y_test_windowed[target])

            if args.include_trees:
                for tree_spec in build_tree_specs(args, d, x_test):
                    print(f"{dataset_variant} {target}: fitting {tree_spec.name}", flush=True)
                    model, fitted_spec = fit_tree_spec(tree_spec, x_fit, y_target, x_test)
                    add_result_row(
                        rows,
                        base=base,
                        spec=fitted_spec,
                        model=model,
                        x_validation=x_validation,
                        y_validation_target=y_validation_target,
                        x_test=x_test,
                        y_test_target=y_test_windowed[target],
                        timing_repeats=args.timing_repeats,
                        checkpoint_path=args.checkpoint_path,
                    )

            if args.include_forests:
                for n_estimators in args.forest_estimators:
                    for leaves in args.forest_leaf_values:
                        name = f"rf_T{n_estimators}_L{leaves}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_leaf_nodes=leaves,
                            min_samples_leaf=args.min_samples_leaf,
                            n_jobs=args.n_jobs,
                            random_state=RANDOM_SEED,
                        )
                        base_spec = ModelSpec(
                            family="tree_ensemble",
                            architecture="random_forest",
                            name=name,
                            compute_proxy_ops=float(n_estimators * np.log2(leaves)),
                            compute_proxy_kind="sum_empirical_tree_path_depths",
                            n_parameters=float(n_estimators * leaves),
                            fit_fn=lambda: model,
                        )
                        model, fitted_spec = fit_forest_model(model, x_fit, y_target, x_test, base_spec)
                        add_result_row(
                            rows,
                            base=base,
                            spec=fitted_spec,
                            model=model,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                for n_estimators in args.forest_estimators:
                    for leaves in args.forest_leaf_values:
                        name = f"extra_T{n_estimators}_L{leaves}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = ExtraTreesClassifier(
                            n_estimators=n_estimators,
                            max_leaf_nodes=leaves,
                            min_samples_leaf=args.min_samples_leaf,
                            n_jobs=args.n_jobs,
                            random_state=RANDOM_SEED,
                        )
                        base_spec = ModelSpec(
                            family="tree_ensemble",
                            architecture="extra_trees",
                            name=name,
                            compute_proxy_ops=float(n_estimators * np.log2(leaves)),
                            compute_proxy_kind="sum_empirical_tree_path_depths",
                            n_parameters=float(n_estimators * leaves),
                            fit_fn=lambda: model,
                        )
                        model, fitted_spec = fit_forest_model(model, x_fit, y_target, x_test, base_spec)
                        add_result_row(
                            rows,
                            base=base,
                            spec=fitted_spec,
                            model=model,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

            if args.include_boosted_trees:
                for max_iter in args.hgb_max_iter_values:
                    for leaves in args.hgb_leaf_values:
                        name = f"hgb_I{max_iter}_L{leaves}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = HistGradientBoostingClassifier(
                            loss="log_loss",
                            learning_rate=args.hgb_learning_rate,
                            max_iter=max_iter,
                            max_leaf_nodes=leaves,
                            min_samples_leaf=args.min_samples_leaf,
                            random_state=RANDOM_SEED,
                        )
                        model.fit(x_fit, y_target)
                        depth_proxy = np.log2(leaves)
                        model_spec = ModelSpec(
                            family="boosted_trees",
                            architecture="hist_gradient_boosting",
                            name=name,
                            compute_proxy_ops=float(max_iter * len(CLASS_ORDER) * depth_proxy),
                            compute_proxy_kind="iterations_times_classes_times_log2_leaves",
                            n_parameters=float(max_iter * len(CLASS_ORDER) * leaves),
                            fit_fn=lambda: model,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=model,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

            if args.include_mlp:
                for hidden in args.mlp_hidden_layers:
                    name = "mlp_" + "x".join(str(width) for width in hidden)
                    print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden,
                        activation="relu",
                        solver="adam",
                        alpha=args.mlp_alpha,
                        batch_size=args.mlp_batch_size,
                        learning_rate_init=args.mlp_learning_rate,
                        max_iter=args.mlp_max_iter,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=5,
                        random_state=RANDOM_SEED,
                    )
                    model.fit(x_fit, y_target)
                    model_spec = ModelSpec(
                        family="mlp",
                        architecture="mlp_classifier",
                        name=name,
                        compute_proxy_ops=float(mlp_compute_ops(d, hidden, len(CLASS_ORDER))),
                        compute_proxy_kind="dense_forward_multiply_adds",
                        n_parameters=float(mlp_parameter_count(d, hidden, len(CLASS_ORDER))),
                        fit_fn=lambda: model,
                    )
                    add_result_row(
                        rows,
                        base=base,
                        spec=model_spec,
                        model=model,
                        x_validation=x_validation,
                        y_validation_target=y_validation_target,
                        x_test=x_test,
                        y_test_target=y_test_windowed[target],
                        timing_repeats=args.timing_repeats,
                        checkpoint_path=args.checkpoint_path,
                        )

            if args.include_random_conv:
                if window < 2:
                    print(f"{dataset_variant} {target}: skipping random_conv features for window={window}", flush=True)
                else:
                    for n_kernels in args.random_conv_kernels:
                        for head in args.random_conv_heads:
                            head = head.strip()
                            if not head:
                                continue
                            name = f"randconv_{head}_K{n_kernels}"
                            print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                            transformer = RandomConvFeatureTransformer(
                                window=window,
                                base_d=len(spec.columns),
                                n_kernels=n_kernels,
                                kernel_sizes=tuple(args.random_conv_kernel_sizes),
                                random_state=RANDOM_SEED,
                            )
                            if head == "sgd":
                                classifier = SGDClassifier(
                                    loss="log_loss",
                                    alpha=args.random_conv_sgd_alpha,
                                    max_iter=args.random_conv_sgd_max_iter,
                                    random_state=RANDOM_SEED,
                                )
                                architecture = "random_conv_sgd"
                            elif head == "logistic":
                                classifier = LogisticRegression(
                                    C=args.random_conv_logistic_c,
                                    max_iter=args.random_conv_logistic_max_iter,
                                    n_jobs=args.n_jobs,
                                    random_state=RANDOM_SEED,
                                )
                                architecture = "random_conv_logistic"
                            else:
                                print(f"skipping random_conv head={head}: expected sgd or logistic", flush=True)
                                continue
                            model = make_pipeline(transformer, StandardScaler(), classifier)
                            model.fit(x_fit, y_target)
                            model.metadata = {
                                "representation_kind": "raw_lob_random_temporal_basis",
                                "random_conv_kernels": n_kernels,
                                "random_conv_kernel_sizes": "x".join(str(size) for size in args.random_conv_kernel_sizes),
                                "random_conv_head": head,
                            }
                            transform_ops = getattr(transformer, "compute_proxy_ops_", float(n_kernels * window))
                            feature_count = getattr(transformer, "n_features_out_", n_kernels * 3)
                            model_spec = ModelSpec(
                                family="random_conv_features",
                                architecture=architecture,
                                name=name,
                                compute_proxy_ops=float(transform_ops + feature_count * len(CLASS_ORDER)),
                                compute_proxy_kind="random_conv_features_plus_linear_logits",
                                n_parameters=float(feature_count * len(CLASS_ORDER)),
                                fit_fn=lambda: model,
                            )
                            add_result_row(
                                rows,
                                base=base,
                                spec=model_spec,
                                model=model,
                                x_validation=x_validation,
                                y_validation_target=y_validation_target,
                                x_test=x_test,
                                y_test_target=y_test_windowed[target],
                                timing_repeats=args.timing_repeats,
                                checkpoint_path=args.checkpoint_path,
                            )

            if args.include_temporal_summary_boosted:
                if window < 2:
                    print(f"{dataset_variant} {target}: skipping temporal summaries for window={window}", flush=True)
                else:
                    summary_transformer = TemporalSummaryTransformer(
                        window=window,
                        base_d=len(spec.columns),
                        summary_windows=tuple(args.temporal_summary_windows),
                    )
                    x_fit_summary = summary_transformer.fit_transform(x_fit)
                    x_validation_summary = summary_transformer.transform(x_validation) if len(x_validation) else np.empty(
                        (0, summary_transformer.n_features_out_),
                        dtype=np.float32,
                    )
                    summary_ops = summary_transformer.compute_proxy_ops_
                    for max_iter in args.temporal_summary_hgb_max_iter_values:
                        for leaves in args.temporal_summary_hgb_leaf_values:
                            name = f"summary_hgb_I{max_iter}_L{leaves}"
                            print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                            classifier = HistGradientBoostingClassifier(
                                loss="log_loss",
                                learning_rate=args.temporal_summary_hgb_learning_rate,
                                max_iter=max_iter,
                                max_leaf_nodes=leaves,
                                min_samples_leaf=args.min_samples_leaf,
                                random_state=RANDOM_SEED,
                            )
                            classifier.fit(x_fit_summary, y_target)
                            model = TransformedClassifier(summary_transformer, classifier)
                            model.metadata = {
                                "representation_kind": "raw_lob_plus_causal_temporal_summary",
                                "temporal_summary_windows": "x".join(str(size) for size in summary_transformer.active_windows_),
                                "temporal_summary_feature_count": summary_transformer.n_features_out_,
                            }
                            tree_ops = max_iter * len(CLASS_ORDER) * np.log2(leaves)
                            model_spec = ModelSpec(
                                family="temporal_summary_boosted",
                                architecture="temporal_summary_hgb",
                                name=name,
                                compute_proxy_ops=float(summary_ops + tree_ops),
                                compute_proxy_kind="temporal_summary_ops_plus_iterations_times_classes_times_log2_leaves",
                                n_parameters=float(summary_transformer.n_features_out_ + max_iter * len(CLASS_ORDER) * leaves),
                                fit_fn=lambda: model,
                            )
                            add_result_row(
                                rows,
                                base=base,
                                spec=model_spec,
                                model=model,
                                x_validation=x_validation,
                                y_validation_target=y_validation_target,
                                x_test=x_test,
                                y_test_target=y_test_windowed[target],
                                timing_repeats=args.timing_repeats,
                                checkpoint_path=args.checkpoint_path,
                            )

                    if args.temporal_summary_include_xgboost:
                        try:
                            from xgboost import XGBClassifier
                        except Exception as exc:
                            print(f"xgboost could not be imported; skipping temporal summary xgboost: {exc}", flush=True)
                            XGBClassifier = None
                        if XGBClassifier is not None:
                            label_map = {-1: 0, 0: 1, 1: 2}
                            inverse_columns = {0: -1, 1: 0, 2: 1}
                            y_xgb = y_target.map(label_map)
                            y_validation_xgb = y_validation_target.map(label_map) if len(y_validation_target) else None
                            for n_estimators in args.temporal_summary_xgb_estimators:
                                for max_depth in args.temporal_summary_xgb_depths:
                                    name = f"summary_xgb_T{n_estimators}_D{max_depth}"
                                    print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                                    xgb_kwargs = {
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "learning_rate": args.temporal_summary_xgb_learning_rate,
                                        "objective": "multi:softprob",
                                        "num_class": len(CLASS_ORDER),
                                        "eval_metric": "mlogloss",
                                        "random_state": RANDOM_SEED,
                                        "n_jobs": args.n_jobs,
                                        "tree_method": args.xgb_tree_method,
                                        "subsample": args.xgb_subsample,
                                        "colsample_bytree": args.xgb_colsample_bytree,
                                        "min_child_weight": args.xgb_min_child_weight,
                                        "reg_lambda": args.xgb_reg_lambda,
                                    }
                                    fit_kwargs = {}
                                    if (
                                        args.temporal_summary_xgb_early_stopping_rounds > 0
                                        and len(x_validation_summary)
                                        and y_validation_xgb is not None
                                    ):
                                        xgb_kwargs["early_stopping_rounds"] = args.temporal_summary_xgb_early_stopping_rounds
                                        fit_kwargs["eval_set"] = [(x_validation_summary, y_validation_xgb)]
                                        fit_kwargs["verbose"] = args.xgb_verbose_eval if args.xgb_verbose_eval > 0 else False
                                    classifier = XGBClassifier(**xgb_kwargs)
                                    classifier.fit(x_fit_summary, y_xgb, **fit_kwargs)
                                    model = TransformedClassifier(summary_transformer, classifier, class_map=inverse_columns)
                                    model.metadata = {
                                        "representation_kind": "raw_lob_plus_causal_temporal_summary",
                                        "temporal_summary_windows": "x".join(str(size) for size in summary_transformer.active_windows_),
                                        "temporal_summary_feature_count": summary_transformer.n_features_out_,
                                    }
                                    best_iteration = getattr(classifier, "best_iteration", None)
                                    effective_estimators = int(best_iteration + 1) if best_iteration is not None else n_estimators
                                    model_spec = ModelSpec(
                                        family="temporal_summary_boosted",
                                        architecture="temporal_summary_xgboost",
                                        name=name,
                                        compute_proxy_ops=float(summary_ops + effective_estimators * max_depth * len(CLASS_ORDER)),
                                        compute_proxy_kind="temporal_summary_ops_plus_effective_trees_times_depth_times_classes",
                                        n_parameters=float(
                                            summary_transformer.n_features_out_ + effective_estimators * (2 ** max_depth)
                                        ),
                                        fit_fn=lambda: model,
                                    )
                                    add_result_row(
                                        rows,
                                        base=base,
                                        spec=model_spec,
                                        model=model,
                                        x_validation=x_validation,
                                        y_validation_target=y_validation_target,
                                        x_test=x_test,
                                        y_test_target=y_test_windowed[target],
                                        timing_repeats=args.timing_repeats,
                                        checkpoint_path=args.checkpoint_path,
                                    )

            if args.include_torch_nn:
                if not torch_available():
                    print("torch is not installed; skipping torch neural families", flush=True)
                else:
                    torch_device = resolve_torch_device(args.torch_device)
                    print(f"{dataset_variant} {target}: torch device {torch_device}", flush=True)
                    x_fit_arr, y_fit_arr = make_torch_arrays(x_fit, y_target)
                    x_validation_arr, y_validation_arr = make_torch_arrays(x_validation, y_validation_target) if len(x_validation) else (
                        np.empty((0, d), dtype=np.float32),
                        np.empty((0,), dtype=np.int64),
                    )
                    x_test_arr, _ = make_torch_arrays(x_test)
                    x_fit_std, x_validation_std, _, mean, std = standardize_arrays(
                        x_fit_arr,
                        x_validation_arr,
                        x_test_arr,
                    )

                    for hidden in args.torch_mlp_hidden_layers:
                        name = "torch_mlp_" + "x".join(str(width) for width in hidden)
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = build_torch_mlp(d, hidden, residual=False)
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="flat",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        wrapper.metadata.update(
                            {
                                "torch_mlp_hidden_layers": "x".join(str(width) for width in hidden),
                            }
                        )
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_mlp",
                            name=name,
                            compute_proxy_ops=float(mlp_compute_ops(d, hidden, len(CLASS_ORDER))),
                            compute_proxy_kind="dense_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )
                        gc.collect()
                        torch_clear_device_cache(torch_device)

                    for hidden in args.torch_residual_hidden_layers:
                        name = "torch_resmlp_" + "x".join(str(width) for width in hidden)
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_mlp(d, hidden, residual=True)
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="flat",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        # Residual blocks have more operations than a plain MLP;
                        # parameter count is the more reliable size proxy here.
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_residual_mlp",
                            name=name,
                            compute_proxy_ops=float(torch_parameter_count(model)),
                            compute_proxy_kind="parameter_count_proxy",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for rank in args.torch_fm_ranks:
                        name = f"torch_fm_R{rank}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = build_torch_factorization_machine(d, rank)
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="flat",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        wrapper.metadata.update({"torch_fm_rank": rank})
                        model_spec = ModelSpec(
                            family="torch_interaction",
                            architecture="torch_factorization_machine",
                            name=name,
                            compute_proxy_ops=float(torch_factorization_machine_compute_ops(d, rank, len(CLASS_ORDER))),
                            compute_proxy_kind="multiclass_factorization_machine_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for transformer_spec in args.torch_tab_transformer_specs:
                        if len(transformer_spec) != 3:
                            print(
                                f"skipping torch_tab_transformer_{'x'.join(str(width) for width in transformer_spec)}: "
                                "specs must be token_dim x layers x heads",
                                flush=True,
                            )
                            continue
                        token_dim, layers, heads = transformer_spec
                        name = f"torch_tabtransformer_D{token_dim}_L{layers}_H{heads}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_tabular_transformer(
                                d,
                                token_dim,
                                layers,
                                heads,
                                args.torch_tab_transformer_ff_multiplier,
                            )
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="flat",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        wrapper.metadata.update(
                            {
                                "torch_tab_transformer_token_dim": token_dim,
                                "torch_tab_transformer_layers": layers,
                                "torch_tab_transformer_heads": heads,
                                "torch_tab_transformer_ff_multiplier": args.torch_tab_transformer_ff_multiplier,
                            }
                        )
                        model_spec = ModelSpec(
                            family="torch_tabular_transformer",
                            architecture="torch_tabular_transformer",
                            name=name,
                            compute_proxy_ops=float(
                                torch_tabular_transformer_compute_ops(
                                    d,
                                    token_dim,
                                    layers,
                                    heads,
                                    args.torch_tab_transformer_ff_multiplier,
                                )
                            ),
                            compute_proxy_kind="feature_token_transformer_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for mlplob_spec in args.torch_mlplob_specs:
                        if len(mlplob_spec) not in {2, 3}:
                            print(
                                f"skipping torch_mlplob_{'x'.join(str(width) for width in mlplob_spec)}: "
                                "specs must be hidden_dim x layers or hidden_dim x layers x expansion",
                                flush=True,
                            )
                            continue
                        hidden_dim = mlplob_spec[0]
                        layers = mlplob_spec[1]
                        expansion = mlplob_spec[2] if len(mlplob_spec) == 3 else args.torch_mlplob_expansion
                        name = f"torch_mlplob_H{hidden_dim}_L{layers}_E{expansion}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_mlplob(
                                len(spec.columns),
                                window,
                                hidden_dim,
                                layers,
                                expansion,
                                args.torch_mlplob_dropout,
                                args.torch_lob_use_bin,
                                args.torch_lob_bin_eps,
                            )
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="mlplob",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        wrapper.metadata.update(
                            {
                                "torch_mlplob_hidden_dim": hidden_dim,
                                "torch_mlplob_layers": layers,
                                "torch_mlplob_expansion": expansion,
                                "torch_mlplob_dropout": args.torch_mlplob_dropout,
                                "torch_lob_use_bin": int(args.torch_lob_use_bin),
                                "torch_lob_bin_eps": args.torch_lob_bin_eps,
                            }
                        )
                        model_spec = ModelSpec(
                            family="torch_lob_mixer",
                            architecture="torch_mlplob",
                            name=name,
                            compute_proxy_ops=float(
                                torch_mlplob_compute_ops(
                                    window,
                                    len(spec.columns),
                                    hidden_dim,
                                    layers,
                                    expansion,
                                )
                            ),
                            compute_proxy_kind="mlplob_mixer_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )
                        gc.collect()
                        torch_clear_device_cache(torch_device)

                    for tlob_spec in args.torch_tlob_specs:
                        if len(tlob_spec) not in {3, 4}:
                            print(
                                f"skipping torch_tlob_{'x'.join(str(width) for width in tlob_spec)}: "
                                "specs must be hidden_dim x layers x heads or hidden_dim x layers x heads x ff_multiplier",
                                flush=True,
                            )
                            continue
                        hidden_dim = tlob_spec[0]
                        layers = tlob_spec[1]
                        heads = tlob_spec[2]
                        ff_multiplier = tlob_spec[3] if len(tlob_spec) == 4 else args.torch_tlob_ff_multiplier
                        name = f"torch_tlob_H{hidden_dim}_L{layers}_A{heads}_F{ff_multiplier}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_tlob(
                                len(spec.columns),
                                window,
                                hidden_dim,
                                layers,
                                heads,
                                ff_multiplier,
                                args.torch_tlob_dropout,
                                args.torch_lob_use_bin,
                                args.torch_lob_bin_eps,
                            )
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="tlob",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        wrapper.metadata.update(
                            {
                                "torch_tlob_hidden_dim": hidden_dim,
                                "torch_tlob_layers": layers,
                                "torch_tlob_heads": heads,
                                "torch_tlob_ff_multiplier": ff_multiplier,
                                "torch_tlob_dropout": args.torch_tlob_dropout,
                                "torch_lob_use_bin": int(args.torch_lob_use_bin),
                                "torch_lob_bin_eps": args.torch_lob_bin_eps,
                            }
                        )
                        model_spec = ModelSpec(
                            family="torch_lob_attention",
                            architecture="torch_tlob",
                            name=name,
                            compute_proxy_ops=float(
                                torch_tlob_compute_ops(
                                    window,
                                    len(spec.columns),
                                    hidden_dim,
                                    layers,
                                    heads,
                                    ff_multiplier,
                                )
                            ),
                            compute_proxy_kind="tlob_dual_axis_attention_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )
                        gc.collect()
                        torch_clear_device_cache(torch_device)

                    for channels in args.torch_cnn_channels:
                        name = "torch_cnn_" + "x".join(str(width) for width in channels)
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = build_torch_temporal_cnn(len(spec.columns), channels, args.torch_cnn_kernel_size)
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="temporal_cnn",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_temporal_cnn",
                            name=name,
                            compute_proxy_ops=float(torch_temporal_cnn_compute_ops(window, len(spec.columns), channels, args.torch_cnn_kernel_size)),
                            compute_proxy_kind="conv1d_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for channels in args.torch_inception_channels:
                        name = "torch_inception_" + "x".join(str(width) for width in channels)
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_inception_time(
                                len(spec.columns),
                                channels,
                                tuple(args.torch_inception_kernel_sizes),
                                args.torch_inception_bottleneck_channels,
                            )
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="temporal_inception",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_inception_time",
                            name=name,
                            compute_proxy_ops=float(
                                torch_inception_time_compute_ops(
                                    window,
                                    len(spec.columns),
                                    channels,
                                    tuple(args.torch_inception_kernel_sizes),
                                    args.torch_inception_bottleneck_channels,
                                )
                            ),
                            compute_proxy_kind="inception_time_conv1d_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for channels in args.torch_tcn_channels:
                        name = "torch_tcn_" + "x".join(str(width) for width in channels)
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_tcn(len(spec.columns), channels, args.torch_tcn_kernel_size)
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="temporal_tcn",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_temporal_tcn",
                            name=name,
                            compute_proxy_ops=float(torch_tcn_compute_ops(window, len(spec.columns), channels, args.torch_tcn_kernel_size)),
                            compute_proxy_kind="dilated_residual_conv1d_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for channels in args.torch_cnn_attention_channels:
                        name = "torch_cnnattn_" + "x".join(str(width) for width in channels)
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_cnn_attention(len(spec.columns), channels, args.torch_cnn_attention_kernel_size)
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="temporal_cnn_attention",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_cnn_attention",
                            name=name,
                            compute_proxy_ops=float(
                                torch_cnn_attention_compute_ops(
                                    window,
                                    len(spec.columns),
                                    channels,
                                    args.torch_cnn_attention_kernel_size,
                                )
                            ),
                            compute_proxy_kind="conv1d_attention_pool_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for recurrent_spec in args.torch_recurrent_specs:
                        if len(recurrent_spec) not in {1, 2}:
                            print(
                                f"skipping recurrent_{'x'.join(str(width) for width in recurrent_spec)}: "
                                "specs must be hidden or hidden x layers",
                                flush=True,
                            )
                            continue
                        hidden = recurrent_spec[0]
                        layers = recurrent_spec[1] if len(recurrent_spec) == 2 else 1
                        for rnn_type in args.torch_recurrent_types:
                            name = f"torch_{rnn_type}_H{hidden}_L{layers}"
                            print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                            try:
                                model = build_torch_recurrent(len(spec.columns), hidden, layers, rnn_type)
                            except ValueError as exc:
                                print(f"skipping {name}: {exc}", flush=True)
                                continue
                            wrapper = fit_torch_classifier(
                                model=model,
                                x_fit=x_fit_std.astype(np.float32),
                                y_fit=y_fit_arr,
                                x_validation=x_validation_std.astype(np.float32),
                                y_validation=y_validation_arr,
                                args=args,
                                input_kind="temporal_recurrent",
                                window=window,
                                base_d=len(spec.columns),
                                mean=mean.astype(np.float32),
                                std=std.astype(np.float32),
                                run_label=f"{dataset_variant}_{target}_{name}",
                            )
                            model_spec = ModelSpec(
                                family="torch_nn",
                                architecture=f"torch_{rnn_type}",
                                name=name,
                                compute_proxy_ops=float(
                                    torch_recurrent_compute_ops(
                                        window,
                                        len(spec.columns),
                                        hidden,
                                        layers,
                                        rnn_type,
                                    )
                                ),
                                compute_proxy_kind=f"{rnn_type}_forward_multiply_adds",
                                n_parameters=float(torch_parameter_count(model)),
                                fit_fn=lambda: wrapper,
                            )
                            add_result_row(
                                rows,
                                base=base,
                                spec=model_spec,
                                model=wrapper,
                                x_validation=x_validation,
                                y_validation_target=y_validation_target,
                                x_test=x_test,
                                y_test_target=y_test_windowed[target],
                                timing_repeats=args.timing_repeats,
                                checkpoint_path=args.checkpoint_path,
                            )

                    for deeplob_spec in args.torch_deeplob_specs:
                        if len(deeplob_spec) != 3:
                            print(
                                f"skipping deeplob_lite_{'x'.join(str(width) for width in deeplob_spec)}: "
                                "specs must be stem_channels x inception_channels x recurrent_hidden",
                                flush=True,
                            )
                            continue
                        stem_channels, inception_channels, recurrent_hidden = deeplob_spec
                        name = f"deeplob_lite_S{stem_channels}_I{inception_channels}_R{recurrent_hidden}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        kernel_sizes = tuple(args.torch_deeplob_kernel_sizes)
                        model = build_torch_deeplob_lite(
                            len(spec.columns),
                            stem_channels,
                            inception_channels,
                            recurrent_hidden,
                            kernel_sizes,
                        )
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="deeplob_lite",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        wrapper.metadata.update(
                            {
                                "torch_deeplob_lite_stem_channels": stem_channels,
                                "torch_deeplob_lite_inception_channels": inception_channels,
                                "torch_deeplob_lite_recurrent_hidden": recurrent_hidden,
                                "torch_deeplob_kernel_sizes": "x".join(str(kernel) for kernel in kernel_sizes),
                            }
                        )
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_deeplob_lite",
                            name=name,
                            compute_proxy_ops=float(
                                torch_deeplob_lite_compute_ops(
                                    window,
                                    len(spec.columns),
                                    stem_channels,
                                    inception_channels,
                                    recurrent_hidden,
                                    kernel_sizes,
                                )
                            ),
                            compute_proxy_kind="cnn_inception_gru_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                    for deeplob_spec in args.torch_deeplob_full_specs:
                        if len(deeplob_spec) != 3:
                            print(
                                f"skipping deeplob_full_{'x'.join(str(width) for width in deeplob_spec)}: "
                                "specs must be conv_filters x inception_filters x lstm_hidden",
                                flush=True,
                            )
                            continue
                        conv_filters, inception_filters, lstm_hidden = deeplob_spec
                        name = f"deeplob_full_C{conv_filters}_I{inception_filters}_L{lstm_hidden}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        try:
                            model = build_torch_deeplob_full(
                                len(spec.columns),
                                conv_filters,
                                inception_filters,
                                lstm_hidden,
                                args.torch_deeplob_full_dropout,
                                args.torch_deeplob_full_norm,
                                args.torch_deeplob_full_conv2_activation,
                            )
                        except ValueError as exc:
                            print(f"skipping {name}: {exc}", flush=True)
                            continue
                        wrapper = fit_torch_classifier(
                            model=model,
                            x_fit=x_fit_std.astype(np.float32),
                            y_fit=y_fit_arr,
                            x_validation=x_validation_std.astype(np.float32),
                            y_validation=y_validation_arr,
                            args=args,
                            input_kind="deeplob_full",
                            window=window,
                            base_d=len(spec.columns),
                            mean=mean.astype(np.float32),
                            std=std.astype(np.float32),
                            run_label=f"{dataset_variant}_{target}_{name}",
                        )
                        wrapper.metadata.update(
                            {
                                "torch_deeplob_full_conv_filters": conv_filters,
                                "torch_deeplob_full_inception_filters": inception_filters,
                                "torch_deeplob_full_lstm_hidden": lstm_hidden,
                                "torch_deeplob_full_dropout": args.torch_deeplob_full_dropout,
                                "torch_deeplob_full_norm": args.torch_deeplob_full_norm,
                                "torch_deeplob_full_conv2_activation": args.torch_deeplob_full_conv2_activation,
                            }
                        )
                        model_spec = ModelSpec(
                            family="torch_nn",
                            architecture="torch_deeplob_full",
                            name=name,
                            compute_proxy_ops=float(
                                torch_deeplob_full_compute_ops(
                                    window,
                                    len(spec.columns),
                                    conv_filters,
                                    inception_filters,
                                    lstm_hidden,
                                )
                            ),
                            compute_proxy_kind="deeplob_conv2d_inception_lstm_forward_multiply_adds",
                            n_parameters=float(torch_parameter_count(model)),
                            fit_fn=lambda: wrapper,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=wrapper,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )
                        gc.collect()
                        torch_clear_device_cache(torch_device)

            if args.include_kernel:
                for n_components in args.kernel_components:
                    for gamma in args.kernel_gammas:
                        name = f"nystroem_rbf_C{n_components}_g{gamma:g}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = make_pipeline(
                            Nystroem(
                                kernel="rbf",
                                gamma=gamma,
                                n_components=n_components,
                                random_state=RANDOM_SEED,
                            ),
                            LogisticRegression(max_iter=args.kernel_logistic_max_iter, random_state=RANDOM_SEED),
                        )
                        model.fit(x_fit, y_target)
                        model_spec = ModelSpec(
                            family="kernel_approx",
                            architecture="nystroem_rbf_logistic",
                            name=name,
                            compute_proxy_ops=float(d * n_components + n_components * len(CLASS_ORDER)),
                            compute_proxy_kind="rbf_features_plus_linear_logits",
                            n_parameters=float(n_components * (d + len(CLASS_ORDER))),
                            fit_fn=lambda: model,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=model,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

                for n_components in args.kernel_components:
                    for gamma in args.kernel_gammas:
                        name = f"rbfsampler_C{n_components}_g{gamma:g}"
                        print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                        model = make_pipeline(
                            RBFSampler(gamma=gamma, n_components=n_components, random_state=RANDOM_SEED),
                            SGDClassifier(
                                loss="log_loss",
                                alpha=args.kernel_sgd_alpha,
                                max_iter=args.kernel_sgd_max_iter,
                                random_state=RANDOM_SEED,
                            ),
                        )
                        model.fit(x_fit, y_target)
                        model_spec = ModelSpec(
                            family="kernel_approx",
                            architecture="rbf_random_features_sgd",
                            name=name,
                            compute_proxy_ops=float(d * n_components + n_components * len(CLASS_ORDER)),
                            compute_proxy_kind="random_rbf_features_plus_linear_logits",
                            n_parameters=float(n_components * (d + len(CLASS_ORDER))),
                            fit_fn=lambda: model,
                        )
                        add_result_row(
                            rows,
                            base=base,
                            spec=model_spec,
                            model=model,
                            x_validation=x_validation,
                            y_validation_target=y_validation_target,
                            x_test=x_test,
                            y_test_target=y_test_windowed[target],
                            timing_repeats=args.timing_repeats,
                            checkpoint_path=args.checkpoint_path,
                        )

            if args.include_xgboost:
                try:
                    from xgboost import XGBClassifier
                except Exception as exc:
                    print(f"xgboost could not be imported; skipping xgboost family: {exc}", flush=True)
                    XGBClassifier = None

                if XGBClassifier is not None:

                    label_map = {-1: 0, 0: 1, 1: 2}
                    inverse_columns = {0: -1, 1: 0, 2: 1}
                    y_xgb = y_target.map(label_map)
                    y_validation_xgb = y_validation_target.map(label_map) if len(y_validation_target) else None
                    for n_estimators in args.xgb_estimators:
                        for max_depth in args.xgb_depths:
                            name = f"xgb_T{n_estimators}_D{max_depth}"
                            print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                            xgb_kwargs = {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "learning_rate": args.xgb_learning_rate,
                                "objective": "multi:softprob",
                                "num_class": len(CLASS_ORDER),
                                "eval_metric": "mlogloss",
                                "random_state": RANDOM_SEED,
                                "n_jobs": args.n_jobs,
                                "tree_method": args.xgb_tree_method,
                                "subsample": args.xgb_subsample,
                                "colsample_bytree": args.xgb_colsample_bytree,
                                "min_child_weight": args.xgb_min_child_weight,
                                "reg_lambda": args.xgb_reg_lambda,
                            }
                            fit_kwargs = {}
                            if args.xgb_early_stopping_rounds > 0 and len(x_validation) and y_validation_xgb is not None:
                                xgb_kwargs["early_stopping_rounds"] = args.xgb_early_stopping_rounds
                                fit_kwargs["eval_set"] = [(x_validation, y_validation_xgb)]
                                fit_kwargs["verbose"] = args.xgb_verbose_eval if args.xgb_verbose_eval > 0 else False
                            model = XGBClassifier(
                                **xgb_kwargs,
                            )
                            model.fit(x_fit, y_xgb, **fit_kwargs)
                            raw_proba = model.predict_proba(x_test)
                            proba = pd.DataFrame(raw_proba, columns=[inverse_columns[i] for i in range(raw_proba.shape[1])])
                            timing_s, latency_us = time_predict_proba(
                                model,
                                x_test,
                                args.timing_repeats,
                                args.timing_mode,
                                args.single_timing_samples,
                            )
                            best_iteration = getattr(model, "best_iteration", None)
                            best_score = getattr(model, "best_score", None)
                            effective_estimators = int(best_iteration + 1) if best_iteration is not None else n_estimators
                            model_spec = ModelSpec(
                                family="boosted_trees",
                                architecture="xgboost",
                                name=name,
                                compute_proxy_ops=float(effective_estimators * max_depth * len(CLASS_ORDER)),
                                compute_proxy_kind="effective_trees_times_depth_times_classes",
                                n_parameters=float(effective_estimators * (2 ** max_depth)),
                                fit_fn=lambda: model,
                            )
                            row = dict(base)
                            row.update(
                                {
                                    "model_family": model_spec.family,
                                    "architecture": model_spec.architecture,
                                    "model_name": model_spec.name,
                                    "compute_proxy_ops": model_spec.compute_proxy_ops,
                                    "compute_proxy_kind": model_spec.compute_proxy_kind,
                                    "n_parameters": model_spec.n_parameters,
                                    "prediction_time_s_median": timing_s,
                                    "latency_us_per_obs_median": latency_us,
                                    "latency_timing_mode": args.timing_mode,
                                    "latency_single_observation_samples": args.single_timing_samples
                                    if args.timing_mode == "single"
                                    else 0,
                                    "latency_timing_device": "default",
                                    "critical_path_latency_us_per_obs": latency_us,
                                    "serial_latency_us_per_obs": latency_us,
                                    "thread_parallel_latency_us_per_obs": latency_us,
                                    "ideal_parallel_latency_us_per_obs": latency_us,
                                    "total_compute_proxy_ops": model_spec.compute_proxy_ops,
                                    "critical_path_compute_proxy_ops": model_spec.compute_proxy_ops,
                                    "parallel_width": 1,
                                    "xgb_configured_estimators": n_estimators,
                                    "xgb_effective_estimators": effective_estimators,
                                    "xgb_max_depth": max_depth,
                                    "xgb_learning_rate": args.xgb_learning_rate,
                                    "xgb_early_stopping_rounds": args.xgb_early_stopping_rounds,
                                    "xgb_best_iteration": best_iteration if best_iteration is not None else np.nan,
                                    "xgb_best_score": float(best_score) if best_score is not None else np.nan,
                                    "xgb_tree_method": args.xgb_tree_method,
                                    "xgb_subsample": args.xgb_subsample,
                                    "xgb_colsample_bytree": args.xgb_colsample_bytree,
                                    "xgb_min_child_weight": args.xgb_min_child_weight,
                                    "xgb_reg_lambda": args.xgb_reg_lambda,
                                }
                            )
                            row.update(audit_forward_compute(model, model_spec, x_test))
                            if len(x_validation):
                                validation_raw_proba = model.predict_proba(x_validation)
                                validation_proba = pd.DataFrame(
                                    validation_raw_proba,
                                    columns=[inverse_columns[i] for i in range(validation_raw_proba.shape[1])],
                                ).reindex(columns=CLASS_ORDER, fill_value=0.0)
                                row.update(prefix_metrics(score_probabilities(validation_proba, y_validation_target), "validation"))
                            row.update(prefix_metrics(score_probabilities(proba.reindex(columns=CLASS_ORDER, fill_value=0.0), y_test_windowed[target]), "test"))
                            row["accuracy"] = row["test_accuracy"]
                            row["balanced_accuracy"] = row["test_balanced_accuracy"]
                            row["log_loss"] = row["test_log_loss"]
                            rows.append(row)
                            if args.checkpoint_path is not None:
                                pd.DataFrame(rows).to_csv(args.checkpoint_path, index=False)

            if args.include_lightgbm:
                try:
                    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
                except Exception as exc:
                    print(f"lightgbm could not be imported; skipping lightgbm family: {exc}", flush=True)
                    LGBMClassifier = None
                if LGBMClassifier is not None:
                    label_map = {-1: 0, 0: 1, 1: 2}
                    inverse_columns = {0: -1, 1: 0, 2: 1}
                    y_lgbm = y_target.map(label_map)
                    y_validation_lgbm = y_validation_target.map(label_map) if len(y_validation_target) else None
                    for n_estimators in args.lgbm_estimators:
                        for leaves in args.lgbm_num_leaves:
                            name = f"lgbm_T{n_estimators}_L{leaves}"
                            print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                            model = LGBMClassifier(
                                n_estimators=n_estimators,
                                num_leaves=leaves,
                                learning_rate=args.lgbm_learning_rate,
                                objective="multiclass",
                                num_class=len(CLASS_ORDER),
                                random_state=RANDOM_SEED,
                                n_jobs=args.n_jobs,
                                subsample=args.lgbm_subsample,
                                colsample_bytree=args.lgbm_colsample_bytree,
                                reg_lambda=args.lgbm_reg_lambda,
                                verbose=-1,
                            )
                            fit_kwargs = {}
                            if args.lgbm_early_stopping_rounds > 0 and len(x_validation) and y_validation_lgbm is not None:
                                fit_kwargs["eval_set"] = [(x_validation, y_validation_lgbm)]
                                fit_kwargs["eval_metric"] = "multi_logloss"
                                fit_kwargs["callbacks"] = [
                                    early_stopping(args.lgbm_early_stopping_rounds, verbose=False),
                                    log_evaluation(period=0),
                                ]
                            model.fit(x_fit, y_lgbm, **fit_kwargs)
                            best_iteration = getattr(model, "best_iteration_", None)
                            effective_estimators = int(best_iteration) if best_iteration else n_estimators
                            wrapper = ClassMappedClassifier(
                                model,
                                class_map=inverse_columns,
                                metadata={
                                    "lgbm_configured_estimators": n_estimators,
                                    "lgbm_effective_estimators": effective_estimators,
                                    "lgbm_num_leaves": leaves,
                                    "lgbm_learning_rate": args.lgbm_learning_rate,
                                    "lgbm_early_stopping_rounds": args.lgbm_early_stopping_rounds,
                                    "lgbm_subsample": args.lgbm_subsample,
                                    "lgbm_colsample_bytree": args.lgbm_colsample_bytree,
                                    "lgbm_reg_lambda": args.lgbm_reg_lambda,
                                },
                            )
                            model_spec = ModelSpec(
                                family="boosted_trees",
                                architecture="lightgbm",
                                name=name,
                                compute_proxy_ops=float(effective_estimators * len(CLASS_ORDER) * np.log2(leaves)),
                                compute_proxy_kind="effective_trees_times_classes_times_log2_leaves",
                                n_parameters=float(effective_estimators * len(CLASS_ORDER) * leaves),
                                fit_fn=lambda: wrapper,
                            )
                            add_result_row(
                                rows,
                                base=base,
                                spec=model_spec,
                                model=wrapper,
                                x_validation=x_validation,
                                y_validation_target=y_validation_target,
                                x_test=x_test,
                                y_test_target=y_test_windowed[target],
                                timing_repeats=args.timing_repeats,
                                checkpoint_path=args.checkpoint_path,
                            )

            if args.include_catboost:
                try:
                    from catboost import CatBoostClassifier
                except Exception as exc:
                    print(f"catboost could not be imported; skipping catboost family: {exc}", flush=True)
                    CatBoostClassifier = None
                if CatBoostClassifier is not None:
                    label_map = {-1: 0, 0: 1, 1: 2}
                    inverse_columns = {0: -1, 1: 0, 2: 1}
                    y_cat = y_target.map(label_map)
                    y_validation_cat = y_validation_target.map(label_map) if len(y_validation_target) else None
                    for iterations in args.catboost_iterations:
                        for depth in args.catboost_depths:
                            name = f"catboost_T{iterations}_D{depth}"
                            print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                            model = CatBoostClassifier(
                                iterations=iterations,
                                depth=depth,
                                learning_rate=args.catboost_learning_rate,
                                loss_function="MultiClass",
                                random_seed=RANDOM_SEED,
                                thread_count=args.n_jobs,
                                verbose=False,
                                allow_writing_files=False,
                            )
                            fit_kwargs = {}
                            if args.catboost_early_stopping_rounds > 0 and len(x_validation) and y_validation_cat is not None:
                                fit_kwargs["eval_set"] = (x_validation, y_validation_cat)
                                fit_kwargs["early_stopping_rounds"] = args.catboost_early_stopping_rounds
                            model.fit(x_fit, y_cat, **fit_kwargs)
                            best_iteration = model.get_best_iteration()
                            effective_iterations = int(best_iteration + 1) if best_iteration is not None and best_iteration >= 0 else iterations
                            wrapper = ClassMappedClassifier(
                                model,
                                class_map=inverse_columns,
                                metadata={
                                    "catboost_configured_iterations": iterations,
                                    "catboost_effective_iterations": effective_iterations,
                                    "catboost_depth": depth,
                                    "catboost_learning_rate": args.catboost_learning_rate,
                                    "catboost_early_stopping_rounds": args.catboost_early_stopping_rounds,
                                },
                            )
                            model_spec = ModelSpec(
                                family="oblivious_tree_ensemble",
                                architecture="catboost_symmetric_trees",
                                name=name,
                                compute_proxy_ops=float(effective_iterations * depth * len(CLASS_ORDER)),
                                compute_proxy_kind="effective_symmetric_trees_times_depth_times_classes",
                                n_parameters=float(effective_iterations * (2 ** depth) * len(CLASS_ORDER)),
                                fit_fn=lambda: wrapper,
                            )
                            add_result_row(
                                rows,
                                base=base,
                                spec=model_spec,
                                model=wrapper,
                                x_validation=x_validation,
                                y_validation_target=y_validation_target,
                                x_test=x_test,
                                y_test_target=y_test_windowed[target],
                                timing_repeats=args.timing_repeats,
                                checkpoint_path=args.checkpoint_path,
                            )

            if args.include_ebm:
                try:
                    from interpret.glassbox import ExplainableBoostingClassifier
                except Exception as exc:
                    print(f"interpret could not be imported; skipping EBM family: {exc}", flush=True)
                    ExplainableBoostingClassifier = None
                if ExplainableBoostingClassifier is not None:
                    for max_rounds in args.ebm_max_rounds:
                        for interactions in args.ebm_interactions:
                            name = f"ebm_R{max_rounds}_I{interactions}"
                            print(f"{dataset_variant} {target}: fitting {name}", flush=True)
                            model = ExplainableBoostingClassifier(
                                interactions=interactions,
                                max_rounds=max_rounds,
                                learning_rate=args.ebm_learning_rate,
                                max_bins=args.ebm_max_bins,
                                n_jobs=args.n_jobs,
                                random_state=RANDOM_SEED,
                            )
                            try:
                                model.fit(x_fit, y_target)
                            except TypeError:
                                model = ExplainableBoostingClassifier(
                                    interactions=interactions,
                                    max_rounds=max_rounds,
                                    learning_rate=args.ebm_learning_rate,
                                    max_bins=args.ebm_max_bins,
                                    random_state=RANDOM_SEED,
                                )
                                model.fit(x_fit, y_target)
                            model.metadata = {
                                "ebm_max_rounds": max_rounds,
                                "ebm_interactions": interactions,
                                "ebm_learning_rate": args.ebm_learning_rate,
                                "ebm_max_bins": args.ebm_max_bins,
                            }
                            main_effect_ops = d
                            interaction_ops = max(0, interactions) * 2
                            model_spec = ModelSpec(
                                family="generalized_additive_model",
                                architecture="explainable_boosting_machine",
                                name=name,
                                compute_proxy_ops=float((main_effect_ops + interaction_ops) * len(CLASS_ORDER)),
                                compute_proxy_kind="ebm_terms_times_classes",
                                n_parameters=float((d + max(0, interactions)) * args.ebm_max_bins * len(CLASS_ORDER)),
                                fit_fn=lambda: model,
                            )
                            add_result_row(
                                rows,
                                base=base,
                                spec=model_spec,
                                model=model,
                                x_validation=x_validation,
                                y_validation_target=y_validation_target,
                                x_test=x_test,
                                y_test_target=y_test_windowed[target],
                                timing_repeats=args.timing_repeats,
                                checkpoint_path=args.checkpoint_path,
                            )

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-output", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--feature-modes", default="supervised_full144")
    parser.add_argument("--top-n", type=parse_int_list, default=parse_int_list("16,40"))
    parser.add_argument("--horizons", default=",".join(HORIZONS))
    parser.add_argument("--window-sizes", type=parse_int_list, default=parse_int_list("1"))
    parser.add_argument("--allow-engineered-windows", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--recompute-feature-ranking", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--timing-mode",
        choices=["batch", "single"],
        default="batch",
        help="batch reports throughput per observation; single reports one-observation online latency.",
    )
    parser.add_argument(
        "--single-timing-samples",
        type=int,
        default=256,
        help="Number of individual observations to sample when --timing-mode single.",
    )
    parser.add_argument(
        "--torch-timing-device",
        choices=["same", "cpu", "mps", "cuda"],
        default="same",
        help="Device used only for timing torch predict_proba; training/evaluation still use --torch-device.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--min-samples-leaf", type=int, default=20)

    parser.add_argument("--no-trees", dest="include_trees", action="store_false")
    parser.add_argument("--no-forests", dest="include_forests", action="store_false")
    parser.add_argument("--no-boosted-trees", dest="include_boosted_trees", action="store_false")
    parser.add_argument("--no-mlp", dest="include_mlp", action="store_false")
    parser.add_argument("--no-kernel", dest="include_kernel", action="store_false")
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--include-lightgbm", action="store_true")
    parser.add_argument("--include-catboost", action="store_true")
    parser.add_argument("--include-ebm", action="store_true")
    parser.add_argument("--include-torch-nn", action="store_true")
    parser.add_argument("--include-random-conv", action="store_true")
    parser.add_argument("--include-temporal-summary-boosted", action="store_true")

    parser.add_argument("--tree-leaf-values", type=parse_int_list, default=parse_int_list("4,8,16,32,64,128,256"))
    parser.add_argument("--forest-estimators", type=parse_int_list, default=parse_int_list("8,16,32,64"))
    parser.add_argument("--forest-leaf-values", type=parse_int_list, default=parse_int_list("8,16,32,64"))
    parser.add_argument("--hgb-max-iter-values", type=parse_int_list, default=parse_int_list("16,32,64,128"))
    parser.add_argument("--hgb-leaf-values", type=parse_int_list, default=parse_int_list("8,16,32,64"))
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--mlp-hidden-layers",
        type=parse_hidden_layers,
        default=parse_hidden_layers("8,16,32,64,32x16,64x32,128x64,128x64x32"),
    )
    parser.add_argument("--mlp-max-iter", type=int, default=100)
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument(
        "--torch-mlp-hidden-layers",
        type=parse_hidden_layers,
        default=parse_hidden_layers("64,128,256,512,128x64,256x128,512x256"),
    )
    parser.add_argument(
        "--torch-residual-hidden-layers",
        type=parse_hidden_layers,
        default=parse_hidden_layers("128x128,256x256,512x512,256x256x256,512x512x512"),
    )
    parser.add_argument("--torch-fm-ranks", type=parse_int_list, default=parse_int_list(""))
    parser.add_argument(
        "--torch-tab-transformer-specs",
        type=parse_hidden_layers,
        default=parse_hidden_layers(""),
    )
    parser.add_argument("--torch-tab-transformer-ff-multiplier", type=int, default=2)
    parser.add_argument(
        "--torch-mlplob-specs",
        type=parse_hidden_layers,
        default=parse_hidden_layers(""),
        help="MLPLOB specs as hidden_dim x layers, optionally hidden_dim x layers x expansion.",
    )
    parser.add_argument("--torch-mlplob-expansion", type=int, default=4)
    parser.add_argument("--torch-mlplob-dropout", type=float, default=0.1)
    parser.add_argument(
        "--torch-tlob-specs",
        type=parse_hidden_layers,
        default=parse_hidden_layers(""),
        help="TLOB specs as hidden_dim x layers x heads, optionally hidden_dim x layers x heads x ff_multiplier.",
    )
    parser.add_argument("--torch-tlob-ff-multiplier", type=int, default=2)
    parser.add_argument("--torch-tlob-dropout", type=float, default=0.1)
    parser.add_argument("--torch-lob-bin-eps", type=float, default=1e-5)
    parser.add_argument("--torch-lob-no-bin", dest="torch_lob_use_bin", action="store_false")
    parser.add_argument(
        "--torch-cnn-channels",
        type=parse_hidden_layers,
        default=parse_hidden_layers("32,64,128,64x64,128x128,256x256"),
    )
    parser.add_argument("--torch-cnn-kernel-size", type=int, default=3)
    parser.add_argument(
        "--torch-inception-channels",
        type=parse_hidden_layers,
        default=parse_hidden_layers(""),
    )
    parser.add_argument("--torch-inception-kernel-sizes", type=parse_int_list, default=parse_int_list("9,19,39"))
    parser.add_argument("--torch-inception-bottleneck-channels", type=int, default=32)
    parser.add_argument(
        "--torch-tcn-channels",
        type=parse_hidden_layers,
        default=parse_hidden_layers("32x32,64x64,128x128,64x64x64,128x128x128"),
    )
    parser.add_argument("--torch-tcn-kernel-size", type=int, default=3)
    parser.add_argument(
        "--torch-cnn-attention-channels",
        type=parse_hidden_layers,
        default=parse_hidden_layers(""),
    )
    parser.add_argument("--torch-cnn-attention-kernel-size", type=int, default=3)
    parser.add_argument(
        "--torch-recurrent-specs",
        type=parse_hidden_layers,
        default=parse_hidden_layers(""),
    )
    parser.add_argument("--torch-recurrent-types", default="gru,lstm")
    parser.add_argument(
        "--torch-deeplob-specs",
        type=parse_hidden_layers,
        default=parse_hidden_layers("32x32x64,64x64x128,128x64x128"),
    )
    parser.add_argument("--torch-deeplob-kernel-sizes", type=parse_int_list, default=parse_int_list("1,3,5"))
    parser.add_argument(
        "--torch-deeplob-full-specs",
        type=parse_hidden_layers,
        default=parse_hidden_layers("32x64x64"),
    )
    parser.add_argument("--torch-deeplob-full-dropout", type=float, default=0.2)
    parser.add_argument("--torch-deeplob-full-norm", choices=["batchnorm", "groupnorm", "none"], default="batchnorm")
    parser.add_argument("--torch-deeplob-full-conv2-activation", choices=["leaky_relu", "tanh"], default="tanh")
    parser.add_argument("--torch-epochs", type=int, default=80)
    parser.add_argument("--torch-batch-size", type=int, default=1024)
    parser.add_argument("--torch-eval-batch-size", type=int, default=2048)
    parser.add_argument("--torch-optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--torch-lr", type=float, default=1e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-adam-eps", type=float, default=1e-8)
    parser.add_argument("--torch-patience", type=int, default=8)
    parser.add_argument("--torch-min-delta", type=float, default=1e-4)
    parser.add_argument("--torch-early-stop-metric", choices=["loss", "accuracy"], default="loss")
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--torch-log-epochs", action="store_true")
    parser.add_argument("--torch-temperature-scaling", action="store_true")
    parser.add_argument("--torch-temperature-max-iter", type=int, default=100)
    parser.add_argument("--torch-history-output-dir", default="")
    parser.add_argument("--kernel-components", type=parse_int_list, default=parse_int_list("16,32,64,128,256"))
    parser.add_argument("--kernel-gammas", type=parse_float_list, default=parse_float_list("0.01,0.1,1.0"))
    parser.add_argument("--kernel-logistic-max-iter", type=int, default=500)
    parser.add_argument("--kernel-sgd-max-iter", type=int, default=1000)
    parser.add_argument("--kernel-sgd-alpha", type=float, default=1e-4)
    parser.add_argument("--random-conv-kernels", type=parse_int_list, default=parse_int_list("256,512,1024,2048"))
    parser.add_argument("--random-conv-kernel-sizes", type=parse_int_list, default=parse_int_list("3,5,7,9,15,21,31"))
    parser.add_argument("--random-conv-heads", default="sgd,logistic")
    parser.add_argument("--random-conv-sgd-max-iter", type=int, default=1000)
    parser.add_argument("--random-conv-sgd-alpha", type=float, default=1e-4)
    parser.add_argument("--random-conv-logistic-max-iter", type=int, default=500)
    parser.add_argument("--random-conv-logistic-c", type=float, default=1.0)
    parser.add_argument("--temporal-summary-windows", type=parse_int_list, default=parse_int_list("3,5,10,20,50,100"))
    parser.add_argument("--temporal-summary-hgb-max-iter-values", type=parse_int_list, default=parse_int_list("64,128,256"))
    parser.add_argument("--temporal-summary-hgb-leaf-values", type=parse_int_list, default=parse_int_list("16,32,64"))
    parser.add_argument("--temporal-summary-hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--temporal-summary-include-xgboost", action="store_true")
    parser.add_argument("--temporal-summary-xgb-estimators", type=parse_int_list, default=parse_int_list("128,256,512"))
    parser.add_argument("--temporal-summary-xgb-depths", type=parse_int_list, default=parse_int_list("3,4,5"))
    parser.add_argument("--temporal-summary-xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--temporal-summary-xgb-early-stopping-rounds", type=int, default=0)
    parser.add_argument("--xgb-estimators", type=parse_int_list, default=parse_int_list("16,32,64,128"))
    parser.add_argument("--xgb-depths", type=parse_int_list, default=parse_int_list("2,3,4,5"))
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=0)
    parser.add_argument("--xgb-verbose-eval", type=int, default=0)
    parser.add_argument("--xgb-tree-method", default="hist")
    parser.add_argument("--xgb-subsample", type=float, default=1.0)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=1.0)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--lgbm-estimators", type=parse_int_list, default=parse_int_list("128,256,512,1024"))
    parser.add_argument("--lgbm-num-leaves", type=parse_int_list, default=parse_int_list("15,31,63,127"))
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=1.0)
    parser.add_argument("--catboost-iterations", type=parse_int_list, default=parse_int_list("128,256,512,1024"))
    parser.add_argument("--catboost-depths", type=parse_int_list, default=parse_int_list("4,6,8"))
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--ebm-max-rounds", type=parse_int_list, default=parse_int_list("500,1000,2000"))
    parser.add_argument("--ebm-interactions", type=parse_int_list, default=parse_int_list("0,5,10,20"))
    parser.add_argument("--ebm-learning-rate", type=float, default=0.01)
    parser.add_argument("--ebm-max-bins", type=int, default=256)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.feature_modes = {part.strip() for part in args.feature_modes.split(",") if part.strip()}
    args.horizons = [part.strip() for part in args.horizons.split(",") if part.strip()]
    args.torch_recurrent_types = [part.strip() for part in args.torch_recurrent_types.split(",") if part.strip()]
    args.random_conv_heads = [part.strip() for part in args.random_conv_heads.split(",") if part.strip()]
    unknown_random_conv_heads = sorted(set(args.random_conv_heads) - {"sgd", "logistic"})
    if unknown_random_conv_heads:
        raise ValueError(f"Unknown random conv heads: {unknown_random_conv_heads}")
    unknown_recurrent_types = sorted(set(args.torch_recurrent_types) - {"gru", "lstm"})
    if unknown_recurrent_types:
        raise ValueError(f"Unknown recurrent types: {unknown_recurrent_types}")
    unknown_horizons = sorted(set(args.horizons) - set(HORIZONS))
    if unknown_horizons:
        raise ValueError(f"Unknown horizons: {unknown_horizons}")

    output_dir = OUTPUT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"architecture_scaling_cf{args.cf}.csv"
    args.checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else None
    figure_dir = (
        Path(args.figure_dir)
        if args.figure_dir
        else OUTPUT_ROOT / "figures" / "architecture_scaling" / f"cf{args.cf}"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_data_root()
    x_train_all, x_test_all, y_train, y_test = load_no_auction_zscore_cf(args.cf, data_root)

    importance_by_source = {}
    if "supervised_lob40" in args.feature_modes:
        importance_by_source["lob40"] = load_or_compute_feature_importance(
            x_train_all.iloc[:, :40],
            y_train,
            output_dir,
            args.cf,
            "lob40",
            args.recompute_feature_ranking,
        )
    if "supervised_full144" in args.feature_modes:
        importance_by_source["full144"] = load_or_compute_feature_importance(
            x_train_all,
            y_train,
            output_dir,
            args.cf,
            "full144",
            args.recompute_feature_ranking,
        )

    feature_specs = make_feature_specs(x_train_all, importance_by_source, args.feature_modes, args.top_n)
    rows: list[dict] = []
    for spec in feature_specs:
        rows.extend(run_feature_spec(spec, x_train_all, x_test_all, y_train, y_test, args))

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    if not results.empty:
        write_plots(results, figure_dir)
    print(f"wrote {len(results)} rows to {output_path}")
    print(f"wrote frontier plots under {figure_dir}")


if __name__ == "__main__":
    main()
