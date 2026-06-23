#!/usr/bin/env python3
"""Structural forward-work proxy from saved result metadata.

The goal is a reproducible architecture-level axis that can be rebuilt from
CSV artifacts without refitting models. It deliberately uses structural
hyperparameters rather than fitted path-depth traversal.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


CLASS_COUNT = 3


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def positive_int(value: Any, default: int | None = None) -> int | None:
    value_float = finite_float(value)
    if not np.isfinite(value_float) or value_float <= 0:
        return default
    return int(round(value_float))


def parse_model_int(pattern: str, model_name: Any, default: int | None = None) -> int | None:
    match = re.search(pattern, str(model_name))
    if not match:
        return default
    return positive_int(match.group(1), default)


def parse_kernel_sizes(value: Any, window: int) -> list[int]:
    sizes = []
    for part in str(value).replace(",", "x").split("x"):
        part = part.strip()
        if not part:
            continue
        parsed = positive_int(part)
        if parsed is not None and 1 <= parsed <= window:
            sizes.append(parsed)
    return sizes


def structural_tree_work(row: pd.Series) -> tuple[float, str]:
    leaves = parse_model_int(r"tree_L(\d+)", row.get("model_name"), positive_int(row.get("n_parameters")))
    if leaves is None and np.isfinite(finite_float(row.get("n_parameters"))):
        leaves = max(2, int(round((finite_float(row.get("n_parameters")) + 1.0) / 2.0)))
    if leaves is None:
        return finite_float(row.get("compute_proxy_ops")), "fallback_compute_proxy"
    return float(math.log2(max(2, leaves)) + 1.0), "balanced_tree_log2_leaves_plus_leaf_lookup"


def structural_hgb_work(row: pd.Series) -> tuple[float, str]:
    iterations = parse_model_int(r"hgb_I(\d+)", row.get("model_name"))
    leaves = parse_model_int(r"_L(\d+)", row.get("model_name"))
    if iterations is None or leaves is None:
        proxy = finite_float(row.get("compute_proxy_ops"))
        return proxy + 2.0 * CLASS_COUNT if np.isfinite(proxy) else proxy, "fallback_compute_proxy_plus_tree_overhead"
    tree_count = iterations * CLASS_COUNT
    return float(tree_count * (math.log2(max(2, leaves)) + 2.0)), "iterations_times_classes_times_log2_leaves_lookup_logit_add"


def structural_catboost_work(row: pd.Series) -> tuple[float, str]:
    iterations = positive_int(row.get("catboost_effective_iterations"))
    if iterations is None:
        iterations = parse_model_int(r"catboost_T(\d+)", row.get("model_name"))
    depth = positive_int(row.get("catboost_depth"))
    if depth is None:
        depth = parse_model_int(r"_D(\d+)", row.get("model_name"))
    if iterations is None or depth is None:
        return finite_float(row.get("compute_proxy_ops")), "fallback_compute_proxy"
    # CatBoost symmetric multiclass rows use one oblivious tree traversal with a vector leaf.
    return float(iterations * (depth + 1.0 + CLASS_COUNT)), "symmetric_trees_depth_leaf_lookup_vector_logit_add"


def structural_random_conv_work(row: pd.Series) -> tuple[float, str]:
    kernels = positive_int(row.get("random_conv_kernels"))
    if kernels is None:
        kernels = parse_model_int(r"_K(\d+)", row.get("model_name"))
    window = positive_int(row.get("input_window"))
    if kernels is None or window is None:
        return finite_float(row.get("compute_proxy_ops")), "fallback_compute_proxy"
    sizes = parse_kernel_sizes(row.get("random_conv_kernel_sizes", ""), window)
    if not sizes:
        sizes = [window]
    per_kernel = []
    for length in sizes:
        responses = max(1, window - length + 1)
        conv_multiply_adds = responses * length
        max_comparisons = max(0, responses - 1)
        mean_additions = max(0, responses - 1)
        positive_comparisons = responses
        positive_additions = max(0, responses - 1)
        per_kernel.append(
            conv_multiply_adds
            + max_comparisons
            + mean_additions
            + positive_comparisons
            + positive_additions
        )
    feature_count = 3 * kernels
    standardization = 2 * feature_count
    linear_head = feature_count * CLASS_COUNT
    return float(kernels * np.mean(per_kernel) + standardization + linear_head), "expected_random_conv_dot_pool_standardize_linear_head"


def structural_mlp_work(row: pd.Series) -> tuple[float, str]:
    proxy = finite_float(row.get("compute_proxy_ops"))
    hidden_text = str(row.get("model_name", "")).replace("torch_mlp_", "").replace("mlp_", "")
    hidden = [positive_int(part) for part in hidden_text.split("x")]
    activation_ops = sum(value for value in hidden if value is not None)
    if np.isfinite(proxy):
        return float(proxy + activation_ops), "dense_multiply_adds_plus_hidden_activation_proxy"
    return proxy, "fallback_compute_proxy"


def structural_work_for_row(row: pd.Series) -> tuple[float, str]:
    architecture = str(row.get("architecture", ""))
    if architecture == "decision_tree":
        return structural_tree_work(row)
    if architecture == "hist_gradient_boosting":
        return structural_hgb_work(row)
    if architecture == "catboost_symmetric_trees":
        return structural_catboost_work(row)
    if architecture in {"random_conv_logistic", "random_conv_sgd"}:
        return structural_random_conv_work(row)
    if architecture in {"mlp_classifier", "torch_mlp"}:
        return structural_mlp_work(row)
    proxy = finite_float(row.get("compute_proxy_ops"))
    if np.isfinite(proxy):
        return proxy, "analytic_compute_proxy_structural"
    audited = finite_float(row.get("audited_forward_ops"))
    return audited, "legacy_audited_forward_ops_fallback"


def apply_structural_forward_work(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    units = []
    methods = []
    for _, row in frame.iterrows():
        value, method = structural_work_for_row(row)
        units.append(value)
        methods.append(method)
    frame["structural_forward_work_units"] = units
    frame["structural_forward_work_method"] = methods
    frame["structural_forward_work_confidence"] = np.where(
        pd.to_numeric(frame["structural_forward_work_units"], errors="coerce").notna(),
        "structural_proxy",
        "missing",
    )
    return frame
