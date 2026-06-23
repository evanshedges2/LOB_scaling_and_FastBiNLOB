#!/usr/bin/env python3
"""Compute-scaling experiments for FI-2010 NoAuction Z-score CF1.

This script turns the exploratory notebook logic into a reproducible table:
one row per symbolic state construction, feature set, target horizon, and grid
point, with both an operation-count proxy and measured inference latency.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


LOB_RAW_FEATURE_COUNT = 40
FULL_FEATURE_COUNT = 144
LABEL_ROW_START = 144
HORIZONS = ["y_10", "y_20", "y_30", "y_50", "y_100"]
CLASS_ORDER = [-1, 0, 1]
LABEL_MAP = {1: 1, 2: 0, 3: -1}
RANDOM_SEED = 0
MERGE_CHUNK_SIZE = 64

MERGE_TARGETS = {
    "single_y_10": ["y_10"],
    "single_y_20": ["y_20"],
    "single_y_30": ["y_30"],
    "single_y_50": ["y_50"],
    "single_y_100": ["y_100"],
    "joint_10_50_100": ["y_10", "y_50", "y_100"],
    "joint_10_30_100": ["y_10", "y_30", "y_100"],
    "joint_all": HORIZONS,
}

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BUNDLE_ROOT / "outputs"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    columns: list[str]
    selection_mode: str
    source_feature_count: int


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_horizon_groups(raw: str) -> list[list[str]]:
    aliases = {
        "all": HORIZONS,
        "joint_all": HORIZONS,
        "10_20": ["y_10", "y_20"],
        "10_50_100": ["y_10", "y_50", "y_100"],
    }
    groups = []
    for group_raw in raw.split(","):
        group_raw = group_raw.strip()
        if not group_raw:
            continue
        if group_raw in aliases:
            group = aliases[group_raw]
        else:
            group = []
            for part in group_raw.replace("+", "|").split("|"):
                part = part.strip()
                if not part:
                    continue
                horizon = part if part.startswith("y_") else f"y_{part}"
                group.append(horizon)
        unknown = sorted(set(group) - set(HORIZONS))
        if unknown:
            raise ValueError(f"Unknown tree product horizons: {unknown}")
        groups.append(group)
    return groups


def horizon_group_name(group: list[str]) -> str:
    return "tree_tuple_" + "_".join(h.replace("y_", "") for h in group)


def find_data_root() -> Path:
    candidates = [
        BUNDLE_ROOT / "BenchmarkDatasets",
        BUNDLE_ROOT / "data" / "BenchmarkDatasets",
        Path("BenchmarkDatasets"),
        Path("experiments/BenchmarkDatasets"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find FI-2010 BenchmarkDatasets directory.")


def load_no_auction_zscore_cf(cf: int, data_root: Path):
    base = data_root / "NoAuction" / "1.NoAuction_Zscore"
    train_path = base / "NoAuction_Zscore_Training" / f"Train_Dst_NoAuction_ZScore_CF_{cf}.txt"
    test_path = base / "NoAuction_Zscore_Testing" / f"Test_Dst_NoAuction_ZScore_CF_{cf}.txt"

    train_raw = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_raw = pd.read_csv(test_path, sep=r"\s+", header=None)

    x_train = train_raw.iloc[:FULL_FEATURE_COUNT].T.reset_index(drop=True)
    x_test = test_raw.iloc[:FULL_FEATURE_COUNT].T.reset_index(drop=True)
    x_train.columns = [f"x_{i}" for i in range(1, FULL_FEATURE_COUNT + 1)]
    x_test.columns = x_train.columns

    y_train = (
        train_raw.iloc[LABEL_ROW_START : LABEL_ROW_START + len(HORIZONS)]
        .T.replace(LABEL_MAP)
        .astype(int)
        .reset_index(drop=True)
    )
    y_test = (
        test_raw.iloc[LABEL_ROW_START : LABEL_ROW_START + len(HORIZONS)]
        .T.replace(LABEL_MAP)
        .astype(int)
        .reset_index(drop=True)
    )
    y_train.columns = HORIZONS
    y_test.columns = HORIZONS
    return x_train, x_test, y_train, y_test


def compute_feature_importance(x_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    rank_train_idx, rank_val_idx = train_test_split(
        np.arange(len(x_train)),
        test_size=0.25,
        random_state=RANDOM_SEED,
        shuffle=False,
    )

    rows = []
    for horizon in HORIZONS:
        model = HistGradientBoostingClassifier(loss="log_loss", max_depth=4, random_state=RANDOM_SEED)
        model.fit(x_train.iloc[rank_train_idx], y_train[horizon].iloc[rank_train_idx])
        perm = permutation_importance(
            model,
            x_train.iloc[rank_val_idx],
            y_train[horizon].iloc[rank_val_idx],
            scoring="neg_log_loss",
            n_repeats=3,
            random_state=RANDOM_SEED,
        )
        for feature, importance, std in zip(x_train.columns, perm.importances_mean, perm.importances_std):
            rows.append(
                {
                    "horizon": horizon,
                    "feature": feature,
                    "importance": importance,
                    "importance_std": std,
                }
            )

    by_horizon = pd.DataFrame(rows)
    out = by_horizon.groupby("feature", as_index=False).agg(
        mean_importance=("importance", "mean"),
        min_importance=("importance", "min"),
        max_importance=("importance", "max"),
    )
    out = out.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def load_or_compute_feature_importance(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    output_dir: Path,
    cf: int,
    feature_source: str,
    recompute: bool,
) -> pd.DataFrame:
    path = output_dir / f"fi2010_feature_importance_{feature_source}_cf{cf}.csv"
    if path.exists() and not recompute:
        return pd.read_csv(path)

    legacy_path = output_dir / f"fi2010_feature_importance_cf{cf}.csv"
    if feature_source == "lob40" and legacy_path.exists() and not recompute:
        legacy = pd.read_csv(legacy_path)
        if set(legacy["feature"]).issubset(set(x_train.columns)):
            legacy.to_csv(path, index=False)
            return legacy

    importance = compute_feature_importance(x_train, y_train)
    importance.to_csv(path, index=False)
    return importance


def make_feature_specs(
    x_train: pd.DataFrame,
    importance_by_source: dict[str, pd.DataFrame],
    feature_modes: set[str],
    top_ns: list[int],
) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    raw40_columns = list(x_train.columns[:LOB_RAW_FEATURE_COUNT])
    full_columns = list(x_train.columns[:FULL_FEATURE_COUNT])

    if "lob40" in feature_modes:
        specs.append(FeatureSpec("lob_raw_40", raw40_columns, "raw_first_40", LOB_RAW_FEATURE_COUNT))

    if "full144" in feature_modes:
        specs.append(FeatureSpec("full_144", full_columns, "raw_full_144", FULL_FEATURE_COUNT))

    if "supervised_lob40" in feature_modes:
        ranked = importance_by_source["lob40"]["feature"].tolist()
        for top_n in top_ns:
            specs.append(
                FeatureSpec(
                    f"supervised_lob40_top_{top_n}",
                    ranked[:top_n],
                    "supervised_train_only_lob40",
                    LOB_RAW_FEATURE_COUNT,
                )
            )

    if "supervised_full144" in feature_modes:
        ranked = importance_by_source["full144"]["feature"].tolist()
        for top_n in top_ns:
            specs.append(
                FeatureSpec(
                    f"supervised_full144_top_{top_n}",
                    ranked[:top_n],
                    "supervised_train_only_full144",
                    FULL_FEATURE_COUNT,
                )
            )

    return specs


def entropy_rows(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-15, 1.0)
    return -(prob * np.log(prob)).sum(axis=1)


def best_merge_pair_from_counts(counts_df: pd.DataFrame, chunk_size: int = MERGE_CHUNK_SIZE):
    counts = counts_df.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1)
    n_classes = counts.shape[1]

    probs = (counts + 1.0) / (row_sums[:, None] + n_classes)
    ent = entropy_rows(probs)

    best_cost = np.inf
    best_i = None
    best_j = None

    for i0 in range(0, len(counts_df), chunk_size):
        i1 = min(i0 + chunk_size, len(counts_df))
        left_counts = counts[i0:i1]
        left_sums = row_sums[i0:i1]
        left_ent = ent[i0:i1]

        total_sums = left_sums[:, None] + row_sums[None, :]
        mix_probs = (left_counts[:, None, :] + counts[None, :, :] + 1.0) / (
            total_sums[:, :, None] + n_classes
        )
        mix_ent = entropy_rows(mix_probs.reshape(-1, n_classes)).reshape(mix_probs.shape[:2])
        costs = total_sums * mix_ent - left_sums[:, None] * left_ent[:, None] - row_sums[None, :] * ent[None, :]

        global_i = np.arange(i0, i1)[:, None]
        global_j = np.arange(len(counts_df))[None, :]
        costs[global_j <= global_i] = np.inf

        local_i, local_j = np.unravel_index(np.argmin(costs), costs.shape)
        local_cost = costs[local_i, local_j]
        if local_cost < best_cost:
            best_cost = local_cost
            best_i = i0 + local_i
            best_j = local_j

    return counts_df.index[best_i], counts_df.index[best_j]


def make_merge_target(y_train: pd.DataFrame, columns: list[str]) -> pd.Series:
    if len(columns) == 1:
        return y_train[columns[0]].astype(str)
    return y_train[columns].astype(str).agg("|".join, axis=1)


def predictive_merge_map(train_states: np.ndarray, merge_target: pd.Series, target_k: int) -> dict[int, int]:
    counts = pd.crosstab(pd.Series(train_states, name="state"), pd.Series(merge_target, name="target"))
    groups = {state: [state] for state in counts.index}

    while len(counts) > target_k:
        left, right = best_merge_pair_from_counts(counts)
        counts.loc[left] = counts.loc[left] + counts.loc[right]
        counts = counts.drop(index=right)
        groups[left] = groups[left] + groups[right]
        del groups[right]

    state_map: dict[int, int] = {}
    for coarse_state, fine_states in enumerate(groups.values()):
        for fine_state in fine_states:
            state_map[int(fine_state)] = coarse_state
    return state_map


def geometric_merge_map(kmeans_model: KMeans, target_k: int) -> dict[int, int]:
    labels = AgglomerativeClustering(n_clusters=target_k, linkage="ward").fit_predict(kmeans_model.cluster_centers_)
    return {fine_state: int(labels[fine_state]) for fine_state in range(len(labels))}


def random_merge_map(train_states: np.ndarray, target_k: int, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    fine_states = np.sort(np.unique(train_states))
    shuffled = rng.permutation(fine_states)
    groups = np.array_split(shuffled, target_k)

    state_map: dict[int, int] = {}
    for coarse_state, group in enumerate(groups):
        for fine_state in group:
            state_map[int(fine_state)] = coarse_state
    return state_map


def apply_state_map(states: np.ndarray, state_map: dict[int, int]) -> np.ndarray:
    mapped = np.empty(len(states), dtype=int)
    fallback = 0
    for idx, state in enumerate(states):
        mapped[idx] = state_map.get(int(state), fallback)
    return mapped


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def score_states(
    train_states: np.ndarray,
    test_states: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    global_probs = y_train.value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0)
    counts = pd.crosstab(pd.Series(train_states, name="state"), pd.Series(y_train, name="target"))
    counts = counts.reindex(columns=CLASS_ORDER, fill_value=0)
    probs = counts.add(1).div(counts.sum(axis=1) + len(CLASS_ORDER), axis=0)

    proba = pd.DataFrame({"state": test_states}).join(probs, on="state")
    proba = proba[CLASS_ORDER].fillna(global_probs)
    pred = proba.idxmax(axis=1).astype(int)
    test_occ = pd.Series(test_states).value_counts(normalize=True)
    aligned_probs = probs.reindex(test_occ.index).fillna(global_probs)
    kl_to_global = (aligned_probs * np.log(aligned_probs.div(global_probs, axis=1))).sum(axis=1)
    entropy_by_state = entropy_rows(aligned_probs.to_numpy())
    weighted_train_entropy = float(np.sum(test_occ.to_numpy() * entropy_by_state))
    total_test_weighted_train_kl = float(np.sum(test_occ * kl_to_global))
    top_state_info_share = float((test_occ * kl_to_global).max() / total_test_weighted_train_kl) if total_test_weighted_train_kl > 0 else np.nan
    sorted_occ = test_occ.sort_values()
    rare_states = sorted_occ[sorted_occ.cumsum() <= 0.20].index
    if len(rare_states) == 0 and len(sorted_occ) > 0:
        rare_states = sorted_occ.head(1).index
    rare_info = float((test_occ.loc[rare_states] * kl_to_global.loc[rare_states]).sum()) if len(rare_states) else np.nan
    rare_info_share = rare_info / total_test_weighted_train_kl if total_test_weighted_train_kl > 0 else np.nan
    train_move_prob = aligned_probs[-1] + aligned_probs[1]
    global_move_prob = global_probs.loc[-1] + global_probs.loc[1]
    max_abs_return_lift = float(
        pd.concat(
            [
                (aligned_probs[-1] - global_probs.loc[-1]).abs(),
                (aligned_probs[0] - global_probs.loc[0]).abs(),
                (aligned_probs[1] - global_probs.loc[1]).abs(),
                (train_move_prob - global_move_prob).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .max()
    )

    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "log_loss": log_loss(y_test, proba, labels=CLASS_ORDER),
        "weighted_train_label_entropy": weighted_train_entropy,
        "total_test_weighted_train_kl": total_test_weighted_train_kl,
        "top_state_info_share": top_state_info_share,
        "rare_info_share": rare_info_share,
        "max_abs_return_lift": max_abs_return_lift,
    }


def score_proba(proba: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    proba = proba.reindex(columns=CLASS_ORDER, fill_value=0)
    pred = proba.idxmax(axis=1).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "log_loss": log_loss(y_test, proba, labels=CLASS_ORDER),
    }


def state_distribution_stats(states: np.ndarray) -> dict[str, float]:
    values = pd.Series(states)
    occ = values.value_counts(normalize=True)
    entropy = -(occ * np.log(occ)).sum()
    return {
        "test_states_used": int(occ.shape[0]),
        "max_state_occupancy": float(occ.max()),
        "effective_num_states": float(np.exp(entropy)),
        "state_occupancy_entropy": float(entropy),
    }


def same_state_rate(states: np.ndarray) -> float:
    if len(states) <= 1:
        return float("nan")
    return float(np.mean(states[1:] == states[:-1]))


def time_assignment(assign_fn: Callable[[], np.ndarray], n_obs: int, repeats: int) -> tuple[float, float]:
    warmup = assign_fn()
    if len(warmup) != n_obs:
        raise ValueError("assignment function returned wrong number of states")

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = assign_fn()
        elapsed = time.perf_counter() - start
        if len(out) != n_obs:
            raise ValueError("assignment function returned wrong number of states")
        timings.append(elapsed)
    median_s = float(np.median(timings))
    return median_s, median_s * 1_000_000 / n_obs


def tree_average_depth(tree: DecisionTreeClassifier, x: pd.DataFrame) -> float:
    path = tree.decision_path(x)
    return float(np.mean(path.getnnz(axis=1) - 1))


def encode_joint_leaf_states(train_leaf_matrix: np.ndarray, test_leaf_matrix: np.ndarray):
    tuple_to_state: dict[tuple[int, ...], int] = {}
    train_states = np.empty(train_leaf_matrix.shape[0], dtype=int)

    for idx, row in enumerate(train_leaf_matrix):
        key = tuple(int(value) for value in row)
        if key not in tuple_to_state:
            tuple_to_state[key] = len(tuple_to_state)
        train_states[idx] = tuple_to_state[key]

    counts = np.bincount(train_states)
    fallback_state = int(np.argmax(counts))
    test_states = np.empty(test_leaf_matrix.shape[0], dtype=int)
    unseen_count = 0

    for idx, row in enumerate(test_leaf_matrix):
        key = tuple(int(value) for value in row)
        state = tuple_to_state.get(key)
        if state is None:
            state = fallback_state
            unseen_count += 1
        test_states[idx] = state

    unseen_rate = unseen_count / len(test_states) if len(test_states) else 0.0
    return train_states, test_states, tuple_to_state, fallback_state, unseen_rate


def predict_joint_leaf_states(
    trees: list[DecisionTreeClassifier],
    x: pd.DataFrame,
    tuple_to_state: dict[tuple[int, ...], int],
    fallback_state: int,
) -> np.ndarray:
    leaf_matrix = np.column_stack([tree.apply(x) for tree in trees])
    states = np.empty(leaf_matrix.shape[0], dtype=int)
    for idx, row in enumerate(leaf_matrix):
        key = tuple(int(value) for value in row)
        states[idx] = tuple_to_state.get(key, fallback_state)
    return states


def add_scored_rows(
    rows: list[dict],
    *,
    base: dict,
    train_states: np.ndarray,
    test_states: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    assign_fn: Callable[[], np.ndarray],
    timing_repeats: int,
):
    timing_s, latency_us = time_assignment(assign_fn, len(test_states), timing_repeats)
    dist = state_distribution_stats(test_states)
    persistence = same_state_rate(test_states)

    for score_target in HORIZONS:
        row = dict(base)
        row["score_target"] = score_target
        row.update(score_states(train_states, test_states, y_train[score_target], y_test[score_target]))
        row.update(dist)
        row["same_state_rate"] = persistence
        row["assignment_time_s_median"] = timing_s
        row["latency_us_per_obs_median"] = latency_us
        rows.append(row)


def add_predictor_row(
    rows: list[dict],
    *,
    base: dict,
    proba: pd.DataFrame,
    y_test_target: pd.Series,
    predict_fn: Callable[[], np.ndarray],
    timing_repeats: int,
):
    timing_s, latency_us = time_assignment(predict_fn, len(y_test_target), timing_repeats)
    row = dict(base)
    row.update(score_proba(proba, y_test_target))
    row["test_states_used"] = np.nan
    row["max_state_occupancy"] = np.nan
    row["effective_num_states"] = np.nan
    row["state_occupancy_entropy"] = np.nan
    row["same_state_rate"] = np.nan
    row["assignment_time_s_median"] = timing_s
    row["latency_us_per_obs_median"] = latency_us
    rows.append(row)


def add_skipped_merge_row(
    rows: list[dict],
    *,
    base: dict,
    score_target: str,
    reason: str,
):
    row = dict(base)
    row["score_target"] = score_target
    row["accuracy"] = np.nan
    row["balanced_accuracy"] = np.nan
    row["log_loss"] = np.nan
    row["test_states_used"] = np.nan
    row["max_state_occupancy"] = np.nan
    row["effective_num_states"] = np.nan
    row["state_occupancy_entropy"] = np.nan
    row["same_state_rate"] = np.nan
    row["assignment_time_s_median"] = np.nan
    row["latency_us_per_obs_median"] = np.nan
    row["skip_reason"] = reason
    rows.append(row)


def run_feature_spec(
    spec: FeatureSpec,
    x_train_all: pd.DataFrame,
    x_test_all: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    args,
) -> list[dict]:
    x_train = x_train_all[spec.columns]
    x_test = x_test_all[spec.columns]
    d = len(spec.columns)
    rows: list[dict] = []

    k_values = args.k_values
    ratios = args.ratios
    all_m_values = sorted(set(k_values + [k * ratio for k in k_values for ratio in ratios]))
    kmeans_cache = {}
    hgb_cache: dict[str, HistGradientBoostingClassifier] = {}

    for m in all_m_values:
        model = KMeans(n_clusters=m, random_state=RANDOM_SEED, n_init=args.kmeans_n_init)
        train_states = model.fit_predict(x_train)
        test_states = model.predict(x_test)
        kmeans_cache[m] = {"model": model, "train_states": train_states, "test_states": test_states}

    def get_hgb(score_target: str) -> HistGradientBoostingClassifier:
        if score_target not in hgb_cache:
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=args.hgb_max_depth,
                max_iter=args.hgb_max_iter,
                random_state=RANDOM_SEED,
            )
            model.fit(x_train, y_train[score_target])
            hgb_cache[score_target] = model
        return hgb_cache[score_target]

    if args.include_non_symbolic:
        for score_target in HORIZONS:
            logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
            logistic.fit(x_train, y_train[score_target])
            logistic_proba = pd.DataFrame(logistic.predict_proba(x_test), columns=logistic.classes_)
            add_predictor_row(
                rows,
                base={
                    "cf": args.cf,
                    "feature_set": spec.name,
                    "feature_selection": spec.selection_mode,
                    "source_feature_count": spec.source_feature_count,
                    "n_feature_rows_available": FULL_FEATURE_COUNT,
                    "d": d,
                    "method_family": "non_symbolic_predictor",
                    "method": "logistic_regression",
                    "fit_target": score_target,
                    "merge_target": "none",
                    "score_target": score_target,
                    "K": np.nan,
                    "M": np.nan,
                    "ratio": np.nan,
                    "compute_proxy_ops": d * len(CLASS_ORDER),
                    "compute_proxy_kind": "linear_classifier_d_times_classes",
                },
                proba=logistic_proba,
                y_test_target=y_test[score_target],
                predict_fn=lambda model=logistic, x_test=x_test: model.predict_proba(x_test),
                timing_repeats=args.timing_repeats,
            )

            hgb = get_hgb(score_target)
            hgb_proba = pd.DataFrame(hgb.predict_proba(x_test), columns=hgb.classes_)
            add_predictor_row(
                rows,
                base={
                    "cf": args.cf,
                    "feature_set": spec.name,
                    "feature_selection": spec.selection_mode,
                    "source_feature_count": spec.source_feature_count,
                    "n_feature_rows_available": FULL_FEATURE_COUNT,
                    "d": d,
                    "method_family": "non_symbolic_predictor",
                    "method": "hist_gradient_boosting",
                    "fit_target": score_target,
                    "merge_target": "none",
                    "score_target": score_target,
                    "K": np.nan,
                    "M": np.nan,
                    "ratio": np.nan,
                    "compute_proxy_ops": args.hgb_max_iter * args.hgb_max_depth * len(CLASS_ORDER),
                    "compute_proxy_kind": "hgb_trees_times_depth_times_classes",
                },
                proba=hgb_proba,
                y_test_target=y_test[score_target],
                predict_fn=lambda model=hgb, x_test=x_test: model.predict_proba(x_test),
                timing_repeats=args.timing_repeats,
            )

            if args.include_mlp:
                direct_mlp = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=args.mlp_max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=5,
                    random_state=RANDOM_SEED,
                )
                direct_mlp.fit(x_train, y_train[score_target])
                mlp_proba = pd.DataFrame(direct_mlp.predict_proba(x_test), columns=direct_mlp.classes_)
                add_predictor_row(
                    rows,
                    base={
                        "cf": args.cf,
                        "feature_set": spec.name,
                        "feature_selection": spec.selection_mode,
                        "source_feature_count": spec.source_feature_count,
                        "n_feature_rows_available": FULL_FEATURE_COUNT,
                        "d": d,
                        "method_family": "non_symbolic_predictor",
                        "method": "small_mlp",
                        "fit_target": score_target,
                        "merge_target": "none",
                        "score_target": score_target,
                        "K": np.nan,
                        "M": np.nan,
                        "ratio": np.nan,
                        "compute_proxy_ops": d * 32 + 32 * 16 + 16 * len(CLASS_ORDER),
                        "compute_proxy_kind": "mlp_32_16_forward",
                    },
                    proba=mlp_proba,
                    y_test_target=y_test[score_target],
                    predict_fn=lambda model=direct_mlp, x_test=x_test: model.predict_proba(x_test),
                    timing_repeats=args.timing_repeats,
                )

    for k in k_values:
        cached = kmeans_cache[k]
        add_scored_rows(
            rows,
            base={
                "cf": args.cf,
                "feature_set": spec.name,
                "feature_selection": spec.selection_mode,
                "source_feature_count": spec.source_feature_count,
                "n_feature_rows_available": FULL_FEATURE_COUNT,
                "d": d,
                "method_family": "unsupervised_geometric",
                "method": "native_kmeans",
                "fit_target": "none",
                "merge_target": "none",
                "K": k,
                "M": k,
                "ratio": 1,
                "compute_proxy_ops": k * d,
                "compute_proxy_kind": "kmeans_distances_K_times_d",
            },
            train_states=cached["train_states"],
            test_states=cached["test_states"],
            y_train=y_train,
            y_test=y_test,
            assign_fn=lambda model=cached["model"], x_test=x_test: model.predict(x_test),
            timing_repeats=args.timing_repeats,
        )

        for fit_target in HORIZONS:
            tree = DecisionTreeClassifier(max_leaf_nodes=k, random_state=RANDOM_SEED)
            tree.fit(x_train, y_train[fit_target])
            train_states = tree.apply(x_train)
            test_states = tree.apply(x_test)
            avg_depth = tree_average_depth(tree, x_test)
            add_scored_rows(
                rows,
                base={
                    "cf": args.cf,
                    "feature_set": spec.name,
                    "feature_selection": spec.selection_mode,
                    "source_feature_count": spec.source_feature_count,
                    "n_feature_rows_available": FULL_FEATURE_COUNT,
                    "d": d,
                    "method_family": "supervised_symbolic",
                    "method": "decision_tree",
                    "fit_target": fit_target,
                    "merge_target": "none",
                    "K": k,
                    "M": k,
                    "ratio": 1,
                    "compute_proxy_ops": avg_depth,
                    "compute_proxy_kind": "empirical_tree_path_depth",
                },
                train_states=train_states,
                test_states=test_states,
                y_train=y_train,
                y_test=y_test,
                assign_fn=lambda tree=tree, x_test=x_test: tree.apply(x_test),
                timing_repeats=args.timing_repeats,
            )

            if args.include_hgb_bins:
                hgb = get_hgb(fit_target)
                train_conf = hgb.predict_proba(x_train).max(axis=1)
                test_conf = hgb.predict_proba(x_test).max(axis=1)
                edges = np.unique(np.quantile(train_conf, np.linspace(0, 1, k + 1)))
                if len(edges) <= 2:
                    train_states = np.zeros(len(train_conf), dtype=int)
                    test_states = np.zeros(len(test_conf), dtype=int)
                else:
                    train_states = np.digitize(train_conf, edges[1:-1], right=True)
                    test_states = np.digitize(test_conf, edges[1:-1], right=True)
                hgb_compute = args.hgb_max_iter * args.hgb_max_depth * len(CLASS_ORDER) + np.log2(k)
                add_scored_rows(
                    rows,
                    base={
                        "cf": args.cf,
                        "feature_set": spec.name,
                        "feature_selection": spec.selection_mode,
                        "source_feature_count": spec.source_feature_count,
                        "n_feature_rows_available": FULL_FEATURE_COUNT,
                        "d": d,
                        "method_family": "supervised_symbolic",
                        "method": "hgb_confidence_bins",
                        "fit_target": fit_target,
                        "merge_target": "none",
                        "K": k,
                        "M": k,
                        "ratio": 1,
                        "compute_proxy_ops": hgb_compute,
                        "compute_proxy_kind": "hgb_predict_plus_confidence_digitize",
                    },
                    train_states=train_states,
                    test_states=test_states,
                    y_train=y_train,
                    y_test=y_test,
                    assign_fn=lambda model=hgb, edges=edges, x_test=x_test: (
                        np.digitize(model.predict_proba(x_test).max(axis=1), edges[1:-1], right=True)
                        if len(edges) > 2
                        else np.zeros(len(x_test), dtype=int)
                    ),
                    timing_repeats=args.timing_repeats,
                )

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=RANDOM_SEED,
            max_iter=args.gmm_max_iter,
            reg_covar=1e-5,
        )
        gmm.fit(x_train)
        train_states = gmm.predict(x_train)
        test_states = gmm.predict(x_test)
        add_scored_rows(
            rows,
            base={
                "cf": args.cf,
                "feature_set": spec.name,
                "feature_selection": spec.selection_mode,
                "source_feature_count": spec.source_feature_count,
                "n_feature_rows_available": FULL_FEATURE_COUNT,
                "d": d,
                "method_family": "latent_mixture",
                "method": "gaussian_mixture_diag",
                "fit_target": "none",
                "merge_target": "none",
                "K": k,
                "M": k,
                "ratio": 1,
                "compute_proxy_ops": k * d,
                "compute_proxy_kind": "diag_gmm_emission_K_times_d",
            },
            train_states=train_states,
            test_states=test_states,
            y_train=y_train,
            y_test=y_test,
            assign_fn=lambda gmm=gmm, x_test=x_test: gmm.predict(x_test),
            timing_repeats=args.timing_repeats,
        )

        if args.include_mlp:
            for fit_target in HORIZONS:
                mlp = MLPClassifier(
                    hidden_layer_sizes=(32, 8),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=args.mlp_max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=5,
                    random_state=RANDOM_SEED,
                )
                mlp.fit(x_train, y_train[fit_target])

                def embed(x, mlp=mlp):
                    hidden_1 = relu(x.to_numpy() @ mlp.coefs_[0] + mlp.intercepts_[0])
                    return relu(hidden_1 @ mlp.coefs_[1] + mlp.intercepts_[1])

                train_embedding = embed(x_train)
                test_embedding = embed(x_test)
                model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=args.kmeans_n_init)
                train_states = model.fit_predict(train_embedding)
                test_states = model.predict(test_embedding)
                compute_ops = d * 32 + 32 * 8 + k * 8

                def assign_mlp(x_test=x_test, mlp=mlp, model=model):
                    hidden_1 = relu(x_test.to_numpy() @ mlp.coefs_[0] + mlp.intercepts_[0])
                    hidden_2 = relu(hidden_1 @ mlp.coefs_[1] + mlp.intercepts_[1])
                    return model.predict(hidden_2)

                add_scored_rows(
                    rows,
                    base={
                        "cf": args.cf,
                        "feature_set": spec.name,
                        "feature_selection": spec.selection_mode,
                        "source_feature_count": spec.source_feature_count,
                        "n_feature_rows_available": FULL_FEATURE_COUNT,
                        "d": d,
                        "method_family": "supervised_embedding",
                        "method": "mlp_embedding_kmeans",
                        "fit_target": fit_target,
                        "merge_target": "none",
                        "K": k,
                        "M": k,
                        "ratio": 1,
                        "compute_proxy_ops": compute_ops,
                        "compute_proxy_kind": "mlp_forward_plus_embedding_kmeans",
                    },
                    train_states=train_states,
                    test_states=test_states,
                    y_train=y_train,
                    y_test=y_test,
                    assign_fn=assign_mlp,
                    timing_repeats=args.timing_repeats,
                )

    if args.include_tree_product:
        for horizon_group in args.tree_product_horizon_groups:
            group_name = horizon_group_name(horizon_group)
            for fine_leaf_count in args.tree_product_leaf_values:
                trees = []
                train_leaf_columns = []
                test_leaf_columns = []
                train_depths = []
                test_depths = []

                for fit_target in horizon_group:
                    tree = DecisionTreeClassifier(max_leaf_nodes=fine_leaf_count, random_state=RANDOM_SEED)
                    tree.fit(x_train, y_train[fit_target])
                    trees.append(tree)
                    train_leaf_columns.append(tree.apply(x_train))
                    test_leaf_columns.append(tree.apply(x_test))
                    train_depths.append(tree_average_depth(tree, x_train))
                    test_depths.append(tree_average_depth(tree, x_test))

                train_leaf_matrix = np.column_stack(train_leaf_columns)
                test_leaf_matrix = np.column_stack(test_leaf_columns)
                fine_train, fine_test, tuple_to_state, fallback_state, unseen_rate = encode_joint_leaf_states(
                    train_leaf_matrix,
                    test_leaf_matrix,
                )
                observed_m = int(len(np.unique(fine_train)))
                train_depth_sum = float(np.sum(train_depths))
                test_depth_sum = float(np.sum(test_depths))
                print(
                    f"  tree_product group={group_name} leaf_count={fine_leaf_count} "
                    f"observed_m={observed_m} unseen_rate={unseen_rate:.6f}"
                )

                add_scored_rows(
                    rows,
                    base={
                        "cf": args.cf,
                        "feature_set": spec.name,
                        "feature_selection": spec.selection_mode,
                        "source_feature_count": spec.source_feature_count,
                        "n_feature_rows_available": FULL_FEATURE_COUNT,
                        "d": d,
                        "method_family": "supervised_composite_symbolic",
                        "method": "tree_product_fine",
                        "fit_target": group_name,
                        "merge_target": "none",
                        "K": observed_m,
                        "M": observed_m,
                        "ratio": 1,
                        "compute_proxy_ops": test_depth_sum,
                        "compute_proxy_kind": "sum_multi_horizon_tree_path_depths",
                        "tree_product_horizons": "+".join(horizon_group),
                        "tree_product_leaf_count": fine_leaf_count,
                        "tree_product_train_depth_sum": train_depth_sum,
                        "tree_product_test_depth_sum": test_depth_sum,
                        "fine_state_unseen_rate": unseen_rate,
                    },
                    train_states=fine_train,
                    test_states=fine_test,
                    y_train=y_train,
                    y_test=y_test,
                    assign_fn=lambda trees=trees, x_test=x_test, tuple_to_state=tuple_to_state, fallback_state=fallback_state: predict_joint_leaf_states(
                        trees,
                        x_test,
                        tuple_to_state,
                        fallback_state,
                    ),
                    timing_repeats=args.timing_repeats,
                )

                for k in k_values:
                    if observed_m <= k:
                        continue
                    for merge_name in args.merge_targets:
                        base = {
                            "cf": args.cf,
                            "feature_set": spec.name,
                            "feature_selection": spec.selection_mode,
                            "source_feature_count": spec.source_feature_count,
                            "n_feature_rows_available": FULL_FEATURE_COUNT,
                            "d": d,
                            "method_family": "supervised_target_aware_compression",
                            "method": "tree_product_predictive_merge",
                            "fit_target": group_name,
                            "merge_target": merge_name,
                            "K": k,
                            "M": observed_m,
                            "ratio": observed_m / k,
                            "compute_proxy_ops": test_depth_sum + 1,
                            "compute_proxy_kind": "sum_multi_horizon_tree_path_depths_plus_lookup",
                            "tree_product_horizons": "+".join(horizon_group),
                            "tree_product_leaf_count": fine_leaf_count,
                            "tree_product_train_depth_sum": train_depth_sum,
                            "tree_product_test_depth_sum": test_depth_sum,
                            "fine_state_unseen_rate": unseen_rate,
                        }
                        if observed_m > args.max_exact_merge_states:
                            reason = (
                                f"observed_m={observed_m} exceeds max_exact_merge_states="
                                f"{args.max_exact_merge_states}; exact greedy merge is too expensive"
                            )
                            print(
                                f"  skipping tree_product merge group={group_name} leaf_count={fine_leaf_count} "
                                f"K={k} target={merge_name}: {reason}"
                            )
                            for score_target in HORIZONS:
                                add_skipped_merge_row(rows, base=base, score_target=score_target, reason=reason)
                            continue

                        print(
                            f"  exact tree_product merge group={group_name} leaf_count={fine_leaf_count} "
                            f"M={observed_m}->K={k} target={merge_name}"
                        )
                        merge_target = make_merge_target(y_train, MERGE_TARGETS[merge_name])
                        state_map = predictive_merge_map(fine_train, merge_target, k)
                        train_states = apply_state_map(fine_train, state_map)
                        test_states = apply_state_map(fine_test, state_map)
                        add_scored_rows(
                            rows,
                            base=base,
                            train_states=train_states,
                            test_states=test_states,
                            y_train=y_train,
                            y_test=y_test,
                            assign_fn=lambda trees=trees, x_test=x_test, tuple_to_state=tuple_to_state, fallback_state=fallback_state, state_map=state_map: apply_state_map(
                                predict_joint_leaf_states(trees, x_test, tuple_to_state, fallback_state),
                                state_map,
                            ),
                            timing_repeats=args.timing_repeats,
                        )

    if args.include_controls or args.include_predictive_merge:
        for k in k_values:
            for ratio in ratios:
                m = k * ratio
                cached = kmeans_cache[m]
                fine_train = cached["train_states"]
                fine_test = cached["test_states"]

                if args.include_controls:
                    control_maps = {
                        "geometric_merge": geometric_merge_map(cached["model"], k),
                        "random_merge": random_merge_map(fine_train, k, seed=10_000 + m + k + d),
                    }
                    for method, state_map in control_maps.items():
                        train_states = apply_state_map(fine_train, state_map)
                        test_states = apply_state_map(fine_test, state_map)
                        add_scored_rows(
                            rows,
                            base={
                                "cf": args.cf,
                                "feature_set": spec.name,
                                "feature_selection": spec.selection_mode,
                                "source_feature_count": spec.source_feature_count,
                                "n_feature_rows_available": FULL_FEATURE_COUNT,
                                "d": d,
                                "method_family": "unsupervised_geometric",
                                "method": method,
                                "fit_target": "none",
                                "merge_target": "none",
                                "K": k,
                                "M": m,
                                "ratio": ratio,
                                "compute_proxy_ops": m * d + 1,
                                "compute_proxy_kind": "fine_kmeans_assignment_plus_lookup",
                            },
                            train_states=train_states,
                            test_states=test_states,
                            y_train=y_train,
                            y_test=y_test,
                            assign_fn=lambda model=cached["model"], state_map=state_map, x_test=x_test: apply_state_map(
                                model.predict(x_test), state_map
                            ),
                            timing_repeats=args.timing_repeats,
                        )

                if args.include_predictive_merge:
                    for merge_name in args.merge_targets:
                        merge_target = make_merge_target(y_train, MERGE_TARGETS[merge_name])
                        state_map = predictive_merge_map(fine_train, merge_target, k)
                        train_states = apply_state_map(fine_train, state_map)
                        test_states = apply_state_map(fine_test, state_map)
                        add_scored_rows(
                            rows,
                            base={
                                "cf": args.cf,
                                "feature_set": spec.name,
                                "feature_selection": spec.selection_mode,
                                "source_feature_count": spec.source_feature_count,
                                "n_feature_rows_available": FULL_FEATURE_COUNT,
                                "d": d,
                                "method_family": "target_aware_compression",
                                "method": "predictive_merge",
                                "fit_target": "none",
                                "merge_target": merge_name,
                                "K": k,
                                "M": m,
                                "ratio": ratio,
                                "compute_proxy_ops": m * d + 1,
                                "compute_proxy_kind": "fine_kmeans_assignment_plus_lookup",
                            },
                            train_states=train_states,
                            test_states=test_states,
                            y_train=y_train,
                            y_test=y_test,
                            assign_fn=lambda model=cached["model"], state_map=state_map, x_test=x_test: apply_state_map(
                                model.predict(x_test), state_map
                            ),
                            timing_repeats=args.timing_repeats,
                        )

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf", type=int, default=1)
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("4,8,16,32,64"))
    parser.add_argument("--ratios", type=parse_int_list, default=parse_int_list("2,4"))
    parser.add_argument("--top-n", type=parse_int_list, default=parse_int_list("16,40"))
    parser.add_argument(
        "--feature-modes",
        default="lob40,supervised_lob40,supervised_full144",
        help="Comma list: lob40,full144,supervised_lob40,supervised_full144",
    )
    parser.add_argument("--merge-targets", default="single_y_10,joint_10_50_100,joint_all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--recompute-feature-ranking", action="store_true")
    parser.add_argument("--no-controls", dest="include_controls", action="store_false")
    parser.add_argument("--no-predictive-merge", dest="include_predictive_merge", action="store_false")
    parser.add_argument("--no-mlp", dest="include_mlp", action="store_false")
    parser.add_argument("--no-non-symbolic", dest="include_non_symbolic", action="store_false")
    parser.add_argument("--no-hgb-bins", dest="include_hgb_bins", action="store_false")
    parser.add_argument("--no-tree-product", dest="include_tree_product", action="store_false")
    parser.add_argument("--tree-product-leaf-values", type=parse_int_list, default=parse_int_list("4,8,16"))
    parser.add_argument(
        "--tree-product-horizon-groups",
        default="10_20",
        help="Comma-separated groups like 10_20, all, or y_10+y_20+y_50.",
    )
    parser.add_argument(
        "--max-exact-merge-states",
        type=int,
        default=512,
        help="Skip exact greedy predictive merges when the fine alphabet is larger than this.",
    )
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--mlp-max-iter", type=int, default=100)
    parser.add_argument("--hgb-max-iter", type=int, default=100)
    parser.add_argument("--hgb-max-depth", type=int, default=4)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.feature_modes = {part.strip() for part in args.feature_modes.split(",") if part.strip()}
    if "all" in args.feature_modes:
        args.feature_modes.remove("all")
        args.feature_modes.add("lob40")
    if "supervised" in args.feature_modes:
        args.feature_modes.remove("supervised")
        args.feature_modes.add("supervised_lob40")
    args.merge_targets = [part.strip() for part in args.merge_targets.split(",") if part.strip()]
    args.tree_product_horizon_groups = parse_horizon_groups(args.tree_product_horizon_groups)

    unknown_targets = sorted(set(args.merge_targets) - set(MERGE_TARGETS))
    if unknown_targets:
        raise ValueError(f"Unknown merge targets: {unknown_targets}")
    known_feature_modes = {"lob40", "full144", "supervised_lob40", "supervised_full144"}
    unknown_feature_modes = sorted(args.feature_modes - known_feature_modes)
    if unknown_feature_modes:
        raise ValueError(f"Unknown feature modes: {unknown_feature_modes}")

    data_root = find_data_root()
    output_dir = OUTPUT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"compute_scaling_cf{args.cf}.csv"

    x_train_all, x_test_all, y_train, y_test = load_no_auction_zscore_cf(args.cf, data_root)
    importance_by_source = {}
    if "supervised_lob40" in args.feature_modes:
        importance_by_source["lob40"] = load_or_compute_feature_importance(
            x_train_all.iloc[:, :LOB_RAW_FEATURE_COUNT],
            y_train,
            output_dir,
            args.cf,
            feature_source="lob40",
            recompute=args.recompute_feature_ranking,
        )
    if "supervised_full144" in args.feature_modes:
        importance_by_source["full144"] = load_or_compute_feature_importance(
            x_train_all.iloc[:, :FULL_FEATURE_COUNT],
            y_train,
            output_dir,
            args.cf,
            feature_source="full144",
            recompute=args.recompute_feature_ranking,
        )
    specs = make_feature_specs(x_train_all, importance_by_source, args.feature_modes, args.top_n)

    all_rows = []
    for spec in specs:
        print(f"running feature_set={spec.name} d={len(spec.columns)}")
        rows = run_feature_spec(spec, x_train_all, x_test_all, y_train, y_test, args)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(output_path, index=False)
        print(f"  cumulative rows={len(all_rows)} -> {output_path}")

    results = pd.DataFrame(all_rows)
    results.to_csv(output_path, index=False)
    print(f"wrote {len(results)} rows to {output_path}")
    print(
        results.sort_values("log_loss")
        .head(12)[
            [
                "score_target",
                "feature_set",
                "method_family",
                "method",
                "fit_target",
                "merge_target",
                "K",
                "M",
                "compute_proxy_ops",
                "latency_us_per_obs_median",
                "log_loss",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
