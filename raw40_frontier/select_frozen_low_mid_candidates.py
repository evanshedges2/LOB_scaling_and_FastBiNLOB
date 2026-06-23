#!/usr/bin/env python3
"""Select frozen low/mid-compute candidates from discovery result CSVs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = BUNDLE_ROOT / "shared"
OUTPUT_ROOT = BUNDLE_ROOT / "outputs"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
ARCHITECTURE_SCALING_SCRIPT = SHARED_DIR / "architecture_scaling_frontiers.py"

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_EXCLUDED_ARCHITECTURES = ("torch_deeplob_full",)
DEFAULT_REQUIRED_WINDOWS = (1, 10, 100)


def parse_csv_paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token[:1].lower() == "w":
            token = token[1:]
        values.append(int(token))
    return values


def safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()


def read_result_csvs(input_dirs: list[Path], skipped_output: Path | None = None) -> pd.DataFrame:
    rows = []
    skipped = []
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*.csv")):
            completed_pair = path.name.endswith("_checkpoint.csv") and path.with_name(
                path.name.replace("_checkpoint", "")
            ).exists()
            if completed_pair:
                continue
            try:
                frame = pd.read_csv(path)
            except EmptyDataError:
                skipped.append({"path": str(path), "reason": "empty_csv"})
                continue
            if frame.empty:
                skipped.append({"path": str(path), "reason": "zero_rows"})
                continue
            frame["selection_source_dir"] = input_dir.name
            frame["selection_source_file"] = path.name
            rows.append(frame)
    if not rows:
        raise SystemExit("No non-empty result CSVs found in --input-dirs.")
    combined = pd.concat(rows, ignore_index=True)
    if skipped and skipped_output is not None:
        skipped_frame = pd.DataFrame(skipped)
        skipped_output.parent.mkdir(parents=True, exist_ok=True)
        skipped_frame.to_csv(skipped_output, index=False)
    return combined


def model_family(row: pd.Series) -> str:
    architecture = str(row.get("architecture", ""))
    if architecture == "decision_tree":
        return "tree"
    if architecture == "hist_gradient_boosting":
        return "hgb"
    if architecture == "catboost_symmetric_trees":
        return "catboost"
    if architecture == "explainable_boosting_machine":
        return "ebm"
    if architecture == "random_conv_logistic":
        return "random_conv"
    if architecture == "torch_mlp":
        return "torch_mlp"
    if architecture == "torch_temporal_tcn":
        return "torch_tcn"
    return architecture or "unknown"


def extract_ints(model_name: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", model_name)]


def backend_args_for(row: pd.Series) -> list[str]:
    """Return model-specific backend flags, excluding CF/horizon/window/output flags."""

    architecture = str(row["architecture"])
    model_name = str(row["model_name"])
    values = extract_ints(model_name)

    if architecture == "decision_tree":
        leaves = values[0]
        return [
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--tree-leaf-values",
            str(leaves),
        ]

    if architecture == "hist_gradient_boosting":
        max_iter, leaves = values[:2]
        return [
            "--no-trees",
            "--no-forests",
            "--no-mlp",
            "--no-kernel",
            "--hgb-max-iter-values",
            str(max_iter),
            "--hgb-leaf-values",
            str(leaves),
            "--hgb-learning-rate",
            "0.05",
        ]

    if architecture == "catboost_symmetric_trees":
        iterations, depth = values[:2]
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-catboost",
            "--catboost-iterations",
            str(iterations),
            "--catboost-depths",
            str(depth),
            "--catboost-learning-rate",
            "0.03",
            "--catboost-early-stopping-rounds",
            "60",
        ]

    if architecture == "explainable_boosting_machine":
        rounds, interactions = values[:2]
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-ebm",
            "--ebm-max-rounds",
            str(rounds),
            "--ebm-interactions",
            str(interactions),
            "--ebm-learning-rate",
            "0.01",
            "--ebm-max-bins",
            "256",
        ]

    if architecture == "random_conv_logistic":
        kernels = values[0]
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-random-conv",
            "--random-conv-kernels",
            str(kernels),
            "--random-conv-kernel-sizes",
            "3,5,7,9,15,21,31,45,63",
            "--random-conv-heads",
            "logistic",
            "--random-conv-logistic-max-iter",
            "800",
            "--random-conv-logistic-c",
            "1.0",
        ]

    if architecture == "torch_mlp":
        spec = model_name.removeprefix("torch_mlp_")
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-torch-nn",
            "--torch-optimizer",
            "adam",
            "--torch-lr",
            "0.0003",
            "--torch-weight-decay",
            "0.0",
            "--torch-batch-size",
            "256",
            "--torch-epochs",
            "100",
            "--torch-patience",
            "12",
            "--torch-min-delta",
            "0.00005",
            "--torch-early-stop-metric",
            "loss",
            "--torch-mlp-hidden-layers",
            spec,
            "--torch-residual-hidden-layers",
            "",
            "--torch-fm-ranks",
            "",
            "--torch-tab-transformer-specs",
            "",
            "--torch-cnn-channels",
            "",
            "--torch-inception-channels",
            "",
            "--torch-tcn-channels",
            "",
            "--torch-cnn-attention-channels",
            "",
            "--torch-recurrent-specs",
            "",
            "--torch-deeplob-specs",
            "",
            "--torch-deeplob-full-specs",
            "",
        ]

    if architecture == "torch_temporal_tcn":
        spec = model_name.removeprefix("torch_tcn_")
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-torch-nn",
            "--torch-optimizer",
            "adam",
            "--torch-lr",
            "0.0003",
            "--torch-weight-decay",
            "0.0",
            "--torch-batch-size",
            "64",
            "--torch-epochs",
            "120",
            "--torch-patience",
            "12",
            "--torch-min-delta",
            "0.00005",
            "--torch-early-stop-metric",
            "loss",
            "--torch-mlp-hidden-layers",
            "",
            "--torch-residual-hidden-layers",
            "",
            "--torch-fm-ranks",
            "",
            "--torch-tab-transformer-specs",
            "",
            "--torch-cnn-channels",
            "",
            "--torch-inception-channels",
            "",
            "--torch-tcn-channels",
            spec,
            "--torch-cnn-attention-channels",
            "",
            "--torch-recurrent-specs",
            "",
            "--torch-deeplob-specs",
            "",
            "--torch-deeplob-full-specs",
            "",
        ]

    return []


def prepare_candidates(
    frame: pd.DataFrame,
    excluded_architectures: set[str],
    selection_metric: str,
    group_cols: list[str],
) -> pd.DataFrame:
    if "structural_forward_work_units" in frame.columns:
        work_col = "structural_forward_work_units"
    elif "forward_work_units" in frame.columns:
        work_col = "forward_work_units"
    else:
        work_col = "audited_forward_ops"
    required_cols = {"architecture", "model_name", "input_window", work_col, "validation_log_loss"}
    if selection_metric == "normalized_validation_log_loss":
        required_cols.add("naive_fit_prior_validation_log_loss")
    missing = sorted(required_cols - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required result columns: {missing}")
    missing_group_cols = sorted(set(group_cols) - set(frame.columns))
    if missing_group_cols:
        raise SystemExit(f"Missing requested --group-cols columns: {missing_group_cols}")

    candidates = frame.copy()
    candidates = candidates[~candidates["architecture"].isin(excluded_architectures)]
    candidates["selection_work_units"] = candidates[work_col].astype(float)
    candidates = candidates[pd.notna(candidates["selection_work_units"])]
    candidates = candidates[pd.notna(candidates["validation_log_loss"])]
    candidates = candidates[candidates["selection_work_units"] > 0]
    candidates = candidates[np.isfinite(candidates["selection_work_units"].astype(float))]
    candidates = candidates[np.isfinite(candidates["validation_log_loss"].astype(float))]

    if selection_metric == "normalized_validation_log_loss":
        candidates = candidates[pd.notna(candidates["naive_fit_prior_validation_log_loss"])]
        candidates = candidates[candidates["naive_fit_prior_validation_log_loss"] > 0]
        candidates["selection_loss"] = (
            candidates["validation_log_loss"].astype(float)
            / candidates["naive_fit_prior_validation_log_loss"].astype(float)
        )
    else:
        candidates["selection_loss"] = candidates["validation_log_loss"].astype(float)

    candidates = candidates[np.isfinite(candidates["selection_loss"].astype(float))]
    candidates["selection_metric"] = selection_metric

    key_cols = group_cols + ["input_window", "architecture", "model_name"]
    candidates = candidates.sort_values(["selection_loss", "selection_work_units"])
    candidates = candidates.drop_duplicates(key_cols, keep="first")
    candidates["family"] = candidates.apply(model_family, axis=1)
    candidates["compute_half_decade"] = (np.log10(candidates["selection_work_units"].astype(float)) * 2).round() / 2
    return candidates.reset_index(drop=True)


def lower_envelope(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["selection_work_units", "selection_loss", "model_name"]).copy()
    best = math.inf
    envelope_rows = []
    envelope_order = 0
    for _, row in ordered.iterrows():
        loss = float(row["selection_loss"])
        if loss < best - 1e-12:
            envelope_order += 1
            out = row.copy()
            out["envelope_order"] = envelope_order
            envelope_rows.append(out)
            best = loss
    if not envelope_rows:
        return pd.DataFrame(columns=list(frame.columns) + ["envelope_order"])
    return pd.DataFrame(envelope_rows)


def repeated_envelopes_one_group(frame: pd.DataFrame, min_candidates: int, min_rounds: int) -> pd.DataFrame:
    remaining = frame.copy()
    selected = []
    round_index = 0
    while len(remaining) and (round_index < min_rounds or len(selected) < min_candidates):
        round_index += 1
        envelope = lower_envelope(remaining)
        if envelope.empty:
            break
        envelope["selection_round"] = round_index
        selected.append(envelope)
        envelope_keys = set(zip(envelope["input_window"], envelope["architecture"], envelope["model_name"]))
        keep_mask = [
            (row.input_window, row.architecture, row.model_name) not in envelope_keys
            for row in remaining.itertuples(index=False)
        ]
        remaining = remaining.loc[keep_mask].copy()
    if not selected:
        return pd.DataFrame()
    pool = pd.concat(selected, ignore_index=True)
    pool["pool_order"] = np.arange(1, len(pool) + 1)
    return pool


def repeated_envelopes(frame: pd.DataFrame, min_candidates: int, min_rounds: int, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        pool = repeated_envelopes_one_group(frame, min_candidates=min_candidates, min_rounds=min_rounds)
        pool["selection_group"] = "all"
        return pool

    selected = []
    groupby_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for group_key, group_frame in frame.groupby(groupby_key, sort=True, dropna=False):
        pool = repeated_envelopes_one_group(group_frame, min_candidates=min_candidates, min_rounds=min_rounds)
        if pool.empty:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_label = "|".join(f"{col}={value}" for col, value in zip(group_cols, group_key))
        pool["selection_group"] = group_label
        selected.append(pool)
    if not selected:
        return pd.DataFrame()
    pool = pd.concat(selected, ignore_index=True)
    pool["pool_order"] = np.arange(1, len(pool) + 1)
    return pool


def add_row(selected: list[pd.Series], keys: set[tuple[int, str, str]], row: pd.Series) -> None:
    key = (int(row["input_window"]), str(row["architecture"]), str(row["model_name"]))
    if key in keys:
        return
    selected.append(row)
    keys.add(key)


def prune_pool(
    pool: pd.DataFrame,
    min_candidates: int,
    max_candidates: int,
    min_improvement: float,
    required_windows: set[int],
    sentinel_architectures: set[str],
) -> pd.DataFrame:
    ordered = pool.sort_values(["selection_group", "selection_round", "envelope_order", "selection_work_units"]).copy()
    selected: list[pd.Series] = []
    selected_keys: set[tuple[int, str, str]] = set()

    # Keep broad compute-band coverage first.
    for _, row in ordered.sort_values(["compute_half_decade", "selection_round", "selection_loss"]).iterrows():
        band = row["compute_half_decade"]
        if any(existing["compute_half_decade"] == band for existing in selected):
            continue
        add_row(selected, selected_keys, row)

    # Keep material improvements along each round's envelope.
    for _, round_frame in ordered.groupby("selection_round", sort=True):
        last_loss: float | None = None
        for _, row in round_frame.sort_values(["envelope_order", "selection_work_units"]).iterrows():
            loss = float(row["selection_loss"])
            if last_loss is None or last_loss - loss >= min_improvement:
                add_row(selected, selected_keys, row)
                last_loss = loss

    # Ensure requested window coverage.
    for window in sorted(required_windows):
        if any(int(row["input_window"]) == window for row in selected):
            continue
        subset = pool[pool["input_window"] == window]
        if len(subset):
            add_row(selected, selected_keys, subset.sort_values(["selection_loss", "selection_work_units"]).iloc[0])

    # Optional weak-family sentinels.
    for architecture in sorted(sentinel_architectures):
        if any(str(row["architecture"]) == architecture for row in selected):
            continue
        subset = pool[pool["architecture"] == architecture]
        if len(subset):
            add_row(selected, selected_keys, subset.sort_values(["selection_loss", "selection_work_units"]).iloc[0])

    # Fill to minimum count by validation quality, then by envelope order.
    for _, row in pool.sort_values(["selection_loss", "selection_work_units"]).iterrows():
        if len(selected) >= min_candidates:
            break
        add_row(selected, selected_keys, row)

    pruned = pd.DataFrame(selected)
    if len(pruned) > max_candidates:
        pruned = pruned.sort_values(["selection_round", "envelope_order", "selection_loss"]).head(max_candidates)

    pruned = pruned.sort_values(["selection_work_units", "validation_log_loss", "selection_round"]).reset_index(drop=True)
    pruned["candidate_rank"] = np.arange(1, len(pruned) + 1)
    return pruned


def finalize_output(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    backend_args_json = []
    backend_args_shell = []
    warnings = []
    for _, row in output.iterrows():
        args = backend_args_for(row)
        backend_args_json.append(json.dumps(args))
        backend_args_shell.append(" ".join(shlex.quote(part) for part in args))
        warnings.append("" if args else "unknown_backend_args")
    output["backend_args"] = backend_args_json
    output["backend_args_shell"] = backend_args_shell
    output["selection_warning"] = warnings
    output["candidate_id"] = [
        f"cand_{rank:03d}_{safe_id(row.architecture)}_w{int(row.input_window)}_{safe_id(row.model_name)}"
        for rank, row in enumerate(output.itertuples(index=False), start=1)
    ]
    output["selection_source"] = output["selection_source_dir"].astype(str) + "/" + output["selection_source_file"].astype(str)
    output = output.rename(
        columns={
            "validation_log_loss": "selection_validation_log_loss",
            "audited_forward_ops": "selection_audited_forward_ops",
            "structural_forward_work_units": "selection_structural_forward_work_units",
            "forward_work_units": "selection_forward_work_units",
        }
    )
    if "selection_structural_forward_work_units" not in output.columns:
        output["selection_structural_forward_work_units"] = output["selection_work_units"]
    if "selection_forward_work_units" not in output.columns:
        output["selection_forward_work_units"] = output["selection_work_units"]

    preferred = [
        "candidate_id",
        "candidate_rank",
        "selection_round",
        "envelope_order",
        "selection_group",
        "input_window",
        "family",
        "architecture",
        "model_name",
        "selection_structural_forward_work_units",
        "selection_forward_work_units",
        "selection_audited_forward_ops",
        "selection_metric",
        "selection_loss",
        "selection_validation_log_loss",
        "naive_fit_prior_validation_log_loss",
        "validation_accuracy",
        "test_log_loss",
        "test_accuracy",
        "test_weighted_f1",
        "latency_us_per_obs_median",
        "n_parameters",
        "compute_half_decade",
        "selection_source",
        "backend_args",
        "backend_args_shell",
        "selection_warning",
    ]
    existing = [col for col in preferred if col in output.columns]
    remaining = [col for col in output.columns if col not in existing]
    return output[existing + remaining]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        required=True,
        help="Comma-separated discovery result directories.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT / "manifests" / "frozen_low_mid_candidates.csv"),
        help="Frozen candidate CSV path.",
    )
    parser.add_argument(
        "--audit-output",
        default=str(OUTPUT_ROOT / "manifests" / "frozen_low_mid_candidate_selection_audit.csv"),
        help="Full repeated-envelope audit CSV path.",
    )
    parser.add_argument(
        "--skipped-output",
        default=str(OUTPUT_ROOT / "manifests" / "frozen_low_mid_candidate_selection_skipped_inputs.csv"),
        help="CSV path for empty/skipped discovery inputs.",
    )
    parser.add_argument("--min-candidates", type=int, default=15)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--min-rounds", type=int, default=2)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument(
        "--selection-metric",
        choices=["validation_log_loss", "normalized_validation_log_loss"],
        default="validation_log_loss",
        help="Metric used to construct validation envelopes.",
    )
    parser.add_argument(
        "--group-cols",
        default="score_target",
        help="Comma-separated columns for independent envelope selection before unioning candidates. Use empty string for global.",
    )
    parser.add_argument("--exclude-architectures", default=",".join(DEFAULT_EXCLUDED_ARCHITECTURES))
    parser.add_argument("--required-windows", default=",".join(str(value) for value in DEFAULT_REQUIRED_WINDOWS))
    parser.add_argument(
        "--sentinel-architectures",
        default="",
        help="Optional comma-separated architectures to force in if present, e.g. explainable_boosting_machine.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dirs = parse_csv_paths(args.input_dirs)
    excluded_architectures = set(parse_csv_strings(args.exclude_architectures))
    required_windows = set(parse_csv_ints(args.required_windows))
    sentinel_architectures = set(parse_csv_strings(args.sentinel_architectures))
    group_cols = parse_csv_strings(args.group_cols)

    raw = read_result_csvs(input_dirs, skipped_output=Path(args.skipped_output))
    candidates = prepare_candidates(
        raw,
        excluded_architectures=excluded_architectures,
        selection_metric=args.selection_metric,
        group_cols=group_cols,
    )
    pool = repeated_envelopes(
        candidates,
        min_candidates=args.min_candidates,
        min_rounds=args.min_rounds,
        group_cols=group_cols,
    )
    if pool.empty:
        raise SystemExit("No validation-envelope candidates selected.")

    pruned = prune_pool(
        pool,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        min_improvement=args.min_improvement,
        required_windows=required_windows,
        sentinel_architectures=sentinel_architectures,
    )

    audit = finalize_output(
        pool.sort_values(["selection_group", "selection_round", "envelope_order", "selection_work_units"]).reset_index(drop=True)
    )
    frozen = finalize_output(pruned)

    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)

    print(f"read_rows={len(raw)}")
    print(f"eligible_unique_candidates={len(candidates)}")
    print(f"audit_candidates={len(audit)}")
    print(f"frozen_candidates={len(frozen)}")
    print(f"wrote {output_path}")
    print(f"wrote {audit_path}")
    print(frozen[[
        "candidate_id",
        "input_window",
        "architecture",
        "model_name",
        "selection_forward_work_units",
        "selection_audited_forward_ops",
        "selection_loss",
        "selection_validation_log_loss",
        "test_log_loss",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
