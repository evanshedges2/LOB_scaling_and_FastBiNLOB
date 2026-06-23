#!/usr/bin/env python3
"""Compare MSE and lower-bound frontier fits for held-out MLPLOB targets."""

from __future__ import annotations

import argparse
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
from scipy.optimize import curve_fit

from analyze_fi2010_raw40_frontier_confirmatory import (
    fit_floor,
    load_results,
    lower_envelope,
    power_law_floor,
    predict_fit,
    r2_score,
)
from structural_forward_work import apply_structural_forward_work


DEFAULT_LOW_MID_DIRS = str(OUTPUT_ROOT / "results")
DEFAULT_TARGET_DIRS = str(OUTPUT_ROOT / "results")
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "analysis" / "raw40_mlplob_mse_frontier"
DEFAULT_CUTOFFS = "3000,10000,30000,100000"
DEFAULT_METRICS = "test_log_loss,validation_log_loss"
DEFAULT_AXIS_COL = "structural_forward_work_units"


def parse_paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def canonicalize_forward_work(frame: pd.DataFrame) -> pd.DataFrame:
    frame = apply_structural_forward_work(frame)
    if "forward_work_units" not in frame.columns:
        frame["forward_work_units"] = frame.get("audited_forward_ops", np.nan)
    if "forward_work_method" not in frame.columns:
        frame["forward_work_method"] = frame.get("compute_audit_method", "legacy_audited_forward_ops")
    if "forward_work_confidence" not in frame.columns:
        frame["forward_work_confidence"] = frame.get("compute_audit_confidence", "legacy")
    return frame


def lower_envelope_axis(group: pd.DataFrame, metric_col: str, axis_col: str) -> pd.DataFrame:
    ordered = group.dropna(subset=[axis_col, metric_col]).sort_values([axis_col, metric_col, "model_name"])
    rows = []
    best = np.inf
    for _, row in ordered.iterrows():
        value = float(row[metric_col])
        if float(row[axis_col]) <= 0:
            continue
        if value < best - 1e-12:
            rows.append(row)
            best = value
    return pd.DataFrame(rows)


def fit_mse_floor(frontier: pd.DataFrame, metric_col: str, cutoff: float, axis_col: str) -> dict:
    scoped = frontier[frontier[axis_col].le(cutoff)].dropna(subset=[metric_col]).copy()
    scoped = scoped[scoped[axis_col].gt(0)]
    base = {
        "cutoff": cutoff,
        "metric": metric_col,
        "axis_col": axis_col,
        "n_fit_points": int(len(scoped)),
        "x_min_fit": np.nan,
        "x_max_fit": np.nan,
        "axis_scale": np.nan,
        "param_0": np.nan,
        "param_1": np.nan,
        "param_2": np.nan,
        "fit_rmse": np.nan,
        "error": "",
    }
    if len(scoped) < 3 or scoped[axis_col].nunique() < 3:
        base["error"] = "need at least three unique frontier points"
        return base

    scale = float(np.median(scoped[axis_col]))
    x = scoped[axis_col].to_numpy(dtype=float) / scale
    y = scoped[metric_col].to_numpy(dtype=float)
    base.update(
        {
            "x_min_fit": float(scoped[axis_col].min()),
            "x_max_fit": float(scoped[axis_col].max()),
            "axis_scale": scale,
        }
    )
    lower = np.array([0.0, 0.0, 0.0], dtype=float)
    upper = np.array([max(1e-12, float(np.min(y))), 10.0, 10.0], dtype=float)
    p0 = np.array(
        [
            max(0.0, float(np.min(y)) - 0.02),
            max(1e-6, float(np.max(y) - np.min(y))),
            0.1,
        ],
        dtype=float,
    )
    p0 = np.minimum(np.maximum(p0, lower), upper)
    try:
        params, _ = curve_fit(
            power_law_floor,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000,
        )
        pred = power_law_floor(x, *params)
        base.update(
            {
                "param_0": float(params[0]),
                "param_1": float(params[1]),
                "param_2": float(params[2]),
                "fit_rmse": float(np.sqrt(np.mean(np.square(pred - y)))),
            }
        )
    except Exception as exc:  # noqa: BLE001
        base["error"] = str(exc)
    return base


def predict_rows(
    low_mid: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    metrics: list[str],
    cutoffs: list[float],
    axis_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    envelope_rows = []
    fit_rows = []
    prediction_rows = []

    for (cf, horizon), group in low_mid.groupby(["cf", "score_target"], sort=True):
        for metric in metrics:
            if metric not in group.columns:
                continue
            envelope = lower_envelope_axis(group, metric, axis_col)
            if envelope.empty:
                continue
            envelope = envelope.copy()
            envelope["frontier_metric"] = metric
            envelope["frontier_axis"] = axis_col
            envelope_rows.append(envelope)

            scoped_targets = targets[
                targets["cf"].eq(int(cf))
                & targets["score_target"].eq(horizon)
                & targets[axis_col].notna()
            ].copy()
            if metric not in scoped_targets.columns:
                continue
            scoped_targets = scoped_targets[scoped_targets[metric].notna()].copy()
            if scoped_targets.empty:
                continue

            for cutoff in cutoffs:
                lower_bound_input = envelope.copy()
                if axis_col != "audited_forward_ops":
                    lower_bound_input["audited_forward_ops"] = lower_bound_input[axis_col]
                for fit_kind, fit in [
                    ("mse", fit_mse_floor(envelope, metric, cutoff, axis_col)),
                    ("lower_bound", fit_floor(lower_bound_input, metric, cutoff)),
                ]:
                    fit["axis_col"] = axis_col
                    fit.update({"cf": int(cf), "score_target": horizon, "fit_kind": fit_kind})
                    fit_rows.append(fit.copy())
                    if fit.get("error"):
                        continue
                    pred = predict_fit(pd.Series(fit), scoped_targets[axis_col])
                    rows = scoped_targets.copy()
                    rows["fit_kind"] = fit_kind
                    rows["metric"] = metric
                    rows["axis_col"] = axis_col
                    rows["cutoff"] = float(cutoff)
                    rows["observed"] = rows[metric].astype(float)
                    rows["predicted"] = pred
                    rows["residual"] = rows["observed"] - rows["predicted"]
                    rows["fit_rmse"] = float(fit["fit_rmse"])
                    rows["n_fit_points"] = int(fit["n_fit_points"])
                    prediction_rows.append(rows)

    envelopes = pd.concat(envelope_rows, ignore_index=True, sort=False) if envelope_rows else pd.DataFrame()
    fits = pd.DataFrame(fit_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True, sort=False) if prediction_rows else pd.DataFrame()
    return envelopes, fits, predictions


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, cutoff, fit_kind), group in predictions.groupby(["metric", "cutoff", "fit_kind"], sort=True):
        residual = group["residual"].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "cutoff": float(cutoff),
                "fit_kind": fit_kind,
                "n": int(len(group)),
                "r2": r2_score(group["observed"], group["predicted"]),
                "mae": float(np.mean(np.abs(residual))),
                "mean_residual": float(np.mean(residual)),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "within_0.05": float(np.mean(np.abs(residual) <= 0.05)),
            }
        )
    return pd.DataFrame(rows)


def full_frontier_in_sample(
    low_mid: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    metrics: list[str],
    axis_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([low_mid, targets], ignore_index=True, sort=False)
    fit_rows = []
    point_rows = []

    for (cf, horizon), group in combined.groupby(["cf", "score_target"], sort=True):
        for metric in metrics:
            if metric not in group.columns:
                continue
            envelope = lower_envelope_axis(group, metric, axis_col)
            if len(envelope) < 3 or envelope[axis_col].nunique() < 3:
                continue
            fit = fit_mse_floor(envelope, metric, float("inf"), axis_col)
            fit.update(
                {
                    "cf": int(cf),
                    "score_target": horizon,
                    "fit_kind": "mse_full_frontier",
                    "fit_scope": "full_frontier_in_sample",
                    "contains_mlplob": bool(envelope["architecture"].eq("torch_mlplob").any()),
                    "n_mlplob_points": int(envelope["architecture"].eq("torch_mlplob").sum()),
                }
            )
            if not fit.get("error"):
                pred = predict_fit(pd.Series(fit), envelope[axis_col])
                observed = envelope[metric].astype(float)
                residual = observed.to_numpy(dtype=float) - pred
                fit["fit_r2"] = r2_score(observed, pd.Series(pred, index=envelope.index))
                fit["fit_mae"] = float(np.mean(np.abs(residual)))
                points = envelope.copy()
                points["metric"] = metric
                points["axis_col"] = axis_col
                points["fit_kind"] = "mse_full_frontier"
                points["fit_scope"] = "full_frontier_in_sample"
                points["is_mlplob_target"] = points["architecture"].eq("torch_mlplob")
                points["observed"] = observed
                points["predicted"] = pred
                points["residual"] = residual
                point_rows.append(points)
            fit_rows.append(fit)

    fits = pd.DataFrame(fit_rows)
    points = pd.concat(point_rows, ignore_index=True, sort=False) if point_rows else pd.DataFrame()
    summary_rows = []
    for metric, group in points.groupby("metric", sort=True):
        residual = group["residual"].to_numpy(dtype=float)
        metric_fits = fits[fits["metric"].eq(metric) & fits["error"].fillna("").eq("")]
        summary_rows.append(
            {
                "metric": metric,
                "axis_col": axis_col,
                "fit_kind": "mse_full_frontier",
                "n_curves": int(metric_fits[["cf", "score_target"]].drop_duplicates().shape[0]),
                "n_frontier_points": int(len(group)),
                "n_mlplob_points": int(group["architecture"].eq("torch_mlplob").sum()),
                "pooled_r2": r2_score(group["observed"], group["predicted"]),
                "pooled_mae": float(np.mean(np.abs(residual))),
                "pooled_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mean_residual": float(np.mean(residual)),
                "median_curve_r2": float(metric_fits["fit_r2"].median()),
                "min_curve_r2": float(metric_fits["fit_r2"].min()),
                "max_curve_r2": float(metric_fits["fit_r2"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return fits, points, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-mid-dirs", default=DEFAULT_LOW_MID_DIRS)
    parser.add_argument("--target-dirs", default=DEFAULT_TARGET_DIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoffs", default=DEFAULT_CUTOFFS)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--axis-col", default=DEFAULT_AXIS_COL)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_mid = canonicalize_forward_work(load_results(parse_paths(args.low_mid_dirs)))
    targets = canonicalize_forward_work(load_results(parse_paths(args.target_dirs)))
    targets = targets[targets["architecture"].eq("torch_mlplob")].copy()
    if args.axis_col not in low_mid.columns:
        raise ValueError(f"low/mid results do not contain axis column: {args.axis_col}")
    if args.axis_col not in targets.columns:
        raise ValueError(f"target results do not contain axis column: {args.axis_col}")

    metrics = parse_strings(args.metrics)
    cutoffs = parse_floats(args.cutoffs)
    envelopes, fits, predictions = predict_rows(
        low_mid,
        targets,
        metrics=metrics,
        cutoffs=cutoffs,
        axis_col=args.axis_col,
    )
    summary = summarize(predictions)
    full_fits, full_points, full_summary = full_frontier_in_sample(
        low_mid,
        targets,
        metrics=metrics,
        axis_col=args.axis_col,
    )

    low_mid.to_csv(args.output_dir / "combined_low_mid_results.csv", index=False)
    targets.to_csv(args.output_dir / "combined_mlplob_targets.csv", index=False)
    envelopes.to_csv(args.output_dir / "mse_extrapolation_frontier_points.csv", index=False)
    fits.to_csv(args.output_dir / "mse_vs_lower_bound_power_law_fit_summary.csv", index=False)
    full_fits.to_csv(args.output_dir / "full_frontier_in_sample_power_law_fits.csv", index=False)
    full_points.to_csv(args.output_dir / "full_frontier_in_sample_power_law_points.csv", index=False)
    full_summary.to_csv(args.output_dir / "full_frontier_in_sample_power_law_summary.csv", index=False)
    predictions[
        [
            "fit_kind",
            "cf",
            "score_target",
            "metric",
            "cutoff",
            "observed",
            "predicted",
            "residual",
            "fit_rmse",
            "n_fit_points",
            "axis_col",
            args.axis_col,
            "audited_forward_ops",
            "model_name",
        ]
    ].to_csv(args.output_dir / "mlplob_mse_vs_lower_bound_power_law_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "mlplob_mse_vs_lower_bound_power_law_summary.csv", index=False)

    print(f"wrote {args.output_dir}")
    print(summary.sort_values(["metric", "cutoff", "fit_kind"]).to_string(index=False))
    if not full_summary.empty:
        print("\nFull-frontier in-sample power-law fit:")
        print(full_summary.sort_values(["metric"]).to_string(index=False))


if __name__ == "__main__":
    main()
