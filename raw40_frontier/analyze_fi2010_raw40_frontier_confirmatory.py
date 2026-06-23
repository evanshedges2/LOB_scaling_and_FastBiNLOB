#!/usr/bin/env python3
"""Analyze clean FI-2010 raw40 confirmatory frontier results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = BUNDLE_ROOT / "shared"
OUTPUT_ROOT = BUNDLE_ROOT / "outputs"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
ARCHITECTURE_SCALING_SCRIPT = SHARED_DIR / "architecture_scaling_frontiers.py"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize


CUTOFFS = (3e3, 1e4, 3e4, 1e5, 3e5, 1e6)
RAW_REPRESENTATIONS = {"raw_lob", "raw_lob_random_temporal_basis"}
DEEPLOB_ARCH = "torch_deeplob_full"
LOWER_BOUND_TOL = 1e-6
LOG_LOSS_METRICS = ("validation_log_loss", "test_log_loss")
NORMALIZED_LOG_LOSS_METRICS = (
    "prior_normalized_validation_log_loss",
    "prior_normalized_test_log_loss",
)
FIT_METRICS = LOG_LOSS_METRICS + NORMALIZED_LOG_LOSS_METRICS


def power_law_floor(x: np.ndarray, floor: float, amp: float, alpha: float) -> np.ndarray:
    return floor + amp * np.power(x, -alpha)


def parse_cutoffs(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def metric_label(metric_col: str) -> str:
    labels = {
        "validation_log_loss": "validation log loss",
        "test_log_loss": "test log loss",
        "prior_normalized_validation_log_loss": "validation log loss / prior",
        "prior_normalized_test_log_loss": "test log loss / prior",
    }
    return labels.get(metric_col, metric_col)


def load_results(input_dirs: list[Path], *, include_checkpoints: bool = False) -> pd.DataFrame:
    paths = sorted(
        path
        for input_dir in input_dirs
        for path in input_dir.glob("*.csv")
        if include_checkpoints or not path.name.endswith("_checkpoint.csv")
    )
    if not paths:
        joined = ", ".join(str(path) for path in input_dirs)
        raise FileNotFoundError(f"no result CSVs found under {joined}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True, sort=False)
    results = results[results["feature_set"].eq("lob_raw_40")].copy()
    if "representation_kind" in results.columns:
        results["representation_kind"] = results["representation_kind"].fillna("raw_lob")
        results = results[results["representation_kind"].isin(RAW_REPRESENTATIONS)].copy()
    results = results.dropna(subset=["cf", "score_target", "audited_forward_ops"])
    results["cf"] = results["cf"].astype(int)
    results["audited_forward_ops"] = results["audited_forward_ops"].astype(float)
    for metric in LOG_LOSS_METRICS:
        if metric in results.columns:
            results[metric] = results[metric].astype(float)
    if {"validation_log_loss", "naive_fit_prior_validation_log_loss"}.issubset(results.columns):
        results["prior_normalized_validation_log_loss"] = (
            results["validation_log_loss"] / results["naive_fit_prior_validation_log_loss"]
        )
    if {"test_log_loss", "naive_fit_prior_test_log_loss"}.issubset(results.columns):
        results["prior_normalized_test_log_loss"] = results["test_log_loss"] / results["naive_fit_prior_test_log_loss"]
    return results


def lower_envelope(group: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    ordered = group.dropna(subset=["audited_forward_ops", metric_col]).sort_values(
        ["audited_forward_ops", metric_col, "model_name"]
    )
    rows = []
    best = np.inf
    for _, row in ordered.iterrows():
        ops = float(row["audited_forward_ops"])
        metric = float(row[metric_col])
        if ops <= 0:
            continue
        if metric < best - 1e-12:
            rows.append(row)
            best = metric
    return pd.DataFrame(rows)


def fit_floor(frontier: pd.DataFrame, metric_col: str, cutoff: float) -> dict:
    scoped = frontier[frontier["audited_forward_ops"].le(cutoff)].dropna(subset=[metric_col]).copy()
    scoped = scoped[scoped["audited_forward_ops"].gt(0)]
    base = {
        "cutoff": cutoff,
        "metric": metric_col,
        "n_fit_points": len(scoped),
        "x_min_fit": np.nan,
        "x_max_fit": np.nan,
        "axis_scale": np.nan,
        "param_0": np.nan,
        "param_1": np.nan,
        "param_2": np.nan,
        "fit_rmse": np.nan,
        "constraint_violation": np.nan,
        "error": "",
    }
    if len(scoped) < 3 or scoped["audited_forward_ops"].nunique() < 3:
        base["error"] = "need at least three unique frontier points"
        return base
    scale = float(np.median(scoped["audited_forward_ops"]))
    x = scoped["audited_forward_ops"].to_numpy(dtype=float) / scale
    y = scoped[metric_col].to_numpy(dtype=float)
    base.update({"x_min_fit": float(scoped["audited_forward_ops"].min()), "x_max_fit": float(scoped["audited_forward_ops"].max()), "axis_scale": scale})
    try:
        lower = np.array([0.0, 0.0, 0.0], dtype=float)
        upper = np.array([float(y.min()), 10.0, 10.0], dtype=float)
        bounds = list(zip(lower, upper))
        p0 = [max(0.0, float(y.min()) - 0.02), max(1e-6, float(y.max() - y.min())), 0.1]
        try:
            initial, _ = curve_fit(
                power_law_floor,
                x,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=100000,
            )
        except Exception:
            initial = np.asarray(p0, dtype=float)
        params0 = np.minimum(np.maximum(initial, lower), upper)

        # Constrain the power law to be a lower bound on every fitted empirical
        # frontier point, then choose the closest curve from below. This is the
        # "greatest lower bound" object for our parametric family.
        for _ in range(80):
            if np.all(power_law_floor(x, *params0) <= y + LOWER_BOUND_TOL):
                break
            params0[1] *= 0.8
            params0[0] = max(lower[0], params0[0] - 0.01)
            params0 = np.minimum(np.maximum(params0, lower), upper)

        def objective(params: np.ndarray) -> float:
            residual = y - power_law_floor(x, *params)
            return float(np.mean(np.square(residual)))

        constraints = [
            {"type": "ineq", "fun": lambda params, idx=idx: y[idx] - power_law_floor(np.array([x[idx]]), *params)[0]}
            for idx in range(len(x))
        ]
        result = minimize(
            objective,
            params0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 10000},
        )
        if not result.success:
            base["error"] = str(result.message)
            return base
        params = np.asarray(result.x, dtype=float)
        pred = power_law_floor(x, *params)
        violation = float(np.max(pred - y))
        if violation > LOWER_BOUND_TOL:
            base["error"] = f"lower-envelope constraint violated by {violation:g}"
            return base
        base.update(
            {
                "param_0": float(params[0]),
                "param_1": float(params[1]),
                "param_2": float(params[2]),
                "fit_rmse": float(np.sqrt(np.mean(np.square(pred - y)))),
                "constraint_violation": violation,
            }
        )
    except Exception as exc:  # noqa: BLE001
        base["error"] = str(exc)
    return base


def predict_fit(fit: pd.Series, ops: pd.Series) -> np.ndarray:
    x = ops.to_numpy(dtype=float) / float(fit["axis_scale"])
    return power_law_floor(x, float(fit["param_0"]), float(fit["param_1"]), float(fit["param_2"]))


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    y = y_true.to_numpy(dtype=float)
    pred = y_pred.to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 2:
        return np.nan
    y = y[mask]
    pred = pred[mask]
    denom = float(np.sum(np.square(y - np.mean(y))))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum(np.square(y - pred)) / denom)


def compute_frontiers_and_fits(results: pd.DataFrame, cutoffs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    envelope_rows = []
    fit_rows = []
    non_deeplob = results[~results["architecture"].eq(DEEPLOB_ARCH)].copy()
    metric_cols = [metric for metric in FIT_METRICS if metric in non_deeplob.columns]
    for (cf, horizon), group in non_deeplob.groupby(["cf", "score_target"], sort=True):
        for metric_col in metric_cols:
            envelope = lower_envelope(group, metric_col)
            if envelope.empty:
                continue
            envelope = envelope.copy()
            envelope["frontier_metric"] = metric_col
            envelope_rows.append(envelope)
            for cutoff in cutoffs:
                fit = fit_floor(envelope, metric_col, cutoff)
                fit.update({"cf": cf, "score_target": horizon})
                fit_rows.append(fit)
    envelopes = pd.concat(envelope_rows, ignore_index=True, sort=False) if envelope_rows else pd.DataFrame()
    fits = pd.DataFrame(fit_rows)
    return envelopes, fits


def compute_deeplob_predictions(results: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    deeplob = results[results["architecture"].eq(DEEPLOB_ARCH)].copy()
    rows = []
    valid_fits = fits[fits["error"].fillna("").eq("")].copy()
    for _, fit in valid_fits.iterrows():
        scoped = deeplob[
            deeplob["cf"].eq(int(fit["cf"]))
            & deeplob["score_target"].eq(fit["score_target"])
        ].copy()
        if scoped.empty:
            continue
        metric = str(fit["metric"])
        if metric not in scoped.columns:
            continue
        scoped = scoped.dropna(subset=[metric, "audited_forward_ops"])
        if scoped.empty:
            continue
        scoped["predicted_metric"] = predict_fit(fit, scoped["audited_forward_ops"])
        scoped["observed_metric"] = scoped[metric].astype(float)
        scoped["prediction_residual"] = scoped["observed_metric"] - scoped["predicted_metric"]
        scoped["fit_cutoff"] = float(fit["cutoff"])
        scoped["fit_metric"] = metric
        scoped["fit_param_0"] = float(fit["param_0"])
        scoped["fit_param_1"] = float(fit["param_1"])
        scoped["fit_param_2"] = float(fit["param_2"])
        rows.append(scoped)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def compute_r2_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_specs = [
        ("pooled", []),
        ("by_horizon", ["score_target"]),
        ("by_cf", ["cf"]),
        ("by_horizon_cf", ["score_target", "cf"]),
    ]
    for (metric, cutoff), metric_group in predictions.groupby(["fit_metric", "fit_cutoff"], sort=True):
        for scope, keys in group_specs:
            if keys:
                grouped = metric_group.groupby(keys, sort=True)
            else:
                grouped = [((), metric_group)]
            for key_values, group in grouped:
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                row = {
                    "fit_metric": metric,
                    "fit_cutoff": cutoff,
                    "scope": scope,
                    "n": int(len(group)),
                    "r2": r2_score(group["observed_metric"], group["predicted_metric"]),
                    "mae": float(np.mean(np.abs(group["prediction_residual"]))),
                    "rmse": float(np.sqrt(np.mean(np.square(group["prediction_residual"])))),
                }
                for key, value in zip(keys, key_values):
                    row[key] = value
                rows.append(row)
    return pd.DataFrame(rows)


def plot_frontier_grid(envelopes: pd.DataFrame, output: Path, metric_col: str) -> None:
    metric_env = envelopes[envelopes["frontier_metric"].eq(metric_col)].copy()
    if metric_env.empty:
        return
    horizons = sorted(metric_env["score_target"].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(9, 3.1 * len(horizons)), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab10")
    for ax, horizon in zip(axes, horizons):
        scoped = metric_env[metric_env["score_target"].eq(horizon)]
        for idx, (cf, group) in enumerate(scoped.groupby("cf", sort=True)):
            ax.plot(
                group["audited_forward_ops"],
                group[metric_col],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=cmap((int(cf) - 1) % 10),
                label=f"CF{int(cf)}",
                alpha=0.9,
            )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(horizon)
        ax.set_ylabel(metric_label(metric_col))
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("audited forward ops per observation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_power_law_curve_grid(
    envelopes: pd.DataFrame,
    fits: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
    metric_col: str,
    cutoff: float,
    x_max: float,
) -> None:
    metric_env = envelopes[envelopes["frontier_metric"].eq(metric_col)].copy()
    metric_fits = fits[
        fits["metric"].eq(metric_col)
        & fits["cutoff"].eq(float(cutoff))
        & fits["error"].fillna("").eq("")
    ].copy()
    if metric_env.empty or metric_fits.empty:
        return
    horizons = sorted(metric_env["score_target"].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(9, 3.2 * len(horizons)), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab10")
    for ax, horizon in zip(axes, horizons):
        horizon_env = metric_env[metric_env["score_target"].eq(horizon)]
        for cf, group in horizon_env.groupby("cf", sort=True):
            color = cmap((int(cf) - 1) % 10)
            ax.scatter(
                group["audited_forward_ops"],
                group[metric_col],
                s=18,
                color=color,
                alpha=0.85,
                label=f"CF{int(cf)} envelope",
            )
            fit_rows = metric_fits[
                metric_fits["cf"].eq(int(cf))
                & metric_fits["score_target"].eq(horizon)
            ]
            if fit_rows.empty:
                continue
            fit = fit_rows.iloc[0]
            x_min = max(float(fit["x_min_fit"]), 1e-12)
            x_hi = max(float(x_max), float(horizon_env["audited_forward_ops"].max()))
            xs = np.logspace(np.log10(x_min), np.log10(x_hi), 240)
            ys = power_law_floor(
                xs / float(fit["axis_scale"]),
                float(fit["param_0"]),
                float(fit["param_1"]),
                float(fit["param_2"]),
            )
            ax.plot(xs, ys, color=color, linewidth=1.25, alpha=0.95)
            ax.axvline(float(fit["x_max_fit"]), color=color, linewidth=0.7, alpha=0.22)
            if not predictions.empty and "fit_metric" in predictions.columns:
                deeplob = predictions[
                    predictions["fit_metric"].eq(metric_col)
                    & predictions["fit_cutoff"].eq(float(cutoff))
                    & predictions["cf"].eq(int(cf))
                    & predictions["score_target"].eq(horizon)
                ]
                if not deeplob.empty:
                    ax.scatter(
                        deeplob["audited_forward_ops"],
                        deeplob["observed_metric"],
                        s=38,
                        marker="X",
                        color=color,
                        edgecolor="black",
                        linewidth=0.45,
                        alpha=0.9,
                        label=f"CF{int(cf)} DeepLOB",
                    )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{horizon} fit cutoff <= {cutoff:g} ops")
        ax.set_ylabel(metric_label(metric_col))
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("audited forward ops per observation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_predicted_vs_observed(predictions: pd.DataFrame, r2_table: pd.DataFrame, output: Path, metric_col: str) -> None:
    if predictions.empty or "fit_metric" not in predictions.columns:
        return
    if r2_table.empty or "fit_metric" not in r2_table.columns:
        return
    scoped = predictions[predictions["fit_metric"].eq(metric_col)].copy()
    pooled = r2_table[(r2_table["fit_metric"].eq(metric_col)) & (r2_table["scope"].eq("pooled"))].copy()
    if scoped.empty or pooled.empty:
        return
    cutoffs = sorted(scoped["fit_cutoff"].unique())
    ncols = 3
    nrows = int(np.ceil(len(cutoffs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, cutoff in zip(axes, cutoffs):
        group = scoped[scoped["fit_cutoff"].eq(cutoff)]
        ax.scatter(group["predicted_metric"], group["observed_metric"], s=18, alpha=0.65)
        lo = float(np.nanmin([group["predicted_metric"].min(), group["observed_metric"].min()]))
        hi = float(np.nanmax([group["predicted_metric"].max(), group["observed_metric"].max()]))
        ax.plot([lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=1)
        r2_rows = pooled[pooled["fit_cutoff"].eq(cutoff)]
        r2 = float(r2_rows["r2"].iloc[0]) if len(r2_rows) else np.nan
        ax.set_title(f"cutoff {cutoff:g}, R2={r2:.3f}")
        ax.set_xlabel(f"predicted {metric_label(metric_col)}")
        ax.set_ylabel(f"observed {metric_label(metric_col)}")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(cutoffs) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", default=",".join(f"{cutoff:g}" for cutoff in CUTOFFS))
    parser.add_argument("--curve-x-max", type=float, default=1e9)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = parse_cutoffs(args.cutoffs)
    results = load_results(args.input_dir)
    results.to_csv(args.output_dir / "combined_confirmatory_results.csv", index=False)

    envelopes, fits = compute_frontiers_and_fits(results, cutoffs)
    envelopes.to_csv(args.output_dir / "lower_envelope_points.csv", index=False)
    fits.to_csv(args.output_dir / "power_law_fit_summary.csv", index=False)

    predictions = compute_deeplob_predictions(results, fits)
    predictions.to_csv(args.output_dir / "deeplob_prediction_residuals.csv", index=False)

    r2_table = compute_r2_table(predictions) if not predictions.empty else pd.DataFrame()
    r2_table.to_csv(args.output_dir / "deeplob_prediction_r2_by_cutoff.csv", index=False)

    plot_frontier_grid(envelopes, args.output_dir / "all_folds_by_horizon_validation_log_loss.png", "validation_log_loss")
    plot_frontier_grid(envelopes, args.output_dir / "all_folds_by_horizon_test_log_loss.png", "test_log_loss")
    plot_frontier_grid(
        envelopes,
        args.output_dir / "all_folds_by_horizon_normalized_validation_loss.png",
        "prior_normalized_validation_log_loss",
    )
    plot_frontier_grid(
        envelopes,
        args.output_dir / "all_folds_by_horizon_normalized_test_loss.png",
        "prior_normalized_test_log_loss",
    )
    for cutoff in cutoffs:
        plot_power_law_curve_grid(
            envelopes,
            fits,
            predictions,
            args.output_dir / f"power_law_curves_validation_log_loss_cutoff_{cutoff:g}.png",
            "validation_log_loss",
            cutoff,
            args.curve_x_max,
        )
        plot_power_law_curve_grid(
            envelopes,
            fits,
            predictions,
            args.output_dir / f"power_law_curves_test_log_loss_cutoff_{cutoff:g}.png",
            "test_log_loss",
            cutoff,
            args.curve_x_max,
        )
        plot_power_law_curve_grid(
            envelopes,
            fits,
            predictions,
            args.output_dir / f"power_law_curves_normalized_validation_loss_cutoff_{cutoff:g}.png",
            "prior_normalized_validation_log_loss",
            cutoff,
            args.curve_x_max,
        )
        plot_power_law_curve_grid(
            envelopes,
            fits,
            predictions,
            args.output_dir / f"power_law_curves_normalized_test_loss_cutoff_{cutoff:g}.png",
            "prior_normalized_test_log_loss",
            cutoff,
            args.curve_x_max,
        )
    plot_predicted_vs_observed(
        predictions,
        r2_table,
        args.output_dir / "deeplob_predicted_vs_observed_validation_log_loss_by_cutoff.png",
        "validation_log_loss",
    )
    plot_predicted_vs_observed(
        predictions,
        r2_table,
        args.output_dir / "deeplob_predicted_vs_observed_test_log_loss_by_cutoff.png",
        "test_log_loss",
    )
    plot_predicted_vs_observed(
        predictions,
        r2_table,
        args.output_dir / "deeplob_predicted_vs_observed_normalized_validation_loss_by_cutoff.png",
        "prior_normalized_validation_log_loss",
    )
    plot_predicted_vs_observed(
        predictions,
        r2_table,
        args.output_dir / "deeplob_predicted_vs_observed_normalized_test_loss_by_cutoff.png",
        "prior_normalized_test_log_loss",
    )
    print(f"wrote confirmatory analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
