#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
EXPERIMENTS = REPO_ROOT / "shared"
for path in [ROOT, EXPERIMENTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lob_latency_models import (  # noqa: E402
    CLASS_ORDER,
    TorchSequenceClassifier,
    build_model,
    candidate_ops,
    parameter_count,
    sequence_tensor_from_flat,
    standardize_fit,
    time_single_observation,
    torch_device,
)

from architecture_scaling_frontiers import (  # noqa: E402
    chronological_train_validation_split,
    find_data_root,
    make_lagged_snapshot_dataset,
    maybe_limit_rows,
    prior_log_loss_from_train,
)
from compute_scaling_cf1 import (  # noqa: E402
    FULL_FEATURE_COUNT,
    HORIZONS as FI_HORIZONS,
    LABEL_MAP,
    LABEL_ROW_START,
)


LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_LABEL = np.array(CLASS_ORDER)


def safe_filename(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return safe.strip("_") or "run"


def score_probabilities(proba: np.ndarray, y_true: pd.Series) -> dict[str, float]:
    pred = pd.Series(INDEX_TO_LABEL[np.asarray(proba).argmax(axis=1)], index=y_true.index)
    per_class_f1 = f1_score(y_true, pred, labels=list(CLASS_ORDER), average=None, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=list(CLASS_ORDER)),
        "macro_f1": f1_score(y_true, pred, labels=list(CLASS_ORDER), average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, labels=list(CLASS_ORDER), average="weighted", zero_division=0),
        "micro_f1": f1_score(y_true, pred, labels=list(CLASS_ORDER), average="micro", zero_division=0),
        "neg_f1": float(per_class_f1[0]),
        "neutral_f1": float(per_class_f1[1]),
        "pos_f1": float(per_class_f1[2]),
        "weighted_precision": precision_score(y_true, pred, labels=list(CLASS_ORDER), average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, pred, labels=list(CLASS_ORDER), average="weighted", zero_division=0),
    }
    for key in ("macro_f1", "weighted_f1", "micro_f1", "neg_f1", "neutral_f1", "pos_f1"):
        metrics[f"{key}_percent"] = 100.0 * float(metrics[key])
    return metrics


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def collect_logits(model, tensor, device, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            logits = model(tensor[start : start + batch_size].to(device))
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(CLASS_ORDER)), dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    import torch

    return torch.softmax(torch.from_numpy(logits), dim=1).numpy()


def class_balance(y: pd.Series, prefix: str) -> dict[str, float]:
    counts = y.value_counts().reindex(CLASS_ORDER, fill_value=0).astype(int)
    total = int(counts.sum())
    out = {f"{prefix}_rows": total}
    for label, stem in [(-1, "down"), (0, "stationary"), (1, "up")]:
        count = int(counts.loc[label])
        out[f"{prefix}_class_count_{stem}"] = count
        out[f"{prefix}_class_fraction_{stem}"] = count / total if total else np.nan
    return out


def normalize_feature_mode(raw: str) -> str:
    mode = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "raw40": "lob40",
        "raw_40": "lob40",
        "lob_40": "lob40",
        "first40": "lob40",
        "first_40": "lob40",
        "all144": "full144",
        "all_144": "full144",
        "full_144": "full144",
        "first144": "full144",
        "first_144": "full144",
    }
    if mode in aliases:
        return aliases[mode]
    if mode == "lob40" or mode == "full144":
        return mode
    if mode.startswith("first_"):
        suffix = mode.removeprefix("first_")
        if suffix.isdigit():
            return f"first{suffix}"
    if mode.startswith("first") and mode.removeprefix("first").isdigit():
        return mode
    raise ValueError(f"unknown feature_mode: {raw}")


def feature_mode_from_config(config: dict) -> str:
    if "feature_mode" in config:
        return normalize_feature_mode(str(config["feature_mode"]))
    return normalize_feature_mode(f"first{int(config.get('base_d', 40))}")


def select_feature_columns(x_all: pd.DataFrame, feature_mode: str) -> list[str]:
    mode = normalize_feature_mode(feature_mode)
    if mode == "lob40":
        count = 40
    elif mode == "full144":
        count = 144
    elif mode.startswith("first") and mode.removeprefix("first").isdigit():
        count = int(mode.removeprefix("first"))
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    if count > x_all.shape[1]:
        raise ValueError(f"feature_mode={feature_mode} needs {count} columns, but data has {x_all.shape[1]}")
    return list(x_all.columns[:count])


def normalize_evaluation_setup(raw: str) -> str:
    setup = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "": "anchored_cf",
        "anchored": "anchored_cf",
        "setup1": "anchored_cf",
        "setup_1": "anchored_cf",
        "cf": "anchored_cf",
        "setup2": "setup2_7d_3d",
        "setup_2": "setup2_7d_3d",
        "deep_learning_setup2": "setup2_7d_3d",
        "7d3d": "setup2_7d_3d",
        "7d_3d": "setup2_7d_3d",
    }
    if setup in aliases:
        return aliases[setup]
    if setup in {"anchored_cf", "setup2_7d_3d"}:
        return setup
    raise ValueError(f"unknown evaluation_setup: {raw}")


def no_auction_zscore_path(data_root: Path, split: str, cf: int) -> Path:
    base = data_root / "NoAuction" / "1.NoAuction_Zscore"
    if split == "train":
        return base / "NoAuction_Zscore_Training" / f"Train_Dst_NoAuction_ZScore_CF_{cf}.txt"
    if split == "test":
        return base / "NoAuction_Zscore_Testing" / f"Test_Dst_NoAuction_ZScore_CF_{cf}.txt"
    raise ValueError(f"unknown split: {split}")


def load_no_auction_zscore_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    x = raw.iloc[:FULL_FEATURE_COUNT].T.reset_index(drop=True)
    x.columns = [f"x_{idx}" for idx in range(1, FULL_FEATURE_COUNT + 1)]
    y = (
        raw.iloc[LABEL_ROW_START : LABEL_ROW_START + len(FI_HORIZONS)]
        .T.replace(LABEL_MAP)
        .astype(int)
        .reset_index(drop=True)
    )
    y.columns = FI_HORIZONS
    return x, y


def load_no_auction_zscore_train_cf(cf: int, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_no_auction_zscore_file(no_auction_zscore_path(data_root, "train", cf))


def load_no_auction_zscore_test_cf(cf: int, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_no_auction_zscore_file(no_auction_zscore_path(data_root, "test", cf))


def build_lagged_from_segments(
    x_segments: list[pd.DataFrame],
    y_segments: list[pd.DataFrame],
    *,
    columns: list[str],
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_parts = []
    y_parts = []
    for x_segment, y_segment in zip(x_segments, y_segments, strict=True):
        if len(x_segment) < window:
            continue
        x_lagged, y_lagged = make_lagged_snapshot_dataset(x_segment[columns], y_segment, window)
        x_parts.append(x_lagged)
        y_parts.append(y_lagged)
    if not x_parts:
        raise ValueError(f"no segment has enough rows for window={window}")
    return (
        pd.concat(x_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
    )


def load_evaluation_split(config: dict, data_root: Path) -> dict[str, object]:
    setup = normalize_evaluation_setup(str(config.get("evaluation_setup", "anchored_cf")))
    cf = int(config["cf"])
    if setup == "anchored_cf":
        x_train, y_train = load_no_auction_zscore_train_cf(cf, data_root)
        x_test, y_test = load_no_auction_zscore_test_cf(cf, data_root)
        return {
            "evaluation_setup": setup,
            "train_cfs": [cf],
            "test_cfs": [cf],
            "preserve_day_boundaries": False,
            "x_train_segments": [x_train],
            "y_train_segments": [y_train],
            "x_test_segments": [x_test],
            "y_test_segments": [y_test],
        }

    train_cf = int(config.get("setup2_train_cf", cf))
    test_cfs = [int(value) for value in config.get("setup2_test_cfs", [7, 8, 9])]
    preserve_day_boundaries = bool(config.get("preserve_day_boundaries", True))
    if train_cf != 7 or test_cfs != [7, 8, 9]:
        raise ValueError("setup2_7d_3d currently expects setup2_train_cf=7 and setup2_test_cfs=[7,8,9]")

    if preserve_day_boundaries:
        train_pairs = [load_no_auction_zscore_train_cf(1, data_root)]
        train_pairs.extend(load_no_auction_zscore_test_cf(day_cf, data_root) for day_cf in range(1, train_cf))
    else:
        train_pairs = [load_no_auction_zscore_train_cf(train_cf, data_root)]
    test_pairs = [load_no_auction_zscore_test_cf(test_cf, data_root) for test_cf in test_cfs]

    return {
        "evaluation_setup": setup,
        "train_cfs": [train_cf],
        "test_cfs": test_cfs,
        "preserve_day_boundaries": preserve_day_boundaries,
        "x_train_segments": [item[0] for item in train_pairs],
        "y_train_segments": [item[1] for item in train_pairs],
        "x_test_segments": [item[0] for item in test_pairs],
        "y_test_segments": [item[1] for item in test_pairs],
    }


def requested_timing_devices(config: dict) -> list[str]:
    training = dict(config["training"])
    latency = dict(config["latency"])
    raw = latency.get("timing_devices", training.get("timing_devices", None))
    if raw is None:
        raw = [training.get("timing_device", "cpu")]
    if isinstance(raw, str):
        devices = [part.strip().lower() for part in raw.split(",") if part.strip()]
    else:
        devices = [str(part).strip().lower() for part in raw if str(part).strip()]
    if not devices:
        devices = ["cpu"]
    unique = []
    for device in devices:
        if device == "auto":
            device = "mps"
        if device not in unique:
            unique.append(device)
    return unique


def timing_device_available(device_name: str) -> tuple[bool, str]:
    import torch

    if device_name == "cpu":
        return True, "available"
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return True, "available"
        if torch.backends.mps.is_built():
            return False, "unavailable_mps_not_available"
        return False, "unavailable_mps_not_built"
    if device_name == "cuda":
        return (True, "available") if torch.cuda.is_available() else (False, "unavailable_cuda_not_available")
    try:
        torch.device(device_name)
    except Exception as exc:
        return False, f"invalid_device:{type(exc).__name__}:{exc}"
    return True, "available"


def time_model_on_requested_devices(
    *,
    model,
    config: dict,
    x_test_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
    base_d: int,
    eval_batch_size: int,
) -> dict[str, object]:
    latency_cfg = dict(config["latency"])
    repeats = int(latency_cfg.get("repeats", 3))
    samples = int(latency_cfg.get("single_timing_samples", 256))
    out: dict[str, object] = {}
    completed_devices = []

    for device_name in requested_timing_devices(config):
        available, status = timing_device_available(device_name)
        out[f"{device_name}_timing_status"] = status
        out[f"{device_name}_single_prediction_time_s_median"] = np.nan
        out[f"{device_name}_single_latency_us_per_obs_median"] = np.nan
        if not available:
            continue
        try:
            timing_device = torch_device(device_name)
            model.to(timing_device)
            classifier = TorchSequenceClassifier(
                model=model,
                device=timing_device,
                window=window,
                base_d=base_d,
                mean=mean,
                std=std,
                batch_size=eval_batch_size,
            )
            timing_s, latency_us = time_single_observation(
                classifier,
                x_test_raw,
                repeats=repeats,
                samples=samples,
            )
            out[f"{device_name}_timing_status"] = "ok"
            out[f"{device_name}_single_prediction_time_s_median"] = timing_s
            out[f"{device_name}_single_latency_us_per_obs_median"] = latency_us
            completed_devices.append(device_name)
        except Exception as exc:
            out[f"{device_name}_timing_status"] = f"error:{type(exc).__name__}:{exc}"

    out["timing_devices_requested"] = json.dumps(requested_timing_devices(config))
    out["timing_devices_completed"] = json.dumps(completed_devices)
    if "cpu_single_latency_us_per_obs_median" not in out:
        out["cpu_timing_status"] = "not_requested"
        out["cpu_single_prediction_time_s_median"] = np.nan
        out["cpu_single_latency_us_per_obs_median"] = np.nan
    return out


def fit_one(
    *,
    config: dict,
    candidate: dict,
    horizon: str,
    x_fit: np.ndarray,
    y_fit: pd.DataFrame,
    x_validation: np.ndarray,
    y_validation: pd.DataFrame,
    x_test: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    history_dir: Path,
    candidate_index: int,
) -> dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    training = dict(config["training"])
    window = int(config["window"])
    base_d = int(config["base_d"])
    d = window * base_d
    seed = int(training.get("seed", 0)) + candidate_index * 1009 + sum(ord(ch) for ch in horizon)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch_device(str(training.get("device", "auto")))
    model = build_model(candidate, window=window, base_d=base_d, training=training).to(device)
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.0003)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
        eps=float(training.get("adam_eps", 1e-8)),
    )
    loss_fn = nn.CrossEntropyLoss()
    batch_size = int(training.get("batch_size", 32))
    eval_batch_size = int(training.get("eval_batch_size", 64))

    y_fit_arr = y_fit[horizon].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    y_validation_arr = y_validation[horizon].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    train_tensor = sequence_tensor_from_flat(x_fit, window=window, base_d=base_d)
    train_y = torch.from_numpy(y_fit_arr)
    validation_tensor = sequence_tensor_from_flat(x_validation, window=window, base_d=base_d)
    validation_y = torch.from_numpy(y_validation_arr)
    loader = DataLoader(
        TensorDataset(train_tensor, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_state = None
    selection_metric = str(training.get("selection_metric", "validation_log_loss"))
    if not selection_metric.startswith("validation_"):
        selection_metric = f"validation_{selection_metric}"
    maximize_selection = not selection_metric.endswith("log_loss")
    best_selection_score = -np.inf if maximize_selection else np.inf
    best_val_loss = np.inf
    best_epoch = 0
    bad_epochs = 0
    history_rows = []
    epochs = int(training.get("epochs", 30))
    patience = int(training.get("patience", 8))
    min_delta = float(training.get("min_delta", 0.00005))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_logits = collect_logits(model, validation_tensor, device, eval_batch_size)
        val_loss = float(loss_fn(torch.from_numpy(val_logits), validation_y).item())
        validation_proba = softmax(val_logits)
        epoch_validation_metrics = prefix_metrics(score_probabilities(validation_proba, y_validation[horizon]), "validation")
        epoch_validation_metrics["validation_log_loss"] = val_loss
        if selection_metric not in epoch_validation_metrics:
            raise ValueError(f"unknown selection_metric={selection_metric}")
        selection_score = float(epoch_validation_metrics[selection_metric])
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        if maximize_selection:
            improved = selection_score > best_selection_score + min_delta
        else:
            improved = selection_score < best_selection_score - min_delta
        if improved:
            best_selection_score = selection_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        best_val_loss = min(best_val_loss, val_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_log_loss": val_loss,
                "validation_weighted_f1": epoch_validation_metrics["validation_weighted_f1"],
                "validation_macro_f1": epoch_validation_metrics["validation_macro_f1"],
                "validation_accuracy": epoch_validation_metrics["validation_accuracy"],
                "selection_metric": selection_metric,
                "selection_score": selection_score,
                "best_selection_score": best_selection_score,
                "best_validation_log_loss": best_val_loss,
                "bad_epochs": bad_epochs,
            }
        )
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_tensor = sequence_tensor_from_flat(x_test, window=window, base_d=base_d)
    validation_logits = collect_logits(model, validation_tensor, device, eval_batch_size)
    test_logits = collect_logits(model, test_tensor, device, eval_batch_size)
    validation_proba = softmax(validation_logits)
    test_proba = softmax(test_logits)

    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{safe_filename(candidate['name'] + '_' + horizon)}_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    timing_metrics = time_model_on_requested_devices(
        model=model,
        config=config,
        x_test_raw=x_test_raw,
        mean=mean,
        std=std,
        window=window,
        base_d=base_d,
        eval_batch_size=eval_batch_size,
    )

    row: dict[str, object] = {
        "experiment": config["experiment"],
        "cf": int(config["cf"]),
        "evaluation_setup": str(config.get("evaluation_setup", "anchored_cf")),
        "train_cfs": json.dumps(config.get("train_cfs", [int(config["cf"])])),
        "test_cfs": json.dumps(config.get("test_cfs", [int(config["cf"])])),
        "preserve_day_boundaries": bool(config.get("preserve_day_boundaries", False)),
        "train_source_rows": int(config.get("train_source_rows", 0)),
        "test_source_rows": int(config.get("test_source_rows", 0)),
        "score_target": horizon,
        "candidate_index": candidate_index,
        "lane": candidate["lane"],
        "architecture": candidate["architecture"],
        "model_name": candidate["name"],
        "feature_mode": str(config.get("feature_mode", "lob40")),
        "feature_count": base_d,
        "input_window": window,
        "base_d": base_d,
        "d": d,
        "fit_rows": len(x_fit),
        "validation_rows": len(x_validation),
        "test_rows": len(x_test),
        "torch_device": str(device),
        "torch_timing_device": ",".join(requested_timing_devices(config)),
        "torch_seed": seed,
        "torch_epochs": epochs,
        "torch_epochs_ran": len(history_rows),
        "torch_best_epoch": best_epoch,
        "torch_stopped_early": int(len(history_rows) < epochs),
        "torch_selection_metric": selection_metric,
        "torch_best_selection_score": best_selection_score,
        "torch_history_path": str(history_path),
        "audited_forward_ops": candidate_ops(candidate, window=window, base_d=base_d),
        "compute_audit_method": f"analytic_{candidate['architecture']}_forward_multiply_adds",
        "compute_audit_confidence": "medium_high",
        "n_parameters": parameter_count(model),
    }
    row.update(timing_metrics)
    row.update(class_balance(y_fit[horizon], "fit"))
    row.update(class_balance(y_validation[horizon], "validation"))
    row.update(class_balance(y_test[horizon], "test"))
    row["naive_fit_prior_validation_log_loss"] = prior_log_loss_from_train(y_fit[horizon], y_validation[horizon])
    row["naive_fit_prior_test_log_loss"] = prior_log_loss_from_train(y_fit[horizon], y_test[horizon])
    row.update(prefix_metrics(score_probabilities(validation_proba, y_validation[horizon]), "validation"))
    row.update(prefix_metrics(score_probabilities(test_proba, y_test[horizon]), "test"))
    row["log_loss"] = row["test_log_loss"]
    row["accuracy"] = row["test_accuracy"]
    row["balanced_accuracy"] = row["test_balanced_accuracy"]

    for key, value in candidate.items():
        if key not in {"name", "lane", "architecture"}:
            row[f"candidate_{key}"] = json.dumps(value) if isinstance(value, list) else value
    return row


def filter_candidates(config: dict, lanes: str, names: str, max_candidates: int) -> list[dict]:
    candidates = list(config["candidates"])
    if lanes:
        wanted = {part.strip() for part in lanes.split(",") if part.strip()}
        candidates = [candidate for candidate in candidates if candidate["lane"] in wanted]
    if names:
        wanted_names = {part.strip() for part in names.split(",") if part.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in wanted_names]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fresh latency-SOTA ladder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--lanes", default="")
    parser.add_argument("--candidate-names", default="")
    parser.add_argument("--horizons", default="")
    parser.add_argument("--cf", type=int, default=0, help="Override the config fold for fast exploration.")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.cf > 0:
        config["cf"] = int(args.cf)
    if args.epochs > 0:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["training"]["device"] = args.device

    candidates = filter_candidates(config, args.lanes, args.candidate_names, args.max_candidates)
    horizons = [part.strip() for part in args.horizons.split(",") if part.strip()] if args.horizons else list(config["horizons"])
    default_feature_mode = feature_mode_from_config(config)
    evaluation_setup = normalize_evaluation_setup(str(config.get("evaluation_setup", "anchored_cf")))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cf": int(config["cf"]),
                    "evaluation_setup": evaluation_setup,
                    "train_cfs": [config.get("setup2_train_cf", int(config["cf"]))]
                    if evaluation_setup == "setup2_7d_3d"
                    else [int(config["cf"])],
                    "test_cfs": config.get("setup2_test_cfs", [7, 8, 9])
                    if evaluation_setup == "setup2_7d_3d"
                    else [int(config["cf"])],
                    "preserve_day_boundaries": bool(config.get("preserve_day_boundaries", evaluation_setup == "setup2_7d_3d")),
                    "candidates": [
                        {
                            "name": c["name"],
                            "feature_mode": normalize_feature_mode(str(c.get("feature_mode", default_feature_mode))),
                        }
                        for c in candidates
                    ],
                    "horizons": horizons,
                },
                indent=2,
            )
        )
        return

    existing = pd.DataFrame()
    rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, str, str]] = set()
    if args.output.exists() and args.output.stat().st_size > 0 and not args.force:
        existing = pd.read_csv(args.output)
        rows.extend(existing.to_dict("records"))
        if "feature_mode" in existing.columns:
            feature_modes = existing["feature_mode"].astype(str).map(normalize_feature_mode)
        else:
            feature_modes = pd.Series([default_feature_mode] * len(existing))
        if "evaluation_setup" in existing.columns:
            evaluation_setups = existing["evaluation_setup"].astype(str).map(normalize_evaluation_setup)
        else:
            evaluation_setups = pd.Series([evaluation_setup] * len(existing))
        completed = set(
            zip(
                evaluation_setups,
                feature_modes,
                existing["model_name"].astype(str),
                existing["score_target"].astype(str),
            )
        )

    cf = int(config["cf"])
    data_root = find_data_root()
    split = load_evaluation_split(config, data_root)
    x_train_segments_all = split["x_train_segments"]
    y_train_segments_all = split["y_train_segments"]
    x_test_segments_all = split["x_test_segments"]
    y_test_segments_all = split["y_test_segments"]
    split_meta = {
        "evaluation_setup": split["evaluation_setup"],
        "train_cfs": split["train_cfs"],
        "test_cfs": split["test_cfs"],
        "preserve_day_boundaries": split["preserve_day_boundaries"],
        "train_source_rows": sum(len(item) for item in x_train_segments_all),
        "test_source_rows": sum(len(item) for item in x_test_segments_all),
    }
    window = int(config["window"])
    prepared: dict[str, dict[str, object]] = {}

    def prepare_feature_mode(feature_mode_raw: str) -> dict[str, object]:
        feature_mode = normalize_feature_mode(feature_mode_raw)
        if feature_mode in prepared:
            return prepared[feature_mode]

        columns = select_feature_columns(x_train_segments_all[0], feature_mode)
        x_train_full, y_train_full = build_lagged_from_segments(
            x_train_segments_all,
            y_train_segments_all,
            columns=columns,
            window=window,
        )
        x_test, y_test = build_lagged_from_segments(
            x_test_segments_all,
            y_test_segments_all,
            columns=columns,
            window=window,
        )
        x_test, y_test = maybe_limit_rows(x_test, y_test, args.max_test_rows)
        x_train, y_train = maybe_limit_rows(x_train_full, y_train_full, args.max_train_rows)
        x_fit_df, y_fit, x_validation_df, y_validation = chronological_train_validation_split(
            x_train,
            y_train,
            float(config["training"].get("validation_fraction", 0.2)),
        )

        x_fit_np = x_fit_df.to_numpy(dtype=np.float32)
        x_validation_np = x_validation_df.to_numpy(dtype=np.float32)
        x_test_np = x_test.to_numpy(dtype=np.float32)
        x_fit_std, x_validation_std, x_test_std, mean, std = standardize_fit(x_fit_np, x_validation_np, x_test_np)
        prepared[feature_mode] = {
            "feature_mode": feature_mode,
            "base_d": len(columns),
            "x_fit_std": x_fit_std,
            "y_fit": y_fit,
            "x_validation_std": x_validation_std,
            "y_validation": y_validation,
            "x_test_std": x_test_std,
            "x_test_np": x_test_np,
            "y_test": y_test,
            "mean": mean,
            "std": std,
        }
        return prepared[feature_mode]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.history_dir.mkdir(parents=True, exist_ok=True)
    for candidate_index, candidate in enumerate(candidates):
        feature_mode = normalize_feature_mode(str(candidate.get("feature_mode", default_feature_mode)))
        data = prepare_feature_mode(feature_mode)
        candidate_config = copy.deepcopy(config)
        candidate_config.update(split_meta)
        candidate_config["feature_mode"] = feature_mode
        candidate_config["base_d"] = int(data["base_d"])
        for horizon in horizons:
            key = (evaluation_setup, feature_mode, candidate["name"], horizon)
            if key in completed:
                print(f"skip existing {feature_mode} {candidate['name']} {horizon}", flush=True)
                continue
            print(f"fit {feature_mode} {candidate['name']} {horizon}", flush=True)
            row = fit_one(
                config=candidate_config,
                candidate=copy.deepcopy(candidate),
                horizon=horizon,
                x_fit=data["x_fit_std"],
                y_fit=data["y_fit"],
                x_validation=data["x_validation_std"],
                y_validation=data["y_validation"],
                x_test=data["x_test_std"],
                x_test_raw=data["x_test_np"],
                y_test=data["y_test"],
                mean=data["mean"],
                std=data["std"],
                history_dir=args.history_dir,
                candidate_index=candidate_index,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(
                f"{candidate['name']} {horizon}: "
                f"val_loss={row['validation_log_loss']:.4f} test_loss={row['test_log_loss']:.4f} "
                f"val_mf1={row['validation_macro_f1']:.4f} test_mf1={row['test_macro_f1']:.4f} "
                f"val_wf1={row['validation_weighted_f1']:.4f} test_wf1={row['test_weighted_f1']:.4f} "
                f"lat={row['cpu_single_latency_us_per_obs_median']:.1f}us",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
