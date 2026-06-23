#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
EXPERIMENTS = REPO_ROOT / "shared"
for path in [ROOT, EXPERIMENTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import single_horizon_ladder as single  # noqa: E402
from architecture_scaling_frontiers import find_data_root  # noqa: E402
from lob_latency_models import build_model, candidate_ops, parameter_count, sequence_tensor_from_flat, torch_device  # noqa: E402
from run_multitask_latency_ladder import build_multitask_model, multitask_fastbinlob_ops  # noqa: E402


def is_multitask(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("architecture", "")) == "multitask_fastbinlob"


def build_latency_model(candidate: dict[str, Any], *, config: dict[str, Any], base_d: int):
    if is_multitask(candidate):
        candidate_config = dict(config)
        candidate_config["base_d"] = base_d
        return build_multitask_model(candidate, config=candidate_config, horizons=list(config["horizons"]))
    return build_model(candidate, window=int(config["window"]), base_d=base_d, training=dict(config["training"]))


def audited_ops(candidate: dict[str, Any], *, config: dict[str, Any], base_d: int) -> int:
    if is_multitask(candidate):
        return multitask_fastbinlob_ops(candidate, window=int(config["window"]), base_d=base_d, horizons=list(config["horizons"]))
    return candidate_ops(candidate, window=int(config["window"]), base_d=base_d)


def load_cf_test_tensor(*, config: dict[str, Any], cf: int, sample_count: int, seed: int):
    data_root = find_data_root()
    x_test, y_test = single.load_no_auction_zscore_test_cf(cf, data_root)
    feature_mode = single.feature_mode_from_config(config)
    columns = single.select_feature_columns(x_test, feature_mode)
    x_lagged, y_lagged = single.build_lagged_from_segments(
        [x_test],
        [y_test],
        columns=columns,
        window=int(config["window"]),
    )
    x_np = x_lagged.to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    n = min(sample_count, len(x_np))
    indices = rng.choice(len(x_np), size=n, replace=False)
    sampled = x_np[indices]
    tensor = sequence_tensor_from_flat(sampled, window=int(config["window"]), base_d=len(columns))
    return {
        "tensor": tensor,
        "sample_indices": indices,
        "feature_mode": feature_mode,
        "base_d": len(columns),
        "available_rows": len(x_np),
        "sample_rows": n,
        "source_test_rows": len(x_test),
        "label_rows": len(y_lagged),
    }


def summarize_latencies(latencies_us: np.ndarray) -> dict[str, float]:
    return {
        "latency_us_mean": float(np.mean(latencies_us)),
        "latency_us_std": float(np.std(latencies_us, ddof=1)) if len(latencies_us) > 1 else 0.0,
        "latency_us_min": float(np.min(latencies_us)),
        "latency_us_p10": float(np.percentile(latencies_us, 10)),
        "latency_us_p25": float(np.percentile(latencies_us, 25)),
        "latency_us_median": float(np.median(latencies_us)),
        "latency_us_p75": float(np.percentile(latencies_us, 75)),
        "latency_us_p90": float(np.percentile(latencies_us, 90)),
        "latency_us_p95": float(np.percentile(latencies_us, 95)),
        "latency_us_p99": float(np.percentile(latencies_us, 99)),
        "latency_us_max": float(np.max(latencies_us)),
    }


def synchronize_device(device_type: str) -> None:
    if device_type == "mps":
        import torch

        torch.mps.synchronize()
    elif device_type == "cuda":
        import torch

        torch.cuda.synchronize()


def time_forward_only(model, tensor, *, warmup: int, delay_s: float) -> np.ndarray:
    import torch

    n = int(tensor.shape[0])
    warmup_count = min(max(0, warmup), n)
    device_type = str(tensor.device.type)
    model.eval()
    with torch.inference_mode():
        for idx in range(warmup_count):
            logits = model(tensor[idx : idx + 1])
            if logits.ndim == 3:
                _ = torch.softmax(logits, dim=2)
            else:
                _ = torch.softmax(logits, dim=1)
    synchronize_device(device_type)

    latencies = np.empty(n, dtype=np.float64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.inference_mode():
            for idx in range(n):
                synchronize_device(device_type)
                start_ns = time.perf_counter_ns()
                logits = model(tensor[idx : idx + 1])
                if logits.ndim == 3:
                    proba = torch.softmax(logits, dim=2)
                else:
                    proba = torch.softmax(logits, dim=1)
                synchronize_device(device_type)
                _ = proba
                end_ns = time.perf_counter_ns()
                latencies[idx] = (end_ns - start_ns) / 1000.0
                if delay_s > 0:
                    time.sleep(delay_s)
    finally:
        if gc_was_enabled:
            gc.enable()
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only single-observation CPU latency on real FI-2010 CF test inputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cf", type=int, default=8)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--delay-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-names", default="")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    config["training"]["device"] = args.device
    candidates = list(config["candidates"])
    if args.candidate_names:
        keep = {part.strip() for part in args.candidate_names.split(",") if part.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in keep]
    if not candidates:
        raise ValueError("no candidates selected")

    import torch

    device = torch_device(args.device)
    data = load_cf_test_tensor(config=config, cf=args.cf, sample_count=args.samples, seed=args.seed)
    tensor = data["tensor"].to(device)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for idx, candidate in enumerate(candidates):
        torch.manual_seed(args.seed + idx)
        model = build_latency_model(candidate, config=config, base_d=int(data["base_d"])).to(device)
        model.eval()
        ops = audited_ops(candidate, config=config, base_d=int(data["base_d"]))
        params = parameter_count(model)
        latencies_us = time_forward_only(model, tensor, warmup=int(args.warmup), delay_s=float(args.delay_ms) / 1000.0)
        row: dict[str, Any] = {
            "experiment": str(config.get("experiment", "real_cf_single_latency")),
            "latency_mode": "forward_only_prebuilt_real_cf_tensor_plus_softmax",
            "device": str(device),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "cf": int(args.cf),
            "split": "test",
            "feature_mode": data["feature_mode"],
            "window": int(config["window"]),
            "base_d": int(data["base_d"]),
            "d": int(config["window"]) * int(data["base_d"]),
            "source_test_rows": int(data["source_test_rows"]),
            "available_lagged_rows": int(data["available_rows"]),
            "sample_rows": int(data["sample_rows"]),
            "sample_seed": int(args.seed),
            "warmup_rows": int(min(max(0, args.warmup), data["sample_rows"])),
            "inter_call_delay_ms": float(args.delay_ms),
            "candidate_index": idx,
            "lane": candidate["lane"],
            "architecture": candidate["architecture"],
            "model_name": candidate["name"],
            "audited_forward_ops": int(ops),
            "n_parameters": int(params),
        }
        row.update(summarize_latencies(latencies_us))
        row["ops_per_us_median"] = float(ops / row["latency_us_median"]) if row["latency_us_median"] > 0 else np.nan
        row["ops_per_us_mean"] = float(ops / row["latency_us_mean"]) if row["latency_us_mean"] > 0 else np.nan
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(
            f"{candidate['name']}: median={row['latency_us_median']:.1f}us "
            f"p90={row['latency_us_p90']:.1f}us p99={row['latency_us_p99']:.1f}us "
            f"ops/us={row['ops_per_us_median']:,.0f}",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
