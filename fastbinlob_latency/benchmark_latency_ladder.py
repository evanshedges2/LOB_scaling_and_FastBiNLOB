#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
EXPERIMENTS = REPO_ROOT / "shared"
for path in [ROOT, EXPERIMENTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lob_latency_models import (  # noqa: E402
    TorchSequenceClassifier,
    build_model,
    candidate_ops,
    parameter_count,
    sequence_tensor_from_flat,
    time_single_observation,
    torch_device,
)
from run_multitask_latency_ladder import build_multitask_model, multitask_fastbinlob_ops  # noqa: E402


def feature_base_d(candidate: dict, config: dict) -> int:
    if "base_d" in candidate:
        return int(candidate["base_d"])
    mode = str(candidate.get("feature_mode", config.get("feature_mode", ""))).lower()
    if mode in {"lob40", "raw40", "first40"}:
        return 40
    if mode in {"full144", "all144"}:
        return 144
    if mode.startswith("first"):
        suffix = mode.removeprefix("first")
        if suffix.isdigit():
            return int(suffix)
    return int(config["base_d"])


def requested_devices(config: dict, raw_devices: str | None) -> list[str]:
    if raw_devices:
        return [part.strip() for part in raw_devices.split(",") if part.strip()]
    latency_cfg = dict(config.get("latency", {}))
    devices = latency_cfg.get("timing_devices", ["cpu"])
    if isinstance(devices, str):
        return [part.strip() for part in devices.split(",") if part.strip()]
    return [str(device) for device in devices]


def device_is_available(raw_device: str) -> tuple[bool, str]:
    import torch

    if raw_device == "mps" and not torch.backends.mps.is_available():
        return False, "unavailable_mps_not_available"
    if raw_device == "cuda" and not torch.cuda.is_available():
        return False, "unavailable_cuda_not_available"
    return True, "ok"


class MultitaskTorchSequenceClassifier:
    def __init__(self, *, model, device, window: int, base_d: int, mean: np.ndarray, std: np.ndarray, batch_size: int):
        self.model = model
        self.device = device
        self.window = window
        self.base_d = base_d
        self.mean = mean
        self.std = std
        self.batch_size = batch_size

    def _prepare(self, x: np.ndarray):
        arr = (np.asarray(x, dtype=np.float32) - self.mean) / self.std
        return sequence_tensor_from_flat(arr, window=self.window, base_d=self.base_d)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        tensor = self._prepare(x)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                logits = self.model(tensor[start : start + self.batch_size].to(self.device))
                chunks.append(logits.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0, 3), dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        logits = self.predict_logits(x)
        return torch.softmax(torch.from_numpy(logits), dim=2).numpy()


def candidate_is_multitask(candidate: dict) -> bool:
    return str(candidate.get("architecture", "")) == "multitask_fastbinlob"


def build_latency_model(candidate: dict, *, config: dict, base_d: int):
    window = int(config["window"])
    training = dict(config["training"])
    if candidate_is_multitask(candidate):
        candidate_config = dict(config)
        candidate_config["base_d"] = base_d
        return build_multitask_model(candidate, config=candidate_config, horizons=list(config.get("horizons", [])))
    return build_model(candidate, window=window, base_d=base_d, training=training)


def latency_candidate_ops(candidate: dict, *, config: dict, base_d: int) -> int:
    window = int(config["window"])
    if candidate_is_multitask(candidate):
        return multitask_fastbinlob_ops(candidate, window=window, base_d=base_d, horizons=list(config.get("horizons", [])))
    return candidate_ops(candidate, window=window, base_d=base_d)


def benchmark_one_device(
    candidate: dict,
    *,
    config: dict,
    base_d: int,
    x: np.ndarray,
    raw_device: str,
    repeats: int,
    samples: int,
    seed: int,
) -> dict:
    import torch

    available, status = device_is_available(raw_device)
    prefix = raw_device.replace(":", "_")
    if not available:
        return {
            f"{prefix}_timing_status": status,
            f"{prefix}_single_prediction_time_s_median": np.nan,
            f"{prefix}_single_latency_us_per_obs_median": np.nan,
            f"{prefix}_ops_per_us": np.nan,
        }

    training = dict(config["training"])
    window = int(config["window"])
    d = window * base_d
    torch.manual_seed(seed)
    device = torch_device(raw_device)
    model = build_latency_model(candidate, config=config, base_d=base_d)
    model.eval()
    model.to(device)
    classifier_cls = MultitaskTorchSequenceClassifier if candidate_is_multitask(candidate) else TorchSequenceClassifier
    classifier = classifier_cls(
        model=model,
        device=device,
        window=window,
        base_d=base_d,
        mean=np.zeros((1, d), dtype=np.float32),
        std=np.ones((1, d), dtype=np.float32),
        batch_size=int(training.get("eval_batch_size", 64)),
    )
    timing_s, latency_us = time_single_observation(classifier, x, repeats=repeats, samples=samples)
    ops = latency_candidate_ops(candidate, config=config, base_d=base_d)
    return {
        f"{prefix}_timing_status": status,
        f"{prefix}_single_prediction_time_s_median": timing_s,
        f"{prefix}_single_latency_us_per_obs_median": latency_us,
        f"{prefix}_ops_per_us": ops / latency_us if latency_us > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic single-observation latency audit for LOB architecture candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default=None, help="Comma-separated timing devices. Defaults to config latency.timing_devices.")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--candidate-names", default=None, help="Optional comma-separated subset of candidate names.")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    window = int(config["window"])
    training = dict(config["training"])
    latency_cfg = dict(config["latency"])
    repeats = int(args.repeats or latency_cfg.get("repeats", 3))
    samples = int(args.samples or latency_cfg.get("single_timing_samples", 256))
    synthetic_rows = int(latency_cfg.get("synthetic_rows", 512))
    devices = requested_devices(config, args.devices)
    candidates = list(config["candidates"])
    if args.candidate_names:
        keep = {part.strip() for part in args.candidate_names.split(",") if part.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in keep]
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    import torch

    rng = np.random.default_rng(args.seed)
    synthetic_by_base_d = {}
    rows = []
    for idx, candidate in enumerate(candidates):
        torch.manual_seed(args.seed + idx)
        base_d = feature_base_d(candidate, config)
        d = window * base_d
        if base_d not in synthetic_by_base_d:
            synthetic_by_base_d[base_d] = rng.standard_normal((synthetic_rows, d), dtype=np.float32)
        x = synthetic_by_base_d[base_d]
        model = build_latency_model(candidate, config=config, base_d=base_d)
        ops = latency_candidate_ops(candidate, config=config, base_d=base_d)
        row = {
            "experiment": config["experiment"],
            "candidate_index": idx,
            "lane": candidate["lane"],
            "feature_mode": candidate.get("feature_mode", config.get("feature_mode", "")),
            "architecture": candidate["architecture"],
            "model_name": candidate["name"],
            "window": window,
            "base_d": base_d,
            "d": d,
            "audited_forward_ops": ops,
            "n_parameters": parameter_count(model),
            "timing_repeats": repeats,
            "timing_samples": samples,
            "synthetic_rows": synthetic_rows,
            "timing_devices": ",".join(devices),
        }
        del model
        for raw_device in devices:
            try:
                row.update(
                    benchmark_one_device(
                        candidate,
                        config=config,
                        base_d=base_d,
                        x=x,
                        raw_device=raw_device,
                        repeats=repeats,
                        samples=samples,
                        seed=args.seed + idx,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                prefix = raw_device.replace(":", "_")
                row.update(
                    {
                        f"{prefix}_timing_status": f"error_{type(exc).__name__}: {exc}",
                        f"{prefix}_single_prediction_time_s_median": np.nan,
                        f"{prefix}_single_latency_us_per_obs_median": np.nan,
                        f"{prefix}_ops_per_us": np.nan,
                    }
                )
        rows.append(row)
        cpu_latency = row.get("cpu_single_latency_us_per_obs_median", np.nan)
        cpu_ops_per_us = row.get("cpu_ops_per_us", np.nan)
        if np.isfinite(cpu_latency):
            print(f"{candidate['name']}: cpu={cpu_latency:.1f} us, ops/us={cpu_ops_per_us:,.0f}, ops={ops:,}", flush=True)
        else:
            print(f"{candidate['name']}: cpu failed, ops={ops:,}", flush=True)

    output = pd.DataFrame(rows)
    if "cpu_ops_per_us" in output:
        output = output.sort_values("cpu_ops_per_us", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
