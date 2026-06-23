#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
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
from lob_latency_models import (  # noqa: E402
    CLASS_ORDER,
    SimpleBiN,
    activation_module,
    parameter_count,
    sequence_tensor_from_flat,
    standardize_fit,
    torch_device,
)
from architecture_scaling_frontiers import chronological_train_validation_split, find_data_root, maybe_limit_rows  # noqa: E402


LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}


def safe_filename(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return safe.strip("_") or "run"


def build_multitask_fastbinlob(
    *,
    base_d: int,
    window: int,
    hidden_dim: int,
    layers: int,
    temporal_expansion: int,
    feature_expansion: int,
    horizons: list[str],
    dropout: float,
    activation: str,
    block_norm: str,
    final_norm: str,
    head_hidden: int,
    gated_residual: bool,
    pooling: str,
    taper_norm: bool,
    pool_windows: tuple[int, ...],
    include_last: bool,
    conv_stem: str,
    conv_kernel: int,
    conv_layers: int,
    conv_scale_init: float,
    dense_stem: str,
    dense_stem_kernel: int,
    dense_stem_layers: int,
    dense_stem_scale_init: float,
    dense_stem_init: str,
    dense_stem_activation: bool,
    dense_stem_dropout: float,
    input_views: tuple[str, ...],
    input_norm: str,
    use_bin: bool,
    bin_eps: float,
):
    import torch
    from torch import nn

    if block_norm not in {"none", "layer"}:
        raise ValueError("multitask_fastbinlob block_norm must be none or layer")
    if final_norm not in {"none", "layer"}:
        raise ValueError("multitask_fastbinlob final_norm must be none or layer")
    if pooling not in {"mean", "multi", "taper"}:
        raise ValueError("multitask_fastbinlob pooling must be mean, multi, or taper")
    if input_norm not in {"none", "simple_bin", "official_bin"}:
        raise ValueError("multitask_fastbinlob input_norm must be none, simple_bin, or official_bin")
    if any(value <= 0 for value in pool_windows):
        raise ValueError("multitask_fastbinlob pool_windows must be positive")
    if conv_stem not in {"none", "depthwise", "separable", "full"}:
        raise ValueError("multitask_fastbinlob conv_stem must be none, depthwise, separable, or full")
    if conv_layers < 0:
        raise ValueError("multitask_fastbinlob conv_layers must be non-negative")
    if conv_kernel <= 0 or conv_kernel % 2 != 1:
        raise ValueError("multitask_fastbinlob conv_kernel must be a positive odd integer")
    if dense_stem not in {"none", "local_linear"}:
        raise ValueError("multitask_fastbinlob dense_stem must be none or local_linear")
    if dense_stem_layers < 0:
        raise ValueError("multitask_fastbinlob dense_stem_layers must be non-negative")
    if dense_stem_kernel <= 0 or dense_stem_kernel % 2 != 1:
        raise ValueError("multitask_fastbinlob dense_stem_kernel must be a positive odd integer")
    if dense_stem_init not in {"local", "identity", "random"}:
        raise ValueError("multitask_fastbinlob dense_stem_init must be local, identity, or random")
    if conv_stem != "none" and conv_layers > 0 and dense_stem != "none" and dense_stem_layers > 0:
        raise ValueError("multitask_fastbinlob cannot use conv_stem and dense_stem at the same time")
    if not input_views:
        raise ValueError("multitask_fastbinlob input_views must not be empty")
    valid_simple_views = {"raw", "delta1"}
    avg_windows = []
    for view in input_views:
        if view in valid_simple_views:
            continue
        match = re.fullmatch(r"(ma|res)(\d+)", view)
        if match is None:
            raise ValueError(f"unknown multitask_fastbinlob input view: {view}")
        width = int(match.group(2))
        if width <= 1 or width > window:
            raise ValueError(f"input view {view} has invalid window width")
        avg_windows.append(width)
    unique_avg_windows = tuple(sorted(set(avg_windows)))

    def make_norm(kind: str):
        return nn.LayerNorm(hidden_dim) if kind == "layer" else nn.Identity()

    def make_dropout():
        return nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def make_stem_dropout():
        return nn.Dropout(dense_stem_dropout) if dense_stem_dropout > 0 else nn.Identity()

    def init_local_temporal_linear(layer: nn.Linear):
        if dense_stem_init == "random":
            return
        with torch.no_grad():
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
            if dense_stem_init == "identity":
                layer.weight.copy_(torch.eye(window, dtype=layer.weight.dtype, device=layer.weight.device))
                return
            radius = dense_stem_kernel // 2
            offsets = list(range(-radius, radius + 1))
            raw = torch.tensor(
                [float(radius + 1 - abs(offset)) for offset in offsets],
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
            weights = raw / raw.sum()
            for out_idx in range(window):
                for offset, value in zip(offsets, weights):
                    in_idx = out_idx + offset
                    if 0 <= in_idx < window:
                        layer.weight[out_idx, in_idx] = value

    def trailing_average_matrix(width: int) -> torch.Tensor:
        matrix = torch.zeros(window, window)
        for out_idx in range(window):
            start = max(0, out_idx - width + 1)
            matrix[out_idx, start : out_idx + 1] = 1.0 / float(out_idx - start + 1)
        return matrix

    pool_factor = 1
    if pooling == "taper":
        if hidden_dim // 4 <= 0 or window // 4 <= 0:
            raise ValueError("taper pooling requires hidden_dim>=4 and window>=4")
        pool_dim = (hidden_dim // 4) * (window // 4)
    elif pooling == "multi":
        pool_factor += len(pool_windows)
        if include_last:
            pool_factor += 1
        pool_dim = hidden_dim * pool_factor
    else:
        pool_dim = hidden_dim

    class OfficialBiN(nn.Module):
        def __init__(self):
            super().__init__()
            self.B1 = nn.Parameter(torch.zeros(window, 1))
            self.l1 = nn.Parameter(torch.empty(window, 1))
            nn.init.xavier_normal_(self.l1)
            self.B2 = nn.Parameter(torch.zeros(base_d, 1))
            self.l2 = nn.Parameter(torch.empty(base_d, 1))
            nn.init.xavier_normal_(self.l2)
            self.y1 = nn.Parameter(torch.tensor([0.5]))
            self.y2 = nn.Parameter(torch.tensor([0.5]))

        def forward(self, x):
            if self.y1[0] < 0:
                self.y1.data.fill_(0.01)
            if self.y2[0] < 0:
                self.y2.data.fill_(0.01)

            time_ones = torch.ones([window, 1], device=x.device, dtype=x.dtype)
            feature_ones = torch.ones([base_d, 1], device=x.device, dtype=x.dtype)

            time_mean = torch.mean(x, dim=2).reshape(x.shape[0], x.shape[1], 1)
            time_std = torch.std(x, dim=2).reshape(x.shape[0], x.shape[1], 1)
            time_std = torch.where(time_std < 1e-4, torch.ones_like(time_std), time_std)
            time_norm = (x - (time_mean @ time_ones.T)) / (time_std @ time_ones.T)
            time_scaled = (self.l2 @ time_ones.T) * time_norm + (self.B2 @ time_ones.T)

            feature_mean = torch.mean(x, dim=1).reshape(x.shape[0], x.shape[2], 1)
            feature_std = torch.std(x, dim=1).reshape(x.shape[0], x.shape[2], 1)
            feature_center = torch.permute(feature_mean @ feature_ones.T, (0, 2, 1))
            feature_scale = torch.permute(feature_std @ feature_ones.T, (0, 2, 1))
            feature_norm = (x - feature_center) / feature_scale
            feature_scaled = (feature_ones @ self.l1.T) * feature_norm + (feature_ones @ self.B1.T)

            return self.y1 * feature_scaled + self.y2 * time_scaled

    class TaperPool(nn.Module):
        def __init__(self):
            super().__init__()
            reduced_hidden = hidden_dim // 4
            reduced_window = window // 4
            self.feature_pre_norm = nn.LayerNorm(hidden_dim) if taper_norm else nn.Identity()
            self.feature_post_norm = nn.LayerNorm(reduced_hidden) if taper_norm else nn.Identity()
            self.temporal_pre_norm = nn.LayerNorm(window) if taper_norm else nn.Identity()
            self.temporal_post_norm = nn.LayerNorm(reduced_window) if taper_norm else nn.Identity()
            self.feature_reduce = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                activation_module(activation),
                make_dropout(),
                nn.Linear(hidden_dim * 2, reduced_hidden),
            )
            self.temporal_reduce = nn.Sequential(
                nn.Linear(window, window * 2),
                activation_module(activation),
                make_dropout(),
                nn.Linear(window * 2, reduced_window),
            )
            self.out_activation = activation_module(activation)

        def forward(self, z):
            z = self.feature_reduce(self.feature_pre_norm(z))
            z = self.out_activation(self.feature_post_norm(z))
            z = self.temporal_reduce(self.temporal_pre_norm(z.transpose(1, 2)))
            z = self.out_activation(self.temporal_post_norm(z))
            return z.flatten(start_dim=1)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_norm = make_norm(block_norm)
            self.feature_mlp = (
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * feature_expansion),
                    activation_module(activation),
                    make_dropout(),
                    nn.Linear(hidden_dim * feature_expansion, hidden_dim),
                    make_dropout(),
                )
                if feature_expansion > 0
                else None
            )
            self.temporal_norm = make_norm(block_norm)
            self.temporal_mlp = nn.Sequential(
                nn.Linear(window, window * temporal_expansion),
                activation_module(activation),
                make_dropout(),
                nn.Linear(window * temporal_expansion, window),
                make_dropout(),
            )
            self.feature_scale = nn.Parameter(torch.ones(())) if gated_residual and feature_expansion > 0 else None
            self.temporal_scale = nn.Parameter(torch.ones(())) if gated_residual else None

        def forward(self, z):
            if self.feature_mlp is not None:
                update = self.feature_mlp(self.feature_norm(z))
                if self.feature_scale is not None:
                    update = self.feature_scale * update
                z = z + update
            update = self.temporal_mlp(self.temporal_norm(z).transpose(1, 2)).transpose(1, 2)
            if self.temporal_scale is not None:
                update = self.temporal_scale * update
            return z + update

    class TemporalConvStem(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            self.scales = nn.ParameterList()
            padding = conv_kernel // 2
            for _ in range(conv_layers):
                if conv_stem == "depthwise":
                    layer = nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, conv_kernel, padding=padding, groups=hidden_dim),
                        activation_module(activation),
                        make_dropout(),
                    )
                elif conv_stem == "separable":
                    layer = nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, conv_kernel, padding=padding, groups=hidden_dim),
                        nn.Conv1d(hidden_dim, hidden_dim, 1),
                        activation_module(activation),
                        make_dropout(),
                    )
                elif conv_stem == "full":
                    layer = nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, conv_kernel, padding=padding),
                        activation_module(activation),
                        make_dropout(),
                    )
                else:
                    layer = nn.Identity()
                self.layers.append(layer)
                self.scales.append(nn.Parameter(torch.tensor(float(conv_scale_init))))

        def forward(self, z):
            if conv_stem == "none" or not self.layers:
                return z
            for layer, scale in zip(self.layers, self.scales):
                update = layer(z.transpose(1, 2)).transpose(1, 2)
                z = z + scale * update
            return z

    class DenseTemporalStem(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            self.scales = nn.ParameterList()
            self.activation = activation_module(activation) if dense_stem_activation else nn.Identity()
            self.dropout = make_stem_dropout()
            for _ in range(dense_stem_layers):
                layer = nn.Linear(window, window, bias=False)
                init_local_temporal_linear(layer)
                self.layers.append(layer)
                self.scales.append(nn.Parameter(torch.tensor(float(dense_stem_scale_init))))

        def forward(self, z):
            if dense_stem == "none" or not self.layers:
                return z
            for layer, scale in zip(self.layers, self.scales):
                update = layer(z.transpose(1, 2)).transpose(1, 2)
                update = self.dropout(self.activation(update))
                z = z + scale * update
            return z

    class InputViewStem(nn.Module):
        def __init__(self):
            super().__init__()
            self.views = tuple(input_views)
            self.avg_windows = unique_avg_windows
            for width in self.avg_windows:
                self.register_buffer(f"avg_{width}", trailing_average_matrix(width), persistent=False)

        def trailing_average(self, x, width: int):
            weight = getattr(self, f"avg_{width}")
            return torch.matmul(x, weight.t())

        def forward(self, x):
            averages = {width: self.trailing_average(x, width) for width in self.avg_windows}
            parts = []
            for view in self.views:
                if view == "raw":
                    parts.append(x)
                elif view == "delta1":
                    zero = torch.zeros_like(x[..., :1])
                    parts.append(torch.cat([zero, x[..., 1:] - x[..., :-1]], dim=-1))
                else:
                    match = re.fullmatch(r"(ma|res)(\d+)", view)
                    if match is None:
                        raise ValueError(f"unknown input view: {view}")
                    kind = match.group(1)
                    width = int(match.group(2))
                    averaged = averages[width]
                    parts.append(averaged if kind == "ma" else x - averaged)
            return torch.cat(parts, dim=1)

    class MultiTaskFastBiNLOB(nn.Module):
        def __init__(self):
            super().__init__()
            self.horizons = tuple(horizons)
            if not use_bin or input_norm == "none":
                self.bin = nn.Identity()
            elif input_norm == "official_bin":
                self.bin = OfficialBiN()
            else:
                self.bin = SimpleBiN.build(base_d, bin_eps)
            self.input_view_stem = InputViewStem()
            self.feature_embed = nn.Linear(base_d * len(input_views), hidden_dim)
            self.time_embed = nn.Parameter(torch.zeros(1, window, hidden_dim))
            self.conv_stem = TemporalConvStem()
            self.dense_stem = DenseTemporalStem()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.final_norm = make_norm(final_norm)
            self.taper_pool = TaperPool() if pooling == "taper" else None
            if head_hidden > 0:
                self.heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(pool_dim, head_hidden),
                            activation_module(activation),
                            make_dropout(),
                            nn.Linear(head_hidden, len(CLASS_ORDER)),
                        )
                        for _ in horizons
                    ]
                )
            else:
                self.heads = nn.ModuleList([nn.Linear(pool_dim, len(CLASS_ORDER)) for _ in horizons])

        def pool(self, z):
            z = self.final_norm(z)
            if self.taper_pool is not None:
                return self.taper_pool(z)
            if pooling == "mean":
                return z.mean(dim=1)
            parts = [z.mean(dim=1)]
            if include_last:
                parts.append(z[:, -1, :])
            for pool_window in pool_windows:
                width = min(int(pool_window), z.shape[1])
                parts.append(z[:, -width:, :].mean(dim=1))
            return torch.cat(parts, dim=1)

        def forward(self, x):
            x = self.input_view_stem(self.bin(x))
            z = self.feature_embed(x.transpose(1, 2)) + self.time_embed
            z = self.conv_stem(z)
            z = self.dense_stem(z)
            for block in self.blocks:
                z = block(z)
            pooled = self.pool(z)
            return torch.stack([head(pooled) for head in self.heads], dim=1)

    return MultiTaskFastBiNLOB()


def build_multitask_model(candidate: dict[str, Any], *, config: dict, horizons: list[str]):
    architecture = str(candidate["architecture"])
    if architecture != "multitask_fastbinlob":
        raise ValueError(f"unknown multitask architecture: {architecture}")
    training = dict(config["training"])
    return build_multitask_fastbinlob(
        base_d=int(config["base_d"]),
        window=int(config["window"]),
        hidden_dim=int(candidate["hidden_dim"]),
        layers=int(candidate["layers"]),
        temporal_expansion=int(candidate.get("temporal_expansion", candidate.get("expansion", 4))),
        feature_expansion=int(candidate.get("feature_expansion", candidate.get("expansion", 1))),
        horizons=horizons,
        dropout=float(candidate.get("dropout", training.get("dropout", 0.1))),
        activation=str(candidate.get("activation", "gelu")),
        block_norm=str(candidate.get("block_norm", "none")),
        final_norm=str(candidate.get("final_norm", "none")),
        head_hidden=int(candidate.get("head_hidden", 0)),
        gated_residual=bool(candidate.get("gated_residual", False)),
        pooling=str(candidate.get("pooling", "mean")),
        taper_norm=bool(candidate.get("taper_norm", False)),
        pool_windows=tuple(int(value) for value in candidate.get("pool_windows", [16, 32])),
        include_last=bool(candidate.get("include_last", True)),
        conv_stem=str(candidate.get("conv_stem", "none")),
        conv_kernel=int(candidate.get("conv_kernel", 3)),
        conv_layers=int(candidate.get("conv_layers", 0)),
        conv_scale_init=float(candidate.get("conv_scale_init", 0.1)),
        dense_stem=str(candidate.get("dense_stem", "none")),
        dense_stem_kernel=int(candidate.get("dense_stem_kernel", 3)),
        dense_stem_layers=int(candidate.get("dense_stem_layers", 0)),
        dense_stem_scale_init=float(candidate.get("dense_stem_scale_init", 0.1)),
        dense_stem_init=str(candidate.get("dense_stem_init", "local")),
        dense_stem_activation=bool(candidate.get("dense_stem_activation", False)),
        dense_stem_dropout=float(candidate.get("dense_stem_dropout", 0.0)),
        input_views=tuple(str(value) for value in candidate.get("input_views", ["raw"])),
        input_norm=str(candidate.get("input_norm", "simple_bin" if bool(training.get("use_bin", True)) else "none")),
        use_bin=bool(training.get("use_bin", True)),
        bin_eps=float(training.get("bin_eps", 1e-5)),
    )


def multitask_fastbinlob_ops(candidate: dict[str, Any], *, window: int, base_d: int, horizons: list[str]) -> int:
    hidden_dim = int(candidate["hidden_dim"])
    layers = int(candidate["layers"])
    temporal_expansion = int(candidate.get("temporal_expansion", candidate.get("expansion", 4)))
    feature_expansion = int(candidate.get("feature_expansion", candidate.get("expansion", 1)))
    head_hidden = int(candidate.get("head_hidden", 0))
    conv_stem = str(candidate.get("conv_stem", "none"))
    conv_kernel = int(candidate.get("conv_kernel", 3))
    conv_layers = int(candidate.get("conv_layers", 0))
    dense_stem = str(candidate.get("dense_stem", "none"))
    dense_stem_layers = int(candidate.get("dense_stem_layers", 0))
    input_views = tuple(str(value) for value in candidate.get("input_views", ["raw"]))
    pooling = str(candidate.get("pooling", "mean"))
    pool_windows = tuple(int(value) for value in candidate.get("pool_windows", [16, 32]))
    include_last = bool(candidate.get("include_last", True))
    if pooling == "taper":
        pool_dim = (hidden_dim // 4) * (window // 4)
    elif pooling == "multi":
        pool_factor = 1 + len(pool_windows) + int(include_last)
        pool_dim = hidden_dim * pool_factor
    else:
        pool_dim = hidden_dim
    embed_ops = window * base_d * len(input_views) * hidden_dim
    avg_windows = set()
    delta_views = 0
    residual_views = 0
    for view in input_views:
        if view == "delta1":
            delta_views += 1
        else:
            match = re.fullmatch(r"(ma|res)(\d+)", view)
            if match is not None:
                avg_windows.add(int(match.group(2)))
                if match.group(1) == "res":
                    residual_views += 1
    input_view_ops = len(avg_windows) * base_d * window * window
    input_view_ops += delta_views * base_d * window
    input_view_ops += residual_views * base_d * window
    if conv_layers <= 0 or conv_stem == "none":
        conv_ops = 0
    elif conv_stem == "depthwise":
        conv_ops = conv_layers * window * hidden_dim * conv_kernel
    elif conv_stem == "separable":
        conv_ops = conv_layers * (window * hidden_dim * conv_kernel + window * hidden_dim * hidden_dim)
    elif conv_stem == "full":
        conv_ops = conv_layers * window * hidden_dim * hidden_dim * conv_kernel
    else:
        raise ValueError(f"unknown conv_stem={conv_stem}")
    if dense_stem_layers <= 0 or dense_stem == "none":
        dense_stem_ops = 0
    elif dense_stem == "local_linear":
        dense_stem_ops = dense_stem_layers * hidden_dim * window * window
    else:
        raise ValueError(f"unknown dense_stem={dense_stem}")
    feature_ops = 2 * window * hidden_dim * hidden_dim * feature_expansion
    temporal_ops = 2 * hidden_dim * window * window * temporal_expansion
    if pooling == "taper":
        reduced_hidden = hidden_dim // 4
        reduced_window = window // 4
        pool_ops = window * (hidden_dim * hidden_dim * 2 + hidden_dim * 2 * reduced_hidden)
        pool_ops += reduced_hidden * (window * window * 2 + window * 2 * reduced_window)
    else:
        pool_ops = 0
    if head_hidden > 0:
        head_ops = len(horizons) * (pool_dim * head_hidden + head_hidden * len(CLASS_ORDER))
    else:
        head_ops = len(horizons) * pool_dim * len(CLASS_ORDER)
    return int(input_view_ops + embed_ops + conv_ops + dense_stem_ops + layers * (feature_ops + temporal_ops) + pool_ops + head_ops)


def collect_multitask_logits(model, tensor, device, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            logits = model(tensor[start : start + batch_size].to(device))
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0, len(CLASS_ORDER)), dtype=np.float32)


def multitask_softmax(logits: np.ndarray) -> np.ndarray:
    import torch

    return torch.softmax(torch.from_numpy(logits), dim=2).numpy()


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def make_loss_state(y_fit_arr: np.ndarray, candidate: dict[str, Any], training: dict[str, Any]) -> dict[str, object]:
    loss_type = str(candidate.get("loss_type", training.get("loss_type", "ce"))).lower()
    class_weighting = str(candidate.get("class_weighting", training.get("class_weighting", "none"))).lower()
    if loss_type in {"class_balanced_focal", "balanced_focal"} and class_weighting == "none":
        class_weighting = "effective"
    if loss_type == "weighted_ce" and class_weighting == "none":
        class_weighting = "balanced"

    n_horizons = y_fit_arr.shape[1]
    counts = np.zeros((n_horizons, len(CLASS_ORDER)), dtype=np.float64)
    for horizon_idx in range(n_horizons):
        counts[horizon_idx] = np.bincount(y_fit_arr[:, horizon_idx], minlength=len(CLASS_ORDER))

    smoothing = float(candidate.get("class_prior_smoothing", training.get("class_prior_smoothing", 1.0)))
    priors = (counts + smoothing) / (counts.sum(axis=1, keepdims=True) + smoothing * len(CLASS_ORDER))
    class_weights = np.ones_like(priors, dtype=np.float64)
    if class_weighting in {"balanced", "inverse"}:
        class_weights = counts.sum(axis=1, keepdims=True) / (len(CLASS_ORDER) * np.maximum(counts, 1.0))
    elif class_weighting == "sqrt_inverse":
        class_weights = np.sqrt(counts.sum(axis=1, keepdims=True) / (len(CLASS_ORDER) * np.maximum(counts, 1.0)))
    elif class_weighting == "effective":
        beta = float(candidate.get("effective_beta", training.get("effective_beta", 0.999)))
        effective = 1.0 - np.power(beta, np.maximum(counts, 1.0))
        class_weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    elif class_weighting != "none":
        raise ValueError(f"unknown class_weighting={class_weighting}")
    class_weights = class_weights / np.maximum(class_weights.mean(axis=1, keepdims=True), 1e-12)
    weight_cap = float(candidate.get("class_weight_cap", training.get("class_weight_cap", 8.0)))
    if weight_cap > 0:
        class_weights = np.minimum(class_weights, weight_cap)

    return {
        "loss_type": loss_type,
        "focal_gamma": float(candidate.get("focal_gamma", training.get("focal_gamma", 0.0))),
        "logit_adjust_tau": float(candidate.get("logit_adjust_tau", training.get("logit_adjust_tau", 0.0))),
        "class_weighting": class_weighting,
        "class_counts": counts.astype(np.float32),
        "class_priors": priors.astype(np.float32),
        "log_priors": np.log(np.maximum(priors, 1e-12)).astype(np.float32),
        "class_weights": class_weights.astype(np.float32),
    }


def tensorize_loss_state(loss_state: dict[str, object], device) -> dict[str, object]:
    import torch

    out = dict(loss_state)
    out["log_priors_t"] = torch.from_numpy(np.asarray(loss_state["log_priors"], dtype=np.float32)).to(device)
    out["class_weights_t"] = torch.from_numpy(np.asarray(loss_state["class_weights"], dtype=np.float32)).to(device)
    return out


def horizon_training_loss(logits, targets, horizon_idx: int, loss_state: dict[str, object]):
    import torch
    from torch.nn import functional as F

    loss_type = str(loss_state["loss_type"])
    horizon_logits = logits[:, horizon_idx, :]
    horizon_targets = targets[:, horizon_idx] if targets.ndim == 2 else targets
    if loss_type in {"logit_adjusted_ce", "balanced_softmax", "logit_adjusted_focal"}:
        horizon_logits = horizon_logits + float(loss_state["logit_adjust_tau"]) * loss_state["log_priors_t"][horizon_idx]

    use_class_weights = str(loss_state["class_weighting"]) != "none"
    weight = loss_state["class_weights_t"][horizon_idx] if use_class_weights else None
    if loss_type in {"focal", "balanced_focal", "class_balanced_focal", "logit_adjusted_focal"}:
        ce = F.cross_entropy(horizon_logits, horizon_targets, weight=weight, reduction="none")
        if float(loss_state["focal_gamma"]) <= 0:
            return ce.mean()
        logp = F.log_softmax(horizon_logits, dim=1)
        pt = logp.gather(1, horizon_targets.view(-1, 1)).exp().squeeze(1)
        return (((1.0 - pt) ** float(loss_state["focal_gamma"])) * ce).mean()
    if loss_type in {"ce", "weighted_ce", "logit_adjusted_ce", "balanced_softmax"}:
        return F.cross_entropy(horizon_logits, horizon_targets, weight=weight)
    raise ValueError(f"unknown loss_type={loss_type}")


def multitask_loss_from_logits(loss_state: dict[str, object], logits: np.ndarray, y_arr: np.ndarray, weights: np.ndarray) -> float:
    import torch

    tensor_state = tensorize_loss_state(loss_state, torch.device("cpu"))
    logits_t = torch.from_numpy(logits)
    y_t = torch.from_numpy(y_arr)
    losses = []
    for idx in range(y_arr.shape[1]):
        loss = horizon_training_loss(logits_t, y_t, idx, tensor_state)
        losses.append(float(loss.item()) * float(weights[idx]))
    return float(np.sum(losses))


def decision_calibration_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, object]:
    raw = candidate.get("decision_calibration", config.get("training", {}).get("decision_calibration", False))
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", True))
        cfg = dict(raw)
    else:
        enabled = bool(raw)
        cfg = {}
    cfg["enabled"] = enabled
    cfg.setdefault("grid_min", -0.8)
    cfg.setdefault("grid_max", 0.8)
    cfg.setdefault("grid_steps", 17)
    return cfg


def tune_class_bias(validation_logits: np.ndarray, y_validation: pd.Series, grid: np.ndarray) -> np.ndarray:
    labels = np.asarray(CLASS_ORDER)
    best_score = -np.inf
    best_bias = np.zeros(len(CLASS_ORDER), dtype=np.float32)
    for neg_bias in grid:
        for pos_bias in grid:
            bias = np.array([neg_bias, 0.0, pos_bias], dtype=np.float32)
            pred = labels[np.argmax(validation_logits + bias, axis=1)]
            score = single.f1_score(y_validation, pred, labels=list(CLASS_ORDER), average="macro", zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_bias = bias
    return best_bias


def score_logits_with_bias(logits: np.ndarray, y_true: pd.Series, bias: np.ndarray) -> dict[str, float]:
    return single.score_probabilities(softmax_2d(logits + bias), y_true)


def decision_calibration_summary(
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    validation_logits: np.ndarray,
    test_logits: np.ndarray,
    y_validation: pd.DataFrame,
    y_test: pd.DataFrame,
    horizons: list[str],
    prefix: str,
) -> dict[str, object]:
    cfg = decision_calibration_config(config, candidate)
    out: dict[str, object] = {f"{prefix}_decision_calibration_enabled": bool(cfg["enabled"])}
    if not bool(cfg["enabled"]):
        return out
    grid = np.linspace(float(cfg["grid_min"]), float(cfg["grid_max"]), int(cfg["grid_steps"]), dtype=np.float32)
    validation_macro_values = []
    test_macro_values = []
    validation_weighted_values = []
    test_weighted_values = []
    validation_micro_values = []
    test_micro_values = []
    test_log_loss_values = []
    for horizon_idx, horizon in enumerate(horizons):
        bias = tune_class_bias(validation_logits[:, horizon_idx, :], y_validation[horizon], grid)
        out[f"{horizon}_{prefix}_bias_neg"] = float(bias[0])
        out[f"{horizon}_{prefix}_bias_neutral"] = float(bias[1])
        out[f"{horizon}_{prefix}_bias_pos"] = float(bias[2])
        validation_metrics = score_logits_with_bias(validation_logits[:, horizon_idx, :], y_validation[horizon], bias)
        test_metrics = score_logits_with_bias(test_logits[:, horizon_idx, :], y_test[horizon], bias)
        for key, value in validation_metrics.items():
            out[f"{horizon}_{prefix}_validation_{key}"] = value
        for key, value in test_metrics.items():
            out[f"{horizon}_{prefix}_test_{key}"] = value
        validation_macro_values.append(validation_metrics["macro_f1"])
        test_macro_values.append(test_metrics["macro_f1"])
        validation_weighted_values.append(validation_metrics["weighted_f1"])
        test_weighted_values.append(test_metrics["weighted_f1"])
        validation_micro_values.append(validation_metrics["micro_f1"])
        test_micro_values.append(test_metrics["micro_f1"])
        test_log_loss_values.append(test_metrics["log_loss"])
    out[f"{prefix}_validation_macro_f1_mean"] = float(np.mean(validation_macro_values))
    out[f"{prefix}_test_macro_f1_mean"] = float(np.mean(test_macro_values))
    out[f"{prefix}_validation_weighted_f1_mean"] = float(np.mean(validation_weighted_values))
    out[f"{prefix}_test_weighted_f1_mean"] = float(np.mean(test_weighted_values))
    out[f"{prefix}_validation_micro_f1_mean"] = float(np.mean(validation_micro_values))
    out[f"{prefix}_test_micro_f1_mean"] = float(np.mean(test_micro_values))
    out[f"{prefix}_test_log_loss_mean"] = float(np.mean(test_log_loss_values))
    return out


def metric_summary(
    *,
    logits: np.ndarray,
    y: pd.DataFrame,
    horizons: list[str],
    prefix: str,
) -> dict[str, float]:
    proba = multitask_softmax(logits)
    out: dict[str, float] = {}
    macro_values = []
    weighted_values = []
    micro_values = []
    log_loss_values = []
    for idx, horizon in enumerate(horizons):
        metrics = single.score_probabilities(proba[:, idx, :], y[horizon])
        for key, value in metrics.items():
            out[f"{horizon}_{prefix}_{key}"] = value
        macro_values.append(metrics["macro_f1"])
        weighted_values.append(metrics["weighted_f1"])
        micro_values.append(metrics["micro_f1"])
        log_loss_values.append(metrics["log_loss"])
    out[f"{prefix}_macro_f1_mean"] = float(np.mean(macro_values))
    out[f"{prefix}_macro_f1_min"] = float(np.min(macro_values))
    out[f"{prefix}_weighted_f1_mean"] = float(np.mean(weighted_values))
    out[f"{prefix}_weighted_f1_min"] = float(np.min(weighted_values))
    out[f"{prefix}_micro_f1_mean"] = float(np.mean(micro_values))
    out[f"{prefix}_micro_f1_min"] = float(np.min(micro_values))
    out[f"{prefix}_log_loss_mean"] = float(np.mean(log_loss_values))
    return out


def time_multitask_model_on_requested_devices(
    *,
    model,
    config: dict,
    x_test_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
    base_d: int,
) -> dict[str, object]:
    import torch

    latency_cfg = dict(config["latency"])
    repeats = int(latency_cfg.get("repeats", 3))
    samples = int(latency_cfg.get("single_timing_samples", 256))
    n_samples = min(len(x_test_raw), max(1, samples))
    indices = np.linspace(0, len(x_test_raw) - 1, n_samples, dtype=int)
    out: dict[str, object] = {}
    completed_devices = []

    for device_name in single.requested_timing_devices(config):
        available, status = single.timing_device_available(device_name)
        out[f"{device_name}_timing_status"] = status
        out[f"{device_name}_single_prediction_time_s_median"] = np.nan
        out[f"{device_name}_single_latency_us_per_obs_median"] = np.nan
        if not available:
            continue
        try:
            device = torch_device(device_name)
            model.to(device)
            model.eval()

            def predict_one(idx: int):
                arr = (np.asarray(x_test_raw[idx : idx + 1], dtype=np.float32) - mean) / std
                tensor = sequence_tensor_from_flat(arr, window=window, base_d=base_d).to(device)
                logits = model(tensor)
                return torch.softmax(logits, dim=2)

            with torch.no_grad():
                warmup = predict_one(int(indices[0]))
            if warmup.shape[0] != 1:
                raise ValueError("multitask warmup returned wrong batch size")
            timings = []
            with torch.no_grad():
                for _ in range(repeats):
                    for idx in indices:
                        start = time.perf_counter()
                        proba = predict_one(int(idx))
                        if device.type == "mps":
                            torch.mps.synchronize()
                        elif device.type == "cuda":
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        if proba.shape[0] != 1:
                            raise ValueError("multitask timing returned wrong batch size")
                        timings.append(elapsed)
            median_s = float(np.median(timings))
            out[f"{device_name}_timing_status"] = "ok"
            out[f"{device_name}_single_prediction_time_s_median"] = median_s
            out[f"{device_name}_single_latency_us_per_obs_median"] = median_s * 1_000_000
            completed_devices.append(device_name)
        except Exception as exc:  # noqa: BLE001
            out[f"{device_name}_timing_status"] = f"error:{type(exc).__name__}:{exc}"

    out["timing_devices_requested"] = json.dumps(single.requested_timing_devices(config))
    out["timing_devices_completed"] = json.dumps(completed_devices)
    if "cpu_single_latency_us_per_obs_median" not in out:
        out["cpu_timing_status"] = "not_requested"
        out["cpu_single_prediction_time_s_median"] = np.nan
        out["cpu_single_latency_us_per_obs_median"] = np.nan
    return out


def prepare_feature_mode(
    *,
    config: dict,
    split: dict[str, object],
    feature_mode_raw: str,
    max_train_rows: int,
    max_test_rows: int,
) -> dict[str, object]:
    feature_mode = single.normalize_feature_mode(feature_mode_raw)
    window = int(config["window"])
    x_train_segments_all = split["x_train_segments"]
    y_train_segments_all = split["y_train_segments"]
    x_test_segments_all = split["x_test_segments"]
    y_test_segments_all = split["y_test_segments"]
    columns = single.select_feature_columns(x_train_segments_all[0], feature_mode)
    x_train_full, y_train_full = single.build_lagged_from_segments(
        x_train_segments_all,
        y_train_segments_all,
        columns=columns,
        window=window,
    )
    x_test, y_test = single.build_lagged_from_segments(
        x_test_segments_all,
        y_test_segments_all,
        columns=columns,
        window=window,
    )
    x_test, y_test = maybe_limit_rows(x_test, y_test, max_test_rows)
    x_train, y_train = maybe_limit_rows(x_train_full, y_train_full, max_train_rows)
    x_fit_df, y_fit, x_validation_df, y_validation = chronological_train_validation_split(
        x_train,
        y_train,
        float(config["training"].get("validation_fraction", 0.2)),
    )
    x_fit_np = x_fit_df.to_numpy(dtype=np.float32)
    x_validation_np = x_validation_df.to_numpy(dtype=np.float32)
    x_test_np = x_test.to_numpy(dtype=np.float32)
    x_fit_std, x_validation_std, x_test_std, mean, std = standardize_fit(x_fit_np, x_validation_np, x_test_np)
    return {
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


def finetune_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, object]:
    raw = candidate.get("finetune", config.get("training", {}).get("finetune", False))
    if isinstance(raw, dict):
        out = dict(raw)
        out["enabled"] = bool(out.get("enabled", True))
    else:
        out = {"enabled": bool(raw)}
    out.setdefault("mode", "head")
    out.setdefault("epochs", 6)
    out.setdefault("patience", 3)
    out.setdefault("learning_rate", 0.0001)
    out.setdefault("optimizer", config.get("training", {}).get("optimizer", "adamw"))
    out.setdefault("weight_decay", 0.0)
    out.setdefault("batch_size", config.get("training", {}).get("batch_size", 32))
    return out


def configure_finetune_parameters(model, horizon_idx: int, mode: str) -> list:
    for param in model.parameters():
        param.requires_grad_(False)
    modules = [model.heads[horizon_idx]]
    if mode == "head_last_block":
        if len(model.blocks) > 0:
            modules.append(model.blocks[-1])
        modules.append(model.final_norm)
    elif mode == "all":
        modules = [model]
    elif mode != "head":
        raise ValueError("finetune.mode must be head, head_last_block, or all")
    params = []
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(True)
            params.append(param)
    return params


def run_single_horizon_finetune(
    *,
    base_model,
    config: dict[str, Any],
    candidate: dict[str, Any],
    horizons: list[str],
    train_tensor,
    train_y,
    validation_tensor,
    y_validation: pd.DataFrame,
    test_tensor,
    y_test: pd.DataFrame,
    loss_state: dict[str, object],
    device,
    history_dir: Path,
    per_horizon_base_states: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = finetune_config(config, candidate)
    out: dict[str, object] = {
        "finetune_enabled": bool(cfg["enabled"]),
        "finetune_mode": str(cfg["mode"]),
        "finetune_epochs": int(cfg["epochs"]),
        "finetune_base_checkpointing": "per_horizon" if per_horizon_base_states else "global",
    }
    if not bool(cfg["enabled"]):
        return out

    batch_size = int(cfg["batch_size"])
    eval_batch_size = int(config["training"].get("eval_batch_size", 64))
    patience = int(cfg["patience"])
    loss_state_device = tensorize_loss_state(loss_state, device)
    loader = DataLoader(TensorDataset(train_tensor, train_y), batch_size=batch_size, shuffle=True)
    validation_macro_values = []
    test_macro_values = []
    calibrated_validation_macro_values = []
    calibrated_test_macro_values = []
    calibrated_enabled = bool(decision_calibration_config(config, candidate)["enabled"])

    for horizon_idx, horizon in enumerate(horizons):
        ft_model = copy.deepcopy(base_model).to(device)
        if per_horizon_base_states and horizon in per_horizon_base_states:
            ft_model.load_state_dict(per_horizon_base_states[horizon])
            ft_model.to(device)
        params = configure_finetune_parameters(ft_model, horizon_idx, str(cfg["mode"]))
        optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
        optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(
            params,
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            eps=float(config["training"].get("adam_eps", 1e-8)),
        )
        best_score = -np.inf
        best_epoch = 0
        best_state = None
        bad_epochs = 0
        history_rows = []
        for epoch in range(1, int(cfg["epochs"]) + 1):
            ft_model.train()
            train_losses = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = ft_model(batch_x.to(device))
                loss = horizon_training_loss(logits, batch_y.to(device), horizon_idx, loss_state_device)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            val_logits = collect_multitask_logits(ft_model, validation_tensor, device, eval_batch_size)
            validation_metrics = single.score_probabilities(softmax_2d(val_logits[:, horizon_idx, :]), y_validation[horizon])
            score = float(validation_metrics["macro_f1"])
            improved = score > best_score + float(config["training"].get("min_delta", 1e-5))
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in ft_model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            history_rows.append(
                {
                    "epoch": epoch,
                    "horizon": horizon,
                    "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                    "validation_macro_f1": score,
                    "best_validation_macro_f1": best_score,
                    "bad_epochs": bad_epochs,
                }
            )
            if bad_epochs >= patience:
                break

        if best_state is not None:
            ft_model.load_state_dict(best_state)
            ft_model.to(device)

        validation_logits = collect_multitask_logits(ft_model, validation_tensor, device, eval_batch_size)
        test_logits = collect_multitask_logits(ft_model, test_tensor, device, eval_batch_size)
        validation_metrics = single.score_probabilities(softmax_2d(validation_logits[:, horizon_idx, :]), y_validation[horizon])
        test_metrics = single.score_probabilities(softmax_2d(test_logits[:, horizon_idx, :]), y_test[horizon])
        for key, value in validation_metrics.items():
            out[f"{horizon}_finetune_validation_{key}"] = value
        for key, value in test_metrics.items():
            out[f"{horizon}_finetune_test_{key}"] = value
        out[f"{horizon}_finetune_best_epoch"] = best_epoch
        validation_macro_values.append(validation_metrics["macro_f1"])
        test_macro_values.append(test_metrics["macro_f1"])

        history_path = history_dir / f"{safe_filename(candidate['name'] + '_' + horizon + '_finetune')}_history.csv"
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        out[f"{horizon}_finetune_history_path"] = str(history_path)

        if calibrated_enabled:
            calibration = decision_calibration_summary(
                config=config,
                candidate=candidate,
                validation_logits=validation_logits[:, horizon_idx : horizon_idx + 1, :],
                test_logits=test_logits[:, horizon_idx : horizon_idx + 1, :],
                y_validation=y_validation,
                y_test=y_test,
                horizons=[horizon],
                prefix="finetune_calibrated",
            )
            out.update(calibration)
            calibrated_validation_macro_values.append(float(calibration["finetune_calibrated_validation_macro_f1_mean"]))
            calibrated_test_macro_values.append(float(calibration["finetune_calibrated_test_macro_f1_mean"]))

    out["finetune_validation_macro_f1_mean"] = float(np.mean(validation_macro_values)) if validation_macro_values else np.nan
    out["finetune_test_macro_f1_mean"] = float(np.mean(test_macro_values)) if test_macro_values else np.nan
    if calibrated_validation_macro_values:
        out["finetune_calibrated_validation_macro_f1_mean"] = float(np.mean(calibrated_validation_macro_values))
        out["finetune_calibrated_test_macro_f1_mean"] = float(np.mean(calibrated_test_macro_values))
    out["finetune_latency_inherits_base_model"] = True
    return out


def evaluate_per_horizon_checkpoints(
    *,
    model,
    states: dict[str, dict[str, object]],
    horizons: list[str],
    validation_tensor,
    y_validation: pd.DataFrame,
    test_tensor,
    y_test: pd.DataFrame,
    device,
    eval_batch_size: int,
) -> dict[str, object]:
    out: dict[str, object] = {
        "horizon_checkpointing_enabled": bool(states),
    }
    if not states:
        return out

    live_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    validation_macro_values = []
    test_macro_values = []
    validation_weighted_values = []
    test_weighted_values = []
    validation_micro_values = []
    test_micro_values = []
    test_log_loss_values = []
    for horizon_idx, horizon in enumerate(horizons):
        state = states.get(horizon)
        if state is None:
            continue
        model.load_state_dict(state)
        model.to(device)
        validation_logits = collect_multitask_logits(model, validation_tensor, device, eval_batch_size)
        test_logits = collect_multitask_logits(model, test_tensor, device, eval_batch_size)
        validation_metrics = single.score_probabilities(softmax_2d(validation_logits[:, horizon_idx, :]), y_validation[horizon])
        test_metrics = single.score_probabilities(softmax_2d(test_logits[:, horizon_idx, :]), y_test[horizon])
        for key, value in validation_metrics.items():
            out[f"{horizon}_horizon_checkpoint_validation_{key}"] = value
        for key, value in test_metrics.items():
            out[f"{horizon}_horizon_checkpoint_test_{key}"] = value
        validation_macro_values.append(validation_metrics["macro_f1"])
        test_macro_values.append(test_metrics["macro_f1"])
        validation_weighted_values.append(validation_metrics["weighted_f1"])
        test_weighted_values.append(test_metrics["weighted_f1"])
        validation_micro_values.append(validation_metrics["micro_f1"])
        test_micro_values.append(test_metrics["micro_f1"])
        test_log_loss_values.append(test_metrics["log_loss"])

    model.load_state_dict(live_state)
    model.to(device)
    if validation_macro_values:
        out["horizon_checkpoint_validation_macro_f1_mean"] = float(np.mean(validation_macro_values))
        out["horizon_checkpoint_test_macro_f1_mean"] = float(np.mean(test_macro_values))
        out["horizon_checkpoint_validation_weighted_f1_mean"] = float(np.mean(validation_weighted_values))
        out["horizon_checkpoint_test_weighted_f1_mean"] = float(np.mean(test_weighted_values))
        out["horizon_checkpoint_validation_micro_f1_mean"] = float(np.mean(validation_micro_values))
        out["horizon_checkpoint_test_micro_f1_mean"] = float(np.mean(test_micro_values))
        out["horizon_checkpoint_test_log_loss_mean"] = float(np.mean(test_log_loss_values))
    return out


def fit_one_multitask(
    *,
    config: dict,
    candidate: dict[str, Any],
    horizons: list[str],
    data: dict[str, object],
    split_meta: dict[str, object],
    history_dir: Path,
    candidate_index: int,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    training = dict(config["training"])
    window = int(config["window"])
    base_d = int(config["base_d"])
    base_seed = int(training.get("seed", 0))
    if "seed" in candidate:
        base_seed = int(candidate["seed"])
    elif "seed_offset" in candidate:
        base_seed += int(candidate["seed_offset"])
    else:
        base_seed += candidate_index * 1009
    seed = base_seed + sum(sum(ord(ch) for ch in h) for h in horizons)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch_device(str(training.get("device", "auto")))
    model = build_multitask_model(candidate, config=config, horizons=horizons).to(device)
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.0003)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
        eps=float(training.get("adam_eps", 1e-8)),
    )
    batch_size = int(training.get("batch_size", 32))
    eval_batch_size = int(training.get("eval_batch_size", 64))
    ema_decay = float(training.get("ema_decay", 0.0))
    ema_state = None
    if ema_decay > 0:
        ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def update_ema_state() -> None:
        if ema_state is None:
            return
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if torch.is_floating_point(value):
                    ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                else:
                    ema_state[key].copy_(value.detach())

    def collect_eval_logits(tensor):
        if ema_state is None:
            return collect_multitask_logits(model, tensor, device, eval_batch_size)
        live_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema_state)
        logits = collect_multitask_logits(model, tensor, device, eval_batch_size)
        model.load_state_dict(live_state)
        return logits

    def snapshot_selection_state():
        state = ema_state if ema_state is not None else model.state_dict()
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    raw_weights = training.get("horizon_loss_weights", [1.0] * len(horizons))
    if len(raw_weights) != len(horizons):
        raise ValueError("training.horizon_loss_weights must match horizons")
    weights = np.asarray(raw_weights, dtype=np.float32)
    weights = weights / weights.sum()

    y_fit = data["y_fit"]
    y_validation = data["y_validation"]
    y_test = data["y_test"]
    y_fit_arr = np.stack([y_fit[h].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64) for h in horizons], axis=1)
    y_validation_arr = np.stack([y_validation[h].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64) for h in horizons], axis=1)
    loss_state = make_loss_state(y_fit_arr, candidate, training)
    loss_state_device = tensorize_loss_state(loss_state, device)
    train_tensor = sequence_tensor_from_flat(data["x_fit_std"], window=window, base_d=base_d)
    train_y = torch.from_numpy(y_fit_arr)
    validation_tensor = sequence_tensor_from_flat(data["x_validation_std"], window=window, base_d=base_d)
    validation_y = torch.from_numpy(y_validation_arr)
    loader = DataLoader(
        TensorDataset(train_tensor, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    selection_metric = str(training.get("selection_metric", "validation_macro_f1_mean"))
    if not selection_metric.startswith("validation_"):
        selection_metric = f"validation_{selection_metric}"
    maximize_selection = not selection_metric.endswith("log_loss_mean")
    best_selection_score = -np.inf if maximize_selection else np.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history_rows = []
    epochs = int(training.get("epochs", 30))
    patience = int(training.get("patience", 8))
    min_delta = float(training.get("min_delta", 0.00005))
    per_horizon_checkpointing = bool(candidate.get("per_horizon_checkpointing", training.get("per_horizon_checkpointing", False)))
    per_horizon_best_scores = {horizon: -np.inf for horizon in horizons}
    per_horizon_best_epochs = {horizon: 0 for horizon in horizons}
    per_horizon_states: dict[str, dict[str, object]] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = 0.0
            for horizon_idx in range(len(horizons)):
                loss = loss + float(weights[horizon_idx]) * horizon_training_loss(logits, batch_y.to(device), horizon_idx, loss_state_device)
            loss.backward()
            optimizer.step()
            update_ema_state()
            train_losses.append(float(loss.detach().cpu().item()))

        val_logits = collect_eval_logits(validation_tensor)
        val_loss_mean = multitask_loss_from_logits(loss_state, val_logits, y_validation_arr, weights)
        epoch_metrics = metric_summary(logits=val_logits, y=y_validation, horizons=horizons, prefix="validation")
        epoch_metrics["validation_log_loss_mean"] = val_loss_mean
        if selection_metric not in epoch_metrics:
            raise ValueError(f"unknown selection_metric={selection_metric}")
        selection_score = float(epoch_metrics[selection_metric])
        if maximize_selection:
            improved = selection_score > best_selection_score + min_delta
        else:
            improved = selection_score < best_selection_score - min_delta
        if improved:
            best_selection_score = selection_score
            best_epoch = epoch
            best_state = snapshot_selection_state()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if per_horizon_checkpointing:
            improved_horizons = []
            for horizon in horizons:
                horizon_score = float(epoch_metrics[f"{horizon}_validation_macro_f1"])
                if horizon_score > per_horizon_best_scores[horizon] + min_delta:
                    per_horizon_best_scores[horizon] = horizon_score
                    per_horizon_best_epochs[horizon] = epoch
                    improved_horizons.append(horizon)
            if improved_horizons:
                epoch_state = snapshot_selection_state()
                for horizon in improved_horizons:
                    per_horizon_states[horizon] = {
                        key: value.detach().cpu().clone() for key, value in epoch_state.items()
                    }

        history_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            "best_selection_score": best_selection_score,
            "bad_epochs": bad_epochs,
        }
        if per_horizon_checkpointing:
            for horizon in horizons:
                history_row[f"{horizon}_best_validation_macro_f1"] = per_horizon_best_scores[horizon]
                history_row[f"{horizon}_best_epoch"] = per_horizon_best_epochs[horizon]
        history_row.update(epoch_metrics)
        history_rows.append(history_row)
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    validation_logits = collect_multitask_logits(model, validation_tensor, device, eval_batch_size)
    test_tensor = sequence_tensor_from_flat(data["x_test_std"], window=window, base_d=base_d)
    test_logits = collect_multitask_logits(model, test_tensor, device, eval_batch_size)
    validation_metrics = metric_summary(logits=validation_logits, y=y_validation, horizons=horizons, prefix="validation")
    test_metrics = metric_summary(logits=test_logits, y=y_test, horizons=horizons, prefix="test")
    calibration_metrics = decision_calibration_summary(
        config=config,
        candidate=candidate,
        validation_logits=validation_logits,
        test_logits=test_logits,
        y_validation=y_validation,
        y_test=y_test,
        horizons=horizons,
        prefix="calibrated",
    )
    horizon_checkpoint_metrics = evaluate_per_horizon_checkpoints(
        model=model,
        states=per_horizon_states if per_horizon_checkpointing else {},
        horizons=horizons,
        validation_tensor=validation_tensor,
        y_validation=y_validation,
        test_tensor=test_tensor,
        y_test=y_test,
        device=device,
        eval_batch_size=eval_batch_size,
    )

    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{safe_filename(candidate['name'] + '_multitask')}_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    timing_metrics = time_multitask_model_on_requested_devices(
        model=model,
        config=config,
        x_test_raw=data["x_test_np"],
        mean=data["mean"],
        std=data["std"],
        window=window,
        base_d=base_d,
    )
    finetune_metrics = run_single_horizon_finetune(
        base_model=model,
        config=config,
        candidate=candidate,
        horizons=horizons,
        train_tensor=train_tensor,
        train_y=train_y,
        validation_tensor=validation_tensor,
        y_validation=y_validation,
        test_tensor=test_tensor,
        y_test=y_test,
        loss_state=loss_state,
        device=device,
        history_dir=history_dir,
        per_horizon_base_states=per_horizon_states if per_horizon_checkpointing else None,
    )

    row: dict[str, object] = {
        "experiment": config["experiment"],
        "cf": int(config["cf"]),
        "evaluation_setup": str(config.get("evaluation_setup", "anchored_cf")),
        "train_cfs": json.dumps(split_meta["train_cfs"]),
        "test_cfs": json.dumps(split_meta["test_cfs"]),
        "preserve_day_boundaries": bool(split_meta["preserve_day_boundaries"]),
        "score_target": "multitask",
        "horizons": json.dumps(horizons),
        "candidate_index": candidate_index,
        "candidate_seed": candidate.get("seed", ""),
        "candidate_seed_offset": candidate.get("seed_offset", ""),
        "lane": candidate["lane"],
        "architecture": candidate["architecture"],
        "model_name": candidate["name"],
        "feature_mode": str(config.get("feature_mode", "full144")),
        "feature_count": base_d,
        "input_window": window,
        "base_d": base_d,
        "d": window * base_d,
        "fit_rows": len(data["x_fit_std"]),
        "validation_rows": len(data["x_validation_std"]),
        "test_rows": len(data["x_test_std"]),
        "train_source_rows": int(split_meta["train_source_rows"]),
        "test_source_rows": int(split_meta["test_source_rows"]),
        "torch_device": str(device),
        "torch_timing_device": ",".join(single.requested_timing_devices(config)),
        "torch_seed": seed,
        "torch_epochs": epochs,
        "torch_epochs_ran": len(history_rows),
        "torch_best_epoch": best_epoch,
        "torch_stopped_early": int(len(history_rows) < epochs),
        "torch_selection_metric": selection_metric,
        "torch_best_selection_score": best_selection_score,
        "torch_history_path": str(history_path),
        "per_horizon_checkpointing": per_horizon_checkpointing,
        "training_optimizer": optimizer_name,
        "training_learning_rate": float(training.get("learning_rate", 0.0003)),
        "training_weight_decay": float(training.get("weight_decay", 0.0001)),
        "training_ema_decay": ema_decay,
        "training_ema_enabled": bool(ema_state is not None),
        "latency_repeats": int(config.get("latency", {}).get("repeats", 3)),
        "latency_single_timing_samples": int(config.get("latency", {}).get("single_timing_samples", 256)),
        "latency_synthetic_rows": int(config.get("latency", {}).get("synthetic_rows", 512)),
        "loss_type": str(loss_state["loss_type"]),
        "loss_class_weighting": str(loss_state["class_weighting"]),
        "loss_focal_gamma": float(loss_state["focal_gamma"]),
        "loss_logit_adjust_tau": float(loss_state["logit_adjust_tau"]),
        "audited_forward_ops": multitask_fastbinlob_ops(candidate, window=window, base_d=base_d, horizons=horizons),
        "compute_audit_method": "analytic_multitask_fastbinlob_joint_forward_multiply_adds",
        "compute_audit_confidence": "medium_high",
        "n_parameters": parameter_count(model),
    }
    row.update(timing_metrics)
    row.update(validation_metrics)
    row.update(test_metrics)
    row.update(calibration_metrics)
    row.update(horizon_checkpoint_metrics)
    row.update(finetune_metrics)
    row["validation_macro_f1_mean"] = validation_metrics["validation_macro_f1_mean"]
    row["test_macro_f1_mean"] = test_metrics["test_macro_f1_mean"]
    row["validation_weighted_f1_mean"] = validation_metrics["validation_weighted_f1_mean"]
    row["test_weighted_f1_mean"] = test_metrics["test_weighted_f1_mean"]
    row["validation_log_loss_mean"] = validation_metrics["validation_log_loss_mean"]
    row["test_log_loss_mean"] = test_metrics["test_log_loss_mean"]
    for horizon in horizons:
        if per_horizon_checkpointing:
            row[f"{horizon}_horizon_checkpoint_best_epoch"] = per_horizon_best_epochs[horizon]
            row[f"{horizon}_horizon_checkpoint_best_validation_macro_f1"] = per_horizon_best_scores[horizon]
        row.update(single.class_balance(y_fit[horizon], f"{horizon}_fit"))
        row.update(single.class_balance(y_validation[horizon], f"{horizon}_validation"))
        row.update(single.class_balance(y_test[horizon], f"{horizon}_test"))
    for key, value in candidate.items():
        if key not in {"name", "lane", "architecture"}:
            row[f"candidate_{key}"] = json.dumps(value) if isinstance(value, list) else value
    return row


def filter_candidates(config: dict, names: str, max_candidates: int) -> list[dict]:
    candidates = list(config["candidates"])
    if names:
        wanted = {part.strip() for part in names.split(",") if part.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in wanted]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run joint-horizon multitask latency-SOTA experiments.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--candidate-names", default="")
    parser.add_argument("--horizons", default="")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.epochs > 0:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["training"]["device"] = args.device
    horizons = [part.strip() for part in args.horizons.split(",") if part.strip()] if args.horizons else list(config["horizons"])
    candidates = filter_candidates(config, args.candidate_names, args.max_candidates)
    default_feature_mode = single.feature_mode_from_config(config)
    evaluation_setup = single.normalize_evaluation_setup(str(config.get("evaluation_setup", "anchored_cf")))

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
                    "horizons": horizons,
                    "candidates": [
                        {
                            "name": candidate["name"],
                            "feature_mode": single.normalize_feature_mode(str(candidate.get("feature_mode", default_feature_mode))),
                        }
                        for candidate in candidates
                    ],
                },
                indent=2,
            )
        )
        return

    existing = pd.DataFrame()
    rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, str]] = set()
    if args.output.exists() and args.output.stat().st_size > 0 and not args.force:
        existing = pd.read_csv(args.output)
        rows.extend(existing.to_dict("records"))
        completed = set(
            zip(
                existing.get("evaluation_setup", pd.Series([evaluation_setup] * len(existing))).astype(str).map(single.normalize_evaluation_setup),
                existing.get("feature_mode", pd.Series([default_feature_mode] * len(existing))).astype(str).map(single.normalize_feature_mode),
                existing["model_name"].astype(str),
            )
        )

    data_root = find_data_root()
    split = single.load_evaluation_split(config, data_root)
    split_meta = {
        "evaluation_setup": split["evaluation_setup"],
        "train_cfs": split["train_cfs"],
        "test_cfs": split["test_cfs"],
        "preserve_day_boundaries": split["preserve_day_boundaries"],
        "train_source_rows": sum(len(item) for item in split["x_train_segments"]),
        "test_source_rows": sum(len(item) for item in split["x_test_segments"]),
    }
    prepared: dict[str, dict[str, object]] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.history_dir.mkdir(parents=True, exist_ok=True)
    for candidate_index, candidate in enumerate(candidates):
        feature_mode = single.normalize_feature_mode(str(candidate.get("feature_mode", default_feature_mode)))
        key = (evaluation_setup, feature_mode, candidate["name"])
        if key in completed:
            print(f"skip existing {feature_mode} {candidate['name']} multitask", flush=True)
            continue
        if feature_mode not in prepared:
            prepared[feature_mode] = prepare_feature_mode(
                config=config,
                split=split,
                feature_mode_raw=feature_mode,
                max_train_rows=args.max_train_rows,
                max_test_rows=args.max_test_rows,
            )
        data = prepared[feature_mode]
        candidate_config = copy.deepcopy(config)
        candidate_config.update(split_meta)
        candidate_config["feature_mode"] = feature_mode
        candidate_config["base_d"] = int(data["base_d"])
        print(f"fit {feature_mode} {candidate['name']} multitask {','.join(horizons)}", flush=True)
        row = fit_one_multitask(
            config=candidate_config,
            candidate=copy.deepcopy(candidate),
            horizons=horizons,
            data=data,
            split_meta=split_meta,
            history_dir=args.history_dir,
            candidate_index=candidate_index,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        parts = [
            f"{h} mf1={row[f'{h}_test_macro_f1']:.4f} wf1={row[f'{h}_test_weighted_f1']:.4f}"
            for h in horizons
        ]
        print(
            f"{candidate['name']} multitask: "
            f"mean_mf1={row['test_macro_f1_mean']:.4f} "
            + " ".join(parts)
            + (f" cal_mean={row['calibrated_test_macro_f1_mean']:.4f}" if "calibrated_test_macro_f1_mean" in row else "")
            + (
                f" hckpt_mean={row['horizon_checkpoint_test_macro_f1_mean']:.4f}"
                if "horizon_checkpoint_test_macro_f1_mean" in row
                else ""
            )
            + (f" ft_mean={row['finetune_test_macro_f1_mean']:.4f}" if "finetune_test_macro_f1_mean" in row else "")
            + (
                f" ft_cal_mean={row['finetune_calibrated_test_macro_f1_mean']:.4f}"
                if "finetune_calibrated_test_macro_f1_mean" in row
                else ""
            )
            + f" lat={row['cpu_single_latency_us_per_obs_median']:.1f}us",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
