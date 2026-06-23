from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


CLASS_ORDER = (-1, 0, 1)


def torch_device(raw: str):
    import torch

    if raw == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(raw)


class SimpleBiN:
    """Factory wrapper to avoid importing torch at module import time."""

    @staticmethod
    def build(base_d: int, bin_eps: float):
        import torch
        from torch import nn

        class _SimpleBiN(nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = nn.Parameter(torch.ones(1, base_d, 1))
                self.beta = nn.Parameter(torch.zeros(1, base_d, 1))
                self.mix_logit = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                time_std = x.std(dim=2, keepdim=True, unbiased=False).clamp_min(bin_eps)
                feature_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(bin_eps)
                time_norm = (x - x.mean(dim=2, keepdim=True)) / time_std
                feature_norm = (x - x.mean(dim=1, keepdim=True)) / feature_std
                mix = torch.sigmoid(self.mix_logit)
                return (mix * time_norm + (1.0 - mix) * feature_norm) * self.gamma + self.beta

        return _SimpleBiN()


def activation_module(kind: str):
    from torch import nn

    if kind == "relu":
        return nn.ReLU()
    if kind == "silu":
        return nn.SiLU()
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation: {kind}")


def build_official_mlplob(*, base_d: int, window: int, hidden_dim: int, layers: int):
    """Device-neutral reproduction of the MLPLOB baseline used for timing."""
    import torch
    from torch import nn

    if hidden_dim <= 0 or layers <= 0:
        raise ValueError("official_mlplob hidden_dim and layers must be positive")
    if hidden_dim // 4 <= 0 or window // 4 <= 0:
        raise ValueError("official_mlplob requires hidden_dim>=4 and window>=4")

    class OfficialBiN(nn.Module):
        def __init__(self):
            super().__init__()
            self.t1 = window
            self.d1 = base_d
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

            t2 = torch.ones([self.t1, 1], device=x.device, dtype=x.dtype)
            x2 = torch.mean(x, dim=2).reshape(x.shape[0], x.shape[1], 1)
            std = torch.std(x, dim=2).reshape(x.shape[0], x.shape[1], 1)
            std = torch.where(std < 1e-4, torch.ones_like(std), std)
            z2 = (x - (x2 @ t2.T)) / (std @ t2.T)
            x_temporal = (self.l2 @ t2.T) * z2 + (self.B2 @ t2.T)

            t1 = torch.ones([self.d1, 1], device=x.device, dtype=x.dtype)
            x1 = torch.mean(x, dim=1).reshape(x.shape[0], x.shape[2], 1)
            std = torch.std(x, dim=1).reshape(x.shape[0], x.shape[2], 1)
            op1 = torch.permute(x1 @ t1.T, (0, 2, 1))
            op2 = torch.permute(std @ t1.T, (0, 2, 1))
            z1 = (x - op1) / op2
            x_feature = (t1 @ self.l1.T) * z1 + (t1 @ self.B1.T)

            return self.y1 * x_feature + self.y2 * x_temporal

    class OfficialMLP(nn.Module):
        def __init__(self, start_dim: int, hidden: int, final_dim: int):
            super().__init__()
            self.layer_norm = nn.LayerNorm(final_dim)
            self.fc = nn.Linear(start_dim, hidden)
            self.fc2 = nn.Linear(hidden, final_dim)
            self.gelu = nn.GELU()

        def forward(self, x):
            residual = x
            x = self.gelu(self.fc(x))
            x = self.fc2(x)
            if x.shape[2] == residual.shape[2]:
                x = x + residual
            return self.gelu(self.layer_norm(x))

    class OfficialMLPLOB(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_layer = OfficialBiN()
            self.layers = nn.ModuleList([nn.Linear(base_d, hidden_dim), nn.GELU()])
            for idx in range(layers):
                if idx != layers - 1:
                    self.layers.append(OfficialMLP(hidden_dim, hidden_dim * 4, hidden_dim))
                    self.layers.append(OfficialMLP(window, window * 4, window))
                else:
                    self.layers.append(OfficialMLP(hidden_dim, hidden_dim * 2, hidden_dim // 4))
                    self.layers.append(OfficialMLP(window, window * 2, window // 4))

            total_dim = (hidden_dim // 4) * (window // 4)
            self.final_layers = nn.ModuleList()
            while total_dim > 128:
                self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
                self.final_layers.append(nn.GELU())
                total_dim = total_dim // 4
            self.final_layers.append(nn.Linear(total_dim, len(CLASS_ORDER)))

        def forward(self, x):
            x = self.norm_layer(x).permute(0, 2, 1)
            for layer in self.layers:
                x = layer(x)
                x = x.permute(0, 2, 1)
            x = x.reshape(x.shape[0], -1)
            for layer in self.final_layers:
                x = layer(x)
            return x

    return OfficialMLPLOB()


def build_tlob(
    *,
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

    class TLOBBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_norm = nn.LayerNorm(hidden_dim)
            self.temporal_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
            self.spatial_norm = nn.LayerNorm(hidden_dim)
            self.spatial_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
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
            spatial_update, _ = self.spatial_attention(spatial_norm, spatial_norm, spatial_norm, need_weights=False)
            spatial = spatial + self.dropout(spatial_update)
            z = spatial.reshape(batch, time_steps, features, width)
            return z + self.ffn(self.ffn_norm(z))

    class TLOB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bin = SimpleBiN.build(base_d, bin_eps) if use_bin else nn.Identity()
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
            return self.output(self.norm(z).mean(dim=(1, 2)))

    return TLOB()


def build_model(candidate: dict[str, Any], *, window: int, base_d: int, training: dict[str, Any]):
    architecture = candidate["architecture"]
    dropout = float(candidate.get("dropout", training.get("dropout", 0.1)))
    if architecture == "official_mlplob":
        return build_official_mlplob(
            base_d=base_d,
            window=window,
            hidden_dim=int(candidate["hidden_dim"]),
            layers=int(candidate["layers"]),
        )
    if architecture in {"tlob", "torch_tlob"}:
        return build_tlob(
            base_d=base_d,
            window=window,
            hidden_dim=int(candidate["hidden_dim"]),
            layers=int(candidate["layers"]),
            heads=int(candidate["heads"]),
            ff_multiplier=int(candidate.get("ff_multiplier", 2)),
            dropout=dropout,
            use_bin=bool(training.get("use_bin", True)),
            bin_eps=float(training.get("bin_eps", 1e-5)),
        )
    raise ValueError(f"unknown paper-core latency architecture: {architecture}")


def official_mlplob_ops(*, window: int, base_d: int, hidden_dim: int, layers: int) -> int:
    embed_ops = window * base_d * hidden_dim
    full_feature_mlp_ops = 2 * window * hidden_dim * hidden_dim * 4
    full_temporal_mlp_ops = 2 * hidden_dim * window * window * 4
    nonfinal_ops = max(0, layers - 1) * (full_feature_mlp_ops + full_temporal_mlp_ops)

    reduced_hidden = hidden_dim // 4
    reduced_window = window // 4
    final_feature_ops = window * (hidden_dim * hidden_dim * 2 + hidden_dim * 2 * reduced_hidden)
    final_temporal_ops = reduced_hidden * (window * window * 2 + window * 2 * reduced_window)

    head_ops = 0
    total_dim = reduced_hidden * reduced_window
    while total_dim > 128:
        next_dim = total_dim // 4
        head_ops += total_dim * next_dim
        total_dim = next_dim
    head_ops += total_dim * len(CLASS_ORDER)
    return int(embed_ops + nonfinal_ops + final_feature_ops + final_temporal_ops + head_ops)


def tlob_ops(*, window: int, base_d: int, hidden_dim: int, layers: int, heads: int, ff_multiplier: int) -> int:
    del heads
    embed_ops = window * base_d * hidden_dim
    temporal_attention_ops = base_d * (window * window * hidden_dim + 4 * window * hidden_dim * hidden_dim)
    spatial_attention_ops = window * (base_d * base_d * hidden_dim + 4 * base_d * hidden_dim * hidden_dim)
    ffn_ops = window * base_d * 2 * hidden_dim * hidden_dim * ff_multiplier
    output_ops = hidden_dim * len(CLASS_ORDER)
    return int(embed_ops + layers * (temporal_attention_ops + spatial_attention_ops + ffn_ops) + output_ops)


def candidate_ops(candidate: dict[str, Any], *, window: int, base_d: int) -> int:
    architecture = candidate["architecture"]
    if architecture == "official_mlplob":
        return official_mlplob_ops(
            window=window,
            base_d=base_d,
            hidden_dim=int(candidate["hidden_dim"]),
            layers=int(candidate["layers"]),
        )
    if architecture in {"tlob", "torch_tlob"}:
        return tlob_ops(
            window=window,
            base_d=base_d,
            hidden_dim=int(candidate["hidden_dim"]),
            layers=int(candidate["layers"]),
            heads=int(candidate["heads"]),
            ff_multiplier=int(candidate.get("ff_multiplier", 2)),
        )
    raise ValueError(f"unknown paper-core latency architecture: {architecture}")


def parameter_count(model) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def sequence_tensor_from_flat(x: np.ndarray, *, window: int, base_d: int):
    import torch

    arr = x.reshape(len(x), window, base_d).transpose(0, 2, 1).copy()
    return torch.from_numpy(arr.astype(np.float32, copy=False))


@dataclass
class TorchSequenceClassifier:
    model: Any
    device: Any
    window: int
    base_d: int
    mean: np.ndarray
    std: np.ndarray
    batch_size: int

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
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(CLASS_ORDER)), dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        logits = self.predict_logits(x)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()


def standardize_fit(x_fit: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = x_fit.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_fit.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = [(x_fit - mean) / std]
    out.extend((item - mean) / std for item in others)
    out.extend([mean, std])
    return tuple(out)


def time_single_observation(
    classifier: TorchSequenceClassifier,
    x: np.ndarray,
    *,
    repeats: int,
    samples: int,
) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    n_samples = min(len(x), max(1, int(samples)))
    indices = np.linspace(0, len(x) - 1, n_samples, dtype=int)
    warmup = classifier.predict_proba(x[indices[0] : indices[0] + 1])
    if len(warmup) != 1:
        raise ValueError("single-observation warmup returned wrong row count")

    timings = []
    for _ in range(repeats):
        for idx in indices:
            start = time.perf_counter()
            out = classifier.predict_proba(x[idx : idx + 1])
            elapsed = time.perf_counter() - start
            if len(out) != 1:
                raise ValueError("single-observation prediction returned wrong row count")
            timings.append(elapsed)
    timing_s = float(np.median(timings))
    return timing_s, timing_s * 1_000_000.0
