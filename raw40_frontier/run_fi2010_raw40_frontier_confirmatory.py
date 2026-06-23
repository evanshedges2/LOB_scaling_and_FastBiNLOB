#!/usr/bin/env python3
"""Clean confirmatory runner for FI-2010 raw40 inference frontiers."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = BUNDLE_ROOT / "shared"
OUTPUT_ROOT = BUNDLE_ROOT / "outputs"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
ARCHITECTURE_SCALING_SCRIPT = SHARED_DIR / "architecture_scaling_frontiers.py"


HORIZONS = ("y_10", "y_20", "y_30", "y_50", "y_100")
PHASES = (
    "trees_hgb",
    "ebm_catboost",
    "catboost_trimmed",
    "random_conv",
    "torch_mlp",
    "torch_tcn",
    "deeplob_ladder",
    "deeplob_large_anchors",
)


@dataclass(frozen=True)
class RunItem:
    phase: str
    cf: int
    horizon: str
    window: int
    seed: int | None
    args: list[str]

    @property
    def stem(self) -> str:
        seed_part = f"_seed{self.seed}" if self.seed is not None else ""
        return f"cf{self.cf}_{self.horizon}_w{self.window}_{self.phase}{seed_part}"


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


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def common_backend_args(args: argparse.Namespace, item: RunItem, result_dir: Path, figure_dir: Path, history_dir: Path) -> list[str]:
    output = result_dir / f"{item.stem}.csv"
    checkpoint = result_dir / f"{item.stem}_checkpoint.csv"
    figures = figure_dir / item.stem
    return [
        str(ARCHITECTURE_SCALING_SCRIPT),
        "--cf",
        str(item.cf),
        "--feature-modes",
        "lob40",
        "--horizons",
        item.horizon,
        "--window-sizes",
        str(item.window),
        "--validation-fraction",
        str(args.validation_fraction),
        "--timing-repeats",
        str(args.timing_repeats),
        "--n-jobs",
        str(args.n_jobs),
        "--min-samples-leaf",
        str(args.min_samples_leaf),
        "--output",
        str(output),
        "--checkpoint-output",
        str(checkpoint),
        "--figure-dir",
        str(figures),
        "--torch-history-output-dir",
        str(history_dir),
    ]


def phase_args(args: argparse.Namespace, item: RunItem, smoke: bool) -> list[str]:
    phase = item.phase
    seed = item.seed
    if phase == "trees_hgb":
        tree_leaves = "2,8" if smoke else "2,4,8,16,32,64,128,256,512"
        hgb_iters = "2" if smoke else "8,16,32,64,128,256,512"
        hgb_leaves = "4" if smoke else "8,16,32,64,128,256"
        return [
            "--no-forests",
            "--no-mlp",
            "--no-kernel",
            "--tree-leaf-values",
            tree_leaves,
            "--hgb-max-iter-values",
            hgb_iters,
            "--hgb-leaf-values",
            hgb_leaves,
            "--hgb-learning-rate",
            "0.05",
        ]

    if phase == "ebm_catboost":
        cat_iters = "2" if smoke else "256,512,1024,2048"
        cat_depths = "2" if smoke else "4,6,8"
        ebm_rounds = "2" if smoke else "500,1000,2000"
        ebm_interactions = "0" if smoke else "0,5,10,20"
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-catboost",
            "--include-ebm",
            "--catboost-iterations",
            cat_iters,
            "--catboost-depths",
            cat_depths,
            "--catboost-learning-rate",
            "0.03",
            "--catboost-early-stopping-rounds",
            "0" if smoke else "60",
            "--ebm-max-rounds",
            ebm_rounds,
            "--ebm-interactions",
            ebm_interactions,
            "--ebm-learning-rate",
            "0.01",
            "--ebm-max-bins",
            "256",
        ]

    if phase == "catboost_trimmed":
        if smoke:
            cat_iters = "2"
            cat_depths = "2"
        elif item.window == 10:
            cat_iters = "512,1024,2048"
            cat_depths = "4,6,8"
        elif item.window == 100:
            cat_iters = "512"
            cat_depths = "4,6,8"
        else:
            cat_iters = "512"
            cat_depths = "4,6"
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-catboost",
            "--catboost-iterations",
            cat_iters,
            "--catboost-depths",
            cat_depths,
            "--catboost-learning-rate",
            "0.03",
            "--catboost-early-stopping-rounds",
            "0" if smoke else "60",
        ]

    if phase == "random_conv":
        kernels = "8" if smoke else "256,512,1024,2048"
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-random-conv",
            "--random-conv-kernels",
            kernels,
            "--random-conv-kernel-sizes",
            "3,5" if smoke else "3,5,7,9,15,21,31,45,63",
            "--random-conv-heads",
            "logistic",
            "--random-conv-logistic-max-iter",
            "20" if smoke else "800",
            "--random-conv-logistic-c",
            "1.0",
        ]

    if phase == "torch_mlp":
        layers = "128" if smoke else "128,256,512,256x128,512x256"
        torch_epochs = "1" if smoke else "100"
        assert seed is not None
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-torch-nn",
            "--torch-device",
            args.torch_device,
            "--torch-seed",
            str(seed),
            "--torch-optimizer",
            "adam",
            "--torch-lr",
            "0.0003",
            "--torch-weight-decay",
            "0.0",
            "--torch-batch-size",
            "256",
            "--torch-eval-batch-size",
            str(args.torch_eval_batch_size),
            "--torch-epochs",
            torch_epochs,
            "--torch-patience",
            "12",
            "--torch-min-delta",
            "0.00005",
            "--torch-early-stop-metric",
            "loss",
            "--torch-mlp-hidden-layers",
            layers,
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

    if phase == "torch_tcn":
        channels = "16x16" if smoke else "16x16,32x32,64x64,32x32x32"
        torch_epochs = "1" if smoke else "120"
        assert seed is not None
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-torch-nn",
            "--torch-device",
            args.torch_device,
            "--torch-seed",
            str(seed),
            "--torch-optimizer",
            "adam",
            "--torch-lr",
            "0.0003",
            "--torch-weight-decay",
            "0.0",
            "--torch-batch-size",
            "64",
            "--torch-eval-batch-size",
            str(args.torch_eval_batch_size),
            "--torch-epochs",
            torch_epochs,
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
            channels,
            "--torch-cnn-attention-channels",
            "",
            "--torch-recurrent-specs",
            "",
            "--torch-deeplob-specs",
            "",
            "--torch-deeplob-full-specs",
            "",
        ]

    if phase in {"deeplob_ladder", "deeplob_large_anchors"}:
        specs = "12x24x48" if smoke else "12x24x48,16x32x64,24x32x64,32x32x64"
        eval_batch = args.torch_eval_batch_size
        if phase == "deeplob_large_anchors":
            specs = "96x128x128"
            eval_batch = min(args.torch_eval_batch_size, 512)
        torch_epochs = "1" if smoke else "120"
        assert seed is not None
        return [
            "--no-trees",
            "--no-forests",
            "--no-boosted-trees",
            "--no-mlp",
            "--no-kernel",
            "--include-torch-nn",
            "--torch-device",
            args.torch_device,
            "--torch-seed",
            str(seed),
            "--torch-optimizer",
            "adam",
            "--torch-lr",
            "0.01",
            "--torch-weight-decay",
            "0.0",
            "--torch-adam-eps",
            "1.0",
            "--torch-batch-size",
            "32",
            "--torch-eval-batch-size",
            str(eval_batch),
            "--torch-epochs",
            torch_epochs,
            "--torch-patience",
            "20",
            "--torch-min-delta",
            "0.0",
            "--torch-early-stop-metric",
            "accuracy",
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
            "",
            "--torch-cnn-attention-channels",
            "",
            "--torch-recurrent-specs",
            "",
            "--torch-deeplob-specs",
            "",
            "--torch-deeplob-full-specs",
            specs,
            "--torch-deeplob-full-dropout",
            "0.2",
            "--torch-deeplob-full-norm",
            "batchnorm",
            "--torch-deeplob-full-conv2-activation",
            "tanh",
        ]

    raise ValueError(f"unknown phase: {phase}")


def build_items(args: argparse.Namespace) -> list[RunItem]:
    cfs = [1, 9] if args.smoke and args.smoke_full else [1] if args.smoke else parse_csv_ints(args.cfs)
    horizons = ["y_10", "y_50"] if args.smoke and args.smoke_full else ["y_10"] if args.smoke else parse_csv_strings(args.horizons)
    windows = parse_csv_ints(args.windows)
    seeds = [1] if args.smoke else parse_csv_ints(args.seeds)
    selected_phases = PHASES if args.phases == "all" else tuple(parse_csv_strings(args.phases))
    unknown = sorted(set(selected_phases) - set(PHASES))
    if unknown:
        raise ValueError(f"unknown phases: {unknown}")

    items: list[RunItem] = []
    for phase in selected_phases:
        if args.smoke and not args.smoke_full and phase == "deeplob_large_anchors":
            continue
        phase_cfs = cfs
        phase_horizons = horizons
        phase_windows = windows
        if args.smoke and not args.smoke_full:
            phase_windows = {
                "trees_hgb": [1],
                "ebm_catboost": [1],
                "catboost_trimmed": [10],
                "random_conv": [10],
                "torch_mlp": [1],
                "torch_tcn": [10],
                "deeplob_ladder": [100],
            }.get(phase, [1])
        if phase == "deeplob_large_anchors" and not args.smoke:
            phase_cfs = [cf for cf in cfs if cf in {1, 5, 9}]
            phase_horizons = [h for h in horizons if h in {"y_10", "y_50", "y_100"}]
        phase_seeds: list[int | None] = [None] if phase in {"trees_hgb", "ebm_catboost", "catboost_trimmed"} else seeds
        for cf in phase_cfs:
            for horizon in phase_horizons:
                if horizon not in HORIZONS:
                    raise ValueError(f"unknown horizon: {horizon}")
                for window in phase_windows:
                    for seed in phase_seeds:
                        items.append(
                            RunItem(
                                phase=phase,
                                cf=cf,
                                horizon=horizon,
                                window=window,
                                seed=seed,
                                args=[],
                            )
                        )
    return items


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", ".venv/bin/python"))
    parser.add_argument("--torch-device", default=os.environ.get("TORCH_DEVICE", "mps"))
    parser.add_argument("--cfs", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--horizons", default=",".join(HORIZONS))
    parser.add_argument("--windows", default="1,10,100")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--phases", default="all", help=f"Comma list or all. Known: {','.join(PHASES)}")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--torch-eval-batch-size", type=int, default=1024)
    parser.add_argument("--result-root", type=Path, default=OUTPUT_ROOT / "results")
    parser.add_argument("--log-root", type=Path, default=OUTPUT_ROOT / "logs")
    parser.add_argument("--figure-root", type=Path, default=OUTPUT_ROOT / "figures" / "architecture_scaling")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-full", action="store_true", help="Use the broader historical smoke grid instead of the quick plumbing smoke.")
    parser.add_argument("--allow-phase-failures", action="store_true")
    args = parser.parse_args()

    run_name = f"fi2010_raw40_frontier_confirmatory_{args.run_tag}"
    result_dir = args.result_root / run_name
    log_dir = args.log_root / run_name
    figure_dir = args.figure_root / run_name
    history_dir = figure_dir / "histories"
    for directory in [result_dir, log_dir, figure_dir, history_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"RUN_TAG={args.run_tag}")
    print(f"RESULT_DIR={result_dir}")
    print(f"LOG_DIR={log_dir}")
    print(f"FIGURE_DIR={figure_dir}")
    print(f"PYTHON_BIN={args.python_bin}")
    print(f"TORCH_DEVICE={args.torch_device}")

    items = build_items(args)
    print(f"planned_items={len(items)}")
    for item in items:
        item_args = common_backend_args(args, item, result_dir, figure_dir, history_dir) + phase_args(args, item, args.smoke)
        output = result_dir / f"{item.stem}.csv"
        checkpoint = result_dir / f"{item.stem}_checkpoint.csv"
        log_path = log_dir / f"{item.stem}.log"
        command = [args.python_bin] + item_args
        command_text = shlex.join(command)
        if output.exists() and output.stat().st_size > 0:
            print(f"Skipping existing {item.stem}: {output}")
            continue
        print(command_text)
        if args.dry_run:
            continue
        status = run_command(command, log_path)
        if status != 0:
            if checkpoint.exists() and checkpoint.stat().st_size > 0 and args.allow_phase_failures:
                shutil.copyfile(checkpoint, output)
                print(f"{item.stem} failed with status {status}; copied checkpoint to {output}")
                continue
            print(f"{item.stem} failed with status {status}", file=sys.stderr)
            raise SystemExit(status)


if __name__ == "__main__":
    main()
