#!/usr/bin/env python3
"""Run FI-2010 raw40 MLPLOB W-size ladder experiments.

This wrapper intentionally runs one (cross-fold, horizon, window) item per
backend command so long MPS jobs are easy to monitor, skip, and resume.
"""

from __future__ import annotations

import argparse
import json
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

import pandas as pd


HORIZONS = ("y_10", "y_20", "y_30", "y_50", "y_100")


@dataclass(frozen=True)
class RunItem:
    cf: int
    horizon: str
    window: int

    @property
    def stem(self) -> str:
        return f"cf{self.cf}_{self.horizon}_w{self.window}_mlplob"


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if token:
            values.append(int(token))
    return values


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_items(args: argparse.Namespace) -> list[RunItem]:
    cfs = parse_csv_ints(args.cfs)
    windows = parse_csv_ints(args.windows)
    horizons = parse_csv_strings(args.horizons)
    unknown = sorted(set(horizons) - set(HORIZONS))
    if unknown:
        raise ValueError(f"unknown horizons: {unknown}")
    return [RunItem(cf=cf, horizon=horizon, window=window) for cf in cfs for window in windows for horizon in horizons]


def disabled_torch_family_args() -> list[str]:
    return [
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
        "",
        "--torch-tlob-specs",
        "",
    ]


def build_backend_command(
    args: argparse.Namespace,
    item: RunItem,
    result_dir: Path,
    figure_dir: Path,
    history_dir: Path,
) -> list[str]:
    output = result_dir / f"{item.stem}.csv"
    checkpoint = result_dir / f"{item.stem}_checkpoint.csv"
    figures = figure_dir / item.stem
    command = [
        args.python_bin,
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
        "--timing-mode",
        args.timing_mode,
        "--single-timing-samples",
        str(args.single_timing_samples),
        "--n-jobs",
        str(args.n_jobs),
        "--output",
        str(output),
        "--checkpoint-output",
        str(checkpoint),
        "--figure-dir",
        str(figures),
        "--torch-history-output-dir",
        str(history_dir),
        "--no-trees",
        "--no-forests",
        "--no-boosted-trees",
        "--no-mlp",
        "--no-kernel",
        "--include-torch-nn",
        "--torch-device",
        args.torch_device,
        "--torch-timing-device",
        args.torch_timing_device,
        "--torch-optimizer",
        args.torch_optimizer,
        "--torch-lr",
        str(args.torch_lr),
        "--torch-weight-decay",
        str(args.torch_weight_decay),
        "--torch-batch-size",
        str(args.torch_batch_size),
        "--torch-eval-batch-size",
        str(args.torch_eval_batch_size),
        "--torch-epochs",
        str(args.torch_epochs),
        "--torch-patience",
        str(args.torch_patience),
        "--torch-min-delta",
        str(args.torch_min_delta),
        "--torch-early-stop-metric",
        args.torch_early_stop_metric,
        "--torch-adam-eps",
        str(args.torch_adam_eps),
        "--torch-mlplob-specs",
        args.torch_mlplob_specs,
        "--torch-mlplob-dropout",
        str(args.torch_mlplob_dropout),
    ]
    command.extend(disabled_torch_family_args())
    if args.torch_log_epochs:
        command.append("--torch-log-epochs")
    if args.torch_temperature_scaling:
        command.append("--torch-temperature-scaling")
    if args.max_train_rows:
        command.extend(["--max-train-rows", str(args.max_train_rows)])
    if args.max_test_rows:
        command.extend(["--max-test-rows", str(args.max_test_rows)])
    return command


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def annotate_output(output_path: Path, args: argparse.Namespace, item: RunItem) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return
    frame = pd.read_csv(output_path)
    if frame.empty:
        return
    insertions = {
        "clean_experiment": "mlplob_size_ladder",
        "clean_run_tag": args.run_tag,
        "clean_requested_cfs": args.cfs,
        "clean_requested_horizons": args.horizons,
        "clean_requested_windows": args.windows,
        "clean_requested_mlplob_specs": args.torch_mlplob_specs,
        "clean_item_cf": item.cf,
        "clean_item_horizon": item.horizon,
        "clean_item_window": item.window,
    }
    for key, value in reversed(list(insertions.items())):
        if key in frame.columns:
            frame[key] = value
        else:
            frame.insert(0, key, value)
    frame.to_csv(output_path, index=False)


def write_manifest(args: argparse.Namespace, items: list[RunItem], result_dir: Path) -> None:
    manifest = {
        "run_tag": args.run_tag,
        "python_bin": args.python_bin,
        "torch_device": args.torch_device,
        "torch_timing_device": args.torch_timing_device,
        "timing_mode": args.timing_mode,
        "single_timing_samples": args.single_timing_samples,
        "cfs": parse_csv_ints(args.cfs),
        "horizons": parse_csv_strings(args.horizons),
        "windows": parse_csv_ints(args.windows),
        "torch_mlplob_specs": args.torch_mlplob_specs,
        "torch_epochs": args.torch_epochs,
        "torch_patience": args.torch_patience,
        "torch_lr": args.torch_lr,
        "torch_optimizer": args.torch_optimizer,
        "torch_batch_size": args.torch_batch_size,
        "torch_eval_batch_size": args.torch_eval_batch_size,
        "planned_items": [item.__dict__ for item in items],
    }
    (result_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", ".venv/bin/python"))
    parser.add_argument("--torch-device", default=os.environ.get("TORCH_DEVICE", "mps"))
    parser.add_argument("--torch-timing-device", choices=["same", "cpu", "mps", "cuda"], default="same")
    parser.add_argument("--cfs", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--horizons", default=",".join(HORIZONS))
    parser.add_argument("--windows", default="64,128,256")
    parser.add_argument("--torch-mlplob-specs", default="40x3x4")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--timing-mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--single-timing-samples", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--torch-optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--torch-lr", type=float, default=0.0003)
    parser.add_argument("--torch-weight-decay", type=float, default=0.0001)
    parser.add_argument("--torch-adam-eps", type=float, default=1e-8)
    parser.add_argument("--torch-batch-size", type=int, default=32)
    parser.add_argument("--torch-eval-batch-size", type=int, default=64)
    parser.add_argument("--torch-epochs", type=int, default=30)
    parser.add_argument("--torch-patience", type=int, default=8)
    parser.add_argument("--torch-min-delta", type=float, default=0.00005)
    parser.add_argument("--torch-early-stop-metric", choices=["loss", "accuracy"], default="loss")
    parser.add_argument("--torch-mlplob-dropout", type=float, default=0.1)
    parser.add_argument("--torch-log-epochs", action="store_true")
    parser.add_argument("--torch-temperature-scaling", action="store_true")
    parser.add_argument("--result-root", type=Path, default=OUTPUT_ROOT / "results")
    parser.add_argument("--log-root", type=Path, default=OUTPUT_ROOT / "logs")
    parser.add_argument("--figure-root", type=Path, default=OUTPUT_ROOT / "figures" / "architecture_scaling")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun items even when output CSVs already exist")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    items = build_items(args)
    run_name = f"fi2010_raw40_mlplob_size_ladder_{args.run_tag}"
    result_dir = args.result_root / run_name
    log_dir = args.log_root / run_name
    figure_dir = args.figure_root / run_name
    history_dir = figure_dir / "histories"
    for directory in [result_dir, log_dir, figure_dir, history_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    write_manifest(args, items, result_dir)

    print(f"RUN_TAG={args.run_tag}")
    print(f"RESULT_DIR={result_dir}")
    print(f"LOG_DIR={log_dir}")
    print(f"FIGURE_DIR={figure_dir}")
    print(f"PYTHON_BIN={args.python_bin}")
    print(f"TORCH_DEVICE={args.torch_device}")
    print(f"MLPLOB_SPECS={args.torch_mlplob_specs}")
    print(f"planned_items={len(items)}")

    for item in items:
        output = result_dir / f"{item.stem}.csv"
        checkpoint = result_dir / f"{item.stem}_checkpoint.csv"
        log_path = log_dir / f"{item.stem}.log"
        command = build_backend_command(args, item, result_dir, figure_dir, history_dir)
        if output.exists() and output.stat().st_size > 0 and not args.force:
            annotate_output(output, args, item)
            print(f"Skipping existing {item.stem}: {output}")
            continue
        print(shlex.join(command))
        if args.dry_run:
            continue
        status = run_command(command, log_path)
        if status != 0:
            if checkpoint.exists() and checkpoint.stat().st_size > 0 and args.allow_failures:
                shutil.copyfile(checkpoint, output)
                annotate_output(output, args, item)
                print(f"{item.stem} failed with status {status}; copied checkpoint to {output}")
                continue
            print(f"{item.stem} failed with status {status}", file=sys.stderr)
            raise SystemExit(status)
        annotate_output(output, args, item)


if __name__ == "__main__":
    main()
