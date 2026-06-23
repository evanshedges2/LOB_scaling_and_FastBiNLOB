#!/usr/bin/env python3
"""Run frozen low/mid FI-2010 raw40 frontier candidates."""

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
    candidate: pd.Series

    @property
    def candidate_id(self) -> str:
        return str(self.candidate["candidate_id"])

    @property
    def window(self) -> int:
        return int(self.candidate["input_window"])

    @property
    def stem(self) -> str:
        return f"cf{self.cf}_{self.horizon}_{self.candidate_id}"


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_backend_args(raw: str) -> list[str]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid backend_args JSON: {raw}") from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"backend_args must be a JSON list of strings: {raw}")
    return values


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    candidates = pd.read_csv(args.candidates)
    required = {"candidate_id", "candidate_rank", "input_window", "architecture", "model_name", "backend_args"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise SystemExit(f"{args.candidates} is missing required columns: {missing}")

    if args.candidate_id_filter:
        allowed = set(parse_csv_strings(args.candidate_id_filter))
        candidates = candidates[candidates["candidate_id"].isin(allowed)]
    if args.candidate_architecture_filter:
        allowed = set(parse_csv_strings(args.candidate_architecture_filter))
        candidates = candidates[candidates["architecture"].isin(allowed)]
    if args.candidate_family_filter:
        allowed = set(parse_csv_strings(args.candidate_family_filter))
        candidates = candidates[candidates["family"].isin(allowed)]
    if args.candidate_rank_filter:
        allowed = set(parse_csv_ints(args.candidate_rank_filter))
        candidates = candidates[candidates["candidate_rank"].astype(int).isin(allowed)]

    candidates = candidates.sort_values("candidate_rank").reset_index(drop=True)
    if args.smoke:
        candidates = smoke_candidates(candidates)
    if candidates.empty:
        raise SystemExit("No candidates selected after filters.")
    return candidates


def smoke_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    selected_ids = set()
    for architecture in [
        "decision_tree",
        "hist_gradient_boosting",
        "catboost_symmetric_trees",
        "random_conv_logistic",
    ]:
        subset = candidates[candidates["architecture"] == architecture]
        if subset.empty:
            continue
        if architecture in {"hist_gradient_boosting", "catboost_symmetric_trees", "random_conv_logistic"}:
            preferred = subset[subset["input_window"].astype(int) == 10]
            if not preferred.empty:
                subset = preferred
        if "selection_structural_forward_work_units" in subset.columns:
            sort_col = "selection_structural_forward_work_units"
        elif "selection_forward_work_units" in subset.columns:
            sort_col = "selection_forward_work_units"
        else:
            sort_col = "selection_audited_forward_ops"
        row = subset.sort_values([sort_col, "candidate_rank"]).iloc[0]
        candidate_id = str(row["candidate_id"])
        if candidate_id not in selected_ids:
            selected_ids.add(candidate_id)
            selected_rows.append(row)
    return pd.DataFrame(selected_rows)


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
        "--timing-mode",
        args.timing_mode,
        "--single-timing-samples",
        str(args.single_timing_samples),
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


def candidate_backend_args(args: argparse.Namespace, candidate: pd.Series) -> list[str]:
    backend_args = parse_backend_args(str(candidate["backend_args"]))
    if "--include-torch-nn" in backend_args:
        if "--torch-device" not in backend_args:
            backend_args.extend(["--torch-device", args.torch_device])
        if "--torch-eval-batch-size" not in backend_args:
            backend_args.extend(["--torch-eval-batch-size", str(args.torch_eval_batch_size)])
    return backend_args


def build_items(args: argparse.Namespace, candidates: pd.DataFrame) -> list[RunItem]:
    cfs = [1] if args.smoke else parse_csv_ints(args.cfs)
    horizons = ["y_10", "y_100"] if args.smoke else parse_csv_strings(args.horizons)
    unknown_horizons = sorted(set(horizons) - set(HORIZONS))
    if unknown_horizons:
        raise ValueError(f"unknown horizons: {unknown_horizons}")

    items: list[RunItem] = []
    for cf in cfs:
        for horizon in horizons:
            for _, candidate in candidates.iterrows():
                items.append(RunItem(cf=cf, horizon=horizon, candidate=candidate))
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


def annotate_output(output_path: Path, candidate: pd.Series) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return
    frame = pd.read_csv(output_path)
    if frame.empty:
        return
    insertions = {
        "frozen_candidate_id": candidate.get("candidate_id", ""),
        "frozen_candidate_rank": candidate.get("candidate_rank", ""),
        "frozen_selection_group": candidate.get("selection_group", ""),
        "frozen_selection_score_target": candidate.get("score_target", ""),
        "frozen_selection_metric": candidate.get("selection_metric", ""),
        "frozen_selection_loss": candidate.get("selection_loss", ""),
        "frozen_selection_validation_log_loss": candidate.get("selection_validation_log_loss", ""),
        "frozen_selection_structural_forward_work_units": candidate.get("selection_structural_forward_work_units", ""),
        "frozen_selection_forward_work_units": candidate.get("selection_forward_work_units", ""),
        "frozen_selection_audited_forward_ops": candidate.get("selection_audited_forward_ops", ""),
        "frozen_selection_source": candidate.get("selection_source", ""),
    }
    for key, value in reversed(list(insertions.items())):
        if key in frame.columns:
            frame[key] = value
        else:
            frame.insert(0, key, value)
    frame.to_csv(output_path, index=False)


def write_manifest(candidates: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--candidates", type=Path, default=OUTPUT_ROOT / "manifests" / "frozen_low_mid_candidates.csv")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", ".venv/bin/python"))
    parser.add_argument("--torch-device", default=os.environ.get("TORCH_DEVICE", "mps"))
    parser.add_argument("--cfs", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--horizons", default=",".join(HORIZONS))
    parser.add_argument("--candidate-id-filter", default="")
    parser.add_argument("--candidate-rank-filter", default="")
    parser.add_argument("--candidate-architecture-filter", default="")
    parser.add_argument("--candidate-family-filter", default="")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--timing-mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--single-timing-samples", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--torch-eval-batch-size", type=int, default=1024)
    parser.add_argument("--result-root", type=Path, default=OUTPUT_ROOT / "results")
    parser.add_argument("--log-root", type=Path, default=OUTPUT_ROOT / "logs")
    parser.add_argument("--figure-root", type=Path, default=OUTPUT_ROOT / "figures" / "architecture_scaling")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidates = load_candidates(args)

    run_name = f"fi2010_raw40_frozen_low_mid_{args.run_tag}"
    result_dir = args.result_root / run_name
    log_dir = args.log_root / run_name
    figure_dir = args.figure_root / run_name
    history_dir = figure_dir / "histories"
    for directory in [result_dir, log_dir, figure_dir, history_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    write_manifest(candidates, result_dir / "frozen_candidates_used.csv")

    print(f"RUN_TAG={args.run_tag}")
    print(f"RESULT_DIR={result_dir}")
    print(f"LOG_DIR={log_dir}")
    print(f"FIGURE_DIR={figure_dir}")
    print(f"PYTHON_BIN={args.python_bin}")
    print(f"TORCH_DEVICE={args.torch_device}")
    print(f"selected_candidates={len(candidates)}")

    items = build_items(args, candidates)
    print(f"planned_items={len(items)}")
    for item in items:
        output = result_dir / f"{item.stem}.csv"
        checkpoint = result_dir / f"{item.stem}_checkpoint.csv"
        log_path = log_dir / f"{item.stem}.log"
        item_args = (
            common_backend_args(args, item, result_dir, figure_dir, history_dir)
            + candidate_backend_args(args, item.candidate)
        )
        command = [args.python_bin] + item_args
        if output.exists() and output.stat().st_size > 0:
            annotate_output(output, item.candidate)
            print(f"Skipping existing {item.stem}: {output}")
            continue
        print(shlex.join(command))
        if args.dry_run:
            continue
        status = run_command(command, log_path)
        if status != 0:
            if checkpoint.exists() and checkpoint.stat().st_size > 0 and args.allow_failures:
                shutil.copyfile(checkpoint, output)
                annotate_output(output, item.candidate)
                print(f"{item.stem} failed with status {status}; copied checkpoint to {output}")
                continue
            print(f"{item.stem} failed with status {status}", file=sys.stderr)
            raise SystemExit(status)
        annotate_output(output, item.candidate)


if __name__ == "__main__":
    main()
