# Inference-Compute Frontiers for Limit Order Book Prediction

Code for reproducing the paper-required FI-2010 experiments behind the raw40 inference-compute frontier and the full144 FastBiNLOB latency results. This is a small methods bundle, not a full project archive.

The raw40 lane runs the screened frontier families from the paper: decision trees, histogram gradient boosting, EBM, CatBoost, random-conv logistic, small neural screened rows, and held-out MLPLOB targets. The full144 lane trains the two FastBiNLOB paper operating points and times the MLPLOB/TLOB/FastBiNLOB latency anchors.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
PY=.venv/bin/python
```

Place FI-2010 `BenchmarkDatasets/` at the repository root or under `data/BenchmarkDatasets/`. Expected files are under:

```text
BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/
BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/
```

All generated files go under `outputs/`, which is ignored.

## Raw40 Frontier

Small plumbing smoke:

```bash
$PY raw40_frontier/run_fi2010_raw40_frontier_confirmatory.py \
  --python-bin "$PY" \
  --run-tag smoke_raw40 \
  --smoke \
  --torch-device cpu
```

Full screened discovery grid:

```bash
$PY raw40_frontier/run_fi2010_raw40_frontier_confirmatory.py \
  --python-bin "$PY" \
  --run-tag raw40_frontier_full \
  --torch-device auto
```

Select the frozen low/mid candidate set from the discovery outputs:

```bash
$PY raw40_frontier/select_frozen_low_mid_candidates.py \
  --input-dirs outputs/results/fi2010_raw40_frontier_confirmatory_raw40_frontier_full
```

Run the selected low/mid candidates and the held-out MLPLOB targets:

```bash
$PY raw40_frontier/run_frozen_low_mid_candidates.py \
  --python-bin "$PY" \
  --run-tag raw40_low_mid \
  --candidates outputs/manifests/frozen_low_mid_candidates.csv \
  --torch-device auto

$PY raw40_frontier/run_mlplob_size_ladder.py \
  --python-bin "$PY" \
  --run-tag raw40_mlplob_targets \
  --torch-device auto
```

Analyze the MLPLOB holdout frontier:

```bash
$PY raw40_frontier/analyze_mlplob_mse_frontier_extrapolation.py \
  --low-mid-dirs outputs/results/fi2010_raw40_frozen_low_mid_raw40_low_mid \
  --target-dirs outputs/results/fi2010_raw40_mlplob_size_ladder_raw40_mlplob_targets \
  --output-dir outputs/analysis/raw40_mlplob_mse_frontier
```

## Full144 FastBiNLOB

Dry-run one training config:

```bash
$PY fastbinlob_latency/run_multitask_latency_ladder.py \
  --config fastbinlob_latency/setup2_full144_h96_mean_core.json \
  --output outputs/results/full144_h96_mean.csv \
  --history-dir outputs/results/histories_full144_h96_mean \
  --dry-run
```

Train the two paper operating points:

```bash
$PY fastbinlob_latency/run_multitask_latency_ladder.py \
  --config fastbinlob_latency/setup2_full144_h96_mean_core.json \
  --output outputs/results/full144_h96_mean.csv \
  --history-dir outputs/results/histories_full144_h96_mean

$PY fastbinlob_latency/run_multitask_latency_ladder.py \
  --config fastbinlob_latency/setup2_full144_h120_taper_core.json \
  --output outputs/results/full144_h120_taper.csv \
  --history-dir outputs/results/histories_full144_h120_taper
```

Run the paper-core latency audit:

```bash
$PY fastbinlob_latency/benchmark_real_cf_single_latency.py \
  --config fastbinlob_latency/setup2_full144_paper_core_cpu_latency.json \
  --output outputs/results/full144_paper_core_real_latency.csv \
  --device cpu \
  --cf 8
```

For a quick model-construction check without FI-2010 tensors:

```bash
$PY fastbinlob_latency/benchmark_latency_ladder.py \
  --config fastbinlob_latency/setup2_full144_paper_core_cpu_latency.json \
  --output outputs/results/full144_paper_core_synthetic_latency.csv \
  --devices cpu \
  --max-candidates 1 \
  --samples 8 \
  --repeats 1
```
