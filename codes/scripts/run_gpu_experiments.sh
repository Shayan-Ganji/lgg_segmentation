#!/bin/bash
# =============================================================================
# BSPC Production GPU Execution Script
# Executes:
#   1. Main Experiment: 5-Fold Patient-Grouped Cross-Validation (Zero Leakage)
#   2. Leakage Experiment: Matched 5-Fold Random-Slice Cross-Validation (Reviewer 3)
# =============================================================================

set -e

echo "==========================================================================="
echo "  STARTING BSPC 2026 JOURNAL EXPERIMENTS (GPU SERVER)"
echo "==========================================================================="

DATA_ROOT="${1:-kaggle_3m}"
DATA_CSV="${2:-kaggle_3m/data.csv}"
OUTPUT_DIR="${3:-codes/results/bspc_production}"

echo "Data Root:   $DATA_ROOT"
echo "Data CSV:    $DATA_CSV"
echo "Output Dir:  $OUTPUT_DIR"

# Step 1: Main Protocol (Patient Grouped, 5 Folds)
echo ""
echo ">>> [1/2] Running Main 5-Fold Patient-Grouped CV..."
python codes/scripts/run_gpu_experiments.py \
    --data_root "$DATA_ROOT" \
    --data_csv "$DATA_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --split_type patient_grouped \
    --epochs 100 \
    --batch_size 16 \
    --base_filters 32 \
    --patience 20 \
    --resume

# Step 2: Matched Leakage Protocol (Random Slice, 5 Folds)
echo ""
echo ">>> [2/2] Running Matched 5-Fold Random-Slice CV (Leakage Quantification)..."
python codes/scripts/run_gpu_experiments.py \
    --data_root "$DATA_ROOT" \
    --data_csv "$DATA_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --split_type random_slice \
    --epochs 100 \
    --batch_size 16 \
    --base_filters 32 \
    --patience 20 \
    --resume

echo ""
echo "==========================================================================="
echo "  ALL GPU EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "  Artifacts saved in: $OUTPUT_DIR"
echo "==========================================================================="
