# =============================================================================
# BSPC Production GPU Execution Script (PowerShell)
# Executes:
#   1. Main Experiment: 5-Fold Patient-Grouped Cross-Validation (Zero Leakage)
#   2. Leakage Experiment: Matched 5-Fold Random-Slice Cross-Validation (Reviewer 3)
# =============================================================================

param (
    [string]$DataRoot = "d:\lgg_segmentation\kaggle_3m",
    [string]$DataCsv = "d:\lgg_segmentation\kaggle_3m\data.csv",
    [string]$OutputDir = "d:\lgg_segmentation\codes\results\bspc_production",
    [string]$PythonExe = "python"
)

Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host "  STARTING BSPC 2026 JOURNAL EXPERIMENTS (GPU SERVER)" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan

Write-Host "Data Root:   $DataRoot"
Write-Host "Data CSV:    $DataCsv"
Write-Host "Output Dir:  $OutputDir"

# Step 1: Main Protocol (Patient Grouped, 5 Folds)
Write-Host "`n>>> [1/2] Running Main 5-Fold Patient-Grouped CV..." -ForegroundColor Green
& $PythonExe codes\scripts\run_gpu_experiments.py `
    --data_root "$DataRoot" `
    --data_csv "$DataCsv" `
    --output_dir "$OutputDir" `
    --split_type patient_grouped `
    --epochs 100 `
    --batch_size 16 `
    --base_filters 32 `
    --patience 20 `
    --resume

# Step 2: Matched Leakage Protocol (Random Slice, 5 Folds)
Write-Host "`n>>> [2/2] Running Matched 5-Fold Random-Slice CV (Leakage Quantification)..." -ForegroundColor Yellow
& $PythonExe codes\scripts\run_gpu_experiments.py `
    --data_root "$DataRoot" `
    --data_csv "$DataCsv" `
    --output_dir "$OutputDir" `
    --split_type random_slice `
    --epochs 100 `
    --batch_size 16 `
    --base_filters 32 `
    --patience 20 `
    --resume

Write-Host "`n===========================================================================" -ForegroundColor Cyan
Write-Host "  ALL GPU EXPERIMENTS COMPLETED SUCCESSFULLY!" -ForegroundColor Cyan
Write-Host "  Artifacts saved in: $OutputDir" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan
