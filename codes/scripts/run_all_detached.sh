#!/usr/bin/env bash
# =============================================================================
# Detached Resumable Production Runner for Ferdowsi Cloud (RTX 3090)
# Runs both patient_grouped (5-fold) and random_slice (5-fold) in background.
# Survives laptop sleep, browser disconnects, and network drops.
# =============================================================================

set -e

# Force unbuffered Python stdout
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0

# Ensure working directory is project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

echo "================================================================================"
echo "🚀 DETACHED PRODUCTION TRAINING STARTED: $(date)"
echo "   Working Directory: $(pwd)"
echo "================================================================================"

# Log GPU details if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Hardware Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "--------------------------------------------------------------------------------"
fi

# -----------------------------------------------------------------------------
# STEP 1: Main Protocol - 5-Fold GroupKFold (Patient-Grouped, Zero Leakage)
# -----------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "▶️ [STEP 1/2] RUNNING EXPERIMENT: patient_grouped (5 FOLDS WITH --resume)"
echo "================================================================================"
python -u codes/scripts/run_gpu_experiments.py \
    --split_type patient_grouped \
    --resume

echo ""
echo "✅ [STEP 1/2] patient_grouped FINISHED AT: $(date)"

# -----------------------------------------------------------------------------
# STEP 2: SOTA Comparison Protocol - 5-Fold KFold (Random-Slice, Leakage Proof)
# -----------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "▶️ [STEP 2/2] RUNNING EXPERIMENT: random_slice (5 FOLDS WITH --resume)"
echo "================================================================================"
python -u codes/scripts/run_gpu_experiments.py \
    --split_type random_slice \
    --resume

echo ""
echo "✅ [STEP 2/2] random_slice FINISHED AT: $(date)"

# -----------------------------------------------------------------------------
# STEP 3: Auto-Package Complete Experimental Results Bundle (< 50MB)
# -----------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "📦 [STEP 3/3] PACKAGING ALL OOF LOGITS, WEIGHTS & METRICS INTO ZIP BUNDLE"
echo "================================================================================"
BUNDLE_NAME="bspc_results_bundle.zip"
rm -f "${BUNDLE_NAME}"

python -c "
import os
import zipfile

bundle = 'bspc_results_bundle.zip'
results_dir = 'codes/results'

with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            # Exclude heavy 32MB weight files (.weights.h5); keep only predictions & metrics
            if f.endswith(('.npz', '.csv', '.json')):
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, '.')
                zipf.write(full_p, rel_p)

size_mb = os.path.getsize(bundle) / (1024 * 1024)
print(f'Results bundle successfully created: {bundle} ({size_mb:.1f} MB)')
"

BUNDLE_SIZE=$(du -h "${BUNDLE_NAME}" 2>/dev/null | cut -f1 || ls -lh "${BUNDLE_NAME}" | awk '{print $5}')
echo "⭐ RESULTS ARCHIVED: ${BUNDLE_NAME} (${BUNDLE_SIZE})"

echo ""
echo "================================================================================"
echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY: $(date)"
echo "   Download '${BUNDLE_NAME}' from JupyterLab file manager to your laptop."
echo "================================================================================"
