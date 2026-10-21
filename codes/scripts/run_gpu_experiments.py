"""
=============================================================================
BSPC Journal Production Pipeline: 5-Fold Resumable GPU Experiment Runner
Outputs:
  - best_model_fold{k}.weights.h5
  - oof_val_predictions_fold{k}.npz  (contains logits, probs, targets, patient_ids, slice_indices)
  - oof_test_predictions_fold{k}.npz (contains logits, probs, targets, patient_ids, slice_indices)
  - fold{k}_history.csv              (epoch-by-epoch loss/dice)
  - cohort_screening_audit.csv       (complete 3,929 slice screening record)
  - experiment_summary.json          (overall OOF performance)
=============================================================================
"""

import os
import sys
import argparse
import json
import time
import functools
import numpy as np
import pandas as pd

# Force unbuffered stdout streaming for real-time remote GPU logs
print = functools.partial(print, flush=True)

# Add codes directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import CFG, seed_everything
from src.dataset import build_dataframe, get_splits
from src.preprocessing import get_or_create_preprocessed_cache
from src.train import train_single_fold

def parse_args():
    parser = argparse.ArgumentParser(description="BSPC Production 5-Fold Resumable Experiment Runner")
    parser.add_argument("--data_root", type=str, default=CFG["DATA_ROOT"], help="Path to kaggle_3m dataset")
    parser.add_argument("--data_csv", type=str, default=CFG["DATA_CSV"], help="Path to data.csv metadata")
    parser.add_argument("--output_dir", type=str, default=CFG["OUTPUT_DIR"], help="Base output directory for results")
    parser.add_argument("--cache_path", type=str, default=CFG["CACHE_PATH"], help="Path to preprocessed cache .npz")
    parser.add_argument("--split_type", type=str, default="patient_grouped", choices=["patient_grouped", "random_slice"],
                        help="Data split type: 'patient_grouped' (main protocol) or 'random_slice' (leakage experiment)")
    parser.add_argument("--epochs", type=int, default=CFG["EPOCHS"], help="Total training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=CFG["BATCH_SIZE"], help="Mini-batch size")
    parser.add_argument("--base_filters", type=int, default=CFG["BASE_FILTERS"], help="Base conv filters (32)")
    parser.add_argument("--patience", type=int, default=CFG["ES_PATIENCE"], help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already completed folds")
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(CFG["SEED"])
    
    # Requirement 2: Separate output directory by split_type to prevent any collision or resume bug
    effective_output_dir = os.path.join(args.output_dir, args.split_type)
    os.makedirs(effective_output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 75)
    print("  BSPC 2026 JOURNAL EXPERIMENT: PRODUCTION RUNNER")
    print("=" * 75)
    print(f"  Split Type:     {args.split_type.upper()}")
    print(f"  Data Root:      {args.data_root}")
    print(f"  Data CSV:       {args.data_csv}")
    print(f"  Base Out Dir:   {args.output_dir}")
    print(f"  Split Out Dir:  {effective_output_dir}")
    print(f"  Cache Path:     {args.cache_path}")
    print(f"  Epochs / Batch: {args.epochs} / {args.batch_size}")
    print(f"  Base Filters:   {args.base_filters}")
    print(f"  Resume Mode:    {args.resume}")
    print("=" * 75)
    
    # 1. Build DataFrame
    df = build_dataframe(data_root=args.data_root, data_csv_path=args.data_csv)
    
    # 2. Requirement 1 & 3: Quality-Aware Preprocessing & Caching + Screening Audit
    # Performed once at startup on all 3,929 slices, caching 128x128 images into RAM/disk.
    audit_csv_path = os.path.join(args.output_dir, "cohort_screening_audit.csv")
    images_cache, masks_cache, df = get_or_create_preprocessed_cache(
        df=df,
        cache_path=args.cache_path,
        audit_csv_path=audit_csv_path
    )
    
    n_flagged = df["is_artifact_flagged"].sum()
    print(f"  Dataset ready: {len(df)} total slices | {n_flagged} artifact-flagged slices ({len(df) - n_flagged} clean).")

    # 3. Generate Matched 5-Fold Splits
    splits = get_splits(df, split_type=args.split_type, n_folds=CFG["N_FOLDS"],
                        inner_val_ratio=CFG["INNER_VAL_RATIO"], seed=CFG["SEED"])
    
    fold_summaries = []
    total_start_time = time.time()
    
    # 4. Iterate over Folds
    for fold_info in splits:
        fold = fold_info["fold"]
        val_npz = os.path.join(effective_output_dir, f"oof_val_predictions_fold{fold}.npz")
        test_npz = os.path.join(effective_output_dir, f"oof_test_predictions_fold{fold}.npz")
        
        # Check if already completed
        if args.resume and os.path.exists(val_npz) and os.path.exists(test_npz):
            print(f"\n[RESUME] Fold {fold} ({args.split_type}) already completed. Loading existing predictions...")
            test_data = np.load(test_npz)
            if "probs" in test_data:
                t_probs = test_data["probs"]
            else:
                t_logits = test_data["logits"].astype(np.float32)
                t_probs = 1.0 / (1.0 + np.exp(-t_logits))
            t_targets = test_data["targets"]
            
            t_bin = (t_probs >= 0.50).astype(np.uint8)
            d_all = [float((2.0 * np.sum(t_targets[i] * t_bin[i]) + 1e-6) / (t_targets[i].sum() + t_bin[i].sum() + 1e-6)) for i in range(len(t_targets))]
            d_pos = [d_all[i] for i in range(len(t_targets)) if t_targets[i].max() > 0]
            
            fold_summaries.append({
                "fold": fold,
                "status": "resumed",
                "test_dice_all": float(np.mean(d_all)),
                "test_dice_pos": float(np.mean(d_pos)) if d_pos else 0.0,
                "train_time_sec": 0.0
            })
            continue

        # Train Fold with cached images (zero disk I/O during training)
        res = train_single_fold(
            fold_info=fold_info,
            df=df,
            output_dir=effective_output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_filters=args.base_filters,
            patience=args.patience,
            images_cache=images_cache,
            masks_cache=masks_cache
        )
        fold_summaries.append(res)
        
    total_elapsed = time.time() - total_start_time
    
    # 5. Aggregate Complete Out-Of-Fold Performance
    mean_dice_all = float(np.mean([f["test_dice_all"] for f in fold_summaries]))
    mean_dice_pos = float(np.mean([f["test_dice_pos"] for f in fold_summaries]))
    
    experiment_summary = {
        "split_type": args.split_type,
        "epochs": args.epochs,
        "base_filters": args.base_filters,
        "total_elapsed_sec": round(total_elapsed, 2),
        "mean_test_dice_all": round(mean_dice_all, 4),
        "mean_test_dice_pos": round(mean_dice_pos, 4),
        "fold_details": fold_summaries
    }
    
    summary_path = os.path.join(effective_output_dir, f"experiment_summary_{args.split_type}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=4)
        
    print("\n" + "=" * 75)
    print(f"  EXPERIMENT [{args.split_type.upper()}] COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print(f"  Summary saved to:    {summary_path}")
    print(f"  Mean OOF Dice_all:   {mean_dice_all:.4f}")
    print(f"  Mean OOF Dice_pos:   {mean_dice_pos:.4f}")
    print(f"  Total Runtime:       {total_elapsed/60.0:.2f} minutes")
    print("=" * 75)

if __name__ == "__main__":
    main()
