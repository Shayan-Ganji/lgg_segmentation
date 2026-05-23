import os
import gc
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CFG, seed_everything
from src.dataset import build_dataframe, build_dataset
from src.train import train_fold
from src.evaluate import tta_predict_batch, run_threshold_grid_search, morph_filter
from src.metrics import hard_dice_coef, hard_iou_coef

def main():
    print("\n" + "=" * 65)
    print("  LAUNCHING PRODUCTION PIPELINE: LIGHTWEIGHT ATTENTION RES-UNet (v5)")
    print("  Protocol: Strict Patient-Level 5-Fold Cross Validation + Rotation TTA")
    print("=" * 65 + "\n")

    seed_everything()
    
    df = build_dataframe()
    
    gkf = GroupKFold(n_splits=CFG["N_FOLDS"])
    folds = list(gkf.split(df, groups=df["patient_id"]))
    
    steps_per_epoch = math.ceil(len(folds[0][0]) / CFG["BATCH_SIZE"])
    
    ckpt_paths = []
    val_soft_dices = []
    
    for fold, (tr_idx, vl_idx) in enumerate(folds):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_vl = df.iloc[vl_idx].reset_index(drop=True)
        
        ckpt = train_fold(fold, df_tr, df_vl, steps_per_epoch)
        ckpt_paths.append(ckpt)
        
    print("\n" + "-" * 65)
    print("  CROSS-VALIDATION TRAINING COMPLETE")
    print("-" * 65)

    N = len(df)
    S = CFG["IMG_SIZE"]
    oof_soft_preds = np.zeros((N, S, S, 1), dtype=np.float32)
    oof_gt_masks = np.zeros((N, S, S, 1), dtype=np.float32)

    print("\n  Starting Out-Of-Fold TTA Ensemble Prediction...")
    for fold, ((_, val_idx), ckpt) in enumerate(zip(folds, ckpt_paths)):
        print(f"    -> Running TTA Inference on Fold {fold+1} using: {ckpt}")
        model = tf.keras.models.load_model(ckpt, compile=False)
        
        df_val = df.iloc[val_idx].reset_index(drop=True)
        ds_val = build_dataset(df_val, training=False, batch_size=CFG["BATCH_SIZE"])
        
        preds_list, gts_list = [], []
        for img_batch, msk_batch in ds_val:
            tta_preds = tta_predict_batch(model, img_batch.numpy())
            preds_list.append(tta_preds)
            gts_list.append(msk_batch.numpy())
            
        oof_soft_preds[val_idx] = np.concatenate(preds_list, axis=0)
        oof_gt_masks[val_idx] = np.concatenate(gts_list, axis=0)
        
        del model; gc.collect(); tf.keras.backend.clear_session()

    best_threshold = run_threshold_grid_search(oof_soft_preds, oof_gt_masks)

    final_dices, final_ious = [], []
    tumor_pos_dices = []

    for i in range(N):
        prob = oof_soft_preds[i, :, :, 0]
        gt = oof_gt_masks[i, :, :, 0].astype(np.uint8)
        
        binary = (prob >= best_threshold).astype(np.uint8)
        binary = morph_filter(binary)
        
        d = hard_dice_coef(gt, binary)
        iou = hard_iou_coef(gt, binary)
        
        final_dices.append(d)
        final_ious.append(iou)
        
        if gt.max() > 0:
            tumor_pos_dices.append(d)

    print("\n" + "=" * 65)
    print("  PRODUCTION FRAMEWORK FINAL HONEST METRICS")
    print("=" * 65)
    print(f"  Optimal Decision Threshold : {best_threshold:.3f}")
    print(f"  Mean Hard Dice (All Slices): {np.mean(final_dices):.4f}")
    print(f"  Mean Hard IoU (All Slices) : {np.mean(final_ious):.4f}")
    print(f"  Mean Hard Dice (Tumor Only): {np.mean(tumor_pos_dices):.4f}")
    print("=" * 65 + "\n")

    df_out = df.copy()
    df_out["hard_dice"] = final_dices
    df_out["hard_iou"] = final_ious
    df_out.to_csv(os.path.join(CFG["OUTPUT_DIR"], "oof_per_slice.csv"), index=False)
    print(f"  Per-slice detailed CSV report saved to {CFG['OUTPUT_DIR']}/oof_per_slice.csv")

if __name__ == "__main__":
    main()