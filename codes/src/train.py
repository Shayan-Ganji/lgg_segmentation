import os
import time
import functools
import numpy as np
import pandas as pd
import tensorflow as tf

# Force unbuffered stdout streaming for real-time remote GPU logs
print = functools.partial(print, flush=True)

from src.config import CFG
from src.dataset import build_dataset
from src.losses import compound_production_loss
from src.metrics import soft_dice_coef, hard_dice_coef, hard_iou_coef
from src.models.attention_resunet import build_attention_res_unet

class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_initial, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_initial = float(lr_initial)
        self.lr_min = float(lr_min)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.lr_initial * (step / tf.maximum(1.0, self.warmup_steps))
        progress = (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(float(np.pi) * progress))
        cosine_lr = self.lr_min + (self.lr_initial - self.lr_min) * cosine_decay
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "lr_initial": self.lr_initial,
            "lr_min": self.lr_min,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }

def predict_split(inference_model, df_subset, batch_size=CFG["BATCH_SIZE"],
                  images_cache: np.ndarray = None, masks_cache: np.ndarray = None):
    """
    Runs deterministic GPU-accelerated inference over df_subset.
    Returns:
      probs: (N, 128, 128) float32
      logits: (N, 128, 128) float32
      targets: (N, 128, 128) uint8
      patient_ids: (N,) string
      slice_indices: (N,) int32
      is_artifact_flagged: (N,) bool
    """
    ds = build_dataset(df_subset, training=False, batch_size=batch_size,
                       images_cache=images_cache, masks_cache=masks_cache)
    
    @tf.function
    def infer_step(x):
        return inference_model(x, training=False)
        
    probs_list = []
    logits_list = []
    targets_list = []
    
    for x_batch, y_batch in ds:
        p_batch, z_batch = infer_step(x_batch)
        probs_list.append(p_batch.numpy().squeeze(-1))
        logits_list.append(z_batch.numpy().squeeze(-1))
        targets_list.append(y_batch.numpy().squeeze(-1).astype(np.uint8))
        
    probs = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, CFG["IMG_SIZE"], CFG["IMG_SIZE"]), np.float32)
    logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, CFG["IMG_SIZE"], CFG["IMG_SIZE"]), np.float32)
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.empty((0, CFG["IMG_SIZE"], CFG["IMG_SIZE"]), np.uint8)
    
    n_slices = len(df_subset)
    probs = probs[:n_slices]
    logits = logits[:n_slices]
    targets = targets[:n_slices]
    
    patient_ids = df_subset["patient_id"].values.astype(str)
    slice_indices = df_subset["slice_idx"].values.astype(np.int32)
    if "is_artifact_flagged" in df_subset.columns:
        is_artifact_flagged = df_subset["is_artifact_flagged"].values.astype(bool)
    else:
        is_artifact_flagged = np.zeros(n_slices, dtype=bool)
    
    return probs, logits, targets, patient_ids, slice_indices, is_artifact_flagged

def train_single_fold(fold_info: dict, df: pd.DataFrame, 
                      output_dir: str, epochs: int = CFG["EPOCHS"], 
                      batch_size: int = CFG["BATCH_SIZE"],
                      base_filters: int = CFG["BASE_FILTERS"],
                      patience: int = CFG["ES_PATIENCE"],
                      images_cache: np.ndarray = None,
                      masks_cache: np.ndarray = None):
    """
    Trains one fold at hardware saturation speed with compiled @tf.function.
    Saves:
      1. Best model weights (best_model_fold{k}.weights.h5)
      2. oof_val_predictions_fold{k}.npz (compressed float16 logits)
      3. oof_test_predictions_fold{k}.npz (compressed float16 logits)
      4. fold{k}_history.csv
    """
    fold = fold_info["fold"]
    print(f"\n{'='*65}\nSTARTING FOLD {fold} / {CFG['N_FOLDS']}\n{'='*65}")
    
    os.makedirs(output_dir, exist_ok=True)
    val_npz_path = os.path.join(output_dir, f"oof_val_predictions_fold{fold}.npz")
    test_npz_path = os.path.join(output_dir, f"oof_test_predictions_fold{fold}.npz")
    weights_path = os.path.join(output_dir, f"best_model_fold{fold}.weights.h5")
    history_csv = os.path.join(output_dir, f"fold{fold}_history.csv")
    
    df_train = df.loc[fold_info["train_fit_idx"]].reset_index(drop=True)
    df_val = df.loc[fold_info["inner_val_idx"]].reset_index(drop=True)
    df_test = df.loc[fold_info["outer_test_idx"]].reset_index(drop=True)
    
    # Artifact screen exclusion from training only (clean train, unfiltered validation/test)
    if "is_artifact_flagged" in df_train.columns:
        n_tr_orig = len(df_train)
        df_train = df_train[~df_train["is_artifact_flagged"]].reset_index(drop=True)
        n_tr_filtered = len(df_train)
        print(f"  Artifact screening in Train set: excluded {n_tr_orig - n_tr_filtered} noisy slices ({n_tr_filtered} clean slices retained).")
        
    print(f"  Train slices: {len(df_train)} (clean) | Inner Val slices: {len(df_val)} (unfiltered) | Outer Test slices: {len(df_test)} (unfiltered)")
    
    # Datasets with precomputed in-memory cache
    ds_train = build_dataset(df_train, training=True, batch_size=batch_size,
                             images_cache=images_cache, masks_cache=masks_cache)
    ds_val = build_dataset(df_val, training=False, batch_size=batch_size,
                           images_cache=images_cache, masks_cache=masks_cache)
    
    # Dual-output Architecture
    train_model, inference_model = build_attention_res_unet(
        img_size=CFG["IMG_SIZE"], in_ch=CFG["INPUT_CHANNELS"],
        base_f=base_filters, groups=CFG["GROUP_NORM_G"], dropout=CFG["DROPOUT"]
    )
    
    steps_per_epoch = max(1, len(df_train) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * CFG["WARMUP_EPOCHS"]
    
    lr_schedule = CosineWarmupSchedule(
        lr_initial=CFG["LR_INITIAL"], lr_min=CFG["LR_MIN"],
        warmup_steps=warmup_steps, total_steps=total_steps
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=CFG["WEIGHT_DECAY"],
        clipnorm=CFG["GRAD_CLIP"]
    )
    
    # Compiled Graph Step Functions for Hardware GPU Saturation (Zero Python/Eager Overhead)
    ds_weight = float(CFG["DS_WEIGHT"])

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            p_main, p_half, p_quart = train_model(x, training=True)
            loss_main = compound_production_loss(y, p_main)
            loss_half = compound_production_loss(y, p_half)
            loss_quart = compound_production_loss(y, p_quart)
            total_loss = loss_main + ds_weight * (loss_half + loss_quart)
            
        grads = tape.gradient(total_loss, train_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, train_model.trainable_variables))
        d_main = soft_dice_coef(y, p_main)
        return total_loss, d_main

    @tf.function
    def val_step(x, y):
        p_main, _, _ = train_model(x, training=False)
        loss = compound_production_loss(y, p_main)
        dice = soft_dice_coef(y, p_main)
        return loss, dice

    train_loss_metric = tf.keras.metrics.Mean()
    train_dice_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()
    val_dice_metric = tf.keras.metrics.Mean()

    # Smart Resume from existing checkpoint (e.g. Fold 2 interrupted after hours of training)
    start_epoch = 1
    best_val_dice = -1.0
    history_records = []
    patience_counter = 0

    if os.path.exists(weights_path):
        print(f"  [SMART RESUME] Existing weights checkpoint detected: {weights_path}")
        try:
            train_model.load_weights(weights_path)
            # Evaluate baseline on inner val set
            val_loss_metric.reset_state()
            val_dice_metric.reset_state()
            for x_v, y_v in ds_val:
                v_loss, v_dice = val_step(x_v, y_v)
                val_loss_metric.update_state(v_loss)
                val_dice_metric.update_state(v_dice)
            best_val_dice = float(val_dice_metric.result())
            print(f"  [SMART RESUME] Baseline Val Soft Dice from loaded weights: {best_val_dice:.4f}")

            if os.path.exists(history_csv):
                try:
                    hist_df = pd.read_csv(history_csv)
                    if len(hist_df) > 0:
                        completed_epochs = int(hist_df["epoch"].max())
                        start_epoch = completed_epochs + 1
                        history_records = hist_df.to_dict("records")
                        print(f"  [SMART RESUME] Found {completed_epochs} previously completed epochs in {history_csv}. Continuing from epoch {start_epoch}...")
                except Exception as e:
                    print(f"  [SMART RESUME] History parse notice: {e}")
        except Exception as e:
            print(f"  [SMART RESUME] Warning: Could not restore weights ({e}), starting fold from scratch.")

    start_train_time = time.time()
    
    # Fast GPU Training Loop
    if start_epoch <= epochs:
        print(f"  Launching GPU Training from epoch {start_epoch} to {epochs} with @tf.function acceleration...")
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            train_loss_metric.reset_state()
            train_dice_metric.reset_state()
            
            # Optimized Training Step (Zero per-batch CPU-GPU sync stalls)
            for x_b, y_b in ds_train:
                t_loss, t_dice = train_step(x_b, y_b)
                train_loss_metric.update_state(t_loss)
                train_dice_metric.update_state(t_dice)
                
            mean_tr_loss = float(train_loss_metric.result())
            mean_tr_dice = float(train_dice_metric.result())
            
            # Optimized Validation Step
            val_loss_metric.reset_state()
            val_dice_metric.reset_state()
            for x_v, y_v in ds_val:
                v_loss, v_dice = val_step(x_v, y_v)
                val_loss_metric.update_state(v_loss)
                val_dice_metric.update_state(v_dice)
                
            mean_val_loss = float(val_loss_metric.result())
            mean_val_dice = float(val_dice_metric.result())
            current_lr = float(lr_schedule(optimizer.iterations))
            
            epoch_sec = time.time() - epoch_start
            
            # Checkpointing on inner_val soft dice
            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                train_model.save_weights(weights_path)
                patience_counter = 0
                flag = " ⭐ (best saved)"
            else:
                patience_counter += 1
                flag = ""
                
            print(f"  Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.1f}s] - LR: {current_lr:.2e} - "
                  f"Train Loss: {mean_tr_loss:.4f} Dice: {mean_tr_dice:.4f} | "
                  f"Val Loss: {mean_val_loss:.4f} Dice: {mean_val_dice:.4f}{flag}")
            
            history_records.append({
                "epoch": epoch,
                "train_loss": mean_tr_loss,
                "train_dice": mean_tr_dice,
                "val_loss": mean_val_loss,
                "val_dice": mean_val_dice,
                "lr": current_lr
            })
            
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (best inner val dice: {best_val_dice:.4f})")
                break
    else:
        print(f"  [SMART RESUME] All {epochs} epochs already completed for Fold {fold}. Proceeding directly to .npz export!")

    pd.DataFrame(history_records).to_csv(history_csv, index=False)
    print(f"  Saved training curve log to: {history_csv}")
    
    # Restore best checkpoint
    print(f"  Restoring best weights from: {weights_path}")
    train_model.load_weights(weights_path)
    
    # Export inner_val predictions (.npz in ultra-compact float16 format, ~10MB)
    print(f"  Exporting inner_val predictions to: {val_npz_path}")
    v_probs, v_logits, v_targets, v_pids, v_sindices, v_flagged = predict_split(
        inference_model, df_val, batch_size, images_cache, masks_cache
    )
    np.savez_compressed(
        val_npz_path,
        logits=v_logits.astype(np.float16),
        targets=v_targets.astype(np.uint8),
        patient_ids=v_pids,
        slice_indices=v_sindices,
        is_artifact_flagged=v_flagged
    )
    
    # Export outer_test predictions (.npz in ultra-compact float16 format, ~15MB)
    print(f"  Exporting outer_test predictions to: {test_npz_path}")
    t_probs, t_logits, t_targets, t_pids, t_sindices, t_flagged = predict_split(
        inference_model, df_test, batch_size, images_cache, masks_cache
    )
    np.savez_compressed(
        test_npz_path,
        logits=t_logits.astype(np.float16),
        targets=t_targets.astype(np.uint8),
        patient_ids=t_pids,
        slice_indices=t_sindices,
        is_artifact_flagged=t_flagged
    )
    
    # Calculate baseline metrics at standard tau = 0.50
    t_pred_bin = (t_probs >= 0.50).astype(np.uint8)
    dice_pos_list = []
    dice_all_list = []
    for i in range(len(t_targets)):
        d = hard_dice_coef(t_targets[i], t_pred_bin[i])
        dice_all_list.append(d)
        if t_targets[i].max() > 0:
            dice_pos_list.append(d)
            
    res_summary = {
        "fold": fold,
        "best_val_dice": best_val_dice,
        "test_dice_all": float(np.mean(dice_all_list)),
        "test_dice_pos": float(np.mean(dice_pos_list)) if dice_pos_list else 0.0,
        "train_time_sec": round(time.time() - start_train_time, 2)
    }
    print(f"  Fold {fold} Finished: Test Dice_all={res_summary['test_dice_all']:.4f}, Test Dice_pos={res_summary['test_dice_pos']:.4f}")
    return res_summary