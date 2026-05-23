import os
import gc
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CFG, seed_everything
from src.dataset import build_dataframe, build_dataset
from src.models.attention_resunet import build_attention_res_unet
from src.losses import compound_production_loss
from src.metrics import soft_dice_coef, hard_dice_coef, hard_iou_coef
from src.evaluate import morph_filter

def main():
    print("\n" + "=" * 65)
    print("  LAUNCHING CONTROLLED EXPERIMENT: SPATIAL DATA LEAKAGE ANALYSIS")
    print("  Protocol: Randomized Slice-Level Split (Replicating SOTA Flaws)")
    print("=" * 65 + "\n")

    seed_everything()
    leaky_output_dir = "./outputs_leaky"
    os.makedirs(leaky_output_dir, exist_ok=True)

    df = build_dataframe()

    df_train, df_test = train_test_split(df, test_size=0.236, random_state=CFG["SEED"], shuffle=True)
    
    tumor_df = df_train[df_train.has_tumor].copy()
    empty_df = df_train[~df_train.has_tumor].copy()
    balanced_tr = pd.concat([tumor_df.sample(frac=2.0, replace=True, random_state=CFG["SEED"]),
                             empty_df.sample(frac=0.40, replace=False, random_state=CFG["SEED"])]).sample(frac=1.0, random_state=CFG["SEED"]).reset_index(drop=True)

    print(f"  Leaky Splitting: Train Slices={len(balanced_tr)} | Test Slices={len(df_test)}")
    
    ds_tr = build_dataset(balanced_tr, training=True, batch_size=CFG["BATCH_SIZE"])
    ds_vl = build_dataset(df_test, training=False, batch_size=CFG["BATCH_SIZE"])

    model = build_attention_res_unet()
    
    opt = tf.keras.optimizers.AdamW(learning_rate=CFG["LR_INITIAL"], weight_decay=CFG["WEIGHT_DECAY"])
    model.compile(
        optimizer=opt,
        loss={"main": compound_production_loss, "ds_half_act": compound_production_loss, "ds_quarter_act": compound_production_loss},
        loss_weights={"main": 1.0, "ds_half_act": CFG["DS_WEIGHT"], "ds_quarter_act": CFG["DS_WEIGHT"]},
        metrics={"main": [soft_dice_coef]}
    )

    def wrap_targets(img, msk):
        return img, {"main": msk, "ds_half_act": msk, "ds_quarter_act": msk}

    print("\n  Starting Leaky Training Process...")
    ckpt_leaky = os.path.join(leaky_output_dir, "leaky_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_leaky, monitor="val_main_soft_dice_coef", mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_main_soft_dice_coef", mode="max", patience=CFG["ES_PATIENCE"], restore_best_weights=True, verbose=1)
    ]

    model.fit(ds_tr.map(wrap_targets), validation_data=ds_vl.map(wrap_targets), epochs=CFG["EPOCHS"], callbacks=callbacks, verbose=1)

    print("\n  Evaluating Leaky Model on Leaky Test Distribution...")
    best_leaky_model = tf.keras.models.load_model(ckpt_leaky, compile=False)
    
    leaky_soft_preds = []
    gt_list = []
    
    for _, row in df_test.iterrows():
        img_raw = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        img_arr = cv2.resize(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), (CFG["IMG_SIZE"], CFG["IMG_SIZE"])).astype(np.float32) / 255.0
        
        msk_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        gt = (cv2.resize(msk_raw, (CFG["IMG_SIZE"], CFG["IMG_SIZE"]), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        
        pred = best_leaky_model(tf.constant(img_arr[np.newaxis]), training=False)
        main_pred = pred[0].numpy()[0, :, :, 0] if isinstance(pred, (list, tuple)) else pred.numpy()[0, :, :, 0]
        
        leaky_soft_preds.append(main_pred)
        gt_list.append(gt)

    best_t, best_d = 0.50, 0.0
    for t in np.arange(0.25, 0.95, 0.05):
        dices = [hard_dice_coef(gt_list[i], morph_filter((leaky_soft_preds[i] >= t).astype(np.uint8))) for i in range(len(gt_list))]
        if np.mean(dices) > best_d:
            best_d, best_t = np.mean(dices), t

    dices = [hard_dice_coef(gt_list[i], morph_filter((leaky_soft_preds[i] >= best_t).astype(np.uint8))) for i in range(len(gt_list))]
    ious = [hard_iou_coef(gt_list[i], morph_filter((leaky_soft_preds[i] >= best_t).astype(np.uint8))) for i in range(len(gt_list))]
    tumor_pos_dices = [dices[i] for i in range(len(gt_list)) if gt_list[i].max() > 0]

    print("\n" + "=" * 65)
    print("  ⚠️ SPATIAL DATA LEAKAGE SIMULATION RESULTS")
    print("=" * 65)
    print(f"  Inflated Threshold         : {best_t:.3f}")
    print(f"  Inflated Hard Dice (All)   : {np.mean(dices):.4f} (Expected: 0.9232)")
    print(f"  Inflated Hard IoU  (All)   : {np.mean(ious):.4f} (Expected: 0.8937)")
    print(f"  Inflated Hard Dice (Tumor+): {np.mean(tumor_pos_dices):.4f} (Expected: 0.8366)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()