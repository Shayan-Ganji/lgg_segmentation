import numpy as np
from scipy.ndimage import label as scipy_label, rotate
import tensorflow as tf
from src.config import CFG
from src.metrics import hard_dice_coef, hard_iou_coef

def morph_filter(binary: np.ndarray, min_px=CFG.get("MIN_COMPONENT_PX", 10)) -> np.ndarray:
    """Removes segmented connected components smaller than min_px threshold."""
    labeled, n = scipy_label(binary)
    out = np.zeros_like(binary)
    for cid in range(1, n + 1):
        if (labeled == cid).sum() >= min_px:
            out[labeled == cid] = 1
    return out.astype(np.uint8)

def tta_predict_batch(model, img_batch) -> np.ndarray:
    """Executes rotational Test-Time Augmentation (TTA) across 5 structural angles."""
    TTA_ANGLES = [-10, -5, 0, 5, 10]
    final_preds = np.zeros_like(img_batch[..., :1], dtype=np.float32)

    for angle in TTA_ANGLES:
        if angle == 0:
            preds = model.predict(img_batch, verbose=0)
            main_pred = preds[0] if isinstance(preds, (list, tuple)) else preds
            final_preds += main_pred
        else:
            # Batch rotation using scipy.ndimage.rotate across spatial axes (1, 2)
            rotated_imgs = rotate(img_batch, angle=angle, axes=(1, 2), reshape=False, order=1, mode='reflect')
            preds = model.predict(rotated_imgs, verbose=0)
            main_pred = preds[0] if isinstance(preds, (list, tuple)) else preds
            inv_preds = rotate(main_pred, angle=-angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)
            final_preds += inv_preds

    return final_preds / len(TTA_ANGLES)

def run_threshold_grid_search(soft_preds: np.ndarray, gt_masks: np.ndarray) -> float:
    """Grid search on decision boundaries [0.30, 0.85] to discover optimal Hard Dice threshold."""
    best_t, best_d = 0.50, 0.0
    print("\n  Executing decision boundary grid search...")
    
    for t in np.arange(0.30, 0.85, 0.05):
        dices = []
        for i in range(len(soft_preds)):
            prob = soft_preds[i, :, :, 0]
            gt = gt_masks[i, :, :, 0].astype(np.uint8)
            binary = morph_filter((prob >= t).astype(np.uint8))
            dices.append(hard_dice_coef(gt, binary))
            
        mean_d = float(np.mean(dices))
        print(f"    Threshold={t:.2f}  ->  Mean Hard Dice={mean_d:.4f}")
        if mean_d > best_d:
            best_d, best_t = mean_d, t
            
    print(f"  ⭐ Optimal decision threshold discovered: {best_t:.2f} (Dice={best_d:.4f})")
    return best_t