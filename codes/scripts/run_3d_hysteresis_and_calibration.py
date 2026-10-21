"""
=============================================================================
BSPC Journal Phase 3: Step 2 - Leak-Free Inner Calibration & 3D Z-Hysteresis
=============================================================================
Executes:
  1. Nested Temperature Scaling: T_k* tuned strictly on inner_val logits
     via NLL minimization; applied to outer_test logits.
     Evaluates NLL, ECE (Expected Calibration Error), and Brier Score.
  2. Nested 2D Threshold Optimization: tau_k* tuned on inner_val vs.
     Leaked Pooled Threshold (tau_leak*) to quantify Delta Optimism
     (direct quantitative answer to Reviewer 1 and Editor).
  3. 3D Z-Continuity Hysteresis:
     - 3D volume assembly (128 x 128 x Z_p) per patient along Z-axis.
     - Dual-threshold hysteresis (tau_high*, tau_low*) with 26-neighborhood
       continuity and minimum volume filter (V_min*) tuned on inner_val.
     - Applied to untouched outer_test 3D volumes.
     - Evaluates Dice_pos, Dice_all, FPR_empty, and boundary slice rescue.
  4. Patient-level clustered bootstrap (10,000 resamples) 95% CIs.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import functools
from scipy.optimize import minimize_scalar
from scipy.ndimage import label, binary_dilation
from scipy.special import expit

print = functools.partial(print, flush=True)

# Add project root and codes to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(CODES_DIR)
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(CODES_DIR, "results", "bspc_production")
PG_DIR = os.path.join(RESULTS_DIR, "patient_grouped")
STRUCT_26 = np.ones((3, 3, 3), dtype=bool)
STRUCT_2D = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def compute_ece(probs, targets, n_bins=15):
    """Computes Expected Calibration Error (ECE) across 15 equal bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i+1]
        mask = (probs >= b_low) & (probs < b_high) if i < n_bins - 1 else (probs >= b_low) & (probs <= b_high)
        n_b = np.sum(mask)
        if n_b > 0:
            acc = np.mean(targets[mask])
            conf = np.mean(probs[mask])
            ece += (n_b / total) * np.abs(acc - conf)
    return float(ece)


def compute_brier(probs, targets):
    """Computes Brier Score (Mean Squared Error of probability)."""
    return float(np.mean((probs - targets) ** 2))


def compute_nll(logits, targets, temperature=1.0):
    """Numerically stable binary cross-entropy (Negative Log-Likelihood)."""
    z = logits / temperature
    return float(np.mean(np.logaddexp(0.0, z) - targets * z))


def fit_temperature(val_logits, val_targets):
    """
    Minimizes NLL strictly on inner_val logits using bounded scalar search.
    """
    flat_logits = val_logits.astype(np.float32).ravel()
    flat_targets = val_targets.astype(np.float32).ravel()

    def obj(T):
        z = flat_logits / T
        return np.mean(np.logaddexp(0.0, z) - flat_targets * z)

    res = minimize_scalar(obj, bounds=(0.5, 4.0), method="bounded")
    T_star = float(res.x)
    return T_star


def assemble_patient_volumes(logits, targets, pids, sindices, T=1.0):
    """
    Groups slices by patient_id, sorts by slice_idx along Z-axis,
    and returns list of patient volume dicts.
    """
    unique_pids = np.unique(pids)
    volumes = []

    for pid in unique_pids:
        idx = np.where(pids == pid)[0]
        sort_order = np.argsort(sindices[idx])
        sorted_idx = idx[sort_order]

        vol_logits = logits[sorted_idx].astype(np.float32)
        vol_probs = expit(vol_logits / T)
        vol_targets = targets[sorted_idx].astype(np.uint8)
        vol_sindices = sindices[sorted_idx].astype(np.int32)

        volumes.append({
            "patient_id": pid,
            "logits": vol_logits,
            "probs": vol_probs,
            "targets": vol_targets,
            "slice_indices": vol_sindices,
            "Z": len(sorted_idx)
        })

    return volumes


def apply_anisotropic_3d_hysteresis(vol_probs, tau_xy, tau_z_low=0.40, min_vol=50, tau_seed=0.70):
    """
    Anisotropic 3D Z-Continuity Hysteresis:
    1. In-plane segmentation with optimal tau_xy.
    2. Z-Continuity Rescue for empty/near-empty slices adjacent to detected tumor slices.
    3. 3D False Positive Component Suppression (min_vol and tau_seed).
    """
    Z = len(vol_probs)
    m0 = (vol_probs >= tau_xy).astype(bool)

    # Step 2: Z-Continuity Rescue ONLY for empty/near-empty slices
    m1 = m0.copy()
    for z in range(Z):
        if np.sum(m0[z]) < 10:
            adj = []
            if z > 0 and np.sum(m0[z-1]) >= 20:
                adj.append(binary_dilation(m0[z-1], structure=STRUCT_2D, iterations=2))
            if z < Z - 1 and np.sum(m0[z+1]) >= 20:
                adj.append(binary_dilation(m0[z+1], structure=STRUCT_2D, iterations=2))
            if adj:
                prior = np.logical_or.reduce(adj)
                m1[z] = m0[z] | (prior & (vol_probs[z] >= tau_z_low))

    # Step 3: 3D False Positive Component Suppression
    lbl, n_lbl = label(m1, structure=STRUCT_26)
    if n_lbl == 0:
        return np.zeros_like(vol_probs, dtype=np.uint8)

    comp_sizes = np.bincount(lbl.ravel())
    valid = [c for c in range(1, n_lbl + 1) if comp_sizes[c] >= min_vol and np.max(vol_probs[lbl == c]) >= tau_seed]
    if len(valid) == 0:
        # Fallback safeguard: if min_vol filtered all components, preserve high-confidence components
        # with peak probability >= tau_seed so small tumors in genuine patients are not obliterated
        strong_candidates = [c for c in range(1, n_lbl + 1) if np.max(vol_probs[lbl == c]) >= tau_seed]
        if strong_candidates:
            valid = strong_candidates
        else:
            return np.zeros_like(vol_probs, dtype=np.uint8)

    pred_3d = np.isin(lbl, valid).astype(np.uint8)
    return pred_3d



def evaluate_slice_metrics(targets, preds_bin):
    """Computes slice-wise Dice, IoU, Precision, Recall."""
    tp = np.sum((targets == 1) & (preds_bin == 1), axis=(1, 2)).astype(np.float32)
    fp = np.sum((targets == 0) & (preds_bin == 1), axis=(1, 2)).astype(np.float32)
    fn = np.sum((targets == 1) & (preds_bin == 0), axis=(1, 2)).astype(np.float32)

    total_pos = tp + fp + fn
    empty_both = (total_pos == 0)

    denom_dice = 2.0 * tp + fp + fn
    dice = np.zeros_like(tp)
    nonzero_d = denom_dice > 0
    dice[nonzero_d] = (2.0 * tp[nonzero_d]) / denom_dice[nonzero_d]
    dice[empty_both] = 1.0

    denom_iou = tp + fp + fn
    iou = np.zeros_like(tp)
    nonzero_u = denom_iou > 0
    iou[nonzero_u] = tp[nonzero_u] / denom_iou[nonzero_u]
    iou[empty_both] = 1.0

    denom_prec = tp + fp
    prec = np.zeros_like(tp)
    nonzero_p = denom_prec > 0
    prec[nonzero_p] = tp[nonzero_p] / denom_prec[nonzero_p]
    prec[empty_both] = 1.0

    denom_rec = tp + fn
    rec = np.zeros_like(tp)
    nonzero_r = denom_rec > 0
    rec[nonzero_r] = tp[nonzero_r] / denom_rec[nonzero_r]
    rec[empty_both] = 1.0

    return dice, iou, prec, rec


def fast_patient_clustered_bootstrap_ci(pids, metric_arr, has_tumor_mask, n_bootstraps=10000, seed=42):
    """Patient-clustered 10,000 bootstrap 95% CIs."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(pids)
    n_patients = len(unique_patients)

    pid_to_int = {pid: i for i, pid in enumerate(unique_patients)}
    slice_p_indices = np.array([pid_to_int[p] for p in pids])

    pat_metric_sum_all = np.zeros(n_patients, dtype=np.float64)
    pat_count_all = np.zeros(n_patients, dtype=np.int64)
    pat_metric_sum_pos = np.zeros(n_patients, dtype=np.float64)
    pat_count_pos = np.zeros(n_patients, dtype=np.int64)

    for i in range(len(pids)):
        p_idx = slice_p_indices[i]
        m = metric_arr[i]
        pat_metric_sum_all[p_idx] += m
        pat_count_all[p_idx] += 1
        if has_tumor_mask[i]:
            pat_metric_sum_pos[p_idx] += m
            pat_count_pos[p_idx] += 1

    boot_p_indices = rng.choice(n_patients, size=(n_bootstraps, n_patients), replace=True)

    sampled_sum_all = np.take(pat_metric_sum_all, boot_p_indices).sum(axis=1)
    sampled_cnt_all = np.take(pat_count_all, boot_p_indices).sum(axis=1)
    boot_means_all = sampled_sum_all / sampled_cnt_all

    sampled_sum_pos = np.take(pat_metric_sum_pos, boot_p_indices).sum(axis=1)
    sampled_cnt_pos = np.take(pat_count_pos, boot_p_indices).sum(axis=1)
    boot_means_pos = sampled_sum_pos / np.maximum(sampled_cnt_pos, 1)

    ci_all = (float(np.percentile(boot_means_all, 2.5)), float(np.percentile(boot_means_all, 97.5)))
    ci_pos = (float(np.percentile(boot_means_pos, 2.5)), float(np.percentile(boot_means_pos, 97.5)))

    return ci_all, ci_pos


def tune_anisotropic_3d_hysteresis_grid(val_vols, tau_xy):
    """
    Grid searches (tau_z_low, min_vol, tau_seed) strictly on inner_val patient volumes
    using the frozen optimal in-plane tau_xy.
    """
    tau_z_candidates = [0.35, 0.40, 0.45, 0.50]
    min_vol_candidates = [20, 50, 80]
    tau_seed_candidates = [0.70, 0.75]

    best_score = -1.0
    best_params = (0.40, 50, 0.70)

    for tz in tau_z_candidates:
        for vm in min_vol_candidates:
            for seed in tau_seed_candidates:
                d_pos_list = []
                for vol in val_vols:
                    p3d = apply_anisotropic_3d_hysteresis(vol["probs"], tau_xy, tau_z_low=tz, min_vol=vm, tau_seed=seed)
                    vt = vol["targets"]
                    has_t = np.sum(vt, axis=(1, 2)) > 0
                    if np.sum(has_t) == 0:
                        continue
                    d, _, _, _ = evaluate_slice_metrics(vt[has_t], p3d[has_t])
                    d_pos_list.extend(d)

                mean_d = float(np.mean(d_pos_list))
                if mean_d > best_score:
                    best_score = mean_d
                    best_params = (tz, vm, seed)

    return best_params, best_score


def main():
    print("=" * 90)
    print("  BSPC PHASE 3: STEP 2 - NESTED CALIBRATION, LEAK-FREE THRESHOLDING & 3D HYSTERESIS")
    print("=" * 90)

    fold_results = []
    
    # Storage for pooled test evaluations
    pooled_test_targets = []
    pooled_test_uncal_probs = []
    pooled_test_cal_probs = []
    pooled_test_pids = []
    pooled_test_sindices = []
    pooled_test_folds = []

    # Final prediction masks across all outer_test slices
    all_pred_2d_base = []
    all_pred_2d_nested = []
    all_pred_3d_hyst = []

    # Boundary slice tracking
    boundary_records = []

    print("\n--- Iterating 5 Folds under Strict Nested Inner-Validation Tuning ---")

    for fold in range(1, 6):
        print(f"\n{'='*30} FOLD {fold} / 5 {'='*30}")

        # 1. Load Inner Val and Outer Test NPZ
        val_path = os.path.join(PG_DIR, f"oof_val_predictions_fold{fold}.npz")
        test_path = os.path.join(PG_DIR, f"oof_test_predictions_fold{fold}.npz")

        val_data = np.load(val_path)
        test_data = np.load(test_path)

        val_logits = val_data["logits"].astype(np.float32)
        val_targets = val_data["targets"].astype(np.uint8)
        val_pids = val_data["patient_ids"].astype(str)
        val_sindices = val_data["slice_indices"].astype(np.int32)

        test_logits = test_data["logits"].astype(np.float32)
        test_targets = test_data["targets"].astype(np.uint8)
        test_pids = test_data["patient_ids"].astype(str)
        test_sindices = test_data["slice_indices"].astype(np.int32)

        print(f"Fold {fold} Data: Inner Val = {len(val_targets)} slices ({len(np.unique(val_pids))} patients) | "
              f"Outer Test = {len(test_targets)} slices ({len(np.unique(test_pids))} patients)")

        # 2. Temperature Scaling Optimization on Inner Val
        T_star = fit_temperature(val_logits, val_targets)
        
        # Calibration Metrics on Inner Val
        val_uncal_p = expit(val_logits)
        val_cal_p = expit(val_logits / T_star)

        val_nll_before = compute_nll(val_logits, val_targets, 1.0)
        val_nll_after = compute_nll(val_logits, val_targets, T_star)
        val_ece_before = compute_ece(val_uncal_p.ravel(), val_targets.ravel())
        val_ece_after = compute_ece(val_cal_p.ravel(), val_targets.ravel())
        val_brier_before = compute_brier(val_uncal_p.ravel(), val_targets.ravel())
        val_brier_after = compute_brier(val_cal_p.ravel(), val_targets.ravel())

        # Calibration Metrics on Untouched Outer Test
        test_uncal_p = expit(test_logits)
        test_cal_p = expit(test_logits / T_star)

        test_nll_before = compute_nll(test_logits, test_targets, 1.0)
        test_nll_after = compute_nll(test_logits, test_targets, T_star)
        test_ece_before = compute_ece(test_uncal_p.ravel(), test_targets.ravel())
        test_ece_after = compute_ece(test_cal_p.ravel(), test_targets.ravel())
        test_brier_before = compute_brier(test_uncal_p.ravel(), test_targets.ravel())
        test_brier_after = compute_brier(test_cal_p.ravel(), test_targets.ravel())

        print(f"  [Temp Scaling] Optimal T* = {T_star:.3f}")
        print(f"    Inner Val : NLL: {val_nll_before:.4f} -> {val_nll_after:.4f} | "
              f"ECE: {val_ece_before*100:.3f}% -> {val_ece_after*100:.3f}% | "
              f"Brier: {val_brier_before:.5f} -> {val_brier_after:.5f}")
        print(f"    Outer Test: NLL: {test_nll_before:.4f} -> {test_nll_after:.4f} | "
              f"ECE: {test_ece_before*100:.3f}% -> {test_ece_after*100:.3f}% | "
              f"Brier: {test_brier_before:.5f} -> {test_brier_after:.5f}")

        # 3. 2D Single Threshold Tuning strictly on Inner Val
        val_has_t = np.sum(val_targets, axis=(1, 2)) > 0
        best_tau_2d = 0.50
        best_val_dice_pos = 0.0

        for tau in np.arange(0.20, 0.90, 0.02):
            p_bin = (val_cal_p[val_has_t] >= tau).astype(np.uint8)
            d, _, _, _ = evaluate_slice_metrics(val_targets[val_has_t], p_bin)
            mean_d = float(np.mean(d))
            if mean_d > best_val_dice_pos:
                best_val_dice_pos = mean_d
                best_tau_2d = float(tau)

        print(f"  [2D Threshold] Optimal tau* on inner_val = {best_tau_2d:.2f} (Val Dice_pos: {best_val_dice_pos*100:.2f}%)")

        # 4. Anisotropic 3D Z-Continuity Hysteresis Tuning strictly on Inner Val
        val_vols = assemble_patient_volumes(val_logits, val_targets, val_pids, val_sindices, T=T_star)
        (tl_star, vm_star, seed_star), val_3d_dice = tune_anisotropic_3d_hysteresis_grid(val_vols, best_tau_2d)
        print(f"  [Anisotropic 3D Hysteresis] Optimal params on inner_val: tau_xy={best_tau_2d:.2f}, "
              f"tau_z_low={tl_star:.2f}, V_min={vm_star:d}, tau_seed={seed_star:.2f} (Val 3D Dice_pos: {val_3d_dice*100:.2f}%)")

        # 5. Evaluate on Outer Test Patients
        test_vols = assemble_patient_volumes(test_logits, test_targets, test_pids, test_sindices, T=T_star)

        fold_pred_2d_base = []
        fold_pred_2d_nest = []
        fold_pred_3d_hyst = []
        fold_targets = []
        fold_pids = []
        fold_sindices = []

        for vol in test_vols:
            vp = vol["probs"]
            vt = vol["targets"]
            pid = vol["patient_id"]
            sind = vol["slice_indices"]

            # Method A: Baseline 2D (uncalibrated, tau = 0.50)
            p_base = (expit(vol["logits"]) >= 0.50).astype(np.uint8)

            # Method B: Nested 2D (calibrated, tau_star)
            p_nest = (vp >= best_tau_2d).astype(np.uint8)

            # Method C: Anisotropic 3D Z-Continuity Hysteresis (frozen parameters)
            p_hyst = apply_anisotropic_3d_hysteresis(vp, best_tau_2d, tl_star, vm_star, seed_star)

            fold_pred_2d_base.append(p_base)
            fold_pred_2d_nest.append(p_nest)
            fold_pred_3d_hyst.append(p_hyst)
            fold_targets.append(vt)
            fold_pids.extend([pid] * len(vt))
            fold_sindices.extend(sind)

            # Boundary Slice Evaluation
            has_t = np.sum(vt, axis=(1, 2)) > 0
            if np.sum(has_t) >= 3:
                pos_z = np.where(has_t)[0]
                boundary_slices = [pos_z[0], pos_z[-1]]
                for b_idx in boundary_slices:
                    tp_base = np.sum((p_base[b_idx] == 1) & (vt[b_idx] == 1))
                    d_base = float(2 * tp_base / (2 * tp_base + np.sum(p_base[b_idx] == 1) + np.sum(vt[b_idx] == 1)))

                    tp_hyst = np.sum((p_hyst[b_idx] == 1) & (vt[b_idx] == 1))
                    d_hyst = float(2 * tp_hyst / (2 * tp_hyst + np.sum(p_hyst[b_idx] == 1) + np.sum(vt[b_idx] == 1)))

                    boundary_records.append({
                        "fold": fold,
                        "patient_id": pid,
                        "slice_idx": int(sind[b_idx]),
                        "dice_2d_base": d_base,
                        "dice_3d_hyst": d_hyst,
                        "rescued": (d_base == 0.0 and d_hyst > 0.0)
                    })

        p_base_arr = np.concatenate(fold_pred_2d_base, axis=0)
        p_nest_arr = np.concatenate(fold_pred_2d_nest, axis=0)
        p_hyst_arr = np.concatenate(fold_pred_3d_hyst, axis=0)
        tgt_arr = np.concatenate(fold_targets, axis=0)

        all_pred_2d_base.append(p_base_arr)
        all_pred_2d_nested.append(p_nest_arr)
        all_pred_3d_hyst.append(p_hyst_arr)

        pooled_test_targets.append(test_targets)
        pooled_test_uncal_probs.append(test_uncal_p)
        pooled_test_cal_probs.append(test_cal_p)
        pooled_test_pids.extend(test_pids)
        pooled_test_sindices.extend(test_sindices)
        pooled_test_folds.extend([fold] * len(test_targets))

        # Fold test performance
        has_t_test = np.sum(tgt_arr, axis=(1, 2)) > 0
        d_base_pos = np.mean(evaluate_slice_metrics(tgt_arr[has_t_test], p_base_arr[has_t_test])[0])
        d_nest_pos = np.mean(evaluate_slice_metrics(tgt_arr[has_t_test], p_nest_arr[has_t_test])[0])
        d_hyst_pos = np.mean(evaluate_slice_metrics(tgt_arr[has_t_test], p_hyst_arr[has_t_test])[0])

        fpr_base_empty = np.mean(np.sum(p_base_arr[~has_t_test], axis=(1, 2)) > 0) * 100.0
        fpr_hyst_empty = np.mean(np.sum(p_hyst_arr[~has_t_test], axis=(1, 2)) > 0) * 100.0

        print(f"  [Outer Test Performance]")
        print(f"    • 2D Baseline (tau=0.50):    Dice_pos = {d_base_pos*100:.2f}% | FPR_empty = {fpr_base_empty:.2f}%")
        print(f"    • 2D Nested (tau={best_tau_2d:.2f}):      Dice_pos = {d_nest_pos*100:.2f}%")
        print(f"    • 3D Z-Hysteresis:           Dice_pos = {d_hyst_pos*100:.2f}% | FPR_empty = {fpr_hyst_empty:.2f}%")

        fold_results.append({
            "fold": fold,
            "T_star": T_star,
            "tau_2d_star": best_tau_2d,
            "tl_star": tl_star,
            "vm_star": vm_star,
            "seed_star": seed_star,
            "test_ece_before": test_ece_before,
            "test_ece_after": test_ece_after,
            "test_brier_before": test_brier_before,
            "test_brier_after": test_brier_after,
            "test_nll_before": test_nll_before,
            "test_nll_after": test_nll_after,
            "dice_base_pos": float(d_base_pos),
            "dice_nest_pos": float(d_nest_pos),
            "dice_hyst_pos": float(d_hyst_pos),
            "fpr_base_empty": float(fpr_base_empty),
            "fpr_hyst_empty": float(fpr_hyst_empty)
        })

    # 6. Global Analysis Across All 3,929 Outer Test Slices
    print("\n" + "=" * 90)
    print("  SUB-STEP 2.2: GLOBAL 5-FOLD POOLED EVALUATION & LEAKAGE OPTIMISM (DELTA OPTIMISM)")
    print("=" * 90)

    all_tgt = np.concatenate(pooled_test_targets, axis=0)
    all_uncal_p = np.concatenate(pooled_test_uncal_probs, axis=0)
    all_cal_p = np.concatenate(pooled_test_cal_probs, axis=0)
    all_base_bin = np.concatenate(all_pred_2d_base, axis=0)
    all_nest_bin = np.concatenate(all_pred_2d_nested, axis=0)
    all_hyst_bin = np.concatenate(all_pred_3d_hyst, axis=0)
    all_pids = np.array(pooled_test_pids)

    has_t = np.sum(all_tgt, axis=(1, 2)) > 0
    empty_mask = ~has_t

    # Search Leaked Pooled Threshold on Outer Test
    best_tau_leaked = 0.50
    best_d_leaked = 0.0
    for tau in np.arange(0.20, 0.95, 0.02):
        p_bin = (all_uncal_p[has_t] >= tau).astype(np.uint8)
        d, _, _, _ = evaluate_slice_metrics(all_tgt[has_t], p_bin)
        mean_d = float(np.mean(d))
        if mean_d > best_d_leaked:
            best_d_leaked = mean_d
            best_tau_leaked = float(tau)

    # Calculate metrics for the 3 core methods
    dice_base, iou_base, prec_base, rec_base = evaluate_slice_metrics(all_tgt, all_base_bin)
    dice_nest, iou_nest, prec_nest, rec_nest = evaluate_slice_metrics(all_tgt, all_nest_bin)
    dice_hyst, iou_hyst, prec_hyst, rec_hyst = evaluate_slice_metrics(all_tgt, all_hyst_bin)

    # 10,000 Patient Bootstrap CIs
    ci_dice_base_all, ci_dice_base_pos = fast_patient_clustered_bootstrap_ci(all_pids, dice_base, has_t)
    ci_dice_nest_all, ci_dice_nest_pos = fast_patient_clustered_bootstrap_ci(all_pids, dice_nest, has_t)
    ci_dice_hyst_all, ci_dice_hyst_pos = fast_patient_clustered_bootstrap_ci(all_pids, dice_hyst, has_t)

    ci_iou_hyst_all, ci_iou_hyst_pos = fast_patient_clustered_bootstrap_ci(all_pids, iou_hyst, has_t)
    ci_prec_hyst_all, ci_prec_hyst_pos = fast_patient_clustered_bootstrap_ci(all_pids, prec_hyst, has_t)
    ci_rec_hyst_all, ci_rec_hyst_pos = fast_patient_clustered_bootstrap_ci(all_pids, rec_hyst, has_t)

    # False Positive Rates on Empty Slices
    fpr_base_1px = np.mean(np.sum(all_base_bin[empty_mask], axis=(1, 2)) > 0) * 100.0
    fpr_nest_1px = np.mean(np.sum(all_nest_bin[empty_mask], axis=(1, 2)) > 0) * 100.0
    fpr_hyst_1px = np.mean(np.sum(all_hyst_bin[empty_mask], axis=(1, 2)) > 0) * 100.0

    delta_optimism = (best_d_leaked - np.mean(dice_nest[has_t])) * 100.0

    print(f"\n1. Leakage Analysis: Honest Nested CV vs Leaked Pooled Search (Direct Reviewer 1 & Editor Answer):")
    print(f"   • Baseline Fixed (tau = 0.50):           Dice_pos = {np.mean(dice_base[has_t])*100:.2f}% "
          f"[{ci_dice_base_pos[0]*100:.2f}% - {ci_dice_base_pos[1]*100:.2f}%]")
    print(f"   • Honest Nested Tuning (tau_k* in val):   Dice_pos = {np.mean(dice_nest[has_t])*100:.2f}% "
          f"[{ci_dice_nest_pos[0]*100:.2f}% - {ci_dice_nest_pos[1]*100:.2f}%]")
    print(f"   • Leaked Pooled Search (tau_leak* on test):Dice_pos = {best_d_leaked*100:.2f}% (tau = {best_tau_leaked:.2f})")
    print(f"   • Artificial Optimism (Delta Optimism):    +{delta_optimism:.3f}% (Statistically unearned inflation)")

    print(f"\n2. Innovation 1: 3D Z-Continuity Hysteresis vs 2D Baselines:")
    print(f"   • 2D Baseline (tau = 0.50):     Dice_pos = {np.mean(dice_base[has_t])*100:.2f}% | "
          f"Dice_all = {np.mean(dice_base)*100:.2f}% | FPR_empty = {fpr_base_1px:.2f}%")
    print(f"   • 2D Nested (tau_k*):           Dice_pos = {np.mean(dice_nest[has_t])*100:.2f}% | "
          f"Dice_all = {np.mean(dice_nest)*100:.2f}% | FPR_empty = {fpr_nest_1px:.2f}%")
    print(f"   • 3D Z-Hysteresis (Novel):      Dice_pos = {np.mean(dice_hyst[has_t])*100:.2f}% "
          f"[{ci_dice_hyst_pos[0]*100:.2f}% - {ci_dice_hyst_pos[1]*100:.2f}%] | "
          f"Dice_all = {np.mean(dice_hyst)*100:.2f}% [{ci_dice_hyst_all[0]*100:.2f}% - {ci_dice_hyst_all[1]*100:.2f}%] | "
          f"FPR_empty = {fpr_hyst_1px:.2f}%")
    print(f"   • Relative Empty Slice False Positive Reduction: "
          f"{(fpr_base_1px - fpr_hyst_1px)/fpr_base_1px*100:.1f}% reduction in false alarm slices!")

    # Boundary Slice Analysis
    df_boundary = pd.DataFrame(boundary_records)
    mean_b_base = df_boundary["dice_2d_base"].mean() * 100.0
    mean_b_hyst = df_boundary["dice_3d_hyst"].mean() * 100.0
    n_rescued = df_boundary["rescued"].sum()
    n_boundary = len(df_boundary)

    print(f"\n3. Boundary Slice Rescue Analysis (Tumor Margins along Z-Axis):")
    print(f"   • Total Evaluated Margin Slices: {n_boundary}")
    print(f"   • Mean Margin Dice (2D Baseline): {mean_b_base:.2f}%")
    print(f"   • Mean Margin Dice (3D Hysteresis): {mean_b_hyst:.2f}% (Delta = {mean_b_hyst - mean_b_base:+.2f}%)")
    print(f"   • Slices Rescued from Zero Detection: {n_rescued} / {n_boundary} ({n_rescued/n_boundary*100:.1f}%)")

    # Global Calibration Summary
    print(f"\n4. Innovation 2: Global Calibration Impact Across All 3,929 Test Slices:")
    mean_ece_before = np.mean([f["test_ece_before"] for f in fold_results]) * 100.0
    mean_ece_after = np.mean([f["test_ece_after"] for f in fold_results]) * 100.0
    mean_brier_before = np.mean([f["test_brier_before"] for f in fold_results])
    mean_brier_after = np.mean([f["test_brier_after"] for f in fold_results])
    mean_nll_before = np.mean([f["test_nll_before"] for f in fold_results])
    mean_nll_after = np.mean([f["test_nll_after"] for f in fold_results])

    print(f"   • Expected Calibration Error (ECE): {mean_ece_before:.3f}% -> {mean_ece_after:.3f}% "
          f"({(mean_ece_before - mean_ece_after)/mean_ece_before*100:.1f}% reduction)")
    print(f"   • Brier Score:                      {mean_brier_before:.5f} -> {mean_brier_after:.5f} "
          f"({(mean_brier_before - mean_brier_after)/mean_brier_before*100:.1f}% improvement)")
    print(f"   • Negative Log-Likelihood (NLL):   {mean_nll_before:.4f} -> {mean_nll_after:.4f} "
          f"({(mean_nll_before - mean_nll_after)/mean_nll_before*100:.1f}% reduction)")

    # Save comprehensive results
    summary_output = {
        "fold_results": fold_results,
        "delta_optimism": delta_optimism,
        "best_tau_leaked": best_tau_leaked,
        "metrics_summary": {
            "2d_baseline": {
                "dice_pos": float(np.mean(dice_base[has_t])),
                "dice_pos_ci": ci_dice_base_pos,
                "dice_all": float(np.mean(dice_base)),
                "dice_all_ci": ci_dice_base_all,
                "fpr_empty": float(fpr_base_1px)
            },
            "2d_nested": {
                "dice_pos": float(np.mean(dice_nest[has_t])),
                "dice_pos_ci": ci_dice_nest_pos,
                "dice_all": float(np.mean(dice_nest)),
                "dice_all_ci": ci_dice_nest_all,
                "fpr_empty": float(fpr_nest_1px)
            },
            "3d_hysteresis": {
                "dice_pos": float(np.mean(dice_hyst[has_t])),
                "dice_pos_ci": ci_dice_hyst_pos,
                "dice_all": float(np.mean(dice_hyst)),
                "dice_all_ci": ci_dice_hyst_all,
                "iou_pos": float(np.mean(iou_hyst[has_t])),
                "iou_pos_ci": ci_iou_hyst_pos,
                "prec_pos": float(np.mean(prec_hyst[has_t])),
                "prec_pos_ci": ci_prec_hyst_pos,
                "rec_pos": float(np.mean(rec_hyst[has_t])),
                "rec_pos_ci": ci_rec_hyst_pos,
                "fpr_empty": float(fpr_hyst_1px)
            }
        },
        "boundary_analysis": {
            "total_boundary_slices": int(n_boundary),
            "mean_dice_2d": float(mean_b_base),
            "mean_dice_3d": float(mean_b_hyst),
            "rescued_count": int(n_rescued),
            "rescued_pct": float(n_rescued / n_boundary * 100.0) if n_boundary > 0 else 0.0
        },
        "calibration_summary": {
            "ece_before": float(mean_ece_before),
            "ece_after": float(mean_ece_after),
            "brier_before": float(mean_brier_before),
            "brier_after": float(mean_brier_after),
            "nll_before": float(mean_nll_before),
            "nll_after": float(mean_nll_after)
        }
    }

    out_json_path = os.path.join(RESULTS_DIR, "3d_hysteresis_and_calibration_summary.json")
    with open(out_json_path, "w") as f:
        json.dump(summary_output, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else int(x) if hasattr(x, '__int__') else str(x))
    print(f"\n[OK] Saved results summary to: {out_json_path}")

    # Also save predictions NPZ for downstream radiomics (Step 4)
    out_npz_path = os.path.join(RESULTS_DIR, "3d_hysteresis_test_predictions.npz")
    np.savez_compressed(
        out_npz_path,
        preds_3d_hyst=all_hyst_bin.astype(np.uint8),
        preds_2d_base=all_base_bin.astype(np.uint8),
        preds_2d_nest=all_nest_bin.astype(np.uint8),
        targets=all_tgt.astype(np.uint8),
        patient_ids=all_pids,
        slice_indices=np.array(pooled_test_sindices, dtype=np.int32),
        folds=np.array(pooled_test_folds, dtype=np.int32)
    )
    print(f"[OK] Saved 3D hysteresis prediction bundle to: {out_npz_path}")

    print("\n" + "=" * 90)
    print("  STEP 2 COMPLETED SUCCESSFULLY!")
    print("=" * 90)


if __name__ == "__main__":
    main()
