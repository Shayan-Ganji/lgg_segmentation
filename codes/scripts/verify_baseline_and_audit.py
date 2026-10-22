"""
Baseline metrics verification and cohort screening audit.
"""


import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

# Add project root and codes to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(CODES_DIR)
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CFG

RESULTS_DIR = os.path.join(CODES_DIR, "results", "bspc_production")
AUDIT_CSV = os.path.join(RESULTS_DIR, "cohort_screening_audit.csv")


def audit_npz_files():
    """
    Verifies all 20 NPZ files (5 val + 5 test for patient_grouped,
    5 val + 5 test for random_slice).
    Checks shape, dtypes, slice count, min/max logits, and NaN/Inf absence.
    """
    print("\n" + "=" * 90)
    print("  SUB-STEP 1.1: AUDIT & INTEGRITY CHECK OF ALL NPZ PREDICTION FILES")
    print("=" * 90)

    protocols = ["patient_grouped", "random_slice"]
    splits = ["val", "test"]
    audit_records = []

    for proto in protocols:
        proto_dir = os.path.join(RESULTS_DIR, proto)
        print(f"\n---> Auditing protocol: {proto} ({proto_dir})")
        for split in splits:
            for fold in range(1, 6):
                fname = f"oof_{split}_predictions_fold{fold}.npz"
                fpath = os.path.join(proto_dir, fname)
                if not os.path.exists(fpath):
                    print(f"  [ERROR] Missing file: {fpath}")
                    continue

                data = np.load(fpath)
                keys = list(data.keys())
                targets = data["targets"]
                logits = data["logits"]
                pids = data["patient_ids"]
                sindices = data["slice_indices"]
                flagged = data["is_artifact_flagged"]

                has_nan_logit = np.isnan(logits).any()
                has_inf_logit = np.isinf(logits).any()
                has_nan_tgt = np.isnan(targets).any()

                record = {
                    "protocol": proto,
                    "split": split,
                    "fold": fold,
                    "slices": targets.shape[0],
                    "unique_patients": len(np.unique(pids)),
                    "logits_dtype": str(logits.dtype),
                    "targets_dtype": str(targets.dtype),
                    "min_logit": float(np.min(logits)),
                    "max_logit": float(np.max(logits)),
                    "pos_slices": int(np.sum(np.sum(targets, axis=(1, 2)) > 0)),
                    "flagged_slices": int(np.sum(flagged)),
                    "integrity_pass": (not has_nan_logit) and (not has_inf_logit) and (not has_nan_tgt)
                }
                audit_records.append(record)

                status = "[PASS]" if record["integrity_pass"] else "[FAIL]"
                print(f"  {status} Fold {fold} {split:4s}: {record['slices']:4d} slices | "
                      f"{record['unique_patients']:2d} patients | {record['pos_slices']:3d} pos | "
                      f"Logits range: [{record['min_logit']:.2f}, {record['max_logit']:.2f}] | "
                      f"Dtype: {record['logits_dtype']}")

    df_audit = pd.DataFrame(audit_records)
    print("\n--- Summary per protocol & split ---")
    summary = df_audit.groupby(["protocol", "split"]).agg(
        total_slices=("slices", "sum"),
        total_pos=("pos_slices", "sum"),
        total_flagged=("flagged_slices", "sum"),
        all_passed=("integrity_pass", "all")
    )
    print(summary)
    return df_audit


def load_split_predictions(protocol_dir):
    """Loads and aggregates all 5 folds of test predictions for a protocol."""
    all_logits = []
    all_probs = []
    all_targets = []
    all_pids = []
    all_sindices = []
    all_flagged = []
    fold_ids = []

    for fold in range(1, 6):
        p = os.path.join(protocol_dir, f"oof_test_predictions_fold{fold}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")
        data = np.load(p)
        n = data["targets"].shape[0]

        logits = data["logits"].astype(np.float32)
        if "probs" in data:
            probs = data["probs"].astype(np.float32)
        else:
            probs = expit(logits)

        targets = data["targets"].astype(np.uint8)
        pids = data["patient_ids"].astype(str)
        sindices = data["slice_indices"].astype(np.int32)
        flagged = data["is_artifact_flagged"].astype(bool)

        all_logits.append(logits)
        all_probs.append(probs)
        all_targets.append(targets)
        all_pids.append(pids)
        all_sindices.append(sindices)
        all_flagged.append(flagged)
        fold_ids.append(np.full(n, fold, dtype=np.int32))

    df_preds = pd.DataFrame({
        "patient_id": np.concatenate(all_pids),
        "slice_idx": np.concatenate(all_sindices),
        "is_artifact_flagged": np.concatenate(all_flagged),
        "fold": np.concatenate(fold_ids)
    })

    logits_arr = np.concatenate(all_logits, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    return df_preds, logits_arr, probs_arr, targets_arr


def compute_slice_metrics(targets, preds_bin):
    """Vectorized per-slice Dice, IoU, Precision, Recall."""
    # targets: (N, 128, 128), preds_bin: (N, 128, 128)
    tp = np.sum((targets == 1) & (preds_bin == 1), axis=(1, 2)).astype(np.float32)
    fp = np.sum((targets == 0) & (preds_bin == 1), axis=(1, 2)).astype(np.float32)
    fn = np.sum((targets == 1) & (preds_bin == 0), axis=(1, 2)).astype(np.float32)
    tn = np.sum((targets == 0) & (preds_bin == 0), axis=(1, 2)).astype(np.float32)

    total_pos = tp + fp + fn
    empty_both = (total_pos == 0)

    # Dice: 2*TP / (2*TP + FP + FN)
    denom_dice = 2.0 * tp + fp + fn
    dice = np.zeros_like(tp)
    nonzero_d = denom_dice > 0
    dice[nonzero_d] = (2.0 * tp[nonzero_d]) / denom_dice[nonzero_d]
    dice[empty_both] = 1.0  # Perfect agreement on empty slices

    # IoU: TP / (TP + FP + FN)
    denom_iou = tp + fp + fn
    iou = np.zeros_like(tp)
    nonzero_u = denom_iou > 0
    iou[nonzero_u] = tp[nonzero_u] / denom_iou[nonzero_u]
    iou[empty_both] = 1.0

    # Precision: TP / (TP + FP)
    denom_prec = tp + fp
    prec = np.zeros_like(tp)
    nonzero_p = denom_prec > 0
    prec[nonzero_p] = tp[nonzero_p] / denom_prec[nonzero_p]
    prec[empty_both] = 1.0

    # Recall: TP / (TP + FN)
    denom_rec = tp + fn
    rec = np.zeros_like(tp)
    nonzero_r = denom_rec > 0
    rec[nonzero_r] = tp[nonzero_r] / denom_rec[nonzero_r]
    rec[empty_both] = 1.0

    return dice, iou, prec, rec


def fast_patient_clustered_bootstrap_ci(df_preds, metric_arr, has_tumor_mask, n_bootstraps=10000, seed=42):
    """
    Patient-level clustered bootstrap with 10,000 resamples.
    Mathematically exact and accelerated by pre-aggregating patient-level sums.
    """
    rng = np.random.RandomState(seed)
    unique_patients = df_preds["patient_id"].unique()
    n_patients = len(unique_patients)

    # Map each slice to its patient index
    pid_to_int = {pid: i for i, pid in enumerate(unique_patients)}
    slice_p_indices = np.array([pid_to_int[p] for p in df_preds["patient_id"].values])

    # Pre-aggregate sums and counts per patient
    pat_metric_sum_all = np.zeros(n_patients, dtype=np.float64)
    pat_count_all = np.zeros(n_patients, dtype=np.int64)

    pat_metric_sum_pos = np.zeros(n_patients, dtype=np.float64)
    pat_count_pos = np.zeros(n_patients, dtype=np.int64)

    for i in range(len(df_preds)):
        p_idx = slice_p_indices[i]
        m = metric_arr[i]
        pat_metric_sum_all[p_idx] += m
        pat_count_all[p_idx] += 1
        if has_tumor_mask[i]:
            pat_metric_sum_pos[p_idx] += m
            pat_count_pos[p_idx] += 1

    # Draw 10,000 bootstrap resamples of patient indices
    boot_p_indices = rng.choice(n_patients, size=(n_bootstraps, n_patients), replace=True)

    # Compute sampled sums and counts
    sampled_sum_all = np.take(pat_metric_sum_all, boot_p_indices).sum(axis=1)
    sampled_cnt_all = np.take(pat_count_all, boot_p_indices).sum(axis=1)
    boot_means_all = sampled_sum_all / sampled_cnt_all

    sampled_sum_pos = np.take(pat_metric_sum_pos, boot_p_indices).sum(axis=1)
    sampled_cnt_pos = np.take(pat_count_pos, boot_p_indices).sum(axis=1)
    boot_means_pos = sampled_sum_pos / np.maximum(sampled_cnt_pos, 1)

    ci_all = (float(np.percentile(boot_means_all, 2.5)), float(np.percentile(boot_means_all, 97.5)))
    ci_pos = (float(np.percentile(boot_means_pos, 2.5)), float(np.percentile(boot_means_pos, 97.5)))

    return ci_all, ci_pos


def evaluate_cohort(name, df, metric_dict, has_tumor_mask, n_bootstraps=10000):
    """Prints and returns formatted table of metrics with 95% bootstrap CIs."""
    print(f"\n{'='*35} {name} {'='*35}")
    n_total = len(df)
    n_pos = int(np.sum(has_tumor_mask))
    n_empty = n_total - n_pos
    n_patients = df["patient_id"].nunique()

    print(f"Cohort Size: {n_total} slices across {n_patients} patients "
          f"({n_pos} positive [{n_pos/n_total*100:.1f}%], {n_empty} empty [{n_empty/n_total*100:.1f}%])")

    results = {}
    for metric_name, arr in metric_dict.items():
        ci_all, ci_pos = fast_patient_clustered_bootstrap_ci(
            df, arr, has_tumor_mask, n_bootstraps=n_bootstraps
        )
        mean_all = float(np.mean(arr))
        mean_pos = float(np.mean(arr[has_tumor_mask])) if n_pos > 0 else 0.0

        results[metric_name] = {
            "mean_all": mean_all,
            "ci_all": ci_all,
            "mean_pos": mean_pos,
            "ci_pos": ci_pos
        }

        print(f"  • {metric_name:18s} | Pos Slices: {mean_pos*100:6.2f}% "
              f"[{ci_pos[0]*100:5.2f}% - {ci_pos[1]*100:5.2f}%] | "
              f"All Slices: {mean_all*100:6.2f}% [{ci_all[0]*100:5.2f}% - {ci_all[1]*100:5.2f}%]")

    return results


def analyze_reviewer5_audit():
    """
    Examines the 200 artifact slices removed during training
    and provides a definitive breakdown for Reviewer 5.
    """
    print("\n" + "=" * 90)
    print("  SUB-STEP 1.3: REVIEWER 5 AUDIT - ANALYSIS OF THE 200 SCREENED-OUT ARTIFACT SLICES")
    print("=" * 90)

    audit_df = pd.read_csv(AUDIT_CSV)
    flagged = audit_df[audit_df["is_artifact_flagged"]].copy()
    unflagged = audit_df[~audit_df["is_artifact_flagged"]].copy()

    total_slices = len(audit_df)
    n_flagged = len(flagged)
    n_flagged_empty = int(np.sum(~flagged["has_tumor"]))
    n_flagged_pos = int(np.sum(flagged["has_tumor"]))

    print(f"\n1. Global Cohort Filtering Breakdown:")
    print(f"   • Total Slices in Dataset:         {total_slices:,}")
    print(f"   • Screened-out Artifact Slices:   {n_flagged:,} ({n_flagged/total_slices*100:.2f}%)")
    print(f"   • High-Quality Training Slices:   {len(unflagged):,} ({len(unflagged)/total_slices*100:.2f}%)")

    print(f"\n2. Tumor Content of the 200 Screened Slices (Direct Answer to Reviewer 5):")
    print(f"   • Pure Empty / Black Slices:       {n_flagged_empty} / {n_flagged} ({n_flagged_empty/n_flagged*100:.1f}%)")
    print(f"   • Tumor-bearing Slices:             {n_flagged_pos} / {n_flagged} ({n_flagged_pos/n_flagged*100:.1f}%)")

    if n_flagged_pos > 0:
        pos_flagged = flagged[flagged["has_tumor"]]
        print(f"     -> Tumor pixel counts in flagged slices:")
        print(f"        Min: {pos_flagged['tumor_pixels'].min()}, "
              f"Median: {pos_flagged['tumor_pixels'].median()}, "
              f"Max: {pos_flagged['tumor_pixels'].max()}")
        print(f"        Slice indices: {pos_flagged['slice_idx'].values}")
        print(f"        Patient IDs: {pos_flagged['patient_id'].unique()}")

    print(f"\n3. Morphological Violation Breakdown among {n_flagged} Flagged Slices:")
    ed_viol = flagged["ED"] < CFG["ED_THRESH"]
    cr_viol = flagged["CR"] < CFG["CR_THRESH"]
    dc_viol = flagged["dc"] > CFG["DC_THRESH"]

    print(f"   • Edge Density (ED < {CFG['ED_THRESH']}):     {ed_viol.sum():3d} slices "
          f"({ed_viol.sum()/n_flagged*100:.1f}%) [Extreme signal absence/air background]")
    print(f"   • Contour Ratio (CR < {CFG['CR_THRESH']}):   {cr_viol.sum():3d} slices "
          f"({cr_viol.sum()/n_flagged*100:.1f}%) [Inadequate brain tissue silhouette]")
    print(f"   • Centroid Dist (dc > {CFG['DC_THRESH']}): {dc_viol.sum():3d} slices "
          f"({dc_viol.sum()/n_flagged*100:.1f}%) [Extreme off-center scanner FOV truncation]")

    # Overlaps
    all_three = (ed_viol & cr_viol & dc_viol).sum()
    ed_and_cr = (ed_viol & cr_viol).sum()
    print(f"   • Combined (ED & CR) violations:   {ed_and_cr:3d} slices ({ed_and_cr/n_flagged*100:.1f}%)")
    print(f"   • All 3 criteria violated:         {all_three:3d} slices ({all_three/n_flagged*100:.1f}%)")

    print(f"\n4. Quality Gate Wavelet Denoising Analysis (Eq. 2 of Manuscript):")
    q_bypassed = audit_df["Q"] >= CFG["TAU_Q"]
    q_denoised = audit_df["Q"] < CFG["TAU_Q"]
    print(f"   • Wavelet Denoised (Q < {CFG['TAU_Q']}):   {q_denoised.sum():,d} / {total_slices:,d} "
          f"({q_denoised.sum()/total_slices*100:.2f}%)")
    print(f"   • Wavelet Bypassed (Q >= {CFG['TAU_Q']}):  {q_bypassed.sum():,d} / {total_slices:,d} "
          f"({q_bypassed.sum()/total_slices*100:.2f}%) [Preserved native sharp edges]")


def main():
    print("=" * 90)
    print("  BSPC PHASE 3: STEP 1 - BASELINE VERIFICATION & COHORT SCREENING AUDIT")
    print("=" * 90)

    # Sub-step 1.1: Audit all 20 NPZ files
    audit_npz_files()

    # Sub-step 1.2: Baseline verification at tau = 0.50
    print("\n" + "=" * 90)
    print("  SUB-STEP 1.2: BASELINE PERFORMANCE AT FIXED TAU = 0.50 (PATIENT-GROUPED PROTOCOL)")
    print("=" * 90)

    pg_dir = os.path.join(RESULTS_DIR, "patient_grouped")
    df_pg, logits_pg, probs_pg, targets_pg = load_split_predictions(pg_dir)

    tumor_px = np.sum(targets_pg, axis=(1, 2))
    has_tumor = tumor_px > 0
    df_pg["has_tumor"] = has_tumor
    df_pg["tumor_pixels"] = tumor_px

    # Compute binary prediction at tau = 0.50
    preds_50 = (probs_pg >= 0.50).astype(np.uint8)
    dice, iou, prec, rec = compute_slice_metrics(targets_pg, preds_50)

    metric_dict = {
        "Dice Similarity": dice,
        "Jaccard / IoU": iou,
        "Precision (PPV)": prec,
        "Recall (Sens)": rec
    }

    # Empty slice FPR
    empty_slices = ~has_tumor
    pred_empty_px = np.sum(preds_50[empty_slices], axis=(1, 2))
    fpr_1px = np.mean(pred_empty_px > 0) * 100.0
    fpr_10px = np.mean(pred_empty_px >= 10) * 100.0
    print(f"\nEmpty Slices False Positive Rate (FPR_empty):")
    print(f"  • Slices with >= 1 false positive pixel:  {fpr_1px:.2f}% ({np.sum(pred_empty_px > 0)} / {len(pred_empty_px)})")
    print(f"  • Slices with >= 10 false positive pixels: {fpr_10px:.2f}% ({np.sum(pred_empty_px >= 10)} / {len(pred_empty_px)})")

    # A. Full Cohort (3,929 slices)
    full_results = evaluate_cohort(
        "FULL COHORT (3,929 Slices - Real TCGA-LGG Dataset)",
        df_pg, metric_dict, has_tumor, n_bootstraps=10000
    )

    # B. Screened Cohort (3,729 slices)
    clean_mask = ~df_pg["is_artifact_flagged"].values
    df_clean = df_pg[clean_mask].copy().reset_index(drop=True)
    has_tumor_clean = has_tumor[clean_mask]
    clean_metric_dict = {k: v[clean_mask] for k, v in metric_dict.items()}

    clean_results = evaluate_cohort(
        "SCREENED COHORT (3,729 Slices - Excluding 200 Artifact Slices)",
        df_clean, clean_metric_dict, has_tumor_clean, n_bootstraps=10000
    )

    # Sub-step 1.3: Reviewer 5 Audit Analysis
    analyze_reviewer5_audit()

    print("\n" + "=" * 90)
    print("  STEP 1 COMPLETED SUCCESSFULLY!")
    print("=" * 90)


if __name__ == "__main__":
    main()
