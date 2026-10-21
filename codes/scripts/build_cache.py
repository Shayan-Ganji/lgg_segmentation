"""
=============================================================================
Precompute Quality-Aware Preprocessed Cache & Screening Audit
Generates:
  1. kaggle_3m/preprocessed_cache_128.npz (Images & Masks cached for GPU training)
  2. codes/results/bspc_production/cohort_screening_audit.csv
  3. kaggle_3m/cohort_screening_audit.csv
=============================================================================
"""

import os
import sys
import time
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import CFG
from src.dataset import build_dataframe
from src.preprocessing import get_or_create_preprocessed_cache

def main():
    print("=" * 70)
    print("  PRECOMPUTING QUALITY-AWARE WAVELET CACHE & SCREENING AUDIT")
    print("=" * 70)
    
    df = build_dataframe(data_root=CFG["DATA_ROOT"], data_csv_path=CFG["DATA_CSV"])
    
    cache_path = CFG["CACHE_PATH"]
    audit_csv_prod = os.path.join(CFG["OUTPUT_DIR"], "cohort_screening_audit.csv")
    audit_csv_root = os.path.join(CFG["DATA_ROOT"], "cohort_screening_audit.csv")
    
    if os.path.exists(cache_path):
        print(f"Removing old cache at: {cache_path}")
        os.remove(cache_path)
        
    t0 = time.time()
    print(f"Starting Wavelet Preprocessing & Screening on {len(df)} slices...")
    imgs, msks, df = get_or_create_preprocessed_cache(
        df=df,
        cache_path=cache_path,
        audit_csv_path=audit_csv_prod
    )
    elapsed = time.time() - t0
    
    shutil.copy2(audit_csv_prod, audit_csv_root)
    
    n_q_pass = int(df["passes_q_gate"].sum())
    n_q_bypass = len(df) - n_q_pass
    n_art = int(df["is_artifact_flagged"].sum())
    n_art_clean = len(df) - n_art
    
    print("\n" + "=" * 70)
    print("  PREPROCESSING & SCREENING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  Total processing time:        {elapsed:.1f}s ({elapsed/60.0:.2f} minutes)")
    print(f"  Preprocessed cache saved to:  {cache_path}")
    print(f"  Cache disk file size:         {os.path.getsize(cache_path) / (1024*1024):.2f} MB")
    print(f"  In-memory uncompressed size:  {imgs.nbytes / (1024*1024):.2f} MB")
    print(f"  Screening audit saved to:     {audit_csv_prod}")
    print(f"  Root copy saved to:           {audit_csv_root}")
    print(f"  Total slices:                 {len(df)}")
    print(f"  Passed Q gate (< 4.0, Denoised):  {n_q_pass} / {len(df)} ({n_q_pass/len(df)*100:.2f}%)")
    print(f"  Bypassed Q gate (>= 4.0, Kept):   {n_q_bypass} / {len(df)} ({n_q_bypass/len(df)*100:.2f}%)")
    print(f"  Artifact-flagged slices:      {n_art} / {len(df)} ({n_art/len(df)*100:.2f}%)")
    print(f"  Clean training slices:        {n_art_clean} / {len(df)} ({n_art_clean/len(df)*100:.2f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
