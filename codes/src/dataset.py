import os
import re
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from sklearn.model_selection import GroupKFold, KFold
except ImportError:
    # Pure NumPy fallback for GroupKFold and KFold (100% offline without scikit-learn)
    class GroupKFold:
        def __init__(self, n_splits=5):
            self.n_splits = n_splits

        def split(self, X, y=None, groups=None):
            groups_arr = np.array(groups if groups is not None else X)
            unique_groups = np.unique(groups_arr)
            folds = np.array_split(unique_groups, self.n_splits)
            for i in range(self.n_splits):
                val_groups = set(folds[i])
                train_idx = np.where(~np.isin(groups_arr, list(val_groups)))[0]
                val_idx = np.where(np.isin(groups_arr, list(val_groups)))[0]
                yield train_idx, val_idx

    class KFold:
        def __init__(self, n_splits=5, shuffle=False, random_state=None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, X, y=None, groups=None):
            n = len(X)
            indices = np.arange(n)
            if self.shuffle:
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(indices)
            folds = np.array_split(indices, self.n_splits)
            for i in range(self.n_splits):
                val_idx = folds[i]
                train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
                yield train_idx, val_idx

from src.config import CFG
from src.augmentation import augment_pair

def parse_slice_idx(filepath: str) -> int:
    """Extract numeric slice index from filenames like TCGA_CS_4941_19960909_14_mask.tif."""
    base = os.path.splitext(os.path.basename(filepath))[0]
    base = base.replace("_mask", "")
    match = re.search(r"_(\d+)$", base)
    if match:
        return int(match.group(1))
    return 0

def read_image_rgb(path: str, target_size: int = CFG["IMG_SIZE"]) -> np.ndarray:
    """Reads RGB image with cv2 or PIL fallback."""
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if Image is not None and os.path.exists(path):
        try:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB").resize((target_size, target_size), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
                return np.array(pil_img, dtype=np.float32) / 255.0
        except Exception:
            pass
    return np.zeros((target_size, target_size, 3), dtype=np.float32)

def read_mask_gray(path: str, target_size: int = CFG["IMG_SIZE"]) -> np.ndarray:
    """Reads grayscale binary mask with cv2 or PIL fallback."""
    if cv2 is not None:
        msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if msk is not None:
            return (cv2.resize(msk, (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
    if Image is not None and os.path.exists(path):
        try:
            with Image.open(path) as pil_msk:
                pil_msk = pil_msk.convert("L").resize((target_size, target_size), Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
                return (np.array(pil_msk) > 127).astype(np.float32)
        except Exception:
            pass
    return np.zeros((target_size, target_size), dtype=np.float32)

def build_dataframe(data_root=CFG["DATA_ROOT"], data_csv_path=CFG["DATA_CSV"]) -> pd.DataFrame:
    """
    Constructs a structured DataFrame of all slices, with numeric Z-axis sorting
    and 100% integration with data.csv clinical/genomic metadata.
    """
    # Check if precomputed audit CSV is available (enables lightweight offline execution)
    audit_csv_candidates = [
        os.path.join(data_root, "cohort_screening_audit.csv"),
        os.path.join(os.path.dirname(data_root), "kaggle_3m", "cohort_screening_audit.csv"),
        os.path.join(os.path.dirname(data_root), "codes", "results", "bspc_production", "cohort_screening_audit.csv"),
        "cohort_screening_audit.csv",
        r"d:\lgg_segmentation\kaggle_3m\cohort_screening_audit.csv"
    ]
    found_audit = None
    for cand in audit_csv_candidates:
        if os.path.exists(cand):
            found_audit = cand
            break
            
    if found_audit:
        print(f"Loading metadata from precomputed audit: '{found_audit}'")
        df = pd.read_csv(found_audit)
        df["patient_prefix"] = df["patient_id"].astype(str).str[:12]
        if os.path.exists(data_csv_path):
            data_df = pd.read_csv(data_csv_path)
            df = df.merge(data_df, left_on="patient_prefix", right_on="Patient", how="left")
            if "Patient" in df.columns:
                df.drop(columns=["Patient"], inplace=True)
        print(f"Loaded {len(df)} slices from {df['patient_id'].nunique()} patients (from audit CSV).")
        return df

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: '{data_root}' and no audit CSV discovered.")

    mask_files = glob.glob(os.path.join(data_root, "*", "*_mask.tif"))
    if not mask_files:
        raise ValueError(f"No *_mask.tif files or cohort_screening_audit.csv found under {data_root}")

    records = []
    for msk_path in mask_files:
        img_path = msk_path.replace("_mask.tif", ".tif")
        if not os.path.exists(img_path):
            continue
            
        patient_folder = os.path.basename(os.path.dirname(msk_path))
        patient_prefix = patient_folder[:12]
        slice_idx = parse_slice_idx(msk_path)
        
        records.append({
            "image_path": img_path,
            "mask_path": msk_path,
            "patient_id": patient_folder,
            "patient_prefix": patient_prefix,
            "slice_idx": slice_idx
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["patient_id", "slice_idx"]).reset_index(drop=True)

    # Attach tumor status
    tumor_flags = []
    tumor_pixels_list = []
    for msk_p in df["mask_path"]:
        msk = read_mask_gray(msk_p)
        px = int(np.sum(msk > 0))
        tumor_pixels_list.append(px)
        tumor_flags.append(px > 0)
        
    df["has_tumor"] = tumor_flags
    df["tumor_pixels"] = tumor_pixels_list

    if os.path.exists(data_csv_path):
        data_df = pd.read_csv(data_csv_path)
        df = df.merge(data_df, left_on="patient_prefix", right_on="Patient", how="left")
        if "Patient" in df.columns:
            df.drop(columns=["Patient"], inplace=True)

    print(f"Loaded {len(df)} slices from {df['patient_id'].nunique()} patients.")
    print(f"  Positive slices: {df['has_tumor'].sum()} ({df['has_tumor'].mean()*100:.1f}%)")
    print(f"  Empty slices:    {(~df['has_tumor']).sum()} ({(~df['has_tumor']).mean()*100:.1f}%)")
    return df

def get_splits(df: pd.DataFrame, split_type: str = "patient_grouped", 
               n_folds: int = CFG["N_FOLDS"], inner_val_ratio: float = CFG["INNER_VAL_RATIO"],
               seed: int = CFG["SEED"]):
    """
    Produces perfectly matched 5-Fold cross-validation splits.
    
    Returns a list of dicts per fold k:
      - 'train_fit_idx': indices for network optimization (oversampled & augmented)
      - 'inner_val_idx': held-out calibration split (15% of train pool, unmodified)
      - 'outer_test_idx': untouched test split (1 fold, unmodified)
    """
    splits = []
    
    if split_type == "patient_grouped":
        # GroupKFold by patient_id: no patient overlap anywhere
        gkf = GroupKFold(n_splits=n_folds)
        unique_patients = np.array(sorted(df["patient_id"].unique()))
        
        patient_folds = list(gkf.split(unique_patients, groups=unique_patients))
        
        for fold, (train_pool_p_idx, test_p_idx) in enumerate(patient_folds):
            test_patients = set(unique_patients[test_p_idx])
            train_pool_patients = unique_patients[train_pool_p_idx]
            
            rng = np.random.RandomState(seed + fold)
            n_inner = int(np.round(len(train_pool_patients) * inner_val_ratio))
            shuffled_p = rng.permutation(train_pool_patients)
            inner_val_patients = set(shuffled_p[:n_inner])
            train_fit_patients = set(shuffled_p[n_inner:])
            
            outer_test_idx = df.index[df["patient_id"].isin(test_patients)].values
            inner_val_idx = df.index[df["patient_id"].isin(inner_val_patients)].values
            train_fit_idx = df.index[df["patient_id"].isin(train_fit_patients)].values
            
            splits.append({
                "fold": fold + 1,
                "train_fit_idx": train_fit_idx,
                "inner_val_idx": inner_val_idx,
                "outer_test_idx": outer_test_idx
            })
            
    elif split_type == "random_slice":
        # Matched 5-Fold KFold across slices (for leakage quantification experiment)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        slice_indices = np.arange(len(df))
        
        for fold, (train_pool_idx, test_idx) in enumerate(kf.split(slice_indices)):
            rng = np.random.RandomState(seed + fold)
            n_inner = int(np.round(len(train_pool_idx) * inner_val_ratio))
            shuffled_idx = rng.permutation(train_pool_idx)
            inner_val_idx = shuffled_idx[:n_inner]
            train_fit_idx = shuffled_idx[n_inner:]
            
            splits.append({
                "fold": fold + 1,
                "train_fit_idx": train_fit_idx,
                "inner_val_idx": inner_val_idx,
                "outer_test_idx": test_idx
            })
    else:
        raise ValueError(f"Unknown split_type: '{split_type}'. Choose 'patient_grouped' or 'random_slice'.")

    return splits

def _py_load_cached(idx_t, ht_t, training: bool, img_cache: np.ndarray, msk_cache: np.ndarray):
    idx = int(idx_t.numpy())
    has_tumor = bool(ht_t.numpy())

    img = img_cache[idx].astype(np.float32) / 255.0
    msk = msk_cache[idx].astype(np.float32)

    if training:
        img, msk = augment_pair(img, msk, has_tumor)
        msk = np.clip(msk * (1.0 - 2 * CFG["LABEL_SMOOTH"]) + CFG["LABEL_SMOOTH"], 0.0, 1.0)

    return img.astype(np.float32), msk[..., np.newaxis].astype(np.float32)

def _py_load_disk(ip_t, mp_t, ht_t, training: bool):
    ip = ip_t.numpy().decode()
    mp = mp_t.numpy().decode()
    has_tumor = bool(ht_t.numpy())

    img = read_image_rgb(ip)
    msk = read_mask_gray(mp)

    if training:
        img, msk = augment_pair(img, msk, has_tumor)
        msk = np.clip(msk * (1.0 - 2 * CFG["LABEL_SMOOTH"]) + CFG["LABEL_SMOOTH"], 0.0, 1.0)

    return img.astype(np.float32), msk[..., np.newaxis].astype(np.float32)

def build_dataset(df_subset: pd.DataFrame, 
                  training: bool, 
                  batch_size: int = CFG["BATCH_SIZE"],
                  images_cache: np.ndarray = None, 
                  masks_cache: np.ndarray = None) -> tf.data.Dataset:
    """Builds an optimized tf.data.Dataset for training or validation with optional in-memory caching."""
    if training:
        # Balanced sampling: double positive slices, retain 40% empty slices
        pos_df = df_subset[df_subset["has_tumor"]]
        neg_df = df_subset[~df_subset["has_tumor"]]
        
        pos_sampled = pos_df.sample(n=len(pos_df)*2, replace=True, random_state=CFG["SEED"])
        neg_sampled = neg_df.sample(frac=0.40, replace=False, random_state=CFG["SEED"])
        balanced_df = pd.concat([pos_sampled, neg_sampled]).sample(frac=1.0, random_state=CFG["SEED"]).reset_index(drop=True)
        target_df = balanced_df
    else:
        target_df = df_subset

    use_cache = (images_cache is not None) and (masks_cache is not None) and ("cache_idx" in target_df.columns)

    if use_cache:
        ds = tf.data.Dataset.from_tensor_slices((
            target_df["cache_idx"].values.astype(np.int32),
            target_df["has_tumor"].values.astype(np.int32)
        ))
        if training:
            ds = ds.shuffle(len(target_df), seed=CFG["SEED"], reshuffle_each_iteration=True)
            
        def _map_fn(idx, ht):
            img, msk = tf.py_function(
                func=lambda i, h: _py_load_cached(i, h, training, images_cache, masks_cache),
                inp=[idx, ht],
                Tout=[tf.float32, tf.float32]
            )
            img.set_shape([CFG["IMG_SIZE"], CFG["IMG_SIZE"], CFG["INPUT_CHANNELS"]])
            msk.set_shape([CFG["IMG_SIZE"], CFG["IMG_SIZE"], 1])
            return img, msk
            
        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((
            target_df["image_path"].values,
            target_df["mask_path"].values,
            target_df["has_tumor"].values.astype(np.int32)
        ))
        if training:
            ds = ds.shuffle(len(target_df), seed=CFG["SEED"], reshuffle_each_iteration=True)
            
        def _map_disk_fn(ip, mp, ht):
            img, msk = tf.py_function(
                func=lambda i, m, h: _py_load_disk(i, m, h, training),
                inp=[ip, mp, ht],
                Tout=[tf.float32, tf.float32]
            )
            img.set_shape([CFG["IMG_SIZE"], CFG["IMG_SIZE"], CFG["INPUT_CHANNELS"]])
            msk.set_shape([CFG["IMG_SIZE"], CFG["IMG_SIZE"], 1])
            return img, msk
            
        ds = ds.map(_map_disk_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=training)
    return ds.prefetch(tf.data.AUTOTUNE)