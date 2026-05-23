import os
import glob
import zipfile
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src.config import CFG
from src.augmentation import augment_pair

def build_dataframe(data_root=CFG["DATA_ROOT"], zip_path=CFG["ZIP_PATH"]) -> pd.DataFrame:
    """Extracts zip and constructs structured records mapping images to masks."""
    if not os.path.exists(data_root):
        if os.path.exists(zip_path):
            print(f"Extracting '{zip_path}' to '{data_root}'...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(data_root)
        else:
            raise FileNotFoundError(f"Neither '{data_root}' nor '{zip_path}' found.")

    mask_files = sorted(glob.glob(os.path.join(data_root, "**", "*_mask.tif"), recursive=True))
    if not mask_files:
        raise ValueError(f"No *_mask.tif found under {data_root}")

    records = []
    for msk_path in mask_files:
        img_path = msk_path.replace("_mask", "")
        patient_id = os.path.basename(os.path.dirname(msk_path))
        if not os.path.exists(img_path):
            continue
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        has_tumor = (msk is not None) and (int(msk.max()) > 0)
        records.append(dict(
            image_path=img_path, mask_path=msk_path,
            patient_id=patient_id, has_tumor=has_tumor
        ))
    return pd.DataFrame(records).reset_index(drop=True)

def _py_load(ip_t, mp_t, ht_t, training: bool):
    ip = ip_t.numpy().decode()
    mp = mp_t.numpy().decode()
    has_tumor = bool(ht_t.numpy())

    img = cv2.imread(ip, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3), np.float32)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CFG["IMG_SIZE"], CFG["IMG_SIZE"]), cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

    msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if msk is None:
        msk = np.zeros((CFG["IMG_SIZE"], CFG["IMG_SIZE"]), np.float32)
    else:
        msk = cv2.resize(msk, (CFG["IMG_SIZE"], CFG["IMG_SIZE"]), cv2.INTER_NEAREST)
        msk = (msk > 127).astype(np.float32)

    if training:
        img, msk = augment_pair(img, msk, has_tumor)
        msk = np.clip(msk * (1.0 - 2 * CFG["LABEL_SMOOTH"]) + CFG["LABEL_SMOOTH"], 0.0, 1.0)

    return img.astype(np.float32), msk[..., np.newaxis].astype(np.float32)

def _tf_load(ip, mp, ht, training: bool):
    img, msk = tf.py_function(
        func=lambda i, m, h: _py_load(i, m, h, training),
        inp=[ip, mp, ht],
        Tout=[tf.float32, tf.float32]
    )
    img.set_shape([CFG["IMG_SIZE"], CFG["IMG_SIZE"], CFG["INPUT_CHANNELS"]])
    msk.set_shape([CFG["IMG_SIZE"], CFG["IMG_SIZE"], 1])
    return img, msk

def build_dataset(df: pd.DataFrame, training: bool, batch_size: int) -> tf.data.Dataset:
    """Builds highly parallelized multi-threaded tf.data engine pipelines."""
    ds = tf.data.Dataset.from_tensor_slices((
        df["image_path"].values,
        df["mask_path"].values,
        df["has_tumor"].values.astype(np.int32)
    ))
    if training:
        ds = ds.shuffle(len(df), seed=CFG["SEED"], reshuffle_each_iteration=True)
    ds = ds.map(lambda i, m, h: _tf_load(i, m, h, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    return ds.prefetch(tf.data.AUTOTUNE)