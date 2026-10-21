import os
import random
import numpy as np
import tensorflow as tf

CFG = {
    # Data & Paths
    "DATA_ROOT": r"d:\lgg_segmentation\kaggle_3m",
    "DATA_CSV": r"d:\lgg_segmentation\kaggle_3m\data.csv",
    "OUTPUT_DIR": r"d:\lgg_segmentation\codes\results\bspc_production",
    "CACHE_PATH": r"d:\lgg_segmentation\kaggle_3m\preprocessed_cache_128.npz",
    
    # Model architecture
    "IMG_SIZE": 128,
    "INPUT_CHANNELS": 3,
    "BASE_FILTERS": 32,
    "GROUP_NORM_G": 8,
    "DROPOUT": 0.25,
    "DS_WEIGHT": 0.30,
    
    # Training dynamics
    "BATCH_SIZE": 16,
    "EPOCHS": 100,
    "WARMUP_EPOCHS": 8,
    "LR_INITIAL": 3e-4,
    "LR_MIN": 1e-6,
    "WEIGHT_DECAY": 1e-4,
    "GRAD_CLIP": 1.0,
    "ES_PATIENCE": 20,
    "SEED": 42,
    "N_FOLDS": 5,
    "INNER_VAL_RATIO": 0.15,
    
    # Loss formulation (Focal Tversky + Binary Focal Loss)
    "FT_ALPHA": 0.80,    # Weight on False Negatives
    "FT_BETA": 0.20,     # Weight on False Positives
    "FT_GAMMA": 0.60,    # Focal exponent (1/gamma = 1.67)
    "BF_GAMMA": 2.0,     # Binary focal loss gamma
    "LOSS_FT_W": 0.70,   # Weight of FT loss
    "LOSS_BF_W": 0.30,   # Weight of BF loss
    "LABEL_SMOOTH": 0.02,
    
    # Quality-aware screening parameters
    "TAU_Q": 4.0,
    "ED_THRESH": 0.003,
    "CR_THRESH": 0.01,
    "DC_THRESH": 0.30,
    
    # Data Augmentation parameters
    "ROT_RANGE": 15.0,
    "ZOOM_MIN": 0.90,
    "ZOOM_MAX": 1.10,
    "ELASTIC_ALPHA": 15.0,
    "ELASTIC_SIGMA": 3.0,
    "BRIGHTNESS_DELTA": 0.10,
    "CONTRAST_RANGE": (0.85, 1.15),
    "NOISE_STD": 0.02,
    "AUG_PROB_TUMOR": 0.90,
    "AUG_PROB_EMPTY": 0.20,
    
    # Post-processing
    "MIN_COMPONENT_PX": 10
}

def seed_everything(seed=CFG["SEED"]):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)