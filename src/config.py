import os
import random
import numpy as np
import tensorflow as tf

CFG = {
    "DATA_ROOT": "./kaggle_3m",
    "ZIP_PATH": "kaggle_3m.zip",
    "OUTPUT_DIR": "./outputs_final",
    "IMG_SIZE": 128,
    "INPUT_CHANNELS": 3,
    "BATCH_SIZE": 16,
    "EPOCHS": 60,
    "WARMUP_EPOCHS": 5,
    "LR_INITIAL": 2e-4,
    "LR_MIN": 1e-6,
    "WEIGHT_DECAY": 1e-4,
    "GRAD_CLIP": 1.0,
    "SEED": 42,
    "N_FOLDS": 5,
    "BASE_FILTERS": 32,
    "GROUP_NORM_G": 8,
    "DROPOUT": 0.25,
    "DS_WEIGHT": 0.30,
    "FT_ALPHA": 0.70,
    "FT_BETA": 0.30,
    "FT_GAMMA": 0.75,
    "BF_GAMMA": 2.0,
    "LOSS_FT_W": 0.65,
    "LOSS_BF_W": 0.35,
    "LABEL_SMOOTH": 0.05,
    "MIN_COMPONENT_PX": 20,
    "ES_PATIENCE": 15
}

def seed_everything():
    random.seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])
    tf.random.set_seed(CFG["SEED"])