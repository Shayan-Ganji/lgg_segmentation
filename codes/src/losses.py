import tensorflow as tf
from src.config import CFG

def focal_tversky_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1.0 - y_true) * y_pred)
    FN = tf.reduce_sum(y_true * (1.0 - y_pred))
    
    tversky_index = (TP + smooth) / (TP + CFG["FT_ALPHA"] * FN + CFG["FT_BETA"] * FP + smooth)
    return tf.pow(1.0 - tversky_index, 1.0 / CFG["FT_GAMMA"])

def binary_focal_loss(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), smooth, 1.0 - smooth)
    
    bce = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1.0 - y_pred)
    return tf.reduce_mean(tf.pow(1.0 - pt, CFG["BF_GAMMA"]) * bce)

def compound_production_loss(y_true, y_pred):
    return CFG["LOSS_FT_W"] * focal_tversky_loss(y_true, y_pred) + \
           CFG["LOSS_BF_W"] * binary_focal_loss(y_true, y_pred)