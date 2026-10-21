import numpy as np
import tensorflow as tf

def soft_dice_coef(y_true, y_pred, smooth=1e-6):
    """Continuous soft dice for live loss graph tracking."""
    if isinstance(y_pred, (list, tuple)):
        y_pred = y_pred[0]
    yt = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    yp = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    return (2.0 * tf.reduce_sum(yt * yp) + smooth) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth)

def hard_dice_coef(gt: np.ndarray, pred: np.ndarray, smooth=1e-6) -> float:
    """Discrete validation hard dice score."""
    intersection = np.sum(gt * pred)
    return float((2.0 * intersection + smooth) / (gt.sum() + pred.sum() + smooth))

def hard_iou_coef(gt: np.ndarray, pred: np.ndarray, smooth=1e-6) -> float:
    """Discrete Jaccard Index (Intersection over Union)."""
    intersection = np.sum(gt * pred)
    union = gt.sum() + pred.sum() - intersection
    return float((intersection + smooth) / (union + smooth))