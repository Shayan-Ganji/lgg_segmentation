import cv2
import numpy as np
from src.config import CFG

def _rotate(img, msk):
    a = np.random.uniform(-CFG["ROT_RANGE"], CFG["ROT_RANGE"])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return img, msk

def _zoom(img, msk):
    s = np.random.uniform(CFG["ZOOM_MIN"], CFG["ZOOM_MAX"])
    h, w = img.shape[:2]
    nh, nw = int(h * s), int(w * s)
    ir = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mr = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if s > 1.0:
        y0 = (nh - h) // 2
        x0 = (nw - w) // 2
        ir, mr = ir[y0:y0+h, x0:x0+w], mr[y0:y0+h, x0:x0+w]
    else:
        ph, pw = h - nh, w - nw
        t, b = ph // 2, ph - ph // 2
        l, r = pw // 2, pw - pw // 2
        ir = cv2.copyMakeBorder(ir, t, b, l, r, cv2.BORDER_REFLECT_101)
        mr = cv2.copyMakeBorder(mr, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    return ir, mr

def _elastic(img, msk):
    a, s = CFG["ELASTIC_ALPHA"], CFG["ELASTIC_SIGMA"]
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * a, (0, 0), s)
    dy = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * a, (0, 0), s)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    mx = (x + dx).astype(np.float32)
    my = (y + dy).astype(np.float32)
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    msk = cv2.remap(msk, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return img, msk

def _photometric(img):
    img = np.clip(img + np.random.uniform(-CFG["BRIGHTNESS_DELTA"], CFG["BRIGHTNESS_DELTA"]), 0.0, 1.0)
    c = np.random.uniform(*CFG["CONTRAST_RANGE"])
    img = np.clip((img - 0.5) * c + 0.5, 0.0, 1.0)
    if np.random.rand() < 0.40:
        img = np.clip(img + np.random.randn(*img.shape).astype(np.float32) * CFG["NOISE_STD"], 0.0, 1.0)
    return img

def augment_pair(img: np.ndarray, msk: np.ndarray, has_tumor: bool) -> tuple:
    """Applies conditional augmentation sequence mapping real-world scanner variances."""
    p = CFG["AUG_PROB_TUMOR"] if has_tumor else CFG["AUG_PROB_EMPTY"]
    if np.random.rand() > p:
        return img, msk
    if np.random.rand() < 0.70:
        img, msk = _rotate(img, msk)
    if np.random.rand() < 0.50:
        img, msk = _zoom(img, msk)
    if np.random.rand() < 0.35:
        img, msk = _elastic(img, msk)
    img = _photometric(img)
    return img, msk