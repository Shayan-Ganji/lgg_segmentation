import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
from src.config import CFG

H, W = CFG["IMG_SIZE"], CFG["IMG_SIZE"]
_Y_GRID, _X_GRID = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

def _rotate(img: np.ndarray, msk: np.ndarray) -> tuple:
    """High-speed affine rotation using SIMD-accelerated PIL."""
    a = float(np.random.uniform(-CFG["ROT_RANGE"], CFG["ROT_RANGE"]))
    im = Image.fromarray((img * 255.0).astype(np.uint8))
    ms = Image.fromarray((msk * 255.0).astype(np.uint8))
    r_im = np.array(im.rotate(a, resample=Image.BILINEAR)).astype(np.float32) / 255.0
    r_ms = (np.array(ms.rotate(a, resample=Image.NEAREST)) > 127).astype(np.float32)
    return r_im, r_ms

def _zoom(img: np.ndarray, msk: np.ndarray) -> tuple:
    """High-speed bilinear zoom with symmetric pad/crop."""
    s = float(np.random.uniform(CFG["ZOOM_MIN"], CFG["ZOOM_MAX"]))
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    
    im = Image.fromarray((img * 255.0).astype(np.uint8)).resize((nw, nh), resample=Image.BILINEAR)
    ms = Image.fromarray((msk * 255.0).astype(np.uint8)).resize((nw, nh), resample=Image.NEAREST)
    arr_im = np.array(im).astype(np.float32) / 255.0
    arr_ms = (np.array(ms) > 127).astype(np.float32)
    
    if s >= 1.0:
        y0 = (nh - h) // 2
        x0 = (nw - w) // 2
        arr_im = arr_im[y0:y0+h, x0:x0+w]
        arr_ms = arr_ms[y0:y0+h, x0:x0+w]
    else:
        ph, pw = h - nh, w - nw
        t, b = ph // 2, ph - ph // 2
        l, r = pw // 2, pw - pw // 2
        arr_im = np.pad(arr_im, ((t, b), (l, r), (0, 0)), mode='reflect')
        arr_ms = np.pad(arr_ms, ((t, b), (l, r)), mode='constant', constant_values=0)
        
    return arr_im[:h, :w], arr_ms[:h, :w]

def _elastic(img: np.ndarray, msk: np.ndarray) -> tuple:
    """Optimized elastic deformation with coarse-grid spline interpolation."""
    a = float(CFG["ELASTIC_ALPHA"])
    h, w = img.shape[:2]
    
    # Coarse grid (32x32) Gaussian noise field upsampled to 128x128
    gh, gw = 32, 32
    dx_coarse = gaussian_filter(np.random.randn(gh, gw).astype(np.float32) * a, sigma=1.0, mode='reflect')
    dy_coarse = gaussian_filter(np.random.randn(gh, gw).astype(np.float32) * a, sigma=1.0, mode='reflect')
    dx = np.array(Image.fromarray(dx_coarse).resize((w, h), resample=Image.BILINEAR))
    dy = np.array(Image.fromarray(dy_coarse).resize((w, h), resample=Image.BILINEAR))
    
    coords = np.array([_Y_GRID + dy, _X_GRID + dx])
    out_img = np.empty_like(img)
    for c in range(img.shape[2]):
        out_img[..., c] = map_coordinates(img[..., c], coords, order=1, mode='reflect')
    out_msk = (map_coordinates(msk, coords, order=0, mode='reflect') > 0.5).astype(np.float32)
    
    return out_img, out_msk

def _photometric(img: np.ndarray) -> np.ndarray:
    """Photometric jitter with brightness, contrast, and gaussian noise."""
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