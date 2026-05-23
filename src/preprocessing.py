import cv2
import numpy as np
from skimage.restoration import denoise_wavelet

def compute_noise_metric(image: np.ndarray) -> float:
    """Computes a hybrid moment-based noise estimation metric."""
    img = image.astype(np.float32)
    
    median_filtered = cv2.medianBlur(image, 3)
    residual_median = img - median_filtered.astype(np.float32)
    noise_median = np.std(residual_median)

    gaussian_filtered = cv2.GaussianBlur(image, (3, 3), 0)
    residual_gaussian = img - gaussian_filtered.astype(np.float32)
    noise_gaussian = np.std(residual_gaussian)
    
    return float((noise_median + noise_gaussian) / 2.0)

def denoise_sure_wavelet(img: np.ndarray) -> np.ndarray:
    """Applies SURE-based wavelet thresholding (BayesShrink soft scaling)."""
    img_float = img.astype(np.float32) / 255.0
    denoised = denoise_wavelet(img_float, channel_axis=None,
                               method='BayesShrink', mode='soft', rescale_sigma=True)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)

def is_snowy_artifact(img: np.ndarray, 
                      canny_low=50, canny_high=150, kernel_size=5,
                      ed_thresh=0.003, ct_thresh=0.01, centroid_dist_thresh=0.3) -> bool:
    """Autonomously isolates muddled and snowy artifacts using morphological metrics."""
    h, w = img.shape
    total_area = h * w

    edges = cv2.Canny(img, canny_low, canny_high)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    edge_density = np.count_nonzero(dilated_edges) / total_area
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return True
        
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_area = cv2.contourArea(largest_contour)
    structure_ratio = largest_contour_area / total_area

    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    center_x, center_y = w // 2, h // 2
    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    max_distance = np.sqrt((center_x)**2 + (center_y)**2)
    norm_distance = distance / max_distance

    if (edge_density < ed_thresh) and (structure_ratio < ct_thresh) and (norm_distance > centroid_dist_thresh):
        return True
    return False