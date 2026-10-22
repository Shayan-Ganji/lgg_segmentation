"""
Generates qualitative segmentation comparison across representative test slices.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.segmentation import find_boundaries

# Set style for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

BASE_DIR = r"d:\lgg_segmentation"
CACHE_PATH = os.path.join(BASE_DIR, "kaggle_3m", "preprocessed_cache_128.npz")
PRED_PATH = os.path.join(BASE_DIR, "codes", "results", "bspc_production", "3d_hysteresis_test_predictions.npz")
FIG_DIR = os.path.join(BASE_DIR, "codes", "results", "bspc_production", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
print("[1/3] Loading cache and predictions...")
cache = np.load(CACHE_PATH)
c_images = cache["images"]
c_pids = cache["patient_ids"]
c_sindices = cache["slice_indices"]
cache_map = {(c_pids[i], c_sindices[i]): i for i in range(len(c_pids))}

preds = np.load(PRED_PATH)
t_pids = preds["patient_ids"]
t_sindices = preds["slice_indices"]
targets = preds["targets"]
p_base = preds["preds_2d_base"]
p_nest = preds["preds_2d_nest"]
p_hyst = preds["preds_3d_hyst"]

# Define 3 selected scenarios
scenarios = [
    {
        "name": "Scenario A: Axial Boundary Slice Rescue",
        "subtitle": "Slice 11 of TCGA_DU_7018 (Axial Margins)",
        "test_idx": 2575,
        "desc_base": "Dice = 63.8%",
        "desc_nest": "Dice = 0.00% (Missed)",
        "desc_hyst": "Dice = 67.2% (Rescued)",
        "note": "2D Nested misses apical tumor; 3D Z-continuity rescues via slice 12 prior."
    },
    {
        "name": "Scenario B: Empty-Slice False Positive Suppression",
        "subtitle": "Slice 24 of TCGA_DU_6405 (Tumor-Free Slice)",
        "test_idx": 169,
        "desc_base": "FP = 48 px (Alarm)",
        "desc_nest": "FP = 21 px (Alarm)",
        "desc_hyst": "TN: 0 px (Clean, Dice=100%)",
        "note": "2D models trigger false alarms on skull artifact; 3D spatial filter suppresses noise."
    },
    {
        "name": "Scenario C: Complex Infiltrating Core Delineation",
        "subtitle": "Slice 9 of TCGA_CS_5393 (Volumetric Core)",
        "test_idx": 8,
        "desc_base": "Dice = 93.6%",
        "desc_nest": "Dice = 95.8%",
        "desc_hyst": "Dice = 96.3% (Optimal)",
        "note": "High geometric fidelity with complex lobulated boundaries."
    }
]

def overlay_mask(img_rgb, mask, color_rgb, alpha=0.45):
    """Overlays semi-transparent mask with crisp solid boundary."""
    out = img_rgb.copy().astype(np.float32) / 255.0
    if np.sum(mask) == 0:
        return np.clip(out, 0.0, 1.0)

    # Fill
    for c in range(3):
        out[:, :, c] = np.where(mask > 0,
                                out[:, :, c] * (1.0 - alpha) + color_rgb[c] * alpha,
                                out[:, :, c])

    # Contour boundary
    bounds = find_boundaries(mask.astype(bool), mode="inner")
    for c in range(3):
        out[:, :, c] = np.where(bounds, color_rgb[c], out[:, :, c])

    return np.clip(out, 0.0, 1.0)

# Colors (RGB normalized [0, 1])
COLOR_GT = [0.15, 0.85, 0.25]    # Green
COLOR_BASE = [0.95, 0.25, 0.20]  # Red
COLOR_NEST = [0.98, 0.65, 0.15]  # Orange
COLOR_HYST = [0.10, 0.70, 0.95]  # Cyan

print("[2/3] Generating 3x5 qualitative panel figure...")
fig, axes = plt.subplots(3, 5, figsize=(15.0, 9.6))

col_titles = [
    "Preprocessed MRI\n(Normalized Composite)",
    "Radiologist GT\n(Ground Truth Mask)",
    "2D Baseline\n(Fixed Tau = 0.50)",
    "2D Nested\n(Inner Val Tau_k*)",
    "Proposed Anisotropic 3D\n(Z-Continuity Hysteresis)"
]

for row_idx, sc in enumerate(scenarios):
    t_idx = sc["test_idx"]
    pid = t_pids[t_idx]
    sidx = t_sindices[t_idx]
    c_idx = cache_map[(pid, sidx)]

    img_raw = c_images[c_idx]  # shape (128, 128, 3)
    # Brightness adjustment for visualization
    img_display = img_raw.copy().astype(np.float32)
    max_val = np.max(img_display)
    if max_val > 0:
        img_display = (img_display / max_val * 255.0).astype(np.uint8)

    y_true = targets[t_idx]
    y_base = p_base[t_idx]
    y_nest = p_nest[t_idx]
    y_hyst = p_hyst[t_idx]

    # Subplots
    # Concise row titles
    row_labels = [
        "Row A: Axial Rescue\n(TCGA_DU_7018, Slice 11)",
        "Row B: Empty-Slice TN\n(TCGA_DU_6405, Slice 24)",
        "Row C: Core Contour\n(TCGA_CS_5393, Slice 9)"
    ]

    # Col 0: MRI
    ax0 = axes[row_idx, 0]
    ax0.imshow(img_display)
    if row_idx == 0:
        ax0.set_title(col_titles[0], fontweight="bold", pad=8)
    ax0.set_ylabel(row_labels[row_idx], fontsize=10.5, fontweight="bold", labelpad=10)
    ax0.set_xlabel("Normalized Input", fontsize=11.2, fontweight="bold", color="#333333", labelpad=6)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Col 1: GT
    ax1 = axes[row_idx, 1]
    overlay_gt = overlay_mask(img_display, y_true, COLOR_GT)
    ax1.imshow(overlay_gt)
    if row_idx == 0:
        ax1.set_title(col_titles[1], fontweight="bold", pad=8)
    gt_px = int(np.sum(y_true))
    txt_gt = f"GT: {gt_px} px" if gt_px > 0 else "GT: 0 px (Empty)"
    ax1.set_xlabel(txt_gt, fontsize=11.5, fontweight="bold", color="#1b5e20", labelpad=6)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Col 2: Baseline
    ax2 = axes[row_idx, 2]
    overlay_b = overlay_mask(img_display, y_base, COLOR_BASE)
    ax2.imshow(overlay_b)
    if row_idx == 0:
        ax2.set_title(col_titles[2], fontweight="bold", pad=8)
    ax2.set_xlabel(sc["desc_base"], fontsize=11.5, fontweight="bold", color="#b71c1c", labelpad=6)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Col 3: Nested
    ax3 = axes[row_idx, 3]
    overlay_n = overlay_mask(img_display, y_nest, COLOR_NEST)
    ax3.imshow(overlay_n)
    if row_idx == 0:
        ax3.set_title(col_titles[3], fontweight="bold", pad=8)
    ax3.set_xlabel(sc["desc_nest"], fontsize=11.5, fontweight="bold", color="#e65100", labelpad=6)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Col 4: Proposed 3D Hysteresis
    ax4 = axes[row_idx, 4]
    overlay_h = overlay_mask(img_display, y_hyst, COLOR_HYST)
    ax4.imshow(overlay_h)
    if row_idx == 0:
        ax4.set_title(col_titles[4], fontweight="bold", pad=8)
    ax4.set_xlabel(sc["desc_hyst"], fontsize=11.5, fontweight="bold", color="#006064", labelpad=6)
    ax4.set_xticks([])
    ax4.set_yticks([])


# Add comprehensive bottom legend
legend_elements = [
    Patch(facecolor=COLOR_GT, edgecolor="black", label="Radiologist Ground Truth (GT)"),
    Patch(facecolor=COLOR_BASE, edgecolor="black", label="2D Baseline (Fixed Tau = 0.50)"),
    Patch(facecolor=COLOR_NEST, edgecolor="black", label="2D Nested Thresholding (Inner Val Tau_k*)"),
    Patch(facecolor=COLOR_HYST, edgecolor="black", label="Proposed Anisotropic 3D Z-Hysteresis")
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.subplots_adjust(wspace=0.04, hspace=0.12)

pdf_out = os.path.join(FIG_DIR, "fig_qualitative_3d_hysteresis.pdf")
png_out = os.path.join(FIG_DIR, "fig_qualitative_3d_hysteresis.png")
fig.savefig(pdf_out, bbox_inches="tight")
fig.savefig(png_out, bbox_inches="tight")

sub_dir = os.path.join(BASE_DIR, "paper_bspc_submission", "figures")
fig.savefig(os.path.join(sub_dir, "fig_qualitative_3d_hysteresis.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(sub_dir, "fig_qualitative_3d_hysteresis.png"), bbox_inches="tight")
plt.close(fig)

print(f"[3/3] Successfully generated qualitative figures:\n  -> {pdf_out}\n  -> {png_out}")
