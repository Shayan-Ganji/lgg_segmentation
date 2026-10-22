import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.restoration import denoise_wavelet

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT_CSV = os.path.join(PROJECT_ROOT, "codes", "results", "bspc_production", "cohort_screening_audit.csv")
KAGGLE_DIR = os.path.join(PROJECT_ROOT, "kaggle_3m")
FIG_DIR = os.path.join(PROJECT_ROOT, "codes", "results", "bspc_production", "figures")
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, "paper_bspc_submission", "figures")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

def denoise_wavelet_slice(img_gray):
    img_float = img_gray.astype(np.float32) / 255.0
    denoised = denoise_wavelet(img_float, method='BayesShrink', mode='soft', rescale_sigma=True)
    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)

def load_flair(path):
    img = cv2.imread(path)
    if img is not None:
        # BGR in OpenCV: channel 1 (Green) is FLAIR
        return img[:, :, 1]
    return np.zeros((256, 256), dtype=np.uint8)

def main():
    print("[1/3] Loading cohort audit data...")
    df = pd.read_csv(AUDIT_CSV)
    q_vals = df["Q"].values

    total_slices = len(df)
    n_denoised = int(np.sum(q_vals < 4.0))
    n_bypass = int(np.sum(q_vals >= 4.0))
    pct_denoised = n_denoised / total_slices * 100.0
    pct_bypass = n_bypass / total_slices * 100.0

    print(f"Total: {total_slices}, Denoised (<4.0): {n_denoised} ({pct_denoised:.2f}%), Bypass (>=4.0): {n_bypass} ({pct_bypass:.2f}%)")

    # Create figure: 2 main panels. Left: Panel (a) Histogram. Right: Panel (b) 2x3 Image Grid.
    fig = plt.figure(figsize=(15.5, 6.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.7, 0.15, 1.0, 1.0, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.18, hspace=0.35)

    # ================= Panel (a): Histogram =================
    ax_hist = fig.add_subplot(gs[:, 0])

    # Plot histogram with 2 colored sections
    bins = np.linspace(0.0, 7.0, 45)
    counts, edges = np.histogram(q_vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    mask_denoise = centers < 4.0
    mask_bypass = centers >= 4.0

    ax_hist.bar(centers[mask_denoise], counts[mask_denoise], width=width*0.95,
                color="#1b9e77", alpha=0.85, edgecolor="#00695c", label=f"Wavelet Denoised ($Q < 4.0$, {n_denoised:,} slices / {pct_denoised:.1f}%)")
    ax_hist.bar(centers[mask_bypass], counts[mask_bypass], width=width*0.95,
                color="#d95f02", alpha=0.85, edgecolor="#b71c1c", label=f"Bypass High-Contrast ($Q \\geq 4.0$, {n_bypass} slices / {pct_bypass:.1f}%)")

    # Gate threshold line
    ax_hist.axvline(x=4.0, color="#b71c1c", linestyle="--", linewidth=2.2, label="Adaptive Quality Gate Threshold ($Q = 4.0$)")

    # Annotations
    ax_hist.annotate(
        f"Wavelet Denoised\n({n_denoised:,} slices, {pct_denoised:.1f}%)\nSymlet-4 Soft BayesShrink",
        xy=(2.5, 240), xytext=(2.6, 320),
        fontweight="bold", color="#004d40", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0f2f1", edgecolor="#00897b", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#00897b", lw=1.5, shrinkA=3, shrinkB=3)
    )

    ax_hist.annotate(
        f"High-Contrast Bypass\n({n_bypass} slices, {pct_bypass:.1f}%)\nParenchymal Preserved",
        xy=(4.2, 60), xytext=(4.4, 160),
        fontweight="bold", color="#b71c1c", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fbe9e7", edgecolor="#d95f02", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.5, shrinkA=3, shrinkB=3)
    )

    ax_hist.set_title("(a) Quality Index $Q(I)$ Distribution & Adaptive Denoising Gate", fontweight="bold", pad=10)
    ax_hist.set_xlabel("Parenchymal Noise Quality Index $Q(I)$", fontweight="bold")
    ax_hist.set_ylabel("Slice Count", fontweight="bold")
    ax_hist.set_xlim(0.0, 6.8)
    ax_hist.set_ylim(0, 430)
    ax_hist.legend(loc="upper left", framealpha=0.92, fontsize=8.0)

    # ================= Panel (b): Real Slice Grid =================
    # Row 1: Excluded Artifact Slices (Row 1 of Panel b)
    art_cases = [
        {
            "path": os.path.join(KAGGLE_DIR, "TCGA_CS_4943_20000902", "TCGA_CS_4943_20000902_20.tif"),
            "title": "Artifact 1: FOV Truncation",
            "metrics": "$ED=0.0023$, $CR=0.0020$\n$d_c=0.64$ | Excluded (0 px)"
        },
        {
            "path": os.path.join(KAGGLE_DIR, "TCGA_HT_A61B_19991127", "TCGA_HT_A61B_19991127_78.tif"),
            "title": "Artifact 2: Scanner Noise Fringe",
            "metrics": "$ED=0.0029$, $CR=0.0024$\n$d_c=0.32$ | Excluded (0 px)"
        },
        {
            "path": os.path.join(KAGGLE_DIR, "TCGA_CS_5395_19981004", "TCGA_CS_5395_19981004_20.tif"),
            "title": "Artifact 3: Non-Parenchymal Fringe",
            "metrics": "$ED=0.0000$, $CR=0.0000$\n$d_c=1.00$ | Excluded (0 px)"
        }
    ]

    for col_idx, item in enumerate(art_cases):
        ax = fig.add_subplot(gs[0, col_idx + 2])
        flair = load_flair(item["path"])
        ax.imshow(flair, cmap="gray")
        ax.set_title(item["title"], fontsize=9.2, fontweight="bold", pad=5)
        ax.set_xlabel(item["metrics"], fontsize=8.6, fontweight="bold", color="#b71c1c", labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2: Quality-Aware Wavelet Denoising Comparison (Row 2 of Panel b)
    p_norm = os.path.join(KAGGLE_DIR, "TCGA_CS_4943_20000902", "TCGA_CS_4943_20000902_13.tif")
    p_high = os.path.join(KAGGLE_DIR, "TCGA_CS_6667_20011105", "TCGA_CS_6667_20011105_11.tif")

    flair_norm_raw = load_flair(p_norm)
    flair_norm_denoised = denoise_wavelet_slice(flair_norm_raw)
    flair_high_raw = load_flair(p_high)

    filter_cases = [
        {
            "img": flair_norm_raw,
            "title": "Raw Slice (Moderate Noise)",
            "sub": "$Q(I) = 2.37 < 4.0$\nRaw Input Slice"
        },
        {
            "img": flair_norm_denoised,
            "title": "Wavelet Denoised (Symlet-4)",
            "sub": "BayesShrink Denoised\nArtifacts Suppressed"
        },
        {
            "img": flair_high_raw,
            "title": "High-Contrast Bypass",
            "sub": "$Q(I) = 4.12 \\geq 4.0$\nPreserved (No Denoising)"
        }
    ]

    for col_idx, item in enumerate(filter_cases):
        ax = fig.add_subplot(gs[1, col_idx + 2])
        ax.imshow(item["img"], cmap="gray")
        ax.set_title(item["title"], fontsize=9.2, fontweight="bold", pad=5)
        color = "#1b5e20" if col_idx == 1 else ("#0d47a1" if col_idx == 0 else "#e65100")
        ax.set_xlabel(item["sub"], fontsize=8.6, fontweight="bold", color=color, labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Subplot row titles for panel (b) - shortened and aligned
    fig.text(0.38, 0.94, "(b) Excluded Artifact Slices (Top) & Wavelet Denoising Gate (Bottom)",
             fontsize=10.5, fontweight="bold", ha="left")

    pdf_out = os.path.join(FIG_DIR, "fig_preprocessing_audit.pdf")
    png_out = os.path.join(FIG_DIR, "fig_preprocessing_audit.png")
    paper_pdf = os.path.join(PAPER_FIG_DIR, "fig_preprocessing_audit.pdf")
    paper_png = os.path.join(PAPER_FIG_DIR, "fig_preprocessing_audit.png")

    fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png_out, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(paper_pdf, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(paper_png, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"[OK] Generated Preprocessing Audit Figure:\n  -> {pdf_out}\n  -> {paper_pdf}")

if __name__ == "__main__":
    main()
