"""
Publication-Quality Figure 2: Architecture & Pipeline Diagram
Full End-to-End Pipeline:
Stage 1: Quality-Aware Screening & Wavelet Preprocessing
Stage 2: Attention Residual U-Net Backbone (8.20M Params)
Stage 3: Nested Calibration & Anisotropic 3D Hysteresis
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIG_DIR = os.path.join(PROJECT_ROOT, "codes", "results", "bspc_production", "figures")
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, "paper_bspc_submission", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# Plot configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300
})

fig, ax = plt.subplots(figsize=(15.5, 9.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Color palette
c_preproc = "#00838f"
c_enc = "#1565c0"
c_dec = "#00838f"
c_bot = "#283593"
c_attn = "#e65100"
c_ds = "#6a1b9a"
c_calib = "#ad1457"
c_hyst = "#2e7d32"
c_arrow = "#455a64"
c_input = "#37474f"

def draw_box(x, y, w, h, title, subtitle="", bg="#ffffff", border="#000000", lw=1.2,
             title_color="#000000", text_color="#212121", rx=1.2, fontsize=9.0,
             title_offset=2.0, subtitle_offset=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}",
                       facecolor=bg, edgecolor=border, linewidth=lw)
    ax.add_patch(p)
    if title:
        y_title = y + h - (title_offset if subtitle else h/2.0)
        ax.text(x + w/2.0, y_title, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=title_color)
    if subtitle:
        y_sub = (y + (h - 2.5)/2.0 - 0.4) if subtitle_offset is None else (y + subtitle_offset)
        ax.text(x + w/2.0, y_sub, subtitle, ha="center", va="center",
                fontsize=fontsize - 1.0, color=text_color)
    return p

def draw_arrow(x1, y1, x2, y2, color=c_arrow, lw=1.5, style="->", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))

# ================= Stage 1: Quality-Aware Screening & Wavelet Preprocessing =================
p1 = FancyBboxPatch((1.5, 49.5), 24.5, 47.5, boxstyle="round,pad=0.3,rounding_size=2.0",
                    facecolor="#f0f7fa", edgecolor=c_preproc, linewidth=1.5)
ax.add_patch(p1)
ax.text(13.75, 94.5, "Stage 1: Quality-Aware Preprocessing", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=c_preproc)

# Input Box
draw_box(3.5, 80.5, 20.5, 10.5, "Raw Multi-Sequence MRI",
         "110 Patients | 3,929 Slices\nPre-contrast, FLAIR, Post-contrast\nResized: $128 \\times 128 \\times 3$",
         bg="#ffffff", border=c_input, title_color=c_input, fontsize=9.2, title_offset=1.8, subtitle_offset=3.8)

# Screening Box - generous vertical separation between title and subtitle
draw_box(3.5, 63.5, 20.5, 14.5, "Morphological Artifact Screen",
         "Canny Edge Density ($ED < 0.003$)\nContour Ratio ($CR < 0.01$)\nCentroid Dist ($d_c > 0.30$)\nFlags 200 noisy slices (199 tumor-free)",
         bg="#e1f5fe", border=c_preproc, title_color=c_preproc, fontsize=8.8, title_offset=2.2, subtitle_offset=4.3)
draw_arrow(13.75, 80.5, 13.75, 78.0)

# Wavelet Gate Box
draw_box(3.5, 51.5, 20.5, 10.5, "Quality-Aware Wavelet Denoising",
         r"$Q(I) = \frac{1}{2}[\sigma(I-\mathcal{M}_{3\times 3}(I)) + \sigma(I-\mathcal{G}_{3\times 3}(I))]$" + "\n" +
         r"If $Q(I) < 4.0$: BayesShrink 2-level (95.1%)" + "\n" +
         r"If $Q(I) \geq 4.0$: High-contrast bypass (4.9%)",
         bg="#e0f2f1", border="#00897b", title_color="#00695c", fontsize=8.6, title_offset=1.7, subtitle_offset=4.0)
draw_arrow(13.75, 63.5, 13.75, 62.0)

# ================= Stage 2: Attention Res-UNet Backbone =================
# Width 38.5 (from 28.0 to 66.5), leaving clean 5.0 corridor to Stage 3 (at 71.5)
p2 = FancyBboxPatch((28.0, 15), 38.5, 82, boxstyle="round,pad=0.3,rounding_size=2.0",
                    facecolor="#f4f7fc", edgecolor=c_enc, linewidth=1.5)
ax.add_patch(p2)
ax.text(47.25, 94.5, "Stage 2: Attention Residual U-Net Backbone (8.20M Params)", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=c_enc)

# Encoders (Left)
draw_box(29.8, 77.0, 15.0, 9.0, "ResBlock 1 ($128^2$)",
         "32 Filters | DropPath: 0.00\nGroupNorm ($G=8$) + GELU",
         bg="#e3f2fd", border=c_enc, title_color=c_enc, fontsize=9.2)

draw_box(29.8, 62.0, 15.0, 9.0, "ResBlock 2 ($64^2$)",
         "64 Filters | DropPath: 0.05\nGroupNorm ($G=8$) + GELU",
         bg="#e3f2fd", border=c_enc, title_color=c_enc, fontsize=9.2)

draw_box(29.8, 47.0, 15.0, 9.0, "ResBlock 3 ($32^2$)",
         "128 Filters | DropPath: 0.10\nGroupNorm ($G=8$) + GELU",
         bg="#e3f2fd", border=c_enc, title_color=c_enc, fontsize=9.2)

draw_box(29.8, 32.0, 15.0, 9.0, "ResBlock 4 ($16^2$)",
         "256 Filters | DropPath: 0.15\nGroupNorm ($G=8$) + GELU",
         bg="#e3f2fd", border=c_enc, title_color=c_enc, fontsize=9.2)

# Downsampling arrows
draw_arrow(37.3, 77.0, 37.3, 71.0, style="->")
ax.text(37.8, 74.0, "MaxPool 2x2", fontsize=8.0, color=c_arrow)
draw_arrow(37.3, 62.0, 37.3, 56.0, style="->")
ax.text(37.8, 59.0, "MaxPool 2x2", fontsize=8.0, color=c_arrow)
draw_arrow(37.3, 47.0, 37.3, 41.0, style="->")
ax.text(37.8, 44.0, "MaxPool 2x2", fontsize=8.0, color=c_arrow)

# Bottleneck (Bottom Center)
draw_box(38.8, 17.5, 19.0, 9.0, "Bottleneck ($8^2$)",
         "512 Filters | DropPath: 0.20\nSpatial Dropout ($p=0.25$)",
         bg="#e8eaf6", border=c_bot, title_color=c_bot, fontsize=9.2)
draw_arrow(37.3, 32.0, 42.0, 26.5, style="->")
ax.text(38.0, 29.0, "MaxPool 2x2", fontsize=8.0, color=c_arrow)

# Decoders (Right) - Width 13.0 (from 52.2 to 65.2)
draw_box(52.2, 32.0, 13.0, 9.0, "Decoder 4 ($16^2$)",
         "256 Filters | ConvTrans 2x2\nResBlock + GN ($G=8$)",
         bg="#e0f7fa", border=c_dec, title_color=c_dec, fontsize=9.0)
draw_arrow(54.0, 26.5, 57.5, 32.0, style="->")
ax.text(56.8, 29.0, "UpConv 2x2", fontsize=8.0, color=c_arrow)

draw_box(52.2, 47.0, 13.0, 9.0, "Decoder 3 ($32^2$)",
         "128 Filters | ConvTrans 2x2\nResBlock + GN ($G=8$)",
         bg="#e0f7fa", border=c_dec, title_color=c_dec, fontsize=9.0)
draw_arrow(58.7, 41.0, 58.7, 47.0, style="->")
ax.text(59.2, 44.0, "UpConv 2x2", fontsize=8.0, color=c_arrow)

draw_box(52.2, 62.0, 13.0, 9.0, "Decoder 2 ($64^2$)",
         "64 Filters | ConvTrans 2x2\nResBlock + GN ($G=8$)",
         bg="#e0f7fa", border=c_dec, title_color=c_dec, fontsize=9.0)
draw_arrow(58.7, 56.0, 58.7, 62.0, style="->")
ax.text(59.2, 59.0, "UpConv 2x2", fontsize=8.0, color=c_arrow)

draw_box(52.2, 77.0, 13.0, 9.0, "Decoder 1 ($128^2$)",
         "32 Filters | ConvTrans 2x2\nResBlock + GN ($G=8$)",
         bg="#e0f7fa", border=c_dec, title_color=c_dec, fontsize=9.0)
draw_arrow(58.7, 71.0, 58.7, 77.0, style="->")
ax.text(59.2, 74.0, "UpConv 2x2", fontsize=8.0, color=c_arrow)

# Attention Gates (Center horizontal bridges)
for y_gate, label in [(36.5, "AG4"), (51.5, "AG3"), (66.5, "AG2"), (81.5, "AG1")]:
    draw_box(45.5, y_gate - 2.0, 5.0, 4.0, label, "", bg="#fff3e0", border=c_attn, title_color=c_attn, fontsize=8.2)
    draw_arrow(44.8, y_gate, 45.5, y_gate, color=c_attn, lw=1.2)
    draw_arrow(50.5, y_gate, 52.2, y_gate, color=c_attn, lw=1.2)

# Deep Supervision branches routed cleanly through open corridor (x=67.8) between Stage 2 and Stage 3
draw_box(50.2, 2.0, 15.0, 10.0, "Deep Supervision Heads",
         "$\\mathcal{L}_{total} = 1.0\\mathcal{L}_{main} + 0.3\\mathcal{L}_{d2} + 0.3\\mathcal{L}_{d3}$\n$\\mathcal{L} = 0.70\\mathcal{L}_{FTL}(\\alpha=0.8,\\beta=0.2) + 0.30\\mathcal{L}_{BF}$",
         bg="#f3e5f5", border=c_ds, title_color=c_ds, fontsize=8.0)
draw_arrow(65.2, 66.5, 67.8, 66.5, color=c_ds, lw=1.2)
draw_arrow(65.2, 51.5, 67.8, 51.5, color=c_ds, lw=1.2)
draw_arrow(67.8, 66.5, 67.8, 7.0, color=c_ds, lw=1.2)
draw_arrow(67.8, 7.0, 65.2, 7.0, color=c_ds, lw=1.2)

# Pipeline connection from Preproc to Backbone
draw_arrow(24.0, 56.5, 26.5, 56.5)
draw_arrow(26.5, 56.5, 26.5, 81.5)
draw_arrow(26.5, 81.5, 29.8, 81.5)

# ================= Stage 3: Nested Calibration & 3D Anisotropic Hysteresis =================
# Starts at 72.2 to 98.5, leaving clean open corridor (66.8 to 71.9)
p3 = FancyBboxPatch((72.2, 15), 26.3, 82, boxstyle="round,pad=0.3,rounding_size=2.0",
                    facecolor="#f9fbf9", edgecolor=c_hyst, linewidth=1.5)
ax.add_patch(p3)
ax.text(85.35, 94.5, "Stage 3: Calibration & Anisotropic 3D Hysteresis", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=c_hyst)

# Connection from Decoder to Calibration with pill badge centered in open corridor (x=69.5)
# Stage 2 boundary is at 66.8, Stage 3 boundary is at 71.9 -> Badge has >2.5 padding on both sides!
draw_arrow(65.2, 81.5, 73.0, 81.5, lw=1.8)
ax.text(69.5, 84.8, "Logits $z(x)$", fontsize=7.6, fontweight="bold", color=c_arrow, ha="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="#90a4ae", alpha=0.98, lw=0.85))

# Nested Temp Scaling Box
draw_box(73.0, 74.0, 23.5, 13.0, "Nested Temperature Scaling ($T^*$)",
         "Fitted on inner validation logits ($T^* = 1.844$)\n$\\hat{P}(x) = \\sigma(z(x) / T^*)$\nReduces ECE by 46.9% ($0.369\\% \\to 0.196\\%$)\nZero impact on ROC / test labels",
         bg="#fce4ec", border=c_calib, title_color=c_calib, fontsize=8.8)
draw_arrow(84.5, 74.0, 84.5, 69.0)

# 1. In-Plane Binarization
draw_box(73.0, 56.5, 23.5, 12.5, "1. In-Plane Binarization (XY-Plane)",
         "$M^{(0)}(x, y, z) = \\mathbb{I}(\\hat{P} \\geq \\tau_{xy, k}^*)$\nFold-tuned $\\tau_{xy}^* = 0.74$ (Inner Val)\nShaves overconfident False-Positive halo\nBoosts Precision: $71.95\\% \\to 76.86\\%$",
         bg="#e8f5e9", border=c_hyst, title_color=c_hyst, fontsize=8.8)
draw_arrow(84.75, 56.5, 84.75, 51.5)

# 2. Z-Continuity Rescue
draw_box(73.0, 37.0, 23.5, 14.5, "2. Anisotropic Axial Boundary Rescue",
         "If slice $z$ empty, check adjacent $z \\pm 1$:\n$S_{prior} = \\text{Dilate}_{2D}(M_{z-1}^{(0)} \\cup M_{z+1}^{(0)}, \\mathcal{K}_{3\\times 3})$\nRescue: $M_z^{(1)} = M_z^{(0)} \\cup (S_{prior} \\cap \\mathbb{I}(\\hat{P}_z \\geq \\tau_{z\\_low}^*))$\nDecouples in-plane $\\tau_{xy}^* = 0.74$ from axial $\\tau_{z\\_low}^* = 0.45$",
         bg="#e8f5e9", border=c_hyst, title_color=c_hyst, fontsize=8.8)
draw_arrow(84.75, 37.0, 84.75, 32.0)

# 3. 3D Volumetric Filter
draw_box(73.0, 17.5, 23.5, 14.5, "3. 3D Component Filtering & Fallback",
         "26-connected 3D component analysis:\nKeep $C_l$ iff $|C_l| \\geq V_{min, k}^*$ AND $\\max \\hat{P} \\geq \\tau_{seed, k}^*$\nFallback: Retain $\\max \\hat{P} \\geq \\tau_{seed}^*$ if volume empty\nReduces empty-slice FPR by 42.4% ($13.65\\% \\to 7.86\\%$)",
         bg="#e8f5e9", border=c_hyst, title_color=c_hyst, fontsize=8.8)

# Output Banner
p_out = FancyBboxPatch((73.0, 3.5), 23.5, 9.0, boxstyle="round,pad=0.2,rounding_size=1.5",
                       facecolor="#1b5e20", edgecolor="#0d5302", linewidth=1.2)
ax.add_patch(p_out)
ax.text(84.75, 8.8, "Volumetric 3D Tumor Mask", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#ffffff")
ax.text(84.75, 5.5, "$\\text{Dice}_{pos}=74.92\\%$, $\\text{Dice}_{all}=86.12\\%$, 3D $\\text{Dice}=77.73\\%$\nClinical $\\text{ICC}=0.8894$, Genomic $p=0.0268$",
        ha="center", va="center", fontsize=8.5, color="#e8f5e9")
draw_arrow(84.75, 17.5, 84.75, 12.5, color="#1b5e20", lw=2.0)

plt.tight_layout()

pdf_out = os.path.join(FIG_DIR, "fig_architecture_pipeline.pdf")
png_out = os.path.join(FIG_DIR, "fig_architecture_pipeline.png")
paper_pdf = os.path.join(PAPER_FIG_DIR, "fig_architecture_pipeline.pdf")
paper_png = os.path.join(PAPER_FIG_DIR, "fig_architecture_pipeline.png")

fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.15)
fig.savefig(png_out, bbox_inches="tight", pad_inches=0.15)
fig.savefig(paper_pdf, bbox_inches="tight", pad_inches=0.15)
fig.savefig(paper_png, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

print(f"[OK] Generated Architecture & Pipeline Diagram:\n  -> {pdf_out}\n  -> {paper_pdf}")
