# Disentangling Evaluation Distortions in 2D-to-3D Brain Tumor MRI Segmentation: A Patient-Grouped Leakage, Calibration, and Volumetric Post-Processing Audit on TCGA-LGG

[![Journal: Springer Nature JIIM](https://img.shields.io/badge/Journal-Springer%20Nature%20JIIM%20(Q1)-blue.svg)](https://link.springer.com/journal/10278)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official Implementation** of the paper:  
> **"Disentangling Evaluation Distortions in 2D-to-3D Brain Tumor MRI Segmentation: A Patient-Grouped Leakage, Calibration, and Volumetric Post-Processing Audit on TCGA-LGG"**  
> *Shayan Ganji Nataj, Keivan Borna\**  
> *Department of Computer Science, Faculty of Mathematics and Computer Science, Kharazmi University, Tehran, Iran*  
> *Journal: Journal of Imaging Informatics in Medicine (JIIM), Springer Nature / SIIM*

---

## 📌 Executive Summary

While deep learning architectures (notably 2D U-Net and Attention Res-UNet variants) report high Dice coefficients on lower-grade glioma (LGG) benchmarks, clinical translation and methodological reproducibility have been compromised by three widespread evaluation distortions:

1. **Slice-Level Data Leakage:** Conventional random-slice splitting places contiguous axial slices from the same patient across training and testing partitions, allowing networks to exploit subject-specific appearance cues. Under identical architectural and hyperparameter controls on the TCGA-LGG cohort ($110$ patients, $3,929$ axial slices), random-slice partitioning artificially inflates test performance by **$+8.79\text{ pp}$** on tumor-bearing slices ($82.74\%$ vs. $73.96\%$) and **$+6.20\text{ pp}$** on all slices ($88.22\%$ vs. $82.02\%$), decorrelating method rankings (Kendall $\tau = -0.067$) and reversing top ranks below $\pi^* = 86.53\%$.
2. **Probability Miscalibration:** Overlap losses (e.g., Focal Tversky with $\alpha=0.8, \beta=0.2$) produce uncalibrated, overconfident voxel probabilities with a characteristic peripheral false-positive halo.
3. **Empty-Slice Metric Dilution & Axial Boundary Discontinuity:** Pure 2D models lack cross-slice volumetric context, yielding elevated false alarms on healthy slices ($13.65\%$) and missing infiltrative apical/basal tumor margins. A $2^3$ factorial Shapley attribution reveals that empty-slice metric dilution ($+8.28\text{ pp}$, $45.7\%$) and inter-slice data leakage ($+7.41\text{ pp}$, $40.8\%$) account for the vast majority of apparent cross-study performance gaps.

This repository provides the complete, mathematically audited, zero-leakage framework resolving these vulnerabilities through **Quality-Aware Preprocessing**, **Nested Temperature / Platt Calibration**, **Distribution-Free Split-Conformal Prediction**, and **Conditional Axial 3D Hysteresis (CAL-3DH)** evaluated under a strict **Patient-Grouped 5-Fold Cross-Validation** protocol.

---

## 🔬 Key Methodological Innovations

```
Raw Multi-Sequence MRI (FLAIR, T1-pre, T1-post) [TCGA-LGG: 110 Patients, 3,929 Slices]
  │
  ▼
[ Stage 1: Quality-Aware Preprocessing ]
  ├── Morphological Artifact Screen (Canny ED < 0.003, CR < 0.01, dc > 0.30) ──> Excludes 200 non-anatomical slices (0.00% impact on positive Dice)
  └── Conditional BayesShrink Wavelet Denoising (Q(I) < 4.0; Symlet-4)
  │
  ▼
[ Stage 2: Attention Res-UNet Backbone ]
  ├── Residual Blocks + Group Normalization (G=8) + GELU Non-Linearity
  ├── Stochastic Depth (DropPath p=0.00 -> 0.20)
  ├── Additive Attention Gates (Oktay et al.)
  └── Multi-Scale Deep Supervision (Auxiliary Decoders 2 & 3)
  │
  ▼
[ Stage 3: Calibration, Conformal Prediction & CAL-3DH Post-Processing ]
  ├── Nested Temperature / Platt Scaling (T* = 1.844 fitted strictly on inner validation folds)
  │     └── -46.9% reduction in active-bin ECE (0.369% -> 0.196%), -32.9% reduction in NLL
  ├── Distribution-Free Split-Conformal Prediction (>= 90% empirical coverage guarantee)
  └── Conditional Axial 3D Hysteresis (CAL-3DH):
        1. In-plane binarization (tau_xy* = 0.74) ──> Shaves peripheral false-positive halo
        2. Axial boundary rescue (tau_z_low* = 0.45, Sprior) ──> Rescues missed margin slices
        3. 3D component filtering (V_min* = 38 voxels, tau_seed* = 0.70) ──> -42.4% false alarms on empty slices
```

---

## 📊 State-of-the-Art Benchmark Comparison (Table 8)

Systematic comparison against published deep learning models evaluated on the standardized TCGA-LGG dataset, stratified by data partitioning protocol and evaluation metric scope:

| Study | Year | Architecture | Post-Proc. / Calibration | Params | Overall $\text{Dice}_{\text{all}}$ (%) | Tumor-Slice $\text{Dice}_{\text{pos}}$ (%) | 3D Volumetric Dice / ICC |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Part A: Image/Slice-Level Partitioning (Patient separation not enforced / Leaky)** | | | | | | | |
| Naser & Deen | 2020 | Modified U-Net | None / None | NR | 84.00 (Soft) | NR | NR |
| LiteMRINet | 2025 | LiteMRINet | None / None | 1.61M | 85.50 (Soft) | NR | NR |
| PoSAM-ULTRA | 2026 | PoSAM-ULTRA | None / PBFO Tuning | NR | 91.40 (Soft) | NR | NR |
| This Work (Fixed $\tau=0.50$) | 2026 | Attn Res U-Net | None / None | 8.20M | 88.22 [Soft: 88.18] | 82.74 | Invalid (Non-contiguous) |
| This Work (Tuned $\tau^*$) | 2026 | Attn Res U-Net | None / Inner $\tau^*$ | 8.20M | 89.83 | 83.75 | Invalid (Non-contiguous) |
| **Part B: Patient-Grouped Partitioning (Zero Subject Overlap Standard)** | | | | | | | |
| Buda et al. (Baseline Benchmark) | 2019 | Standard 2D U-Net | None / Fixed $\tau=0.50$ | 7.80M | NR | NR | 82.00% (Med: 85.0%) / NR |
| This Work (2D Baseline) | 2026 | Attn Res U-Net | None / Fixed $\tau=0.50$ | 8.20M | 82.02 | 73.96 | 76.00% / 0.8652 |
| This Work (2D Nested) | 2026 | Attn Res U-Net | Temp ($T^*$) / Nested $\tau_k^*$ | 8.20M | 84.45 | 74.60 | 77.57% / 0.8860 |
| **CAL-3DH (Proposed Framework)** | **2026** | **Attn Res U-Net** | **Temp ($T^*$) / CAL-3DH** | **8.20M** | **86.12** | **74.92** | **77.73% / 0.8894** |

> **Key Clinical Finding:** Under strict zero-leakage patient-grouped cross-validation with hard binarization, CAL-3DH achieves an overall Dice of **86.12%** and tumor-slice Dice of **74.92%**, with a median 3D volumetric Dice of **84.42%** and exceptional clinical volumetric agreement of $\text{ICC}(2, 1) = 0.8894$ [95% CI: $0.8407$--$0.9238$ via 10,000 patient-clustered bootstrap resamples].

---

## 📈 Quantitative Audit Highlights

- **Controlled Leakage Audit & Attribution:**
  - Tumor-Slice Dice Inflation: $+8.79\text{ pp}$ ($82.74\%$ vs. $73.96\%$)
  - Overall Dice Inflation: $+6.20\text{ pp}$ ($88.22\%$ vs. $82.02\%$)
  - Ranking Decorrelation: Kendall $\tau = -0.067$, with top ranks reversed below $\pi^* = 86.53\%$
  - Linear Mixed-Effects Analysis: Subject random intercept explains negligible residual leakage variance ($p = 0.509$) with high split-half reliability ($r_{sb} = 0.844$)
  - Cooperative Game-Theoretic Shapley Attribution ($+18.14\text{ pp}$ total apparent gap):
    - Empty-Slice Metric Dilution: $\mathbf{+8.28\text{ pp}}$ ($45.7\%$)
    - Inter-Slice Data Leakage: $\mathbf{+7.41\text{ pp}}$ ($40.8\%$)
    - Post-Processing & Thresholding: $\mathbf{+2.45\text{ pp}}$ ($13.5\%$)
- **Nested Calibration & Conformal Prediction:**
  - Active-Bin Expected Calibration Error (ECE): $0.369\% \to 0.196\%$ (**$-46.9\%$** relative reduction; Affine Platt: $0.130\%$)
  - Brier Score: $0.00375 \to 0.00338$ (**$-9.9\%$**)
  - Distribution-Free Split-Conformal Coverage: $90.08\% \pm 1.86\%$ at nominal $90\%$ confidence level (finite-sample corrected: $90.21\% \pm 1.88\%$; empirical 95% interval: $[86.43\%, 93.79\%]$)
- **Conditional Axial 3D Hysteresis (CAL-3DH):**
  - Empty-Slice False Positive Rate: $13.65\% \to 7.86\%$ (**$-42.4\%$** relative reduction)
  - Boundary Slice Recovery: Rescues $54.2\%$ of infiltrative apical/basal marginal slices
  - Clinical Volumetric Agreement: $\text{ICC}(2, 1) = 0.8894$, Pearson $r = 0.8886$, Bland-Altman mean bias $= +143.7$ voxels
- **Multi-Institutional Generalization Audit:**
  - Evaluated across 5 independent tissue source sites (`TCGA-DU`, `TCGA-HT`, `TCGA-CS`, `TCGA-FG`, `TCGA-EZ`), demonstrating invariant calibration fidelity across multi-scanner protocols
- **Downstream Radiogenomic Association:**
  - Preserves biological tumor burden rank ordering across molecular platforms (*COCCluster*):  
    $\text{COC 1 } (7{,}098.0\text{ voxels}) > \text{COC 3 } (5{,}509.6\text{ voxels}) > \text{COC 2 } (4{,}834.5\text{ voxels})$, Jonckheere-Terpstra / Kruskal-Wallis $p = 0.0268$.

---

## 📁 Repository Structure

```
lgg_segmentation/
├── JIIM_Submission_Ready/               # Formal Submission Package for Springer Nature JIIM
│   ├── manuscript_blinded.tex / .pdf   # Blinded Manuscript (47 pages, audited, zero line clipping)
│   ├── manuscript_full.tex / .pdf      # Full Author Manuscript (48 pages, complete affiliations)
│   ├── title_page.tex / .pdf           # Formal Title Page & Declarations (strictly 1 page)
│   ├── cover_letter_jiim.tex / .pdf    # Submission Cover Letter to Editor-in-Chief (strictly 1 page)
│   └── references.bib                  # Curated LaTeX Bibliography
├── codes/
│   ├── src/
│   │   ├── config.py                   # Central hyperparameter configuration
│   │   ├── dataset.py                  # Zero-leakage patient-grouped data loaders
│   │   ├── preprocessing.py            # Morphological screening & BayesShrink wavelet denoising
│   │   ├── augmentation.py             # Elastic, affine, and intensity augmentation (pure SciPy/NumPy)
│   │   ├── losses.py                   # Focal Tversky Loss + Binary Focal Loss + Deep Supervision
│   │   ├── metrics.py                  # Dice, IoU, HD95, ASD, ICC, Betti-0, ECE, Brier Score
│   │   ├── postprocessing.py           # CAL-3DH conditional axial hysteresis & 3D components
│   │   ├── train.py                    # High-throughput GPU training loop with warmup & cosine decay
│   │   └── models/
│   │       └── attention_resunet.py    # Attn Res-UNet (GN, GELU, DropPath, Deep Supervision)
│   ├── scripts/
│   │   ├── run_gpu_experiments.py      # Master runner for 5-fold training (Patient-Grouped & Random-Slice)
│   │   ├── run_leakage_audit_and_figures.py # Audits leakage, calibration, generates publication figures
│   │   ├── run_3d_radiomics_and_clinical.py # Patient-level 3D metrics, Bland-Altman, genomic analysis
│   │   └── generate_architecture_pipeline_figure.py # Generates Stage 1-3 pipeline diagram
│   └── results/
│       └── bspc_production/            # Experimental logs, quantitative CSV tables & figures
├── LICENSE                             # MIT Open-Source License
└── README.md                           # Documentation and quick-start guide
```

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/Shayan-Ganji/lgg_segmentation.git
cd lgg_segmentation

# Create and activate Python 3.10 environment
conda create -n lgg_env python=3.10 -y
conda activate lgg_env

# Install dependencies (standard scientific Python stack, zero proprietary C++ dependencies)
pip install numpy scipy pandas matplotlib scikit-learn pillow tensorflow==2.18.0
```

### 2. Dataset Preparation

Download the standardized TCGA-LGG MRI collection (Buda et al.) from [The Cancer Imaging Archive (TCIA)](https://doi.org/10.7937/K9/TCIA.2016.L4LTD3TK) or mirrored on [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation), and extract it into `kaggle_3m/`.

### 3. Training & Evaluation

To execute the honest, zero-leakage 5-fold patient-grouped cross-validation:

```bash
python codes/scripts/run_gpu_experiments.py --protocol patient_grouped --epochs 100 --batch_size 16
```

To execute the matched 5-fold random-slice leakage audit:

```bash
python codes/scripts/run_gpu_experiments.py --protocol random_slice --epochs 100 --batch_size 16
```

### 4. Auditing, Calibration, and Publication Figures

```bash
# Execute Leakage Audit, Nested Calibration, and Reliability Diagrams
python codes/scripts/run_leakage_audit_and_figures.py

# Execute 3D Volumetric Metrics, Bland-Altman Agreement, and Radiogenomic Association
python codes/scripts/run_3d_radiomics_and_clinical.py

# Generate Stage 1-3 Architecture and Pipeline Diagram
python codes/scripts/generate_architecture_pipeline_figure.py
```

---

## 📜 Citation

If you find this work, methodological audit, or codebase useful in your research, please cite:

```bibtex
@article{ganji2026disentangling,
  title={Disentangling Evaluation Distortions in 2D-to-3D Brain Tumor MRI Segmentation: A Patient-Grouped Leakage, Calibration, and Volumetric Post-Processing Audit on TCGA-LGG},
  author={Ganji Nataj, Shayan and Borna, Keivan},
  journal={Journal of Imaging Informatics in Medicine},
  year={2026},
  publisher={Springer Nature}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
