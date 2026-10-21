# Calibration-Aware Anisotropic 3D Hysteresis and Leakage Auditing for Volumetric Lower-Grade Glioma Segmentation and Radiogenomic Profiling in Brain MRI

[![Journal: Elsevier BSPC](https://img.shields.io/badge/Journal-Elsevier%20BSPC%20(Q1)-blue.svg)](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official Implementation** of the paper:  
> **"Calibration-Aware Anisotropic 3D Hysteresis and Leakage Auditing for Volumetric Lower-Grade Glioma Segmentation and Radiogenomic Profiling in Brain MRI"**  
> *Shayan Ganji Nataj, Keivan Borna*  
> *Department of Computer Science, Faculty of Mathematics and Computer Science, Kharazmi University, Tehran, Iran*

---

## 📌 Executive Summary

While deep learning architectures (notably 2D U-Net variants) report high Dice coefficients on lower-grade glioma (LGG) benchmarks, clinical translation has been compromised by three methodological vulnerabilities:
1. **Slice-Level Data Leakage:** Conventional random slice splitting places contiguous axial slices from the same patient across training and testing partitions, allowing networks to memorize subject-specific anatomy and artificially inflating test scores by **$+8.79\%$** on positive slices ($82.74\%$ vs. $73.96\%$).
2. **Probability Miscalibration:** Overlap losses (e.g., Focal Tversky with $\alpha=0.8, \beta=0.2$) produce uncalibrated overconfident probabilities with a peripheral false-positive halo.
3. **Axial Boundary Discontinuity:** Pure 2D models lack cross-slice volumetric context, yielding high false alarms on healthy slices ($13.65\%$) and missing infiltrative apical/basal margins.

This repository provides the complete, end-to-end framework resolving these challenges through **Quality-Aware Preprocessing**, **Nested Temperature Scaling**, and an **Anisotropic 3D Z-Continuity Hysteresis Algorithm** under a strict **Patient-Grouped 5-Fold Cross-Validation** protocol on the full TCGA-LGG benchmark ($110$ patients, $3,929$ axial slices).

---

## 🔬 Key Methodological Innovations

```
Raw Multi-Sequence MRI (T1-pre, FLAIR, T1-post)
  │
  ▼
[ Stage 1: Quality-Aware Preprocessing ]
  ├── Morphological Artifact Screen (Canny ED < 0.003, CR < 0.01, dc > 0.30) ──> Excludes 200 non-anatomical slices (0.00% impact on Dice_pos)
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
[ Stage 3: Calibration & Volumetric Post-Processing ]
  ├── Nested Temperature Scaling (T* = 1.844 fitted strictly on inner_val) ──> 46.9% reduction in pooled ECE
  └── Anisotropic 3D Z-Continuity Hysteresis:
        1. In-plane binarization (tau_xy* = 0.74) ──> Shaves peripheral FP halo
        2. Axial boundary rescue (tau_z_low* = 0.45, Sprior) ──> Rescues missed margin slices
        3. 3D component filtering (V_min* = 38, tau_seed* = 0.70) ──> -42.4% false alarms on empty slices
```

---

## 📊 State-of-the-Art Benchmark Comparison (Table 8)

Systematic comparison against published models evaluated on the standardized TCGA-LGG dataset (Buda et al.), stratified by data partitioning protocol:

| Study | Year | Architecture | Partitioning Protocol | Params | Overall $\text{Dice}_{\text{all}}$ (%) | Tumor-Slice $\text{Dice}_{\text{pos}}$ (%) | 3D Volumetric Dice / ICC |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Part A: Random-Slice (Leaky Regime)** | | | | | | | |
| LiteMRINet | 2025 | LiteMRINet | Random-Slice (Leaked) | 1.61M | 85.50 (Soft) | NR | NR |
| PoSAM-ULTRA | 2026 | PoSAM-ULTRA | Random-Slice (Leaked) | NR | 91.40 (Soft) | NR | NR |
| **This Work (Matched Audit, Fixed $\tau=0.50$)** | 2026 | Attn Res-UNet | Random-Slice (Leaked) | 8.20M | 88.22 [Soft: 88.18] | 82.74 | Invalid (Non-contiguous) |
| **This Work (Matched Audit, Tuned $\tau^*$)** | 2026 | Attn Res-UNet | Random-Slice (Leaked) | 8.20M | 89.83 | 83.75 | Invalid (Non-contiguous) |
| **Part B: Patient-Grouped (Zero-Leakage Standard)** | | | | | | | |
| Buda et al. (Baseline Benchmark) | 2019 | Standard 2D U-Net | Patient-Grouped | 7.80M | NR | 72.10 | NR |
| This Work (2D Baseline, $\tau=0.50$) | 2026 | Attn Res-UNet | Patient-Grouped | 8.20M | 82.02 | 73.96 | 76.00% / 0.8652 |
| This Work (2D Nested, $\tau_k^*$) | 2026 | Attn Res-UNet | Patient-Grouped | 8.20M | 84.45 | 74.60 | 77.57% / 0.8860 |
| **This Work (Proposed 3D Z-Hysteresis)** | **2026** | **Attn Res-UNet** | **Patient-Grouped** | **8.20M** | **86.12** | **74.92** | **77.73% / 0.8894** |

> **Key Finding:** Even under the strict zero-leakage patient-grouped standard with hard binarization, our proposed framework achieves an overall Dice of **86.12%**, surpassing **LiteMRINet (85.50%)** (which benefited from slice-level data leakage and soft Dice), while delivering a **+2.82%** boost over the leak-free Buda et al. benchmark on tumor-bearing slices and clinical agreement of $\text{ICC}(2, 1) = 0.8894$.

---

## 📈 Quantitative Summary

- **Controlled Leakage Audit:**
  - Tumor-Slice Dice Inflation: $+8.79\%$ ($82.74\%$ vs. $73.96\%$)
  - Overall Dice Inflation: $+6.20\%$ ($88.22\%$ vs. $82.02\%$)
  - Optimism Bias from Test Sweeping: $\Delta\text{Optimism} = +0.063\%$
- **Nested Calibration ($T^* = 1.844$):**
  - Pooled Cohort ECE: $0.369\% \to 0.196\%$ (**$-46.9\%$** relative reduction)
  - Negative Log-Likelihood: $0.0230 \to 0.0154$ (**$-32.9\%$**)
- **Anisotropic 3D Z-Hysteresis:**
  - Positive-Slice Dice: $74.92\%$ [95% CI: $70.82\%$--$78.39\%$]
  - Overall Dice: $86.12\%$ [95% CI: $84.32\%$--$87.75\%$]
  - Empty-Slice False Positive Rate: $13.65\% \to 7.86\%$ (**$-42.4\%$** relative reduction)
  - 3D Volumetric Agreement: $\text{ICC}(2, 1) = 0.8894$, Pearson $r = 0.8886$, Bland-Altman mean bias $= +143.7$ voxels
- **Radiogenomic Validation:**
  - Preserves exact biological rank ordering across molecular platforms (*COCCluster*):  
    $\text{COC 1 } (7{,}098.0) > \text{COC 3 } (5{,}509.6) > \text{COC 2 } (4{,}834.5\text{ voxels})$, $p = 0.0268$.

---

## 📁 Repository Structure

```
lgg_segmentation/
├── codes/
│   ├── src/
│   │   ├── config.py                 # Central hyperparameter configuration
│   │   ├── dataset.py                # Zero-leakage data loaders & cached pipeline
│   │   ├── preprocessing.py          # Morphological screening & wavelet denoising
│   │   ├── augmentation.py           # Elastic, affine, and intensity augmentation (pure SciPy/NumPy)
│   │   ├── losses.py                 # Focal Tversky Loss + Binary Focal Loss + Deep Supervision
│   │   ├── metrics.py                # Dice, IoU, HD95, ASD, ICC, and calibration (ECE/Brier)
│   │   ├── postprocessing.py         # Anisotropic 3D Z-continuity hysteresis & components
│   │   ├── train.py                  # High-throughput GPU training loop with warmup & cosine decay
│   │   └── models/
│   │       └── attention_resunet.py  # Attn Res-UNet (GN, GELU, DropPath, Deep Sup)
│   ├── scripts/
│   │   ├── run_gpu_experiments.py    # Master runner for 5-fold training (Patient-Grouped & Random-Slice)
│   │   ├── run_leakage_audit_and_figures.py # Audits leakage, calibration, generates Figs 2, 3, 4
│   │   ├── run_3d_radiomics_and_clinical.py # Patient-level 3D metrics, Figs 5, 6, genomic tests
│   │   └── generate_architecture_pipeline_figure.py # Generates Fig 1 pipeline diagram
│   └── results/
│       └── bspc_production/
│           ├── cohort_screening_audit.csv       # 200 artifact slices audit
│           ├── data_leakage_audit_comparison.csv# Quantitative leakage audit table
│           ├── 3d_patient_level_metrics.csv     # 110-patient 3D metrics
│           ├── figures/                         # 300 DPI publication figures
│           ├── patient_grouped/                 # 5-fold training logs & histories
│           └── random_slice/                    # 5-fold random-slice training logs
├── paper_bspc_submission/
│   ├── manuscript_bspc.tex           # Complete LaTeX manuscript (Elsevier 3p 2-column)
│   ├── references.bib                # Fully formatted bibliography
│   ├── cover_letter.tex              # Formal submission cover letter
│   ├── response_to_reviewers.tex     # Comprehensive point-by-point rebuttal
│   └── figures/                      # 300 DPI vector PDF figures
├── ferdowsi_launcher.ipynb           # Cloud GPU / Ferdowsi server notebook launcher
├── .gitignore                        # Standard ignore rules excluding >50MB binaries
└── README.md
```

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/Shayan-Ganji/lgg_segmentation.git
cd lgg_segmentation

# Create and activate environment
conda create -n lgg_env python=3.10 -y
conda activate lgg_env

# Install dependencies (zero external C++ or non-standard dependencies required)
pip install numpy scipy pandas matplotlib scikit-learn pillow tensorflow==2.18.0
```

### 2. Dataset Preparation

Download the TCGA-LGG MRI dataset from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) and extract it into `kaggle_3m/`.

### 3. Training & Evaluation

To run the complete 5-fold patient-grouped cross-validation:

```bash
python codes/scripts/run_gpu_experiments.py --protocol patient_grouped --epochs 100 --batch_size 16
```

To run the matched 5-fold random-slice leakage audit:

```bash
python codes/scripts/run_gpu_experiments.py --protocol random_slice --epochs 100 --batch_size 16
```

### 4. Auditing, Calibration, and Publication Figures

```bash
# Execute Step 3: Leakage Audit, Temperature Scaling Calibration, and Figure Generation
python codes/scripts/run_leakage_audit_and_figures.py

# Execute Step 4: 3D Volumetric Metrics, Bland-Altman, and Radiogenomic Association
python codes/scripts/run_3d_radiomics_and_clinical.py

# Generate Stage 1-3 Architecture and Pipeline Diagram
python codes/scripts/generate_architecture_pipeline_figure.py
```

---

## 📜 Citation

If you find this work or codebase helpful for your research, please cite our paper:

```bibtex
@article{ganji2026calibration,
  title={Calibration-Aware Anisotropic 3D Hysteresis and Leakage Auditing for Volumetric Lower-Grade Glioma Segmentation and Radiogenomic Profiling in Brain MRI},
  author={Ganji Nataj, Shayan and Borna, Keivan},
  journal={Biomedical Signal Processing and Control},
  year={2026},
  publisher={Elsevier}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
