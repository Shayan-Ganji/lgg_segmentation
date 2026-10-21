# Cover Letter for Journal Submission

**To:** Editor-in-Chief, *Biomedical Signal Processing and Control* (Elsevier)  
**Date:** October 21, 2026  
**Subject:** Submission of Original Research Article  

**Title:** Calibration-Aware Anisotropic 3D Hysteresis and Leakage Auditing for Volumetric Lower-Grade Glioma Segmentation and Radiogenomic Profiling in Brain MRI  
**Authors:** Shayan Ganji Nataj, Keivan Borna (Corresponding Author: `borna@khu.ac.ir`)  
**Affiliation:** Department of Computer Science, Faculty of Mathematics and Computer Science, Kharazmi University, Tehran, Iran  

---

Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript entitled **"Calibration-Aware Anisotropic 3D Hysteresis and Leakage Auditing for Volumetric Lower-Grade Glioma Segmentation and Radiogenomic Profiling in Brain MRI"** for publication consideration as a regular research article in *Biomedical Signal Processing and Control*.

Lower-grade gliomas (LGG, WHO grades II and III) represent heterogeneous infiltrative brain tumors where precise computer-assisted delineation is critical for surgical resection and radiotherapy planning. Despite intensive deep learning research, two major methodological pitfalls pervasive in medical imaging literature frequently undermine clinical translation:
1. **Slice-level data leakage**, where random slice splits allow networks to memorize anatomical features across contiguous axial slices from the same subject.
2. **Post-hoc threshold tuning** directly on the testing cohort, introducing artificial optimism.

Furthermore, 2D networks evaluated on volumetric MRI encounter axial discontinuity at apical and basal tumor margins while suffering elevated false-positive alarms across non-neoplastic slices (which comprise 65.1% of the TCGA-LGG imaging volume).

In this study, we address these challenges through a mathematically rigorous, leakage-free pipeline and methodological auditing framework:
- **Controlled Leakage Quantification:** We empirically demonstrate that conventional random slice splitting artificially inflates the positive-slice Dice coefficient by **+8.79%** (73.96% vs. 82.74%) due to patient identity memorization, explaining why many published 2D models fail to generalize to independent patients.
- **Nested Inner-Fold Calibration:** Using bounded scalar temperature scaling fitted strictly on inner validation logits, we reduce the pooled cohort Expected Calibration Error (ECE) by **46.9%** (0.369% to 0.196%) and the mean per-fold ECE by **31.7%** (0.372% to 0.254%), narrowing the empirical overconfidence gap without altering class boundaries.
- **Anisotropic 3D Z-Continuity Hysteresis:** By decoupling horizontal in-plane contouring ($\tau_{xy}^*$) from axial rescue ($\tau_{z\_low}^*$) and enforcing 3D volumetric spatial support ($V_{min}^*$, $\tau_{seed}^*$), the proposed framework achieves a positive-slice Dice of **74.92%** [95% CI: 70.82%–78.39%], overall Dice of **86.12%**, and a **42.4% relative reduction** in empty-slice false alarm rates (13.65% to 7.86%).
- **Clinical and Radiogenomic Concordance:** Patient-level volumetric evaluation across 110 subjects demonstrates excellent clinical agreement with radiologist ground truth (two-way random absolute agreement $\text{ICC}(2, 1) = \mathbf{0.8894}$, Pearson $r = 0.8886$, Bland-Altman mean bias $= +143.7$ voxels). Crucially, the model-segmented volumes faithfully reproduce the statistically significant volumetric distinction across TCGA pan-glioma molecular consensus clusters (*COCCluster*, $p = 0.0268$ vs. ground truth $p = 0.0169$) with exact biological rank ordering ($\text{COC 1} > \text{COC 3} > \text{COC 2}$).

We confirm that this manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have approved the final manuscript and concur with its submission.

Thank you very much for your time and editorial consideration.

Sincerely,

**Keivan Borna**, Ph.D. (Corresponding Author)  
Associate Professor of Computer Science  
Department of Computer Science, Faculty of Mathematics and Computer Science  
Kharazmi University, Tehran, Iran  
Email: `borna@khu.ac.ir`  

**Shayan Ganji Nataj**  
Department of Computer Science  
Kharazmi University, Tehran, Iran  
Email: `shayan.ganji@aut.ac.ir`  
