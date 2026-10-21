# Point-by-Point Response to Reviewers and Editor

**Manuscript Title:** Calibration-Aware Anisotropic 3D Hysteresis and Leakage Auditing for Volumetric Lower-Grade Glioma Segmentation and Radiogenomic Profiling in Brain MRI  
**Authors:** Shayan Ganji Nataj, Keivan Borna (Corresponding Author: `borna@khu.ac.ir`)  
**Target Journal:** *Biomedical Signal Processing and Control* (Elsevier)

---

## Response to the Associate Editor

### Comment 1
> *The authors must ensure complete methodological rigor, eliminating any possibility of threshold tuning leakage or ambiguous ensembling, and clearly demonstrate the clinical and 3D utility of the proposed segmentation system.*

**Response:**  
We deeply appreciate the Editor's guidance. In this revised submission, we have addressed all methodological concerns:
1. **Strict Nested Cross-Validation:** All hyperparameter tuning—including temperature scaling calibration ($T_k^*$), horizontal in-plane binarization ($\tau_{xy, k}^*$), axial rescue threshold ($\tau_{z\_low, k}^*$), and 3D component filtering ($V_{min, k}^*, \tau_{seed, k}^*$)—is performed exclusively on the internal validation set (`inner_val`) of each fold. The test set (`outer_test`) remains untouched until final evaluation. We formally quantify the artificial optimism of post-hoc test tuning as $\Delta\text{Optimism} = +0.063\%$ in Section 5.2.
2. **Unambiguous Evaluation Mapping:** We explicitly clarify that out-of-fold (OOF) predictions are generated strictly by the dedicated model trained for that fold (a 1-to-1 patient mapping), completely refuting any 5-model ensemble averaging ambiguity (Section 3.3).
3. **Clinical and 3D Volumetric Agreement:** We extend the 2D network predictions into complete 3D volumes for all 110 TCGA-LGG patients, evaluating 3D Volumetric Dice (77.73%), 95% Hausdorff Distance (10.97 voxels), Average Surface Distance (3.10 voxels), Intraclass Correlation ($\text{ICC}(2, 1) = 0.8894$), and molecular subtype concordance ($p = 0.0268$ across *COCCluster* cohorts).

---

## Response to Reviewer 1

### Comment 1: Threshold Optimization and Optimism Bias
> *The selection of the decision threshold appears to have been optimized on the testing data, which introduces data leakage and optimistic performance reporting. The authors should use a nested validation framework or report fixed-threshold baselines.*

**Response:**  
We fully agree with the Reviewer. In the revised manuscript, we implemented a strict nested 5-fold cross-validation scheme:
- In each fold $k$, the optimal threshold $\tau_{xy, k}^*$ is selected solely by maximizing positive-slice Dice on the unseen `inner_val` partition.
- The test partition is evaluated exclusively using the frozen $\tau_{xy, k}^*$.
- To directly address the Reviewer's concern, Section 5.2 and Table 5.3 present a formal comparison between:
  1. Fixed Baseline ($\tau=0.50$): $\text{Dice}_{\text{pos}} = 73.96\%$ [95% CI: 70.10%–77.15%]
  2. Nested Honest Search ($\tau_k^*$): $\text{Dice}_{\text{pos}} = 74.60\%$ [95% CI: 70.45%–78.08%]
  3. Leaked Pooled Search ($\tau_{\text{leak}}^* = 0.88$ on test data): $\text{Dice}_{\text{pos}} = 74.66\%$
- We define and report the **Optimism Bias**:
  $$\Delta\text{Optimism} = \text{Dice}(\tau_{\text{leak}}^*) - \text{Dice}(\tau_k^*) = +0.063\%$$
  This rigorously proves that our reported gains are genuinely leak-free.

---

### Comment 2: Ensemble Ambiguity in OOF Predictions
> *It is unclear whether out-of-fold (OOF) results were produced by averaging predictions across all 5 fold models (an ensemble) or by evaluating each test fold independently on its corresponding trained model.*

**Response:**  
We thank the Reviewer for highlighting this ambiguity. We have rewritten Section 3.3 to state the mathematical mapping unambiguously:
$$\hat{Y}_i = \mathcal{M}_k(X_i; \theta_k^*), \quad \forall i \in \mathcal{D}_{\text{test}}^{(k)}, \quad k \in \{1, 2, 3, 4, 5\}$$
Every patient slice belongs to exactly one test fold $\mathcal{D}_{\text{test}}^{(k)}$ and is segmented solely by the single network $\mathcal{M}_k$ trained on folds $j \neq k$. No model averaging, soft blending, or test-time ensembling was performed. The resulting 3,929 out-of-fold predictions represent true prospective generalization to unseen patients.

---

### Comment 3: Slice Leakage vs. Patient-Grouped Splitting
> *Many previous 2D MRI segmentation papers split datasets at the slice level rather than the patient level, leading to memorization of contiguous slices. The authors should evaluate whether slice-level splitting significantly inflates results.*

**Response:**  
The Reviewer raises a foundational concern in medical image analysis. To answer this definitively, we conducted an aligned, matched 5-fold cross-validation experiment comparing:
- **Patient-Grouped Split (Honest):** All slices belonging to any given patient are assigned strictly to a single fold.
- **Random-Slice Split (Data-Leaked):** Slices from the same patient are randomly dispersed across training and testing partitions.

As reported in Section 5.4 and Table 5.4, the random slice split yields an artificial positive-slice Dice inflation of **+8.79%** (82.74% vs. 73.96%) and an overall Dice inflation of **+6.20%** (88.22% vs. 82.02%). This controlled experiment demonstrates that superior Dice numbers in prior 2D literature often stem from slice leakage rather than architectural innovation.

---

### Comment 4: Validation Measurement Protocol and Data Augmentation
> *Were validation loss and Dice curves computed with stochastic data augmentation enabled or in pure inference mode? Augmentation during validation distorts performance curves.*

**Response:**  
In our revised pipeline (Section 3.2), validation and test evaluation are conducted in **pure inference mode** without data augmentation. Stochastic transformations (affine rotations, scaling, elastic deformation, and horizontal flips) are active strictly during forward-backward training steps. Slices in `inner_val` and `outer_test` are fed in deterministic axial sequence using preprocessed, unaugmented MRI tensors. The convergence history across all 5 folds is illustrated in Figure 5.

---

### Comment 5: Artifact Screening Audit and Slices Removed
> *The manuscript mentions excluding 200 artifact slices during training. The authors must audit whether this exclusion introduces selection bias or discards small neoplastic lesions.*

**Response:**  
We have conducted a slice-by-slice morphological and histopathological audit across all 3,929 slices (Section 2.2 and Table 2.1):
1. **Empty Slices:** Out of the 200 excluded slices, exactly **199 slices (99.50%)** are completely tumor-free ($y = 0$). Only 1 slice contained a marginal lesion (0.50%, Slice 23 of TCGA_DU_5855) where severe gradient coil clipping truncated the brain parenchyma.
2. **Zero Change on Positive Slices:** Evaluating the model on the full cohort (3,929 slices) versus the screened cohort (3,729 slices) results in **identical positive-slice Dice of 73.96%** (a change of exactly $0.00\%$).
3. **Wavelet Denoising Gate:** Denoising is selectively applied only when slice quality index $Q(I) < 4.0$ (3,738 slices, 95.14%), while high-quality slices ($Q \ge 4.0$, 191 slices, 4.86%) bypass filtering to preserve fine parenchymal microstructures.

---

### Comment 6: Calibration and Uncertainty Estimation
> *Deep segmentation models trained with overlap-based loss functions (such as Focal Tversky) tend to be severely uncalibrated and overconfident. How does the proposed method address probabilistic reliability?*

**Response:**  
We thoroughly investigated probability calibration in Section 4.1. Because the Focal Tversky loss ($\alpha=0.8, \beta=0.2$) prioritizes foreground precision, raw network outputs exhibit severe overconfidence. We applied nested scalar temperature scaling:
$$\hat{P}_{i, z} = \sigma\left(\frac{z_{i}}{\hat{T}_k^*}\right), \quad \hat{T}_k^* = \arg\min_{T > 0} \mathcal{L}_{\text{NLL}}\left(\frac{z_{\text{val}, k}}{T}, y_{\text{val}, k}\right)$$
Across the 5 folds, the optimal temperature averaged $\hat{T}^* = 1.844$, confirming that the raw network was overconfident ($T > 1$). On untouched test data:
- Pooled cohort Expected Calibration Error (ECE) dropped by **46.9%** (0.369% to 0.196%).
- Mean per-fold ECE dropped by **31.7%** (0.372% to 0.254%).
- Brier Score improved from 0.00375 to 0.00338 (-9.9%), and NLL decreased from 0.0230 to 0.0154 (-32.9%).
- Reliability diagrams (Figure 4) confirm that empirical probability bins align closely with the diagonal identity line after temperature scaling.

---

### Comment 7: Handling of Non-Tumor Slices in Evaluation
> *In brain MRI, many slices contain no tumor. How are empty slices defined and penalized in Dice calculations, and what is the false-positive rate on healthy slices?*

**Response:**  
In Section 3.4, we explicitly state our mathematical convention for slice-level metrics:
$$\text{Dice}(Y, \hat{Y}) = \begin{cases}
1.0, & \text{if } |Y| = 0 \text{ and } |\hat{Y}| = 0 \text{ (True Negative)} \\
0.0, & \text{if } |Y| = 0 \text{ and } |\hat{Y}| > 0 \text{ (False Alarm)} \\
\frac{2|Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}, & \text{if } |Y| > 0
\end{cases}$$
In the TCGA-LGG cohort, 2,556 out of 3,929 slices (65.05%) are completely healthy. To prevent metric distortion, we report:
- $\text{Dice}_{\text{pos}}$ (evaluated strictly on the 1,373 tumor-bearing slices).
- $\text{Dice}_{\text{all}}$ (evaluated across all 3,929 slices including true negatives).
- $\text{FPR}_{\text{empty}}$: The false alarm rate on empty slices.

Crucially, our Proposed Anisotropic 3D Z-Hysteresis reduces $\text{FPR}_{\text{empty}}$ from **13.65% to 7.86%** (a **42.4% relative reduction** in false alarm slices), while boosting $\text{Dice}_{\text{all}}$ from 82.02% to **86.12%**.

---

### Comment 8: Fair Comparison with State-of-the-Art
> *The authors should provide a comprehensive and fair comparison with previous literature on the TCGA-LGG benchmark, detailing splitting regimes, post-processing methods, and model complexities.*

**Response:**  
In Section 7.1 and Table 7.1, we present an exhaustive comparison against 10 recent published studies on the TCGA-LGG cohort. We explicitly categorize literature by:
1. **Split Protocol:** Patient-Grouped vs. Random Slice.
2. **Thresholding Protocol:** Fixed/Nested vs. Post-hoc Tuned.
3. **Parameter Footprint:** Lightweight (4–8M) vs. Bulky 3D networks (>30M).
4. **Post-Processing Strategy:** 2D morphological filtering vs. Proposed 3D Z-Continuity Hysteresis.

Under strictly honest patient-grouped cross-validation, our framework establishes a new state-of-the-art benchmark ($\text{Dice}_{\text{pos}} = 74.92\%$, $\text{Dice}_{\text{all}} = 86.12\%$, 3D Volumetric $\text{Dice} = 77.73\%$, $\text{ICC} = 0.8894$) while operating with an efficient 7.8M parameter budget.
