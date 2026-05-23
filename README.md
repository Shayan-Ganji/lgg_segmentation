# A Progressive Learning Framework for Lower-Grade Glioma (LGG) Brain MRI Segmentation

## Project Overview
This repository presents a robust, modular, and progressive deep learning framework engineered for Fluid-Attenuated Inversion Recovery (FLAIR) Brain MRI segmentation utilizing the Lower-Grade Glioma (LGG) benchmark dataset. Navigating from high-capacity vanilla networks to a highly optimized **Lightweight Attention Res-UNet**, this project systematically addresses clinical resource constraints, model logit calibration, and architectural gradient guidance. 

Crucially, this framework introduces a rigorous methodological critique regarding evaluation protocols in medical imaging, providing the first comprehensive quantification of **Spatial Data Leakage** within this benchmark. We empirically demonstrate that randomized slice-level partitioning results in a severe evaluation bias due to anatomical memorization, inflating metrics falsely while masking genuine generalizability limitations.

---

## Performance and Ablation Analysis
The systematic evolution of the framework across six distinct experimental configurations is quantified below. To maintain strict scientific integrity, metrics are categorized into live back-end soft monitoring coefficients and clinical hard binarized segmentation scores.

| Model Version | Architectural Configuration | Evaluation Split Protocol | Patient Overlap | Overall Hard Dice | Tumor-Positive Hard Dice |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **v1** | Standard 2D U-Net (31.03M Params) | Exploratory Random | ~100% | ~0.4000 (Soft) | N/A |
| **v2** | v1 + Layer-Selective L-BFGS | Exploratory Random | ~100% | 0.5372 (Soft) | N/A |
| **v3** | nnU-Net Style + Deep Supervision | Exploratory Random | ~100% | 0.7417 (Hard) | N/A |
| **v4** | Attention Res-UNet (4.82M Params) | Strict Patient GroupKFold-5 | 0% | 0.8847 (Hard) | 0.7268 $\pm$ 0.2659 |
| **v5** | **v4 + Rotational Test-Time Augmentation** | **Strict Patient GroupKFold-5** | **0%** | **0.8991 (Hard)** | **0.7738 $\pm$ 0.2532** |
| **Leaky** | Attention Res-UNet (4.82M Params) | Randomized Slice Split | ~100% | *0.9232 (Inflated)*| *0.8366 (Inflated)* |

*Methodological Insight: Spatial data leakage artificially inflates the Hard Dice score by **+2.65%** across all slices and by **+6.31%** strictly on tumor-positive slices. Under a rigorous clinical isolation protocol devoid of data leakage, our production framework (89.91% Dice) significantly outperforms recent published benchmarks like LiteMRINet (85.50%) which relied on leaky slice-level distributions.*

---

## Repository Directory Tree
```text
lgg_segmentation/
├── src/                          # Core functional package
│   ├── config.py                 # Configuration manager & Hyperparameters
│   ├── losses.py                 # Asymmetric and Compound loss definitions
│   ├── preprocessing.py          # Automated quality assessment & artifact removal
│   ├── augmentation.py           # Pure OpenCV fast geometric/photometric operations
│   ├── metrics.py                # Soft monitoring and clinical hard metrics calculation
│   ├── dataset.py                # Multi-threaded tf.data engine pipeline
│   ├── train.py                  # WarmupCosineDecay scheduling & fold trainer execution
│   ├── evaluate.py               # Rotational TTA expectation & morphological filter
│   └── models/                   # Architecture matrix
│       ├── unet.py               # Baseline 31M model (v1/v2)
│       ├── deep_supervised_unet.py # nnU-Net style architecture (v3)
│       └── attention_resunet.py  # Final SOTA Production model (v4/v5)
├── notebooks/                    # Chronological experimental history
│   ├── v1_baseline_unet.ipynb
│   ├── v2_adam_lbfgs_hybrid.ipynb
│   ├── v3_deep_supervised_unet.ipynb
│   ├── v4_attention_resunet_clean.ipynb
│   ├── v5_balanced_tta.ipynb
│   └── leakage_analysis.ipynb
├── configs/
│   └── config_production.yaml    # External pipeline configuration file
├── scripts/                      # CLI executable scripts
│   ├── train_v5.py               # Executing production 5-Fold GroupKFold loop
│   └── leakage_experiment.py     # Reproducing spatial data leakage ablation analysis
├── results/                      # Output verification logs and reports
│   ├── v4_production/
│   ├── v5_tta/
│   └── leaky_experiment/
├── requirements.txt              # Pipeline dependencies
└── README.md                     # Main project documentation
Usage and Execution Guide1. Environment InstallationClone the repository and install the verified architectural dependencies:Bashgit clone [https://github.com/Shayan-Ganji/lgg_segmentation.git](https://github.com/Shayan-Ganji/lgg_segmentation.git)
cd lgg_segmentation
pip install -r requirements.txt
2. Dataset ManagementAcquire the clinical brain MRI dataset. Place the compressed kaggle_3m.zip file directly in the repository root directory. The internal data pipeline implemented in src/dataset.py handles automated, safe extraction securely upon execution.3. Executing Production Cross-Validation Pipeline (v5)To initialize the robust 5-Fold Patient-Level training loop managed by parallel prefetching, threshold grid searching, and rotational TTA evaluation:Bashpython scripts/train_v5.py
4. Reproducing Spatial Data Leakage Ablation AnalysisTo quantitatively reproduce our critique regarding randomly split SOTA baselines and output the metric inflation verification reports directly into the results/ directory:Bashpython scripts/leakage_experiment.py
Chronological Experimental ProgressionVersion 1 — Baseline U-Net DevelopmentMethodology: Implemented quality-aware data preprocessing utilizing dual moment-based noise tracking. Deployed a standard 2D U-Net configuration involving 31.03 million trainable parameters operating on grayscale fields.Findings: Revealed a catastrophic overfitting regime. Training Dice converged to 0.7205 while validation metrics stagnated at 0.3935 due to extreme parameter co-adaptation under constrained sample scales.Version 2 — Layer-Selective Hybrid Optimization (Convex Calibration)Methodology: Implemented a two-phase optimization topology. Following initial global exploration via Adam, all core spatial layers were frozen. A second-order quasi-Newton optimizer (L-BFGS) was applied exclusively to the final $1\times1$ classification layer to minimize validation Binary Cross-Entropy loss via a Platt Scaling variant ($w=0.8557, b=-0.3269$).Findings: Mathematical logit calibration successfully dropped Test BCE to 0.0223 and scaled Test Dice to 0.5372, validating logit scaling efficacy while highlighting the frozen feature extraction bottleneck.Version 3 — Deeply Supervised nnU-Net Style ParadigmMethodology: Re-engineered the expanding path to embed four hierarchical multi-resolution auxiliary decoding heads optimized via dynamically scaled loss weights ($\lambda = \{1.0, 0.5, 0.25, 0.125\}$). Volatile Batch Normalization layers were replaced with custom, batch-size independent Channel Group Normalization ($G=8$).Findings: Mitigated mid-network gradient degradation on micro-lesions, forcing direct gradient injection into early encoder states, which propelled the Test Dice coefficient to 0.7417.Version 4 & v5 — Final Production Attention Res-UNetMethodology: Transformed the pipeline into a highly regularized, lightweight architecture containing 4.82 million parameters. Integrated Stochastic Depth regularizers (linearly scaling to a 0.20 path drop probability) and additive Attention Gates (AGs) along skip connections. Enforced a clinical patient-isolated GroupKFold protocol paired with Asymmetric Rotational Test-Time Augmentation ($\theta \in \{-10^\circ, -5^\circ, 0^\circ, 5^\circ, 10^\circ\}$) and connected component post-filtering.Result: Achieved a clean, leakage-free Hard Dice score of 0.8991 across the entire distribution, and 0.7738 strictly on tumor-positive zones.Technical Implementation HighlightsAsymmetric Compound Loss Formulation: To combat severe pixel-level class imbalance (where over 65% of data represents empty tissue background), the optimization trajectory is governed by a heavily skewed Focal Tversky Loss ($\alpha=0.70, \beta=0.30, \gamma=0.75$) coupled with Binary Focal Loss ($\gamma=2.0$) to prioritize sparse tumor voxel recall.Bimodal Failure Topology: Our analysis details that 2D convolutional models processing single-channel FLAIR structures exhibit a bimodal error distribution rather than a standard Gaussian profile. The network either resolves glioma masses with exceptional accuracy (Dice $\ge$ 0.90) or fails entirely (Dice = 0) on low-contrast micro-lesions due to the loss of 3D spatial volumetric context, dragging down the clinical mean score.Methodological Analysis: Spatial Data LeakageEnforcing data isolation at the patient level is a strict necessity to prevent neural networks from artificially inflating performance by memorizing specific skull contours and background anatomical shapes.Evaluation Protocol SplitPatient Overlap RateTumor-Positive Hard DiceClinical GeneralizabilityData IntegrityRandomized Slice-Level Split$\sim 100\%$0.8366 (Inflated Bias)Deficient / Non-ViableCompromisedPatient-Level GroupKFold-50%0.7738 (Honest Metric)High / ViablePreserved
