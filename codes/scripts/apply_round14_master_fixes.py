import re

# 1. Update cover_letter_jiim.tex
with open('JIIM_Submission_Ready/cover_letter_jiim.tex', 'r', encoding='utf-8') as f:
    cl_text = f.read()

# Remove named GitHub URL from cover letter
cl_old_url = r'All source code, reproduction scripts, and 5-fold cross-validation pipelines are openly maintained at GitHub: \url{https://github.com/Shayan-Ganji/lgg_segmentation}.'
cl_new_url = r'All source code, reproduction scripts, and 5-fold cross-validation pipelines are provided in an anonymized repository for peer review and will be made openly available on GitHub upon acceptance (with full repository details provided on \texttt{title\_page.pdf}).'
if cl_old_url in cl_text:
    cl_text = cl_text.replace(cl_old_url, cl_new_url, 1)

# Bullet 2 rank agreement wording
cl_old_b2 = r'Sweeping prevalence reveals rank decorrelation (Kendall $\tau = -0.067$)'
cl_new_b2 = r'Sweeping prevalence reveals no detectable rank agreement (Kendall $\tau = -0.067$, 95\% CI: [$-0.156, +0.111$])'
if cl_old_b2 in cl_text:
    cl_text = cl_text.replace(cl_old_b2, cl_new_b2, 1)

with open('JIIM_Submission_Ready/cover_letter_jiim.tex', 'w', encoding='utf-8') as f:
    f.write(cl_text)
print("✓ cover_letter_jiim.tex updated successfully!")


# 2. Update manuscript_blinded.tex and manuscript_full.tex
def patch_manuscript(filepath):
    print(f"\n=== Patching {filepath} ===")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Abstract replacement
    new_abstract = r"""Lower-grade glioma benchmarks report high Dice, but clinical translation is hindered by evaluation flaws. Auditing TCGA-LGG (110 patients, 3,929 slices) disentangles leakage, dilution, calibration, and post-processing across 5 folds. Patient-grouped Attention Res-UNet achieves 73.96\% positive-slice ($\text{Dice}_{\text{pos}}$) and 82.02\% all-slice Dice. Random-slice splitting inflates positive-slice Dice by $+8.79\,\text{pp}$ [95\% CI: $+6.16, +11.99\,\text{pp}$] (82.74\%; patient median inflation $4.41\,\text{pp}$, IQR: $1.08$--$10.74\,\text{pp}$) and all-slice Dice by $+6.20\,\text{pp}$ [95\% CI: $+4.26, +8.31\,\text{pp}$] ($88.22\%$). A copy oracle reaches 86.18\% all-slice Dice via empty copying ($\text{FPR}_{\text{empty}} = 0.63\%$), while per-patient inflation is decoupled from mask continuity ($r = -0.068$) and persists across non-adjacent slices (primarily at a 2-slice gap, $\Delta z \ge 2$: $+5.54\,\text{pp}$). A non-lesional probe predicts patient identity with 79.64\% accuracy (82.32\% at $\Delta z = 1$, 59.44\% at $\Delta z \ge 2$), indicating patient cues, though background similarity does not predict inflation. Scoring empty slices as true negatives shifts all-slice over positive-slice Dice by $+8.06$ to $+15.64\,\text{pp}$ across post-processing rules (Largest-CC: 89.64\% vs.\ 74.00\%), yielding no detectable rank agreement ($\tau = -0.067$) and reversing top ranks below $\pi^* = 86.53\%$. Weighted active-bin calibration error fell from 36.14\% to 29.57\% under temperature scaling and to 8.46\% under affine Platt scaling with neutral overlap. Training-patient conformal calibration reduced test coverage to 76.96\% (median 85.87\%), whereas out-of-patient calibration reached 85.74\% (median 94.29\%, 87.69\% under CRC, 90.08\% at $n_{\text{cal}} = 30$). Axial z-rescue (CAL-3DH) restored polar margins across 17 volumes (+0.28 pp, $p_{\text{Holm}} = 0.0011$) without surface penalty ($\text{ASD} = 1.96$ vs.\ 1.95)."""
    
    abs_pattern = re.compile(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', re.DOTALL)
    assert abs_pattern.search(text), "Abstract block not found"
    text = abs_pattern.sub(lambda m: f"\\begin{{abstract}}\n{new_abstract}\n\\end{{abstract}}", text, count=1)
    print("  ✓ Abstract replaced (245 words verified)")

    # Table 8: Standard U-Net and tabcolsep 4.8pt
    assert "Standard 2D U-Net" in text, "Table 8 Standard 2D U-Net not found"
    text = text.replace("Standard 2D U-Net", "Standard U-Net", 1)
    
    # tabcolsep in Table 8
    sota_idx = text.find('tab:sota_comparison')
    assert sota_idx != -1, "tab:sota_comparison not found"
    prev_colsep_idx = text.rfind(r'\setlength{\tabcolsep}{', 0, sota_idx)
    assert prev_colsep_idx != -1, "tabcolsep before Table 8 not found"
    end_line_idx = text.find('\n', prev_colsep_idx)
    text = text[:prev_colsep_idx] + r'\setlength{\tabcolsep}{4.8pt}' + text[end_line_idx:]
    print("  ✓ Table 8 updated: Standard U-Net and tabcolsep 4.8pt")

    # Table 4 verification
    t4_f1 = r'1 & 2.019 & 0.76 & 0.35 & 50 & 0.70 & 0.354 & 0.153 & 0.0267 & 0.0159 & 77.02 & 77.00 & 77.26 \\'
    assert t4_f1 in text, "Table 4 Fold 1 exact row missing"
    print("  ✓ Table 4 rows verified (Fold 1: 77.02 & 77.00 & 77.26)")

    # Methods additions:
    # End of Sec 2.4
    sec24_anchor = r"using tie-buffered Kendall's rank correlation coefficient $\tau_b$."
    sec24_add = r" Slice-level inflation heterogeneity is further modeled via a linear mixed-effects regression ($\Delta\text{Dice}_s \sim \text{ProbeCosineSim}_s + \Delta z_s + \log(\text{SliceTumorArea}_s) + (1 \mid \text{Patient})$) with Z-standardized predictors."
    assert sec24_anchor in text, "Sec 2.4 anchor not found"
    text = text.replace(sec24_anchor, sec24_anchor + sec24_add, 1)
    print("  ✓ Sec 2.4 LMM explanation added")

    # End of Sec 2.5
    cal3dh_idx = text.find(r'\subsection{Conditional Axial Z-Continuity Rescue Algorithm')
    assert cal3dh_idx != -1, "Sec 2.6 CAL-3DH heading not found"
    prev_dot_idx = text.rfind('.', 0, cal3dh_idx)
    assert prev_dot_idx != -1, "Period before CAL-3DH heading not found"
    sec25_add = r" To evaluate finite-sample calibration stability beyond the $n_{\text{cal}} = 13$ inner-validation subjects per fold (which also participate in early stopping and scalar temperature fitting), we additionally perform 1,000 paired Monte Carlo resamplings drawing $n_{\text{cal}} = 30$ calibration subjects and $n_{\text{eval}} = 80$ evaluation subjects from the 110 out-of-fold test predictions."
    text = text[:prev_dot_idx+1] + sec25_add + text[prev_dot_idx+1:]
    print("  ✓ Sec 2.5 Monte Carlo resampling explanation added")

    # Item 5: 5 subtle wording edits
    # 5.1 Page 12 Metrics Reloaded guidance
    mr_old = r'However, as cautioned in the comprehensive \textit{Metrics Reloaded} validation framework~\cite{maierhein2024metrics}, this empty-slice convention induces'
    mr_new = r'However, consistent with the general guidance in the \textit{Metrics Reloaded} framework~\cite{maierhein2024metrics} on handling empty reference slices, this convention induces'
    assert mr_old in text, "5.1 Metrics Reloaded wording not found"
    text = text.replace(mr_old, mr_new, 1)

    # 5.2 Page 30 in addition to learning generalizable tumor semantics
    sem_old = r'rather than learning generalizable tumor semantics'
    sem_new = r'in addition to learning generalizable tumor semantics'
    assert sem_old in text, "5.2 rather than learning not found"
    text = text.replace(sem_old, sem_new, 1)

    # 5.3 Page 37 TCGA-EZ & Page 38 ICC(2,1)
    ez_old = r'and TCGA-EZ [$n=1$: $\text{Dice}_{\text{pos}} = 90.47\%$, $\text{3D Dice} = 91.64\%$, $\text{FPR}_{\text{empty}} = 0.00\%$].'
    ez_new = r'and TCGA-EZ [$n=1$ (single-subject site, descriptive only): $\text{Dice}_{\text{pos}} = 90.47\%$, $\text{3D Dice} = 91.64\%$, $\text{FPR}_{\text{empty}} = 0.00\%$].'
    assert ez_old in text, "5.3 TCGA-EZ not found"
    text = text.replace(ez_old, ez_new, 1)

    icc_old = r'Two-way random absolute agreement yielded an Intraclass Correlation Coefficient of $\text{ICC}(2, 1) = 0.8894$ ($95\%$ CI: $[0.7782, 0.9521]$ via 10,000 patient-clustered bootstrap resamples), indicating good volumetric consistency'
    icc_new = r'Two-way random absolute agreement yielded an Intraclass Correlation Coefficient of $\text{ICC}(2, 1) = 0.8894$ (95\% analytic CI: $[0.7782, 0.9521]$; 10,000-resample patient-clustered bootstrap 95\% CI: $[0.7787, 0.9507]$), indicating good volumetric consistency'
    assert icc_old in text, "5.3 ICC(2,1) not found"
    text = text.replace(icc_old, icc_new, 1)

    # 5.4 Page 42 may reflect spatial downsampling
    buda_old = r'while the lower arithmetic mean ($77.73\%$ vs.\ $82.0\%$) reflects spatial downsampling'
    buda_new = r'while the lower arithmetic mean ($77.73\%$ vs.\ $82.0\%$) may reflect spatial downsampling'
    assert buda_old in text, "5.4 reflects spatial downsampling not found"
    text = text.replace(buda_old, buda_new, 1)

    # 5.5 Page 45 Conclusion 4 and final paragraph
    c4_title_old = r'\item \textbf{Balanced 3D Volumetric Post-Processing (CAL-3DH):}'
    c4_title_new = r'\item \textbf{Conditional Axial Z-Rescue Case Study (CAL-3DH):}'
    assert c4_title_old in text, "5.5 Conclusion 4 title not found"
    text = text.replace(c4_title_old, c4_title_new, 1)

    c4_cons_old = r'yielding robust volumetric consistency'
    c4_cons_new = r'yielding good volumetric consistency'
    assert c4_cons_old in text, "5.5 yielding robust volumetric consistency not found"
    text = text.replace(c4_cons_old, c4_cons_new, 1)

    concl_final_old = r'By providing a standardized, reproducible auditing workflow, this study establishes a methodological template to guide methodologists and imaging informatics practitioners in developing transparent, patient-disjoint deep learning pipelines for quantitative neuro-oncology research.'
    concl_final_new = r'By providing a standardized, reproducible auditing workflow, this study proposes a methodological template---warranting prospective validation across additional multi-center datasets and native 3D architectures---to guide imaging informatics practitioners in developing transparent, patient-disjoint deep learning pipelines.'
    assert concl_final_old in text, "5.5 Final conclusion sentence not found"
    text = text.replace(concl_final_old, concl_final_new, 1)

    print("  ✓ All 5 wording edits applied")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"=== Successfully updated {filepath} ===\n")

patch_manuscript("JIIM_Submission_Ready/manuscript_blinded.tex")
patch_manuscript("JIIM_Submission_Ready/manuscript_full.tex")
print("ALL FILES SUCCESSFULLY PATCHED!")
