import sys
import os
import fitz

sys.stdout.reconfigure(encoding="utf-8")

def check_pdf(pdf_path, checks, name):
    print(f"\n================ VERIFYING {name} ================")
    doc = fitz.open(pdf_path)
    full_text = ""
    for i, p in enumerate(doc):
        t = p.get_text()
        full_text += f"\n--- PAGE {i+1} ---\n" + t
    
    for label, pattern in checks:
        found = pattern in full_text
        status = "[PASS]" if found else "[FAIL]"
        print(f"{status} {label}")
        if not found:
            print(f"       Pattern not found: {pattern[:60]}...")

manuscript_checks = [
    ("Abstract: Pearson r paren", "Pearson r = 0.8886)"),
    ("Abstract: Optimism bias", "+0.063%"),
    ("Intro: Primary slice leakage engine", "primary engine of artificial performance inflation"),
    ("Intro: BSPC journal scope", "Biomedical Signal Processing and Control"),
    ("Section 2.2: Centroid Distance paren", "Centroid Distance (dc), defined"),
    ("Section 3.3: 75/13/22 patient split", "22 patients (20% of the 110-subject cohort"),
    ("Section 5.2: ECE background voxel dilution", "over 98.5% of MRI voxels belong to background air"),
    ("Section 5.3: Optimism bias plateau narrative", "+0.063% contrasts sharply with the massive +8.79%"),
    ("Section 5.4: 148 false alarm slices", "148 false alarm slices relative to the baseline (-42.4%"),
    ("Section 5.4: 61 fewer false alarm slices", "61 fewer false alarm slices compared to 2D Nested (-23.3%"),
    ("Section 5.4: Wilcoxon test p < 0.0001", "p < 0.0001, paired Wilcoxon signed-rank test"),
    ("Section 5.4: 5/5 folds", "5/5 folds in Table 4"),
    ("Table 5 Caption", "Table 5: Comprehensive comparison of 2D Baseline, 2D Nested"),
    ("Section 6.2: Spearman rho", "0.8819"),
    ("Table 8: Soft bracket", "88.22 [Soft: 88.18]")
]

check_pdf("paper_bspc_submission/manuscript_bspc.pdf", manuscript_checks, "MANUSCRIPT PDF")

cl_checks = [
    ("Cover letter: BSPC scope statement", "Biomedical Signal Processing and Control"),
    ("Cover letter: 95% CI bracket", "[95% CI: 70.82%"),
    ("Cover letter: Optimism bias equation", "0.063%"),
    ("Cover letter: Table 5", "Table 5")
]

check_pdf("paper_bspc_submission/cover_letter.pdf", cl_checks, "COVER LETTER PDF")
