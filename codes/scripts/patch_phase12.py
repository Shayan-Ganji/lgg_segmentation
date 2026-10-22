import os
import re

def patch_file(filepath, replacements):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    initial = content
    for target, rep in replacements:
        if target not in content:
            print(f"[WARN] Target not found in {filepath}:\n{target[:80]}...")
        else:
            content = content.replace(target, rep)
            print(f"[OK] Replaced in {filepath}: {target[:40]}... -> {rep[:40]}...")
            
    if content != initial:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[SUCCESS] Updated {filepath}")
    else:
        print(f"[NO CHANGE] {filepath}")

def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sub_dir = os.path.join(root, "paper_bspc_submission")
    
    # 1. manuscript_bspc.tex
    tex_path = os.path.join(sub_dir, "manuscript_bspc.tex")
    tex_replacements = [
        # Abstract line: Delta\text{Optimism} -> Delta\mathrm{Optimism}
        (
            r"(optimism bias $\Delta\text{Optimism} = +0.063\%$), legitimizing",
            r"(optimism bias $\Delta\mathrm{Optimism} = +0.063\%$), legitimizing"
        ),
        # Section 1 Item 3: \tau_{z,\mathrm{low}}^* and $FPR_{\mathrm{empty}} = 7.86\%$).
        (
            r"($\text{Dice}_{\text{pos}} = 74.92\%$, $\text{Dice}_{\text{all}} = 86.12\%$, $\text{FPR}_{\text{empty}} = 7.86\%$).",
            r"($\text{Dice}_{\text{pos}} = 74.92\%$, $\text{Dice}_{\text{all}} = 86.12\%$, $FPR_{\mathrm{empty}} = 7.86\%$).",
        ),
        (
            r"($V_{min}^*, \tau_{seed}^*$)",
            r"($V_{\mathrm{min}}^*, \tau_{\mathrm{seed}}^*$)"
        ),
        # Section 1 end: verifiable clinical readiness -> translational clinical readiness
        (
            "advancing translational computational oncology toward verifiable clinical readiness.",
            "advancing translational computational oncology toward translational clinical readiness."
        ),
        # Table 5 header: (\tau_k^*) -> (\tau_{xy}^*)
        (
            r"\shortstack{\textbf{2D Nested} \\ ($\tau_k^*$)}",
            r"\shortstack{\textbf{2D Nested} \\ ($\tau_{xy}^*$)}"
        ),
        # Table 7 row 1: 2D Baseline (\tau=0.50) -> 2D Baseline (\tau = 0.50)
        (
            r"2D Baseline ($\tau=0.50$)",
            r"2D Baseline ($\tau = 0.50$)"
        ),
    ]
    patch_file(tex_path, tex_replacements)
    
    # 2. cover_letter.tex
    cl_path = os.path.join(sub_dir, "cover_letter.tex")
    cl_replacements = [
        (
            "establishing a clinically verifiable benchmark for translational neuro-oncology.",
            "establishing a reproducible volumetric benchmark for translational neuro-oncology."
        )
    ]
    patch_file(cl_path, cl_replacements)

if __name__ == "__main__":
    main()
