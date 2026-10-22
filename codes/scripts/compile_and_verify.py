import os
import subprocess
import fitz

sub_dir = os.path.abspath("paper_bspc_submission")
pdflatex = r"C:\Users\USER\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
bibtex = r"C:\Users\USER\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe"

def run_cmd(cmd, desc):
    print(f"\n--- {desc} ---")
    res = subprocess.run(cmd, cwd=sub_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"FAILED (code {res.returncode})")
        # print last 30 lines of stdout
        print("\n".join(res.stdout.splitlines()[-30:]))
        raise RuntimeError(f"{desc} failed")
    else:
        print(f"SUCCESS (code {res.returncode})")
        # search for warnings or errors in output
        lines = res.stdout.splitlines()
        err_lines = [l for l in lines if "Error" in l or "Fatal" in l]
        if err_lines:
            print("Errors/Warnings:", err_lines)

# 1. Compile manuscript_bspc
print("================ COMPILING MANUSCRIPT ================")
run_cmd([pdflatex, "-interaction=nonstopmode", "manuscript_bspc.tex"], "pdflatex pass 1")
run_cmd([bibtex, "manuscript_bspc"], "bibtex")
run_cmd([pdflatex, "-interaction=nonstopmode", "manuscript_bspc.tex"], "pdflatex pass 2")
run_cmd([pdflatex, "-interaction=nonstopmode", "manuscript_bspc.tex"], "pdflatex pass 3")

# 2. Compile cover_letter
print("\n================ COMPILING COVER LETTER ================")
run_cmd([pdflatex, "-interaction=nonstopmode", "cover_letter.tex"], "pdflatex cover_letter pass 1")
run_cmd([pdflatex, "-interaction=nonstopmode", "cover_letter.tex"], "pdflatex cover_letter pass 2")

# 3. Compile response_to_reviewers
print("\n================ COMPILING RESPONSE TO REVIEWERS ================")
run_cmd([pdflatex, "-interaction=nonstopmode", "response_to_reviewers.tex"], "pdflatex response pass 1")
run_cmd([pdflatex, "-interaction=nonstopmode", "response_to_reviewers.tex"], "pdflatex response pass 2")

# 4. Verify outputs
print("\n================ VERIFYING PDF OUTPUTS ================")
for pdf_name in ["manuscript_bspc.pdf", "cover_letter.pdf", "response_to_reviewers.pdf"]:
    p = os.path.join(sub_dir, pdf_name)
    if os.path.exists(p):
        doc = fitz.open(p)
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"[OK] {pdf_name}: {len(doc)} pages, {size_mb:.2f} MB")
    else:
        print(f"[ERROR] {pdf_name} NOT FOUND!")

print("\nAll documents compiled successfully!")
