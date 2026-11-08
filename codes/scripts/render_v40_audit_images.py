import fitz

artifact_dir = r"C:\Users\USER\.antigravity_account2\.gemini\antigravity\brain\168aaa96-b2ea-4739-8d4b-ce110269e17a"

def render_page(pdf_path, page_num, out_filename):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=150)
    out_path = f"{artifact_dir}/{out_filename}"
    pix.save(out_path)
    print(f"Rendered {pdf_path} p.{page_num} -> {out_filename}")

render_page("JIIM_Submission_Ready/manuscript_blinded.pdf", 1, "v40_blinded_p01_abstract.png")
render_page("JIIM_Submission_Ready/manuscript_blinded.pdf", 21, "v40_blinded_p21_table4.png")
render_page("JIIM_Submission_Ready/manuscript_blinded.pdf", 43, "v40_blinded_p43_table8.png")
render_page("JIIM_Submission_Ready/manuscript_blinded.pdf", 45, "v40_blinded_p45_decl.png")
render_page("JIIM_Submission_Ready/manuscript_blinded.pdf", 47, "v40_blinded_p47_ref25.png")

render_page("JIIM_Submission_Ready/manuscript_full.pdf", 44, "v40_full_p44_table8.png")
render_page("JIIM_Submission_Ready/manuscript_full.pdf", 46, "v40_full_p46_decl.png")
render_page("JIIM_Submission_Ready/manuscript_full.pdf", 48, "v40_full_p48_ref25.png")

render_page("JIIM_Submission_Ready/title_page.pdf", 1, "v40_title_page.png")
render_page("JIIM_Submission_Ready/cover_letter_jiim.pdf", 1, "v40_cover_letter.png")

print("All v40 audit images rendered successfully!")
