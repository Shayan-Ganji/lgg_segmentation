with open('JIIM_Submission_Ready/manuscript_blinded.tex', 'r', encoding='utf-8') as f:
    text = f.read()

old_tab = r'\begin{tabular}{@{}l c l l r r r r r@{}}'
new_tab = r'\begin{tabular}{@{}l c l l r r r r @{\hspace{1.2em}} r@{}}'

assert old_tab in text
text = text.replace(old_tab, new_tab, 1)

with open('JIIM_Submission_Ready/manuscript_blinded.tex', 'w', encoding='utf-8') as f:
    f.write(text)

with open('JIIM_Submission_Ready/manuscript_full.tex', 'r', encoding='utf-8') as f:
    text_f = f.read()
assert old_tab in text_f
text_f = text_f.replace(old_tab, new_tab, 1)
with open('JIIM_Submission_Ready/manuscript_full.tex', 'w', encoding='utf-8') as f:
    f.write(text_f)

print("Added @{\\hspace{1.2em}} between columns 8 and 9!")
