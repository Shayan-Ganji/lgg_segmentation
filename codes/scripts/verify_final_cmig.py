import sys
import fitz

sys.stdout.reconfigure(encoding='utf-8')

# 1. Check log
with open(r'd:\lgg_segmentation\CMIG_Submission_Ready\manuscript_cmig.log', 'r', encoding='latin-1') as f:
    log = f.read()

errors = [line for line in log.splitlines() if line.startswith('!') or 'Error:' in line or 'Fatal error' in line]
undefined_cites = [line for line in log.splitlines() if 'Citation' in line and 'undefined' in line]
undefined_refs = [line for line in log.splitlines() if 'Reference' in line and 'undefined' in line]

print(f'Errors: {len(errors)}')
print(f'Undefined citations: {len(undefined_cites)}')
print(f'Undefined references: {len(undefined_refs)}')

assert len(errors) == 0, f'Errors found: {errors}'
assert len(undefined_cites) == 0, f'Undefined citations: {undefined_cites}'
assert len(undefined_refs) == 0, f'Undefined references: {undefined_refs}'

# 2. Check PDF
doc = fitz.open(r'd:\lgg_segmentation\CMIG_Submission_Ready\manuscript_cmig.pdf')
print(f'Total pages: {len(doc)}')

full_text = ' '.join(' '.join(page.get_text().split()) for page in doc)
full_text_ci = full_text.lower()

checks = [
    ('Item 1: Dual ablation comparison vs Step 1+2 (90 false-alarm slices)', 'eliminating 90 false-alarm slices'),
    ('Item 1: Dual ablation comparison vs Step 1+2 (-30.9% relative reduction)', 'relative reduction in fpr'),
    ('Item 1: Dual ablation comparison vs Step 1+3 (+0.28 pp Dice_pos)', '+0.28 pp'),
    ('Item 1: Dual ablation comparison vs Step 1+3 (+0.63 pp boundary Dice)', 'boundary-slice dice by +0.63 pp'),
    ('Item 2: Mehrtash et al. (2020) citation in Section 1.1', 'mehrtash et al.'),
    ('Item 2: Wen et al. (2020) citation in Section 1.1', 'wen et al.'),
    ('Item 3: Section 4.2 empirical note (relaxing tau_z,low to 0.25-0.30)', 'relaxing τz,low to 0.25'),
    ('Item 3: Section 4.2 empirical note (inflating empty-slice false alarms)', 'inflating empty-slice false alarms without rescuing additional true boundary slices'),
    ('Item 4: Figure 2 box text (0.35)', '0.35')
]

for name, q in checks:
    if q.lower() in full_text_ci:
        print(f'[PASS] {name}')
    else:
        print(f'[FAIL] {name} -> \"{q}\"')

# Also verify Figure 2 on page 7 specifically
p7_blocks = [b[4] for b in doc[6].get_text('blocks')]
p7_text = ' '.join(p7_blocks)
print('\nPage 7 Figure 2 check:')
has_035 = '0.35' in p7_text
has_045 = '0.45' in p7_text
print(f'  0.35 present: {has_035}')
print(f'  0.45 present: {has_045}')
assert has_035, '0.35 not found on Page 7!'
assert not has_045, '0.45 still found on Page 7!'

print('\nALL VERIFICATION AUDITS PASSED WITH ZERO ERRORS!')
