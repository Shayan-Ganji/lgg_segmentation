import json

path = r'd:\lgg_segmentation\ferdowsi_launcher.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source_text = "".join(cell.get('source', []))
        if 'pip install' in source_text:
            cell['source'] = [
                "!pip install --quiet opencv-python-headless scikit-image scikit-learn pandas matplotlib tqdm\n",
                'print("✅ All dependencies verified.")'
            ]
            print("Successfully updated pip install cell!")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Saved updated ferdowsi_launcher.ipynb to disk.")
