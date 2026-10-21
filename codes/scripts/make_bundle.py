import zipfile
import os

zip_path = r"d:\lgg_segmentation\upload_to_server.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

print("Building upload_to_server.zip bundle...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    # 1. Launcher notebook
    z.write(r"d:\lgg_segmentation\ferdowsi_launcher.ipynb", "ferdowsi_launcher.ipynb")
    
    # 2. kaggle_3m essentials
    z.write(r"d:\lgg_segmentation\kaggle_3m\data.csv", "kaggle_3m/data.csv")
    z.write(r"d:\lgg_segmentation\kaggle_3m\cohort_screening_audit.csv", "kaggle_3m/cohort_screening_audit.csv")
    z.write(r"d:\lgg_segmentation\kaggle_3m\preprocessed_cache_128.npz", "kaggle_3m/preprocessed_cache_128.npz")
    
    # 3. codes folder
    for root, dirs, files in os.walk(r"d:\lgg_segmentation\codes"):
        if "__pycache__" in root or "results" in root:
            continue
        for file in files:
            full_p = os.path.join(root, file)
            rel_p = os.path.relpath(full_p, r"d:\lgg_segmentation")
            z.write(full_p, rel_p)

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Bundle successfully created: {zip_path} ({size_mb:.2f} MB)")

# Verification
with zipfile.ZipFile(zip_path, "r") as z:
    nb = z.read("ferdowsi_launcher.ipynb").decode("utf-8")
    assert "opencv-python-headless" in nb, "opencv-python-headless missing in zip!"
    assert "--upgrade" not in nb, "--upgrade found in zip!"
    print("VERIFICATION PASSED: upload_to_server.zip contains verified ferdowsi_launcher.ipynb!")
