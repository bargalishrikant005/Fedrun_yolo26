import os
import shutil
import zipfile
import glob
import numpy as np
import nibabel as nib

# Configurations
RAW_DIR = "/Users/shrikant/Downloads/BraTS-PEDs-v1/Training"
SUBSET_DIR = "/Users/shrikant/Downloads/BraTS-Balanced-20"
ZIP_PATH = "/Users/shrikant/Downloads/BraTS-Balanced-20.zip"
TARGET_COUNT = 20

print("🔍 Scanning BraTS-PEDs-v1 Training directory...")
patient_dirs = sorted([
    os.path.join(RAW_DIR, d)
    for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith("BraTS")
])

print(f"Found {len(patient_dirs)} patients. Analyzing class distributions...")

patient_stats = []

for idx, pdir in enumerate(patient_dirs):
    patient_id = os.path.basename(pdir)
    # Find segmentation file
    seg_files = glob.glob(os.path.join(pdir, "*-seg.nii.gz")) + glob.glob(os.path.join(pdir, "*_seg.nii.gz"))
    if not seg_files:
        continue
        
    try:
        # Load segmentation mask
        seg_img = nib.load(seg_files[0])
        seg_data = seg_img.get_fdata(dtype=np.float32).astype(np.uint8)
        
        # Calculate voxel counts for each class: 1=SNFH, 2=Tumor Core (TC), 3=Enhancing Tumor (ET)
        c1 = int((seg_data == 1).sum())
        c2 = int((seg_data == 2).sum())
        c3 = int((seg_data == 3).sum())
        total_tumor = c1 + c2 + c3
        
        if total_tumor > 0:
            patient_stats.append({
                "id": patient_id,
                "path": pdir,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "total": total_tumor,
                # Score patients that have a healthy representation of all 3 classes (no class is zero)
                "has_all": c1 > 500 and c2 > 500 and c3 > 500
            })
    except Exception as e:
        print(f"Warning: Could not process {patient_id}: {e}")

# Selection Strategy:
# 1. First, select patients that have substantial volumes of ALL three classes to avoid empty class labels
all_classes_present = [p for p in patient_stats if p["has_all"]]
print(f"Found {len(all_classes_present)} patients with all 3 tumor categories represented.")

# If we have enough, select from this list. Sort by total volume to ensure solid training signals.
if len(all_classes_present) >= TARGET_COUNT:
    # Sort by total volume, take the middle-to-high range to avoid outliers
    all_classes_present.sort(key=lambda x: x["total"], reverse=True)
    selected = all_classes_present[:TARGET_COUNT]
else:
    # Fallback: Sort all patients by total volume and select top ones
    print("Warning: Less than 20 patients have all 3 classes. Selecting top general tumor volumes instead.")
    patient_stats.sort(key=lambda x: x["total"], reverse=True)
    selected = patient_stats[:TARGET_COUNT]

print(f"\n✅ Selected {len(selected)} balanced patients:")
for p in selected:
    print(f"  - {p['id']}: SNFH={p['c1']}, TC={p['c2']}, ET={p['c3']} (Total={p['total']})")

# Copying files
dest_training = os.path.join(SUBSET_DIR, "Training")
os.makedirs(dest_training, exist_ok=True)

print(f"\n📂 Copying files to {dest_training}...")
for p in selected:
    src_pdir = p["path"]
    dest_pdir = os.path.join(dest_training, p["id"])
    if os.path.exists(dest_pdir):
        shutil.rmtree(dest_pdir)
    shutil.copytree(src_pdir, dest_pdir)

# Zipping subset
print(f"\n🤐 Zipping dataset to {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBSET_DIR):
        for file in files:
            filepath = os.path.join(root, file)
            # Make the path relative to the parent of SUBSET_DIR so it unzips nicely
            relpath = os.path.relpath(filepath, os.path.dirname(SUBSET_DIR))
            zipf.write(filepath, relpath)

print(f"\n🎉 SUCCESS: Balanced 20-patient dataset zip created at: {ZIP_PATH}")
# Print total size of zip
zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"📦 Total Zip Size: {zip_size_mb:.2f} MB")
