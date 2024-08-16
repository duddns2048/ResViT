import os
import shutil
import json

# Paths
src_root = "../../Datasets/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
dest_root = "../split/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
split_file = "../../Datasets/BraTS2023/case_ids/splits_brats.json"

# Load the split information from JSON
with open(split_file, 'r') as f:
    splits = json.load(f)

# Function to copy folders
def copy_folders(folder_list, destination):
    for folder_name in folder_list:
        src_folder = os.path.join(src_root, folder_name)
        dest_folder = os.path.join(destination, folder_name)
        if os.path.exists(src_folder):
            shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
        else:
            print("the source directory doesn't exist!")

# Copy training folders
for fold in splits.keys():
    fold_dir = os.path.join(dest_root,fold)
    os.makedirs(fold_dir, exist_ok=True)
    copy_folders(splits[fold]["train"], os.path.join(fold_dir, "train"))
    copy_folders(splits[fold]["test"], os.path.join(fold_dir, "test"))

print("Folders copied successfully!")
