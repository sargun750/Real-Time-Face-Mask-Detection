import os
import shutil

from sklearn.model_selection import train_test_split

# Paths (UPDATE THESE)
extracted_dir = "RMFD"  # Parent folder of RMFD_masked_face_dataset
output_dir = "dataset"

def copy_images(src_dir, dst_dir_class, split_ratio=0.8):
    images = []
    for root, _, files in os.walk(src_dir):
        for file in files:

            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                images.append(os.path.join(root, file))
    
    print(f"Found {len(images)} images in {src_dir}")
    
    if len(images) == 0:
        raise ValueError(f"No images in {src_dir}! Check path and file extensions.")

    train_files, val_files = train_test_split(images, train_size=split_ratio, random_state=42)
    
    for src in train_files:
        dst = os.path.join(output_dir, "train", dst_dir_class, os.path.basename(src))
        shutil.copy(src, dst)
        
    for src in val_files:
        dst = os.path.join(output_dir, "val", dst_dir_class, os.path.basename(src))
        shutil.copy(src, dst)

# Copy masked faces (handles subfolders like AFDB_masked_face_0001)
copy_images(
    src_dir = os.path.join(extracted_dir, "AFDB_masked_face_dataset"),
    dst_dir_class = "mask"
)

# Copy unmasked faces
copy_images(
    src_dir = os.path.join(extracted_dir, "AFDB_face_dataset"),
    dst_dir_class = "no_mask"
)

print("Dataset organized successfully!")