import os
import random
import shutil
from tools.create_lmdb_dataset import createDataset

# 1. Split train, test and val data

data_dir = "./securimage-data"  # Path to the directory containing the image files
train_dir = "./securimage-data/train"  # Path to the directory for the train set
test_dir = "./securimage-data/test"  # Path to the directory for the test set
val_dir = "./securimage-data/val"  # Path to the directory for the validation set

train_ratio = 0.7  # Proportion of images for the train set
test_ratio = 0.2  # Proportion of images for the test set
val_ratio = 0.1  # Proportion of images for the validation set

# Create directories for train, test, and validation sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of image files in data directory
image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

# Shuffle the image files randomly
random.shuffle(image_files)

# Calculate number of images for each set
num_images = len(image_files)
num_train = int(train_ratio * num_images)
num_test = int(test_ratio * num_images)
num_val = int(val_ratio * num_images)

# Move images to train set directory
for file_name in image_files[:num_train]:
    src_path = os.path.join(data_dir, file_name)
    dst_path = os.path.join(train_dir, file_name)
    shutil.move(src_path, dst_path)

# Move images to test set directory
for file_name in image_files[num_train:num_train + num_test]:
    src_path = os.path.join(data_dir, file_name)
    dst_path = os.path.join(test_dir, file_name)
    shutil.move(src_path, dst_path)

# Move images to validation set directory
for file_name in image_files[num_train + num_test:num_train + num_test + num_val]:
    src_path = os.path.join(data_dir, file_name)
    dst_path = os.path.join(val_dir, file_name)
    shutil.move(src_path, dst_path)

print(f"Images in '{data_dir}' directory have been split into train, test, and val sets with proportions of {train_ratio}, {test_ratio}, {val_ratio}, respectively.")
print(f"Train set images moved to '{train_dir}' directory.")
print(f"Test set images moved to '{test_dir}' directory.")
print(f"Validation set images moved to '{val_dir}' directory.")

# 2. Create ground truth files

train_gt_file = os.path.join(train_dir, "ground_truth.txt")
with open(train_gt_file, 'w') as f:
    for file_name in os.listdir(train_dir):
        if file_name.endswith(".png"):
            label = file_name.split(".")[0]  # Extract label from file name
            f.write(f"{file_name} {label}\n")  # Write image file name and label to ground truth file

test_gt_file = os.path.join(test_dir, "ground_truth.txt")
with open(test_gt_file, 'w') as f:
    for file_name in os.listdir(test_dir):
        if file_name.endswith(".png"):
            label = file_name.split(".")[0]  # Extract label from file name
            f.write(f"{file_name} {label}\n")  # Write image file name and label to ground truth file

val_gt_file = os.path.join(val_dir, "ground_truth.txt")
with open(val_gt_file, 'w') as f:
    for file_name in os.listdir(val_dir):
        if file_name.endswith(".png"):
            label = file_name.split(".")[0]  # Extract label from file name
            f.write(f"{file_name} {label}\n")  # Write image file name and label to ground truth file


print(f"Train set ground truth file '{train_gt_file}' created successfully!")
print(f"Test set ground truth file '{test_gt_file}' created successfully!")
print(f"Validation set ground truth file '{val_gt_file}' created successfully!")


# 3. Create LMDB dataset

train_lmdb_output_path = './data/train/real'
test_lmdb_output_path = './data/test/real'
val_lmdb_output_path = './data/val'

os.makedirs(train_lmdb_output_path, exist_ok=True)
os.makedirs(test_lmdb_output_path, exist_ok=True)
os.makedirs(val_lmdb_output_path, exist_ok=True)

createDataset(train_dir, train_gt_file, train_lmdb_output_path)
createDataset(test_dir, test_gt_file, test_lmdb_output_path)
createDataset(val_dir, val_gt_file, val_lmdb_output_path)