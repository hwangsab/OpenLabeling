# Sabrina Hwang
# 08/01/2023

# Splits the output images and respective labels into a training and validation
# set. Also generates the appropriate YAML file that is needed for YOLOv5 model
# training from a custom dataset. 
# The generation of a training and validatoin set preserves the original output
# folder generated from OpenLabeling

# To use, edit the paths within the box as needed to the appropriate locations. 
# Within the `main` directory of OpenLabeling, run the command
# `python train_test_split.py` for your labelled images to be split.

import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

#***#***#***#***#***#***# Change these values as needed #***#***#***#***#***#***#
										                                                            #
# Define the paths to the input and output directories from OpenLabeling
input_dir = "/home/<username>/OpenLabeling/main/input"
output_dir = "/home/<username>/OpenLabeling/main/output/YOLO_darknet"
class_list_file = "/home/<username>/OpenLabeling/main/class_list.txt"
										                                                            #
# Define a name for your dataset:
dataset_name = "fruits"	
										                                                            #
# Define the path for the datasets directory
datasets_dir = "/home/<username>/datasets"
										                                                            #
#***#***#***#***#***#***#***#***#***#***#***#***#***#***#***#***#***#***#***#***#

# Read the class names from the class_list_file
with open(class_list_file, 'r') as f:
    class_names = f.read().splitlines()

# Create a dictionary to store the image and label file paths for each sample
data_dict = {}

# Read the image and label file paths and store them in the dictionary
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_file = os.path.join(input_dir, filename)
        label_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                label_lines = f.readlines()
            data_dict[filename] = (image_file, label_lines)

# Split the data into training and validation sets while preserving the image-label correspondence
train_data, val_data = train_test_split(list(data_dict.values()), test_size=0.2, random_state=42)

# Define the paths for the training and validation directories in YOLOv5 format
train_output_dir_labels = os.path.join(datasets_dir, f"{dataset_name}/labels/train")
val_output_dir_labels = os.path.join(datasets_dir, f"{dataset_name}/labels/val")
train_output_dir_images = os.path.join(datasets_dir, f"{dataset_name}/images/train")
val_output_dir_images = os.path.join(datasets_dir, f"{dataset_name}/images/val")

# Create the training and validation directories if they don't exist
os.makedirs(train_output_dir_labels, exist_ok=True)
os.makedirs(val_output_dir_labels, exist_ok=True)
os.makedirs(train_output_dir_images, exist_ok=True)
os.makedirs(val_output_dir_images, exist_ok=True)

# Write the training image and label files in YOLO format
for image_file, label_lines in train_data:
    shutil.copy(image_file, os.path.join(train_output_dir_images, os.path.basename(image_file)))
    with open(os.path.join(train_output_dir_labels, os.path.splitext(os.path.basename(image_file))[0] + ".txt"), 'w') as f:
        f.writelines(label_lines)

# Write the validation image and label files in YOLO format
for image_file, label_lines in val_data:
    shutil.copy(image_file, os.path.join(val_output_dir_images, os.path.basename(image_file)))
    with open(os.path.join(val_output_dir_labels, os.path.splitext(os.path.basename(image_file))[0] + ".txt"), 'w') as f:
        f.writelines(label_lines)

# Generate the YAML file
data_yaml = {
    'path': datasets_dir,
    'train': train_output_dir_images,
    'val': val_output_dir_images,
    'test': '',  # Optional: Add relative path to the test images directory if available
    'names': class_names,  # Classes with class names from class_list_file
}

# Save the YAML to a file
with open(os.path.join(datasets_dir, f"{dataset_name}.yaml"), 'w') as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False)

print("Data split into training and validation sets in YOLO format.")
print("YAML file generated successfully.")
