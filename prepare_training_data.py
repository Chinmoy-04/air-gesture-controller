"""
prepare_training_data.py — Split labelled dataset and generate dataset.yaml.

Expects:
    updated dataset/images/   — JPEG/PNG images
    updated dataset/labels/   — matching YOLO .txt label files
    updated dataset/classes.txt

Produces:
    yolo_data/train/  (80% of images)
    yolo_data/val/    (20% of images)
    dataset.yaml      — Ultralytics training config

Run:  python prepare_training_data.py
"""

import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_label_pairs(images_folder: str, labels_folder: str) -> list[tuple[str, str]]:
    """Return (image_filename, label_filename) pairs with matching label files."""
    pairs: list[tuple[str, str]] = []
    for filename in os.listdir(images_folder):
        if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_name = f"{Path(filename).stem}.txt"
        label_path = os.path.join(labels_folder, label_name)
        if os.path.isfile(label_path):
            pairs.append((filename, label_name))
    return pairs


def _prepare_output_dirs(output_folder: str) -> tuple[str, str]:
    """Create fresh train/val folder trees under *output_folder*."""
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")

    for folder in (train_folder, val_folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        for subfolder in ("images", "labels"):
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    return train_folder, val_folder


def split_data(input_folder, output_folder, train_ratio=0.8, seed=42):
    """
    Splits data into training and validation sets.

    Args:
    - input_folder (str): Path to the folder containing the 'images' and 'labels' subfolders.
    - output_folder (str): Path where the train and val directories will be created.
    - train_ratio (float): Ratio of data used for training (default 0.8).
    - seed (int): Random seed for reproducible splits.
    """

    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist.")

    # Check if images and labels folders exist
    images_folder = os.path.join(input_folder, 'images')
    labels_folder = os.path.join(input_folder, 'labels')

    if not os.path.exists(images_folder):
        raise ValueError(f"Images folder '{images_folder}' does not exist.")
    if not os.path.exists(labels_folder):
        raise ValueError(f"Labels folder '{labels_folder}' does not exist.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_folder, val_folder = _prepare_output_dirs(output_folder)

    combined = _image_label_pairs(images_folder, labels_folder)
    if not combined:
        raise ValueError("No valid image-label file pairs found.")

    print(f"Found {len(combined)} image-label pairs.")
    ext_counts: dict[str, int] = {}
    for img_file, _ in combined:
        ext = Path(img_file).suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print(f"Image types: {ext_counts}")

    random.seed(seed)
    random.shuffle(combined)
    image_files, label_files = zip(*combined)

    total_count = len(image_files)
    train_count = int(total_count * train_ratio)
    val_count = total_count - train_count

    # Copy training data to train folder
    for i in tqdm(range(train_count), desc="Copying training data"):
        img_file = image_files[i]
        lbl_file = label_files[i]

        # Copy image
        shutil.copy(os.path.join(images_folder, img_file), os.path.join(train_folder, 'images', img_file))
        # Copy label
        shutil.copy(os.path.join(labels_folder, lbl_file), os.path.join(train_folder, 'labels', lbl_file))

    # Copy validation data to val folder
    for i in tqdm(range(train_count, total_count), desc="Copying validation data"):
        img_file = image_files[i]
        lbl_file = label_files[i]

        # Copy image
        shutil.copy(os.path.join(images_folder, img_file), os.path.join(val_folder, 'images', img_file))
        # Copy label
        shutil.copy(os.path.join(labels_folder, lbl_file), os.path.join(val_folder, 'labels', lbl_file))

    print(f"Data split complete: {train_count} images for training and {val_count} images for validation.")


def create_data_yaml(path_to_classes_txt, path_to_data_yaml, dataset_root=None):
    """
    Creates a YOLO data configuration YAML file from a classes.txt file.

    Args:
    - path_to_classes_txt (str): Path to the classes.txt file containing class names (one per line)
    - path_to_data_yaml (str): Path where the data.yaml file will be created
    - dataset_root (str | None): Absolute path written to the YAML 'path' field (defaults to yolo_data parent)
    """
    # Check if the classes.txt file exists
    if not os.path.exists(path_to_classes_txt):
        print(f'Error: {path_to_classes_txt} not found! Please ensure the classes.txt file exists.')
        return

    # Read class names from classes.txt
    with open(path_to_classes_txt, 'r') as f:
        classes = []
        for line in f.readlines():
            if len(line.strip()) == 0: 
                continue  # Ignore empty lines
            classes.append(line.strip())
    
    if not classes:
        print('Error: classes.txt is empty or no valid classes were found.')
        return

    # Get number of classes
    number_of_classes = len(classes)

    if dataset_root is None:
        dataset_root = str(Path(path_to_data_yaml).resolve().parent / "yolo_data")

    data = {
        "path": str(Path(dataset_root).as_posix()),
        "train": "train/images",
        "val": "val/images",
        "nc": number_of_classes,
        "names": classes,
    }

    # Write the dictionary to a YAML file
    try:
        with open(path_to_data_yaml, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f'Successfully created config file at {path_to_data_yaml}')
    except Exception as e:
        print(f'Error writing to {path_to_data_yaml}: {e}')



# ============================================================================
# Main execution section
# ============================================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent
    datapath = PROJECT_ROOT / "updated dataset"
    outputpath = PROJECT_ROOT / "yolo_data"
    train_pct = 0.8
    split_seed = 42

    print(f"Input dataset : {datapath}")
    print(f"Output folder : {outputpath}")
    print(f"Train / val   : {train_pct:.0%} / {1 - train_pct:.0%}\n")

    split_data(str(datapath), str(outputpath), train_ratio=train_pct, seed=split_seed)

    create_data_yaml(
        str(datapath / "classes.txt"),
        str(PROJECT_ROOT / "dataset.yaml"),
        dataset_root=str(outputpath),
    )

