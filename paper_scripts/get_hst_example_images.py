#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
# !/usr/bin/env python3
"""
Generate example images with weak and strong augmentations from a folder of JPEG images.

This script creates examples of weakly and strongly augmented images from a given folder,
identifying specific images as anomalies based on their filenames.
"""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.append("/media/home/AnomalyMatch")
sys.path.append("../")
from anomaly_match.datasets.augmentation.randaugment import RandAugment

# Constants
ANOMALY_FILENAMES = [
    "4228766080.jpeg",
    "4001312789523.jpeg",
    "6000444310288.jpeg",
    "4000931204239.jpeg",
]
NUM_EXAMPLES = 10  # Number of examples to generate for each class (if available)
IMAGE_SIZE = (224, 224)  # Size to resize images to (if needed)
FIGURE_DPI = 100


def setup_output_directories():
    """Create output directories for the example images."""
    output_dir = Path("figures/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_sample_images(images_dir, anomaly_filenames=ANOMALY_FILENAMES, num_examples=NUM_EXAMPLES):
    """
    Get sample images from both anomaly and normal classes.

    Args:
        images_dir (str): Directory containing the images
        anomaly_filenames (list): List of filenames that are anomalies
        num_examples (int): Number of examples to get for each category

    Returns:
        tuple: Lists of anomaly and normal samples
    """
    anomaly_list = []
    normal_list = []

    # List all jpeg files in the directory
    all_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg"))]

    # Separate anomaly and normal files
    anomaly_files = [f for f in all_files if f in anomaly_filenames]
    normal_files = [f for f in all_files if f not in anomaly_filenames]

    # Sample from each group (if needed)
    if len(anomaly_files) > num_examples:
        random.seed(42)
        anomaly_files = random.sample(anomaly_files, num_examples)

    if len(normal_files) > num_examples:
        random.seed(42)
        normal_files = random.sample(normal_files, num_examples)

    # Create sample dictionaries
    for filename in anomaly_files:
        img_path = os.path.join(images_dir, filename)
        if os.path.exists(img_path):
            anomaly_list.append(
                {
                    "path": img_path,
                    "filename": filename,
                    "is_anomaly": True,
                    "class_name": "Anomaly",
                }
            )

    for filename in normal_files:
        img_path = os.path.join(images_dir, filename)
        if os.path.exists(img_path):
            normal_list.append(
                {
                    "path": img_path,
                    "filename": filename,
                    "is_anomaly": False,
                    "class_name": "Nominal",
                }
            )

    return anomaly_list, normal_list


def create_augmentations():
    """
    Create weak and strong transformations for image augmentation.

    Returns:
        tuple: (weak_transform, strong_transform)
    """
    # Define a simple weak transform (as used in BasicDataset)
    weak_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    # Create strong transform (following BasicDataset approach)
    strong_transform = transforms.Compose(
        [
            RandAugment(3, 5),  # Apply RandAugment as first step
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    return weak_transform, strong_transform


def apply_transforms(image, weak_transform, strong_transform, num_versions=3):
    """
    Apply weak and strong transformations to an image, creating multiple versions.

    Args:
        image (PIL.Image): Image to transform
        weak_transform: Weak transformation pipeline
        strong_transform: Strong transformation pipeline
        num_versions (int): Number of different augmentation versions to create

    Returns:
        tuple: (original_tensor, list of weak_tensors, list of strong_tensors)
    """
    # Convert to tensor for the original image
    to_tensor = transforms.ToTensor()
    original_tensor = to_tensor(image)

    # Apply weak and strong transforms multiple times
    weak_tensors = []
    strong_tensors = []

    for _ in range(num_versions):
        weak_tensors.append(weak_transform(image))
        strong_tensors.append(strong_transform(image))

    return original_tensor, weak_tensors, strong_tensors


def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image.

    Args:
        tensor (torch.Tensor): Image tensor

    Returns:
        PIL.Image: PIL Image
    """
    # Convert tensor to numpy array and then to PIL Image
    np_image = tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC

    # Clip values to [0, 1] and then scale to [0, 255]
    np_image = np.clip(np_image, 0, 1) * 255
    np_image = np_image.astype(np.uint8)

    return Image.fromarray(np_image)


def save_examples(samples, output_dir, weak_transform, strong_transform, image_size=IMAGE_SIZE):
    """
    Save example images with their weak and strong augmentations.

    Args:
        samples (list): List of sample dictionaries
        output_dir (Path): Output directory path
        weak_transform: Weak transformation pipeline
        strong_transform: Strong transformation pipeline
        image_size (tuple): Size to resize images to
    """
    for i, sample in enumerate(samples):
        # Load image
        try:
            image = Image.open(sample["path"])
            image = image.convert("RGB")

            # Resize if needed
            if image_size:
                image = image.resize(image_size, Image.BILINEAR)

            # Apply transformations (get 3 versions of each)
            original_tensor, weak_tensors, strong_tensors = apply_transforms(
                image, weak_transform, strong_transform, num_versions=3
            )

            # Convert back to PIL images for saving
            original_image = tensor_to_image(original_tensor)

            # Generate base filename
            class_type = "anomaly" if sample["is_anomaly"] else "nominal"
            base_filename = f"{class_type}_{i + 1}"

            # Save original image (only once)
            original_image.save(os.path.join(output_dir, f"{base_filename}_original.png"))
            save_annotated_image(
                original_image,
                os.path.join(output_dir, f"{base_filename}_original_annotated.png"),
                sample["class_name"],
                sample["is_anomaly"],
            )

            # Save multiple versions of weak and strong augmentations
            for version, (weak_tensor, strong_tensor) in enumerate(
                zip(weak_tensors, strong_tensors), 1
            ):
                # Convert tensors to images
                weak_image = tensor_to_image(weak_tensor)
                strong_image = tensor_to_image(strong_tensor)

                # Save images with version numbers
                weak_image.save(os.path.join(output_dir, f"{base_filename}_weak_v{version}.png"))
                strong_image.save(
                    os.path.join(output_dir, f"{base_filename}_strong_v{version}.png")
                )

                # Save annotated versions
                save_annotated_image(
                    weak_image,
                    os.path.join(output_dir, f"{base_filename}_weak_v{version}_annotated.png"),
                    f"{sample['class_name']} (weak v{version})",
                    sample["is_anomaly"],
                )
                save_annotated_image(
                    strong_image,
                    os.path.join(output_dir, f"{base_filename}_strong_v{version}_annotated.png"),
                    f"{sample['class_name']} (strong v{version})",
                    sample["is_anomaly"],
                )

            print(f"Saved examples for {base_filename} (3 versions)")

        except Exception as e:
            print(f"Error processing {sample['path']}: {e}")


def save_annotated_image(image, output_path, class_name, is_anomaly):
    """
    Save image with class annotation overlay.

    Args:
        image (PIL.Image): Image to annotate
        output_path (str): Path to save the annotated image
        class_name (str): Class name to display
        is_anomaly (bool): Whether this is an anomaly class
    """
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIGURE_DPI)

    # Display the image
    ax.imshow(img_array)

    # Set background color based on anomaly status
    bg_color = "red" if is_anomaly else "black"

    # Create background rectangle for text
    rect = Rectangle((0, 0), img_array.shape[1], 20, color=bg_color, alpha=0.6)
    ax.add_patch(rect)

    # Add text annotation
    ax.text(
        img_array.shape[1] / 2,
        10,
        class_name,
        color="white",
        fontsize=40,
        ha="center",
        va="center",
    )

    # Remove axes and white space
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save and close
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    """Main function to generate example images."""
    # Get the images directory from command line args or use default
    if len(sys.argv) > 1:
        images_dir = sys.argv[1]
    else:
        images_dir = "hst"  # Default directory

    # Create output directory
    output_dir = setup_output_directories()

    # Get sample images
    anomaly_samples, normal_samples = get_sample_images(images_dir)
    print(f"Found {len(anomaly_samples)} anomaly samples and {len(normal_samples)} normal samples")

    # Create transformations
    weak_transform, strong_transform = create_augmentations()

    # Generate and save example images
    save_examples(anomaly_samples, output_dir, weak_transform, strong_transform)
    save_examples(normal_samples, output_dir, weak_transform, strong_transform)

    print(f"All example images saved to {output_dir}")


if __name__ == "__main__":
    main()
