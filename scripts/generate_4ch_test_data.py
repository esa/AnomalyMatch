#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Generate 4-channel test images with distinct per-channel content.

Each channel has a different geometric pattern so multispectral images
are visually distinguishable from grayscale:
  Ch0: Concentric circles (radial gradient)
  Ch1: Diagonal stripes
  Ch2: Checkerboard pattern
  Ch3: Gaussian blob (random position per image)
"""

import os

import numpy as np
import pandas as pd
import tifffile


def make_radial_gradient(size, center_x, center_y, scale=1.0):
    """Concentric circles centered at (center_x, center_y)."""
    y, x = np.mgrid[0:size, 0:size]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return np.clip((np.cos(dist * scale * 0.1) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)


def make_diagonal_stripes(size, frequency=0.05, angle=0.0):
    """Diagonal stripe pattern."""
    y, x = np.mgrid[0:size, 0:size]
    val = np.sin(2 * np.pi * frequency * (x * np.cos(angle) + y * np.sin(angle)))
    return np.clip((val * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)


def make_checkerboard(size, block_size=16):
    """Checkerboard pattern."""
    y, x = np.mgrid[0:size, 0:size]
    checker = ((x // block_size) + (y // block_size)) % 2
    return (checker * 255).astype(np.uint8)


def make_gaussian_blob(size, cx, cy, sigma=20, intensity=200):
    """Gaussian blob at (cx, cy)."""
    y, x = np.mgrid[0:size, 0:size]
    blob = intensity * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return np.clip(blob, 0, 255).astype(np.uint8)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    ms_subdir = os.path.join(repo_root, "tests", "test_data", "multispectral_4ch")
    os.makedirs(ms_subdir, exist_ok=True)

    # Clean old files
    for f in os.listdir(ms_subdir):
        os.remove(os.path.join(ms_subdir, f))

    size = 64
    n_images = 10
    np.random.seed(42)

    source_names = [
        "Abell2390_VIS_2",
        "Abell2390_VIS_3",
        "Abell2390_VIS_5",
        "Abell2390_VIS_6",
        "Abell2390_VIS_2021",
        "Abell2390_VIS_2172",
        "Abell2390_VIS_4989",
        "Abell2390_VIS_5260",
        "Abell2390_VIS_5783",
        "Abell2390_VIS_6212",
    ]

    generated_filenames = []

    for i, name in enumerate(source_names[:n_images]):
        # Each image gets slightly different parameters for variety
        cx = np.random.randint(size // 4, 3 * size // 4)
        cy = np.random.randint(size // 4, 3 * size // 4)
        angle = np.random.uniform(0, np.pi)
        block = np.random.choice([8, 12, 16])

        ch0 = make_radial_gradient(size, cx, cy, scale=1.0 + i * 0.3)
        ch1 = make_diagonal_stripes(size, frequency=0.04 + i * 0.01, angle=angle)
        ch2 = make_checkerboard(size, block_size=block)
        ch3 = make_gaussian_blob(size, cx, cy, sigma=10 + i * 2)

        ms_img = np.stack([ch0, ch1, ch2, ch3], axis=-1)  # (H, W, 4)

        filename = f"{name}_4ch.tiff"
        tifffile.imwrite(os.path.join(ms_subdir, filename), ms_img)
        generated_filenames.append(filename)
        print(f"  Created {filename} with shape {ms_img.shape}")

    # Create labeled_data.csv - label first 6, leave 4 unlabeled
    labeled_filenames = generated_filenames[:6]
    labels = ["anomaly"] * 3 + ["normal"] * 3
    df = pd.DataFrame({"filename": labeled_filenames, "label": labels})
    df.to_csv(os.path.join(ms_subdir, "labeled_data.csv"), index=False)
    print(f"Created labeled_data.csv with {len(labeled_filenames)} labeled entries")

    # Create metadata.csv for all images
    metadata_rows = []
    base_ra, base_dec = 328.4034, 17.6950
    for i, filename in enumerate(generated_filenames):
        metadata_rows.append(
            {
                "filename": filename,
                "sourceID": f"MS_{i:05d}",
                "ra": base_ra + i * 0.002,
                "dec": base_dec + i * 0.001,
                "custom_metadata": f"4ch test source {i}",
            }
        )
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(os.path.join(ms_subdir, "metadata.csv"), index=False)
    print(f"Created metadata.csv with {len(metadata_rows)} entries")

    print(f"\nDone! Test data saved to: {ms_subdir}")


if __name__ == "__main__":
    main()
