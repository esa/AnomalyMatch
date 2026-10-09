#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Generate small test datasets for prediction feature testing.

Creates Zarr test data from the first 100 images in data/small_hdf5/. The
cutana catalogue test data is generated separately by
tests/test_data/cutana_catalogue/generate_test_data.py, and the shared model
checkpoint by tests/test_data/generate_test_model.py.

Usage:
    python scripts/generate_prediction_test_data.py
"""

import io
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

N_IMAGES = 100
OUTPUT_DIR = REPO_ROOT / "tests" / "test_data"
SOURCE_HDF5 = REPO_ROOT / "data" / "small_hdf5"


def find_source_hdf5() -> Path:
    """Find the batch_0 HDF5 file in the source directory.

    Returns:
        Path to the first batch_0 HDF5 file found.

    Raises:
        FileNotFoundError: If no batch_0 HDF5 files exist in the source directory.
    """
    candidates = sorted(SOURCE_HDF5.glob("batch_0_*.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"No batch_0_*.h5 files found in {SOURCE_HDF5}. "
            "Download the small_hdf5 test data first."
        )
    return candidates[0]


def extract_images(source_path: Path) -> tuple[list[bytes], list[str], np.ndarray]:
    """Extract first N_IMAGES from the source HDF5 file.

    Returns:
        Tuple of (jpeg_bytes_list, filename_list, decoded_pixels_array).
    """
    print(f"Reading {N_IMAGES} images from {source_path.name}...")
    with h5py.File(source_path, "r") as h5f:
        raw_images = h5f["images"][:N_IMAGES]
        raw_filenames = h5f["filenames"][:N_IMAGES]

    jpeg_bytes_list = [bytes(img) for img in raw_images]
    filenames = [fn.decode("utf-8") if isinstance(fn, bytes) else fn for fn in raw_filenames]

    # Decode JPEGs to pixel arrays for Zarr
    pixels = []
    for i, jpg_bytes in enumerate(jpeg_bytes_list):
        img = Image.open(io.BytesIO(jpg_bytes))
        pixels.append(np.array(img))
    pixels_array = np.stack(pixels)  # (N, H, W, 3)
    print(f"  Decoded pixel array shape: {pixels_array.shape}")

    return jpeg_bytes_list, filenames, pixels_array


def create_zarr(pixels_array: np.ndarray, filenames: list[str]) -> None:
    """Create test_images.zarr with raw pixel data and metadata parquet."""
    zarr_path = OUTPUT_DIR / "zarr" / "test_images.zarr"
    parquet_path = OUTPUT_DIR / "zarr" / "test_images_metadata.parquet"

    print(f"Creating {zarr_path.name}...")
    root = zarr.open_group(str(zarr_path), mode="w")
    arr = root.create_array(
        "images",
        shape=pixels_array.shape,
        chunks=(10, pixels_array.shape[1], pixels_array.shape[2], pixels_array.shape[3]),
        dtype=pixels_array.dtype,
    )
    arr[:] = pixels_array

    size_mb = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"  Created {zarr_path.name} ({size_mb:.1f} MB, shape {pixels_array.shape})")

    print(f"Creating {parquet_path.name}...")
    df = pd.DataFrame({"filename": filenames})
    df.to_parquet(parquet_path, index=False)
    print(f"  Created {parquet_path.name} ({len(df)} rows)")


def create_cutana_catalogue() -> None:
    """Generate cutana catalogue test data (FITS tile + CSV).

    Delegates to tests/test_data/cutana_catalogue/generate_test_data.py
    which creates a self-contained FITS tile and catalogue CSV.
    """
    import subprocess

    script = REPO_ROOT / "tests" / "test_data" / "cutana_catalogue" / "generate_test_data.py"
    print(f"Running {script.relative_to(REPO_ROOT)}...")
    subprocess.check_call([sys.executable, str(script)], cwd=str(REPO_ROOT))


def main():
    """Generate all prediction test data."""
    print("=" * 60)
    print("Generating prediction test data")
    print("=" * 60)

    source_path = find_source_hdf5()
    jpeg_bytes_list, filenames, pixels_array = extract_images(source_path)

    create_zarr(pixels_array, filenames)
    create_cutana_catalogue()

    print()
    print("=" * 60)
    print("Done! Generated files:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file() and p.name != "generate_test_data.py":
            rel = p.relative_to(OUTPUT_DIR)
            size = p.stat().st_size
            if size > 1024 * 1024:
                print(f"  {rel} ({size / (1024 * 1024):.1f} MB)")
            else:
                print(f"  {rel} ({size / 1024:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
