#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Shared pytest fixtures for AnomalyMatch tests."""

import os

import pytest


@pytest.fixture(scope="function")
def multispectral_test_data():
    """Provide 4-channel test images from permanent test data.

    Uses pre-generated 4-channel TIFF images from tests/test_data/multispectral_4ch/

    Yields:
        tuple: (data_dir, image_filenames, labeled_data_path, num_channels)
            - data_dir: Path to directory containing 4-channel test images
            - image_filenames: List of image filenames
            - labeled_data_path: Path to the labeled_data.csv file
            - num_channels: Number of channels (4)
    """
    # Get path to 4-channel test images
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data", "multispectral_4ch")

    if not os.path.exists(test_data_dir):
        pytest.skip(
            "4-channel test data not found. Run 'python scripts/generate_4ch_test_data.py' first."
        )

    # Get all .tiff files (the format that supports 4+ channels)
    tiff_images = [f for f in os.listdir(test_data_dir) if f.endswith(".tiff")]

    if not tiff_images:
        pytest.skip("No .tiff test images found in tests/test_data/multispectral_4ch/")

    csv_path = os.path.join(test_data_dir, "labeled_data.csv")
    if not os.path.exists(csv_path):
        pytest.skip("labeled_data.csv not found in tests/test_data/multispectral_4ch/")

    yield test_data_dir, sorted(tiff_images), csv_path, 4


@pytest.fixture(scope="function")
def multispectral_config(multispectral_test_data):
    """Create a configuration for multispectral testing.

    Args:
        multispectral_test_data: The multispectral test data fixture.

    Yields:
        DotMap: Configuration object set up for 4-channel images.
    """
    import anomaly_match as am

    data_dir, filenames, csv_path, num_channels = multispectral_test_data

    cfg = am.get_default_cfg()
    cfg.data_dir = data_dir
    cfg.normalisation.image_size = [64, 64]
    cfg.net = "test-cnn"
    cfg.pretrained = False
    cfg.num_train_iter = 2
    cfg.num_workers = 0
    cfg.test_ratio = 0.3
    cfg.N_to_load = 10
    cfg.normalisation.fits_extension = None
    cfg.label_file = csv_path
    cfg.seed = 42
    # n_output_channels is auto-detected from images by AnomalyDetectionDataset

    yield cfg
