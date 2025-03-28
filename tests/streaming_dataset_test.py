#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import pytest
import numpy as np
import os
import torch
import imageio.v2 as imageio
from anomaly_match.datasets.StreamingDataset import StreamingDataset


@pytest.fixture
def sample_files(tmp_path):
    """Create sample image files for testing."""
    # Create test images in tmp_path
    file_list = []
    for i in range(10):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        filepath = os.path.join(tmp_path, f"test_img_{i}.jpeg")
        imageio.imwrite(filepath, img)
        file_list.append(filepath)
    return file_list


@pytest.fixture
def streaming_dataset(sample_files):
    return StreamingDataset(
        file_list=sample_files,
        size=[64, 64],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        prefetch_size=5,
    )


def test_streaming_dataset_initialization(streaming_dataset):
    assert streaming_dataset.size == [64, 64]
    assert len(streaming_dataset.file_list) > 0


def test_streaming_dataset_iteration(streaming_dataset):
    item = next(iter(streaming_dataset))
    img_tensor, label, filename = item
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 64, 64)
    assert isinstance(filename, str)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Force cleanup of any remaining threads
    import gc

    gc.collect()
