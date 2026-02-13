#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for data loading utilities."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anomaly_match.datasets.BasicDataset import BasicDataset
from anomaly_match.datasets.data_utils import get_data_loader, get_sampler_by_name
from anomaly_match.image_processing.transforms import get_prediction_transforms


@pytest.fixture
def simple_dataset():
    """Create a simple BasicDataset for testing."""
    imgs = np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8)
    filenames = [f"img_{i}.jpg" for i in range(20)]
    targets = [0] * 10 + [1] * 10
    transform = get_prediction_transforms(num_channels=3)
    return BasicDataset(imgs, filenames, targets, num_classes=2, transform=transform)


class TestGetSamplerByName:
    def test_random_sampler(self):
        sampler = get_sampler_by_name("RandomSampler")
        assert sampler is torch.utils.data.sampler.RandomSampler

    def test_sequential_sampler(self):
        sampler = get_sampler_by_name("SequentialSampler")
        assert sampler is torch.utils.data.sampler.SequentialSampler

    def test_invalid_sampler_raises(self):
        with pytest.raises(AttributeError, match="not found"):
            get_sampler_by_name("NonexistentSampler")


class TestGetDataLoader:
    def test_requires_batch_size(self, simple_dataset):
        with pytest.raises(AssertionError, match="Batch size must be specified"):
            get_data_loader(simple_dataset, batch_size=None)

    def test_basic_dataloader(self, simple_dataset):
        loader = get_data_loader(simple_dataset, batch_size=4, num_workers=0)
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 4

    def test_shuffle_dataloader(self, simple_dataset):
        loader = get_data_loader(simple_dataset, batch_size=4, shuffle=True, num_workers=0)
        assert isinstance(loader, DataLoader)

    def test_weighted_sampler(self, simple_dataset):
        loader = get_data_loader(
            simple_dataset,
            batch_size=4,
            use_weighted_sampler=True,
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)

    def test_weighted_sampler_with_num_iters(self, simple_dataset):
        loader = get_data_loader(
            simple_dataset,
            batch_size=4,
            use_weighted_sampler=True,
            num_iters=10,
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)

    def test_weighted_sampler_with_num_epochs(self, simple_dataset):
        loader = get_data_loader(
            simple_dataset,
            batch_size=4,
            use_weighted_sampler=True,
            num_epochs=2,
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)

    def test_random_sampler_by_name(self, simple_dataset):
        loader = get_data_loader(
            simple_dataset,
            batch_size=4,
            data_sampler="RandomSampler",
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)

    def test_unsupported_sampler_raises(self, simple_dataset):
        with pytest.raises(RuntimeError, match="not fully implemented"):
            get_data_loader(
                simple_dataset,
                batch_size=4,
                data_sampler="SequentialSampler",
                num_workers=0,
            )


class TestWeightedSamplerSingleClass:
    def test_single_class_uniform_weights(self):
        """Weighted sampler with only one class should use uniform weights."""
        imgs = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        filenames = [f"img_{i}.jpg" for i in range(10)]
        targets = [0] * 10  # All same class
        transform = get_prediction_transforms(num_channels=3)
        dataset = BasicDataset(imgs, filenames, targets, num_classes=2, transform=transform)
        loader = get_data_loader(
            dataset,
            batch_size=4,
            use_weighted_sampler=True,
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)
