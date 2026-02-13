#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for image transformation pipelines."""

import numpy as np
import torch

from anomaly_match.image_processing.transforms import (
    NumpyRandomHorizontalFlip,
    NumpyRandomTranslate,
    NumpyToTensor,
    get_prediction_transforms,
    get_strong_transforms,
    get_weak_transforms,
)


class TestNumpyToTensor:
    def test_converts_hwc_to_chw(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = NumpyToTensor()(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_normalizes_to_float(self):
        img = np.full((32, 32, 3), 255, dtype=np.uint8)
        tensor = NumpyToTensor()(img)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.ones(3, 32, 32))

    def test_handles_4_channels(self):
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        tensor = NumpyToTensor()(img)
        assert tensor.shape == (4, 64, 64)

    def test_passthrough_non_numpy(self):
        tensor = torch.randn(3, 64, 64)
        result = NumpyToTensor()(tensor)
        assert torch.equal(result, tensor)


class TestNumpyRandomHorizontalFlip:
    def test_output_shape_preserved(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = NumpyRandomHorizontalFlip(p=1.0)(img)
        assert result.shape == img.shape

    def test_flip_with_p_one(self):
        img = np.zeros((4, 4, 1), dtype=np.uint8)
        img[0, 0, 0] = 255
        result = NumpyRandomHorizontalFlip(p=1.0)(img)
        assert result[0, 3, 0] == 255

    def test_no_flip_with_p_zero(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = NumpyRandomHorizontalFlip(p=0.0)(img)
        assert np.array_equal(result, img)

    def test_handles_tensor(self):
        tensor = torch.randn(3, 64, 64)
        result = NumpyRandomHorizontalFlip(p=1.0)(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape


class TestNumpyRandomTranslate:
    def test_output_shape_preserved(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = NumpyRandomTranslate()(img)
        assert result.shape == img.shape

    def test_handles_tensor(self):
        tensor = torch.randn(3, 64, 64)
        result = NumpyRandomTranslate()(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape


class TestGetWeakTransforms:
    def test_rgb_returns_compose(self):
        transform = get_weak_transforms(num_channels=3)
        assert transform is not None

    def test_rgb_output_tensor(self):
        transform = get_weak_transforms(num_channels=3)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3

    def test_multispectral_output_tensor(self):
        transform = get_weak_transforms(num_channels=4)
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 4


class TestGetPredictionTransforms:
    def test_rgb_output_tensor(self):
        transform = get_prediction_transforms(num_channels=3)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 64, 64)

    def test_multispectral_output_tensor(self):
        transform = get_prediction_transforms(num_channels=4)
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 64, 64)


class TestGetStrongTransforms:
    def test_rgb_returns_compose(self):
        transform = get_strong_transforms(num_channels=3)
        assert transform is not None

    def test_rgb_output_tensor(self):
        from PIL import Image

        transform = get_strong_transforms(num_channels=3)
        # RandAugment for RGB expects PIL Image input
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3

    def test_multispectral_output_tensor(self):
        transform = get_strong_transforms(num_channels=4)
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 4
