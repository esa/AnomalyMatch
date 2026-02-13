#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for display transform functions."""

import numpy as np
import pytest
from PIL import Image

from anomaly_match_ui.utils.display_transforms import (
    apply_transforms_ui,
    display_image_normalisation,
    prepare_for_display,
)


class TestPrepareForDisplay:
    def test_rgb_passthrough(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = prepare_for_display(img)
        assert result.shape == (64, 64, 3)
        assert np.array_equal(result, img)

    def test_grayscale_to_rgb(self):
        img = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
        result = prepare_for_display(img)
        assert result.shape == (64, 64, 3)
        # All channels should be the same
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])

    def test_2d_grayscale(self):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = prepare_for_display(img)
        assert result.shape == (64, 64, 3)

    def test_2_channel_to_rgb(self):
        img = np.random.randint(0, 255, (64, 64, 2), dtype=np.uint8)
        result = prepare_for_display(img)
        assert result.shape == (64, 64, 3)

    def test_4_channel_default_mapping(self):
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = prepare_for_display(img)
        assert result.shape == (64, 64, 3)
        # Default mapping uses first 3 channels
        assert np.array_equal(result, img[:, :, :3])

    def test_4_channel_custom_mapping(self):
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = prepare_for_display(img, rgb_mapping=[1, 2, 3])
        assert result.shape == (64, 64, 3)
        assert np.array_equal(result[:, :, 0], img[:, :, 1])
        assert np.array_equal(result[:, :, 1], img[:, :, 2])
        assert np.array_equal(result[:, :, 2], img[:, :, 3])

    def test_invalid_rgb_mapping_length(self):
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="must have 3 elements"):
            prepare_for_display(img, rgb_mapping=[0, 1])

    def test_invalid_rgb_mapping_index(self):
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="exceed channel count"):
            prepare_for_display(img, rgb_mapping=[0, 1, 5])

    def test_float_to_uint8_conversion(self):
        img = np.random.random((64, 64, 3)).astype(np.float32)
        result = prepare_for_display(img)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_pil_image_input(self):
        pil_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = prepare_for_display(pil_img)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_invalid_input_type(self):
        with pytest.raises(ValueError, match="Expected numpy array or PIL Image"):
            prepare_for_display("not_an_image")


class TestDisplayImageNormalisation:
    def test_basic_normalisation(self):
        img = np.random.random((64, 64, 3)).astype(np.float64) * 255
        result = display_image_normalisation(img)
        assert isinstance(result, Image.Image)

    def test_constant_image(self):
        img = np.full((64, 64, 3), 0.5, dtype=np.float64)
        result = display_image_normalisation(img)
        assert isinstance(result, Image.Image)

    def test_handles_nan(self):
        img = np.random.random((64, 64, 3))
        img[10, 10, 0] = np.nan
        result = display_image_normalisation(img)
        assert isinstance(result, Image.Image)

    def test_handles_inf(self):
        img = np.random.random((64, 64, 3))
        img[10, 10, 0] = np.inf
        result = display_image_normalisation(img)
        assert isinstance(result, Image.Image)


class TestApplyTransformsUI:
    @pytest.fixture
    def sample_pil_image(self):
        return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

    def test_no_transforms(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=False,
            brightness=1.0,
            contrast=1.0,
            unsharp_mask_applied=False,
        )
        assert isinstance(result, Image.Image)

    def test_invert(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=True,
            brightness=1.0,
            contrast=1.0,
            unsharp_mask_applied=False,
        )
        assert isinstance(result, Image.Image)

    def test_brightness(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=False,
            brightness=1.5,
            contrast=1.0,
            unsharp_mask_applied=False,
        )
        assert isinstance(result, Image.Image)

    def test_contrast(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=False,
            brightness=1.0,
            contrast=1.5,
            unsharp_mask_applied=False,
        )
        assert isinstance(result, Image.Image)

    def test_unsharp_mask(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=False,
            brightness=1.0,
            contrast=1.0,
            unsharp_mask_applied=True,
        )
        assert isinstance(result, Image.Image)

    def test_channel_toggling_hide_red(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=False,
            brightness=1.0,
            contrast=1.0,
            unsharp_mask_applied=False,
            show_r=False,
        )
        result_array = np.array(result)
        assert np.all(result_array[:, :, 0] == 0)

    def test_channel_visibility_list(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=False,
            brightness=1.0,
            contrast=1.0,
            unsharp_mask_applied=False,
            channel_visibility=[True, False, True],
        )
        result_array = np.array(result)
        assert np.all(result_array[:, :, 1] == 0)

    def test_all_transforms_combined(self, sample_pil_image):
        result = apply_transforms_ui(
            sample_pil_image,
            invert=True,
            brightness=1.2,
            contrast=0.8,
            unsharp_mask_applied=True,
        )
        assert isinstance(result, Image.Image)
