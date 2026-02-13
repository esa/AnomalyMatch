#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import numpy as np
import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.utils.get_default_cfg import get_default_cfg
from prediction_utils import convert_cutana_cutout, create_cutana_format_cfg


@pytest.fixture
def format_cfg():
    """Create a CONVERSION_ONLY format config for testing."""
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    cfg.num_workers = 0
    return create_cutana_format_cfg(cfg)


class TestCreateCutanaFormatCfg:
    """Tests for create_cutana_format_cfg."""

    def test_returns_config_with_conversion_only(self):
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [64, 64]
        cfg.normalisation.n_output_channels = 3
        cfg.num_workers = 0
        result = create_cutana_format_cfg(cfg)

        assert result.fitsbolt_cfg.normalisation_method == NormalisationMethod.CONVERSION_ONLY

    def test_preserves_image_size(self):
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [128, 128]
        cfg.normalisation.n_output_channels = 3
        cfg.num_workers = 0
        result = create_cutana_format_cfg(cfg)

        assert result.fitsbolt_cfg.size == [128, 128]

    def test_preserves_output_channels(self):
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [64, 64]
        cfg.normalisation.n_output_channels = 3
        cfg.num_workers = 0
        result = create_cutana_format_cfg(cfg)

        assert result.fitsbolt_cfg.n_output_channels == 3


class TestConvertCutanaCutout:
    """Tests for convert_cutana_cutout CHW/HWC handling and format conversion."""

    def test_hwc_input_preserved(self, format_cfg):
        """HWC input should pass through without transpose."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = convert_cutana_cutout(image, format_cfg)

        assert result.shape[0] == 64  # H
        assert result.shape[1] == 64  # W

    def test_chw_input_transposed_to_hwc(self, format_cfg):
        """CHW input should be transposed to HWC before processing."""
        # Create a CHW image (3, 64, 64) with distinct channel values
        image = np.zeros((3, 64, 64), dtype=np.uint8)
        image[0] = 100  # R channel
        image[1] = 150  # G channel
        image[2] = 200  # B channel

        result = convert_cutana_cutout(image, format_cfg)

        # Result should be HWC
        assert result.ndim == 3
        assert result.shape[2] == 3  # channels last

    def test_single_channel_chw_transposed(self, format_cfg):
        """Single-channel CHW (1, H, W) should be detected and transposed."""
        image = np.random.randint(0, 255, (1, 64, 64), dtype=np.uint8)
        result = convert_cutana_cutout(image, format_cfg)

        assert result.ndim == 3
        assert result.shape[2] == 3  # replicated to 3 channels by fitsbolt

    def test_list_input_converted_to_array(self, format_cfg):
        """Non-ndarray input should be converted before processing."""
        image_list = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8).tolist()
        result = convert_cutana_cutout(image_list, format_cfg)

        assert isinstance(result, np.ndarray)

    def test_float_input_converted_to_uint8(self, format_cfg):
        """Float input (from cutana normalisation) should be converted to uint8."""
        image = np.random.random((64, 64, 3)).astype(np.float32)
        result = convert_cutana_cutout(image, format_cfg)

        assert result.dtype == np.uint8

    def test_output_matches_configured_size(self):
        """Output should be resized to the configured image_size."""
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [32, 32]
        cfg.normalisation.n_output_channels = 3
        cfg.num_workers = 0
        small_cfg = create_cutana_format_cfg(cfg)

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = convert_cutana_cutout(image, small_cfg)

        assert result.shape[0] == 32
        assert result.shape[1] == 32
