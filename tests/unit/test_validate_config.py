#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for configuration validation edge cases."""

import pytest
from loguru import logger

from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.utils.validate_config import (
    _get_all_keys,
    _get_nested_value,
    validate_config,
)


@pytest.fixture
def caplog(caplog):
    """Configure loguru to use the caplog handler."""
    handler_id = logger.add(caplog.handler)
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def valid_cfg():
    """Return a valid default config with image_size set."""
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [64, 64]
    return cfg


class TestGetNestedValue:
    def test_simple_key(self, valid_cfg):
        assert _get_nested_value(valid_cfg, "seed") == 42

    def test_nested_key(self, valid_cfg):
        assert _get_nested_value(valid_cfg, "normalisation.image_size") == [64, 64]

    def test_missing_key_raises(self, valid_cfg):
        with pytest.raises(ValueError, match="Missing key in config"):
            _get_nested_value(valid_cfg, "nonexistent.key")


class TestGetAllKeys:
    def test_returns_top_level_keys(self, valid_cfg):
        keys = _get_all_keys(valid_cfg)
        assert "seed" in keys
        assert "batch_size" in keys
        assert "net" in keys

    def test_returns_nested_keys(self, valid_cfg):
        keys = _get_all_keys(valid_cfg)
        assert "normalisation" in keys
        assert "normalisation.image_size" in keys
        assert "normalisation.n_output_channels" in keys


class TestValidateConfigRequired:
    def test_missing_required_string(self, valid_cfg):
        del valid_cfg["name"]
        with pytest.raises(ValueError, match="Missing required parameter"):
            validate_config(valid_cfg)

    def test_missing_required_integer(self, valid_cfg):
        del valid_cfg["batch_size"]
        with pytest.raises(ValueError, match="Missing required parameter"):
            validate_config(valid_cfg)


class TestValidateConfigTypes:
    def test_string_type_mismatch(self, valid_cfg):
        valid_cfg.name = 123
        with pytest.raises(ValueError, match="must be a string"):
            validate_config(valid_cfg)

    def test_int_type_mismatch(self, valid_cfg):
        valid_cfg.batch_size = "not_an_int"
        with pytest.raises(ValueError, match="must be an integer"):
            validate_config(valid_cfg)

    def test_float_type_mismatch(self, valid_cfg):
        valid_cfg.test_ratio = "not_a_float"
        with pytest.raises(ValueError, match="must be a number"):
            validate_config(valid_cfg)

    def test_bool_type_mismatch(self, valid_cfg):
        valid_cfg.pin_memory = "not_a_bool"
        with pytest.raises(ValueError, match="must be a boolean"):
            validate_config(valid_cfg)


class TestValidateConfigRanges:
    def test_int_below_minimum(self, valid_cfg):
        valid_cfg.batch_size = 0
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_config(valid_cfg)

    def test_float_below_minimum(self, valid_cfg):
        valid_cfg.test_ratio = -0.1
        with pytest.raises(ValueError, match="must be >= 0.0"):
            validate_config(valid_cfg)

    def test_float_above_maximum(self, valid_cfg):
        valid_cfg.test_ratio = 1.5
        with pytest.raises(ValueError, match="must be <= 1.0"):
            validate_config(valid_cfg)

    def test_n_to_load_below_minimum(self, valid_cfg):
        valid_cfg.N_to_load = 5
        with pytest.raises(ValueError, match="must be >= 10"):
            validate_config(valid_cfg)


class TestValidateConfigAllowedValues:
    def test_invalid_optimizer(self, valid_cfg):
        valid_cfg.opt = "RMSProp"
        with pytest.raises(ValueError, match="must be one of"):
            validate_config(valid_cfg)

    def test_invalid_net(self, valid_cfg):
        valid_cfg.net = "resnet50"
        with pytest.raises(ValueError, match="must be one of"):
            validate_config(valid_cfg)

    def test_valid_optimizer_sgd(self, valid_cfg):
        valid_cfg.opt = "SGD"
        validate_config(valid_cfg)

    def test_valid_optimizer_adam(self, valid_cfg):
        valid_cfg.opt = "Adam"
        validate_config(valid_cfg)


class TestValidateConfigSpecialTypes:
    def test_invalid_image_size_not_list(self, valid_cfg):
        valid_cfg.normalisation.image_size = 64
        with pytest.raises(ValueError, match="must be a list or tuple of length 2"):
            validate_config(valid_cfg)

    def test_invalid_image_size_wrong_length(self, valid_cfg):
        valid_cfg.normalisation.image_size = [64, 64, 64]
        with pytest.raises(ValueError, match="must be a list or tuple of length 2"):
            validate_config(valid_cfg)

    def test_invalid_eval_iter(self, valid_cfg):
        valid_cfg.num_eval_iter = 0
        with pytest.raises(ValueError, match="must be an integer > 0 or -1"):
            validate_config(valid_cfg)

    def test_valid_eval_iter_negative_one(self, valid_cfg):
        valid_cfg.num_eval_iter = -1
        validate_config(valid_cfg)

    def test_valid_eval_iter_positive(self, valid_cfg):
        valid_cfg.num_eval_iter = 10
        validate_config(valid_cfg)

    def test_normalisation_not_dotmap(self, valid_cfg):
        valid_cfg.normalisation = "not_a_dotmap"
        with pytest.raises(ValueError, match="must be a DotMap"):
            validate_config(valid_cfg)


class TestValidateConfigPaths:
    def test_skip_path_checks(self, valid_cfg):
        valid_cfg.data_dir = "/nonexistent/path"
        validate_config(valid_cfg, check_paths=False)

    def test_invalid_directory_path(self, valid_cfg):
        valid_cfg.data_dir = "/definitely/nonexistent/path"
        with pytest.raises(ValueError, match="directory does not exist"):
            validate_config(valid_cfg, check_paths=True)

    def test_invalid_file_path(self, valid_cfg):
        valid_cfg.label_file = "/nonexistent/file.csv"
        with pytest.raises(ValueError, match="file does not exist"):
            validate_config(valid_cfg, check_paths=True)


class TestValidateConfigOptional:
    def test_optional_none_metadata_file(self, valid_cfg):
        valid_cfg.metadata_file = None
        validate_config(valid_cfg)

    def test_optional_none_prediction_search_dir(self, valid_cfg):
        valid_cfg.prediction_search_dir = None
        validate_config(valid_cfg)


class TestFitsExtensionChannelAutoAdjust:
    """Regression tests for auto-adjusting n_output_channels from fits_extension."""

    def test_fits_extension_4_auto_adjusts_n_output_channels(self, valid_cfg):
        """fits_extension=[0,1,2,3] with default n_output_channels=3 should auto-adjust to 4."""
        import numpy as np

        valid_cfg.normalisation.fits_extension = [0, 1, 2, 3]
        assert valid_cfg.normalisation.n_output_channels == 3
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 4
        assert valid_cfg.num_channels == 4
        np.testing.assert_array_equal(valid_cfg.normalisation.channel_combination, np.eye(4))

    def test_fits_extension_matching_channels_creates_identity(self, valid_cfg):
        """fits_extension=[0,1,2] with n_output_channels=3 should create identity matrix."""
        import numpy as np

        valid_cfg.normalisation.fits_extension = [0, 1, 2]
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 3
        np.testing.assert_array_equal(valid_cfg.normalisation.channel_combination, np.eye(3))

    def test_fits_extension_single_no_adjustment(self, valid_cfg):
        """Single fits_extension should not trigger adjustment."""
        valid_cfg.normalisation.fits_extension = [0]
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 3

    def test_channel_combination_provided_no_adjustment(self, valid_cfg):
        """Explicit channel_combination should prevent auto-adjustment."""
        import numpy as np

        valid_cfg.normalisation.fits_extension = [0, 1, 2, 3]
        valid_cfg.normalisation.channel_combination = np.eye(3, 4)
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 3

    def test_asinh_params_extended_on_adjustment(self, valid_cfg):
        """Per-channel asinh params should be extended when channels are added."""
        valid_cfg.normalisation.fits_extension = [0, 1, 2, 3]
        valid_cfg.normalisation.norm_asinh_scale = [0.7, 0.7, 0.7]
        valid_cfg.normalisation.norm_asinh_clip = [99.8, 99.8, 99.8]
        validate_config(valid_cfg)
        assert len(valid_cfg.normalisation.norm_asinh_scale) == 4
        assert len(valid_cfg.normalisation.norm_asinh_clip) == 4

    def test_n_output_channels_inferred_from_channel_combination(self, valid_cfg):
        """n_output_channels should be inferred from channel_combination shape."""
        import numpy as np

        valid_cfg.normalisation.fits_extension = [0, 1, 2, 3]
        # User provides 1x4 matrix but doesn't set n_output_channels (defaults to 3)
        valid_cfg.normalisation.channel_combination = np.array([[0.25, 0.25, 0.25, 0.25]])
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 1
        assert valid_cfg.num_channels == 1
        assert len(valid_cfg.normalisation.norm_asinh_scale) == 1
        assert len(valid_cfg.normalisation.norm_asinh_clip) == 1

    def test_n_output_channels_inferred_3x4_matrix(self, valid_cfg):
        """3x4 channel_combination with default n_output_channels=3 should not change."""
        import numpy as np

        valid_cfg.normalisation.fits_extension = [0, 1, 2, 3]
        valid_cfg.normalisation.channel_combination = np.eye(3, 4)
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 3

    def test_n_output_channels_none_with_multi_ext_auto_infers(self, valid_cfg):
        """n_output_channels=None with multiple fits_extension should auto-infer."""
        import numpy as np

        valid_cfg.normalisation.fits_extension = [0, 1, 2, 3]
        valid_cfg.normalisation.n_output_channels = None
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 4
        np.testing.assert_array_equal(valid_cfg.normalisation.channel_combination, np.eye(4))

    def test_n_output_channels_none_single_ext_raises(self, valid_cfg):
        """n_output_channels=None with single fits_extension should raise ValueError."""
        valid_cfg.normalisation.fits_extension = [0]
        valid_cfg.normalisation.n_output_channels = None
        with pytest.raises(ValueError, match="n_output_channels is None"):
            validate_config(valid_cfg)


class TestValidateConfigUnexpectedKeys:
    def test_warns_on_unexpected_keys(self, valid_cfg, caplog):
        valid_cfg.unexpected_key = "some_value"
        validate_config(valid_cfg)
        assert "Found unexpected keys in config" in caplog.text
