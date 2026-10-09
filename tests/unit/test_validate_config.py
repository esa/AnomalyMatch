#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for configuration validation edge cases."""

import numpy as np
import pytest
from dotmap import DotMap
from loguru import logger

from anomaly_match.data_io.load_images import get_fitsbolt_config
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

    def test_cutout_padding_factor_accepts_sub_unit_values(self, valid_cfg):
        """Padding factors < 1.0 are valid (range is [0.25, 10.0])."""
        valid_cfg.normalisation.cutout_padding_factor = 0.75
        validate_config(valid_cfg)  # must not raise

    def test_cutout_padding_factor_accepts_minimum(self, valid_cfg):
        valid_cfg.normalisation.cutout_padding_factor = 0.25
        validate_config(valid_cfg)  # must not raise

    def test_cutout_padding_factor_below_minimum_raises(self, valid_cfg):
        valid_cfg.normalisation.cutout_padding_factor = 0.1
        with pytest.raises(ValueError, match="cutout_padding_factor.*must be >= 0.25"):
            validate_config(valid_cfg)

    def test_cutout_padding_factor_above_maximum_raises(self, valid_cfg):
        valid_cfg.normalisation.cutout_padding_factor = 11.0
        with pytest.raises(ValueError, match="cutout_padding_factor.*must be <= 10.0"):
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


class TestValidateAgreesWithFitsboltConfigBuild:
    """A config that validates must also build a fitsbolt config.

    validate_config and get_fitsbolt_config judge the same matrix by different rules,
    so it is possible to make validation pass and then have the run die at
    get_fitsbolt_config inside the training subprocess. These pin the two together.
    """

    @pytest.mark.parametrize(
        "fits_extension",
        [None, [0, 1, 2], ["VIS", "NIR-H", "NIR-J"]],
        ids=["non-fits", "integer-extensions", "named-extensions"],
    )
    @pytest.mark.parametrize(
        "matrix",
        [
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, -0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ],
        ids=["blank-row", "negative-weight"],
    )
    def test_validation_matches_fitsbolt_config_build(self, valid_cfg, fits_extension, matrix):
        """Blank rows pass everywhere; negative weights only where AnomalyMatch combines."""
        valid_cfg.normalisation.fits_extension = fits_extension
        valid_cfg.normalisation.channel_combination = matrix
        fitsbolt_applies_matrix = fits_extension is not None
        if fitsbolt_applies_matrix and (matrix < 0).any():
            with pytest.raises(ValueError, match="negative"):
                validate_config(valid_cfg)
            with pytest.raises(ValueError, match="negative"):
                get_fitsbolt_config(valid_cfg)
            return
        validate_config(valid_cfg)
        get_fitsbolt_config(valid_cfg)


class TestValidateConfigUnexpectedKeys:
    def test_warns_on_unexpected_keys(self, valid_cfg, caplog):
        valid_cfg.unexpected_key = "some_value"
        validate_config(valid_cfg)
        assert "Found unexpected keys in config" in caplog.text


class TestChannelCombinationWithoutFitsExtension:
    """Non-FITS sources (Cutana, Zarr, image folders) leave ``fits_extension`` unset.

    They apply the band-mixing matrix themselves, so fitsbolt is handed ``None`` and
    its validator — which rejects negative weights that are legal on this path —
    never sees the matrix. AnomalyMatch checks the matrix's structure itself instead.
    """

    def test_empty_dotmap_matrix_treated_as_none(self, valid_cfg):
        """``DotMap.copy()`` turns a None channel_combination into an empty DotMap().

        fitsbolt would otherwise report a nonsense shape for it, so the helper must
        collapse anything that is not a real matrix back to None.
        """
        # Single extension: more than one would hit the auto-identity branch and
        # replace the DotMap with np.eye() before the helper ever sees it.
        valid_cfg.normalisation.fits_extension = [0]
        valid_cfg.normalisation.channel_combination = DotMap()
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 3

    @pytest.mark.parametrize(
        "matrix, expected",
        [
            ("not-a-matrix", "must be a numpy array"),
            (np.array([["a", "b"], ["c", "d"]]), "must contain numbers"),
            (np.ones((2, 3, 4)), "must be 2-D"),
            (np.zeros((3, 0)), "at least one row and one column"),
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite"),
            (np.array([[np.inf, 0.0], [0.0, 1.0]]), "finite"),
            ([[1.0, 0.0], [1.0]], "rectangular"),
        ],
    )
    def test_malformed_matrix_rejected(self, valid_cfg, matrix, expected):
        """AnomalyMatch replaces the structural checks fitsbolt no longer performs here.

        Without them a malformed matrix passes config validation and only fails much
        later, inside the decoder.
        """
        valid_cfg.normalisation.n_output_channels = 3
        valid_cfg.normalisation.channel_combination = matrix
        with pytest.raises(ValueError, match=expected):
            validate_config(valid_cfg)

    def test_negative_weights_warn_but_pass(self, valid_cfg, caplog):
        """Negative weights are legal here, unlike under fitsbolt, which rejects them.

        The existing warning about negative weights being clipped only makes sense if
        they are allowed to reach the decoder at all.
        """
        valid_cfg.normalisation.channel_combination = np.array([[1.0, -0.5, 0.0, 0.0]])
        validate_config(valid_cfg)
        assert "negative weights" in caplog.text
        assert valid_cfg.normalisation.n_output_channels == 1

    def test_matrix_still_checked_against_fits_extension(self, valid_cfg):
        """FITS sources still get fitsbolt's column check — this PR must not weaken it.

        Matches fitsbolt's column cross-check specifically: every structural error from
        ``_validate_channel_combination`` also names ``channel_combination``.
        """
        valid_cfg.normalisation.fits_extension = [0, 1]
        valid_cfg.normalisation.channel_combination = np.eye(3, 4)
        with pytest.raises(ValueError, match=r"channel_combination\.shape\[1\]"):
            validate_config(valid_cfg)

    def test_list_matrix_infers_like_ndarray(self, valid_cfg):
        """A nested list and the equivalent ndarray must validate identically."""
        valid_cfg.normalisation.n_output_channels = 3
        valid_cfg.normalisation.channel_combination = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        validate_config(valid_cfg)
        assert valid_cfg.normalisation.n_output_channels == 2
        assert isinstance(valid_cfg.normalisation.channel_combination, np.ndarray)
