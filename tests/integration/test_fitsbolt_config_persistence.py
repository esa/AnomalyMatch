#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Tests for fitsbolt configuration persistence in model checkpoints.

The fitsbolt DotMap configuration can be pickled directly via torch.save/load
without explicit serialization.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from dotmap import DotMap
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.cfg.create_config import validate_config
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.data_io.load_images import get_fitsbolt_config


class TestFitsboltConfigPickling:
    """Test cases for fitsbolt config pickling via torch.save/load."""

    def test_pickle_roundtrip_basic(self):
        """Test basic pickle roundtrip via torch checkpoint."""
        original_cfg = fb_create_cfg(
            output_dtype=np.uint8,
            size=[64, 64],
            n_output_channels=3,
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            num_workers=4,
        )

        # Save via torch
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": original_cfg}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            loaded_cfg = loaded["fitsbolt_cfg"]

            assert loaded_cfg.size == original_cfg.size
            assert loaded_cfg.n_output_channels == original_cfg.n_output_channels
            assert loaded_cfg.num_workers == original_cfg.num_workers
            assert loaded_cfg.normalisation_method == original_cfg.normalisation_method
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_pickle_numpy_dtype(self):
        """Test pickling of numpy dtypes."""
        original_cfg = fb_create_cfg(
            output_dtype=np.float32,
            size=[128, 128],
            n_output_channels=3,
        )

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": original_cfg}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            loaded_cfg = loaded["fitsbolt_cfg"]

            assert loaded_cfg.output_dtype == np.float32
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_pickle_all_normalisation_methods(self):
        """Test pickling with all normalisation methods."""
        for method in NormalisationMethod:
            original_cfg = fb_create_cfg(
                output_dtype=np.uint8,
                size=[64, 64],
                n_output_channels=3,
                normalisation_method=method,
            )

            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
                checkpoint_path = f.name

            try:
                torch.save({"fitsbolt_cfg": original_cfg}, checkpoint_path)
                loaded = torch.load(checkpoint_path, weights_only=False)
                loaded_cfg = loaded["fitsbolt_cfg"]

                assert loaded_cfg.normalisation_method == method
            finally:
                Path(checkpoint_path).unlink(missing_ok=True)

    def test_pickle_channel_combination(self):
        """Test pickling of numpy array channel_combination."""
        original_cfg = fb_create_cfg(
            output_dtype=np.uint8,
            size=[64, 64],
            fits_extension=[0, 1, 2],
            n_output_channels=3,
            channel_combination=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": original_cfg}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            loaded_cfg = loaded["fitsbolt_cfg"]

            np.testing.assert_array_equal(
                loaded_cfg.channel_combination, original_cfg.channel_combination
            )
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_pickle_asinh_settings(self):
        """Test pickling of asinh normalisation settings."""
        original_cfg = fb_create_cfg(
            output_dtype=np.uint8,
            size=[64, 64],
            n_output_channels=3,
            normalisation_method=NormalisationMethod.ASINH,
            norm_asinh_scale=[0.5, 0.6, 0.7],
            norm_asinh_clip=[99.0, 99.5, 99.8],
        )

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": original_cfg}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            loaded_cfg = loaded["fitsbolt_cfg"]

            assert loaded_cfg.normalisation.asinh_scale == original_cfg.normalisation.asinh_scale
            assert loaded_cfg.normalisation.asinh_clip == original_cfg.normalisation.asinh_clip
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestFitsboltConfigValidation:
    """Test cases for fitsbolt config validation after pickling."""

    def test_validate_pickled_config(self):
        """Test that pickled config passes fitsbolt validation."""
        original_cfg = fb_create_cfg(
            output_dtype=np.uint8,
            size=[64, 64],
            n_output_channels=3,
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            num_workers=4,
        )

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": original_cfg}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            loaded_cfg = loaded["fitsbolt_cfg"]

            # Validate using fitsbolt's validate_config
            validate_config(loaded_cfg)
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestFitsboltConfigCompatibility:
    """Test compatibility with fitsbolt's create_config function."""

    def test_compatibility_with_fits_extension_settings(self):
        """Test pickling with various fits_extension configurations."""
        # Single integer extension
        cfg1 = fb_create_cfg(
            output_dtype=np.uint8,
            size=[64, 64],
            n_output_channels=3,
            fits_extension=0,
        )

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": cfg1}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            validate_config(loaded["fitsbolt_cfg"])
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

        # List of extensions
        cfg2 = fb_create_cfg(
            output_dtype=np.uint8,
            size=[64, 64],
            n_output_channels=3,
            fits_extension=[0, 1, 2],
        )

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": cfg2}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            validate_config(loaded["fitsbolt_cfg"])
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestGetFitsboltConfigIntegration:
    """Test get_fitsbolt_config integration with pickling."""

    def test_get_fitsbolt_config_pickling(self):
        """Test that config from get_fitsbolt_config can be pickled."""
        # Create an AnomalyMatch-style config
        cfg = DotMap()
        cfg.normalisation = DotMap()
        cfg.normalisation.output_dtype = np.uint8
        cfg.normalisation.image_size = [64, 64]
        cfg.normalisation.fits_extension = None
        cfg.normalisation.interpolation_order = 1
        cfg.normalisation.n_output_channels = 3
        cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
        cfg.normalisation.channel_combination = None
        cfg.normalisation.norm_maximum_value = None
        cfg.normalisation.norm_minimum_value = None
        cfg.normalisation.norm_log_calculate_minimum_value = False
        cfg.normalisation.norm_crop_for_maximum_value = None
        cfg.normalisation.norm_asinh_scale = [0.7]
        cfg.normalisation.norm_asinh_clip = [99.8]
        cfg.num_workers = 4

        # Get fitsbolt config
        cfg = get_fitsbolt_config(cfg)

        # Save and load via torch
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            torch.save({"fitsbolt_cfg": cfg.fitsbolt_cfg}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=False)
            loaded_cfg = loaded["fitsbolt_cfg"]

            # Validate
            validate_config(loaded_cfg)

            # Verify key properties
            assert loaded_cfg.size == [64, 64]
            assert loaded_cfg.n_output_channels == 3
            assert loaded_cfg.normalisation_method == NormalisationMethod.CONVERSION_ONLY
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestFitsboltConfigE2EWithCheckpoint:
    """End-to-end tests for fitsbolt config persistence with model checkpoints."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_fitsbolt_config_in_checkpoint_dict(self):
        """Test that fitsbolt config can be saved and loaded in a checkpoint-like dict."""
        # Create a fitsbolt config
        fitsbolt_cfg = fb_create_cfg(
            output_dtype=np.uint8,
            size=[64, 64],
            n_output_channels=3,
            normalisation_method=NormalisationMethod.ASINH,
            norm_asinh_scale=[0.5, 0.6, 0.7],
            norm_asinh_clip=[99.0, 99.5, 99.8],
        )

        # Create a mock checkpoint
        checkpoint = {
            "model_state": {"dummy": "data"},
            "optimizer_state": None,
            "fitsbolt_cfg": fitsbolt_cfg,
        }

        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
        loaded_fitsbolt_cfg = loaded_checkpoint["fitsbolt_cfg"]

        # Verify
        assert loaded_fitsbolt_cfg.size == [64, 64]
        assert loaded_fitsbolt_cfg.n_output_channels == 3
        assert loaded_fitsbolt_cfg.normalisation_method == NormalisationMethod.ASINH
        assert loaded_fitsbolt_cfg.normalisation.asinh_scale == [0.5, 0.6, 0.7]
        assert loaded_fitsbolt_cfg.normalisation.asinh_clip == [99.0, 99.5, 99.8]

        # Validate loaded config
        validate_config(loaded_fitsbolt_cfg)

    def test_backward_compatibility_checkpoint_without_fitsbolt(self):
        """Test loading checkpoints that don't have fitsbolt_cfg."""
        # Create a mock checkpoint without fitsbolt_cfg (legacy format)
        checkpoint = {
            "model_state": {"dummy": "data"},
            "optimizer_state": None,
        }

        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / "legacy_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Check that fitsbolt_cfg is not present
        assert "fitsbolt_cfg" not in loaded_checkpoint

        # Accessing non-existent key should return None via .get()
        result = loaded_checkpoint.get("fitsbolt_cfg")
        assert result is None
