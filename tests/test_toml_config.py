#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import tempfile
import toml
from pathlib import Path
from dotmap import DotMap

from anomaly_match.data_io.save_config import (
    save_config_toml,
    _convert_enum_to_string,
)
from anomaly_match.image_processing.NormalisationMethod import NormalisationMethod


class TestTOMLConfigUtils:
    """Test TOML configuration utilities."""

    def test_convert_enum_to_string(self):
        """Test converting NormalisationMethod enum to string."""
        config = {"normalisation_method": NormalisationMethod.ASINH, "other": "value"}
        result = _convert_enum_to_string(config)
        assert result["normalisation_method"] == "ASINH"
        assert result["other"] == "value"

    def test_convert_enum_to_string_no_enum(self):
        """Test converting config without enum."""
        config = {"batch_size": 32, "lr": 0.001}
        result = _convert_enum_to_string(config)
        assert result == config


class TestSaveConfigTOML:
    """Test saving configuration to TOML."""

    def test_save_dotmap_config(self):
        """Test saving DotMap configuration."""
        config = DotMap(
            {
                "name": "test_session",
                "normalisation_method": NormalisationMethod.ZSCALE,
                "batch_size": 64,
                "normalisation": {"maximum_value": 100.0, "minimum_value": 0.0},
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_config_toml(config, temp_path)

            # Verify file was created
            assert temp_path.exists()

            # Verify content
            with open(temp_path, "r") as f:
                content = toml.load(f)

            assert content["name"] == "test_session"
            assert content["normalisation_method"] == "ZSCALE"
            assert content["batch_size"] == 64
            assert content["normalisation"]["maximum_value"] == 100.0

        finally:
            temp_path.unlink()

    def test_save_dict_config(self):
        """Test saving dictionary configuration."""
        config = {"name": "test_session", "batch_size": 32, "lr": 0.001}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_config_toml(config, temp_path)

            # Verify content
            with open(temp_path, "r") as f:
                content = toml.load(f)

            # The saved config will have critical optional fields added as "null"
            expected_config = config.copy()
            expected_config.update(
                {
                    "fits_extension": "null",
                    "metadata_file": "null",
                    "prediction_search_dir": "null",
                    "model_path": "null",
                    "normalisation_method": "null",
                    "normalisation.maximum_value": "null",
                    "normalisation.minimum_value": "null",
                    "normalisation.crop_for_maximum_value": "null",
                    "normalisation.log_calculate_minimum_value": "null",
                    "interpolation_order": "null",
                }
            )

            assert content == expected_config

        finally:
            temp_path.unlink()


class TestConfigIntegration:
    """Test configuration integration and consistency."""

    def test_config_types_and_structure(self):
        """Test that the default config has the expected structure and types."""
        from anomaly_match.utils.get_default_cfg import get_default_cfg

        config = get_default_cfg()

        # Check that it's a DotMap
        assert isinstance(config, DotMap)

        # Check for required keys that should exist in default config
        required_keys = [
            "size",
            "net",
            "batch_size",
            "name",
        ]  # Removed 'device' as it doesn't exist
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

        # Check types
        assert isinstance(config.size, list)
        assert len(config.size) == 2
        assert isinstance(config.net, str)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.name, str)

        # Check that size contains valid dimensions
        assert all(isinstance(dim, int) and dim > 0 for dim in config.size)
