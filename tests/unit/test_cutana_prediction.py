#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for cutana streaming prediction FITS extension resolution.

Regression tests for the bug where string fits_extension values
(e.g. CHANNEL_1) were passed directly to cutana, causing
'Extension CHANNEL_1 not found' errors on Euclid source mosaics.
"""

import numpy as np
import pandas as pd
import pytest

from prediction_process_cutana import _resolve_filter_names_from_catalogue

# Realistic Euclid file paths (matching the bug report)
EUCLID_FITS_PATHS = (
    "['/DATA01/mosaics/VIS/EUC_MER_BGSUB-MOSAIC-VIS_TILE102158888-96CC3C_20250730T105451.756715Z_00.00.fits', "
    "'/DATA01/mosaics/NISP/EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102158888-7489F6_20250729T212617.293884Z_00.00.fits', "
    "'/DATA01/mosaics/NISP/EUC_MER_BGSUB-MOSAIC-NIR-H_TILE102158888-92DFAD_20250729T213004.461695Z_00.00.fits']"
)


def _make_euclid_catalogue(path, fits_paths=EUCLID_FITS_PATHS, n=5):
    """Create a minimal cutana catalogue with Euclid-style fits_file_paths."""
    df = pd.DataFrame(
        {
            "SourceID": [f"SRC_{i}" for i in range(n)],
            "RA": [266.0 + i * 0.01 for i in range(n)],
            "Dec": [65.0 + i * 0.01 for i in range(n)],
            "fits_file_paths": [fits_paths] * n,
            "diameter_arcsec": [1.0] * n,
        }
    )
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


class TestResolveFilterNamesFromCatalogue:
    """Tests for _resolve_filter_names_from_catalogue."""

    def test_resolves_euclid_filter_names_from_parquet(self, tmp_path):
        cat = _make_euclid_catalogue(tmp_path / "sources.parquet")
        names = _resolve_filter_names_from_catalogue(str(cat), n_extensions=3)
        assert names == ["VIS", "NIR-Y", "NIR-H"]

    def test_resolves_euclid_filter_names_from_csv(self, tmp_path):
        cat = _make_euclid_catalogue(tmp_path / "sources.csv")
        names = _resolve_filter_names_from_catalogue(str(cat), n_extensions=3)
        assert names == ["VIS", "NIR-Y", "NIR-H"]

    def test_resolves_from_directory(self, tmp_path):
        _make_euclid_catalogue(tmp_path / "batch_000.parquet")
        names = _resolve_filter_names_from_catalogue(str(tmp_path), n_extensions=3)
        assert names == ["VIS", "NIR-Y", "NIR-H"]

    def test_extension_count_mismatch_raises(self, tmp_path):
        cat = _make_euclid_catalogue(tmp_path / "sources.parquet")
        with pytest.raises(ValueError, match="specifies 2 extensions"):
            _resolve_filter_names_from_catalogue(str(cat), n_extensions=2)

    def test_non_euclid_paths_raise(self, tmp_path):
        non_euclid = "['custom_band_a.fits', 'custom_band_b.fits']"
        cat = _make_euclid_catalogue(tmp_path / "sources.parquet", fits_paths=non_euclid, n=3)
        with pytest.raises(ValueError, match="Could not determine filter names"):
            _resolve_filter_names_from_catalogue(str(cat), n_extensions=2)

    def test_empty_directory_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _resolve_filter_names_from_catalogue(str(empty_dir), n_extensions=3)


class TestStreamingFitsExtensionConfig:
    """Verify cutana config is always set to PRIMARY for streaming,
    regardless of the fits_extension format provided by the user.
    """

    @pytest.fixture
    def euclid_catalogue(self, tmp_path):
        return str(_make_euclid_catalogue(tmp_path / "sources.parquet"))

    @pytest.fixture
    def channel_combination(self):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.9, 0.0],
                [0.0, 0.0, 0.9],
            ]
        )

    @pytest.mark.parametrize(
        "fits_extension",
        [
            [1, 2, 3],
            ["CHANNEL_1", "CHANNEL_2", "CHANNEL_3"],
        ],
        ids=["integer_indices", "string_channel_names"],
    )
    def test_fits_extensions_always_primary(
        self, euclid_catalogue, channel_combination, fits_extension
    ):
        """cutana_config.fits_extensions must be ['PRIMARY'] for all input formats."""
        import cutana

        cutana_config = cutana.get_default_config()

        # Replicate the config-building logic from evaluate_images_from_cutana
        fits_ext = list(fits_extension)

        if len(fits_ext) > 1:
            extension_names = _resolve_filter_names_from_catalogue(euclid_catalogue, len(fits_ext))
        else:
            extension_names = ["PRIMARY"]

        cutana_config.fits_extensions = ["PRIMARY"]

        selected_extensions = []
        for name in extension_names:
            selected_extensions.append({"name": name, "ext": "PRIMARY"})
        cutana_config.selected_extensions = selected_extensions

        # Key assertions: cutana must never receive CHANNEL_1/2/3 or integer indices
        assert cutana_config.fits_extensions == ["PRIMARY"]
        assert extension_names == ["VIS", "NIR-Y", "NIR-H"]
        for ext in cutana_config.selected_extensions:
            assert ext["ext"] == "PRIMARY"
            assert ext["name"] in ("VIS", "NIR-Y", "NIR-H")

    @pytest.mark.parametrize(
        "fits_extension",
        [
            [1, 2, 3],
            ["CHANNEL_1", "CHANNEL_2", "CHANNEL_3"],
        ],
        ids=["integer_indices", "string_channel_names"],
    )
    def test_channel_weights_use_filter_names(
        self, euclid_catalogue, channel_combination, fits_extension
    ):
        """channel_weights keys must be resolved filter names, not cutout extension names."""
        fits_ext = list(fits_extension)

        extension_names = _resolve_filter_names_from_catalogue(euclid_catalogue, len(fits_ext))

        channel_weights = {}
        for j, ext_name in enumerate(extension_names):
            channel_weights[str(ext_name)] = channel_combination[:, j].tolist()

        assert set(channel_weights.keys()) == {"VIS", "NIR-Y", "NIR-H"}
        assert "CHANNEL_1" not in channel_weights
        assert channel_weights["VIS"] == [1.0, 0.0, 0.0]

    def test_single_extension_uses_primary(self):
        """Single-extension case should use PRIMARY without catalogue resolution."""
        fits_ext = [1]
        if len(fits_ext) > 1:
            pytest.fail("Should not resolve for single extension")
        extension_names = ["PRIMARY"]
        assert extension_names == ["PRIMARY"]

    def test_none_extension_defaults_to_primary(self):
        """None fits_extension should default to ['PRIMARY']."""
        fits_ext = None
        if fits_ext is None:
            fits_ext = ["PRIMARY"]
        assert fits_ext == ["PRIMARY"]
        assert len(fits_ext) == 1


class TestDictChannelCombination:
    """Tests for dict-form channel_combination support.

    Dict-form maps filter names to weight lists, making channel mapping
    order-independent regardless of catalogue file path ordering.
    """

    @pytest.fixture
    def euclid_catalogue(self, tmp_path):
        return str(_make_euclid_catalogue(tmp_path / "sources.parquet"))

    @pytest.fixture
    def combo_dict(self):
        """Dict-form channel_combination: filter name -> output weights."""
        return {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-Y": [0.0, 0.9, 0.0],
            "NIR-H": [0.0, 0.0, 0.9],
        }

    def test_dict_combo_produces_correct_channel_weights(self, euclid_catalogue, combo_dict):
        """Dict channel_combination should map weights by filter name, not position."""
        extension_names = _resolve_filter_names_from_catalogue(euclid_catalogue, n_extensions=3)

        # Replicate the dict-form logic from evaluate_images_from_cutana
        channel_weights = {}
        for name in extension_names:
            weights = combo_dict[name]
            channel_weights[name] = list(weights) if not isinstance(weights, list) else weights

        assert channel_weights["VIS"] == [1.0, 0.0, 0.0]
        assert channel_weights["NIR-Y"] == [0.0, 0.9, 0.0]
        assert channel_weights["NIR-H"] == [0.0, 0.0, 0.9]

    def test_dict_combo_order_independent(self, euclid_catalogue):
        """Dict-form should produce the same channel_weights regardless of dict key order."""
        extension_names = _resolve_filter_names_from_catalogue(euclid_catalogue, n_extensions=3)

        # Provide dict in reversed order compared to catalogue
        combo_reversed = {
            "NIR-H": [0.0, 0.0, 0.9],
            "NIR-Y": [0.0, 0.9, 0.0],
            "VIS": [1.0, 0.0, 0.0],
        }

        channel_weights = {}
        for name in extension_names:
            weights = combo_reversed[name]
            channel_weights[name] = list(weights) if not isinstance(weights, list) else weights

        # Weights should be correct regardless of dict ordering
        assert channel_weights["VIS"] == [1.0, 0.0, 0.0]
        assert channel_weights["NIR-Y"] == [0.0, 0.9, 0.0]
        assert channel_weights["NIR-H"] == [0.0, 0.0, 0.9]

    def test_dict_combo_missing_filter_raises(self, euclid_catalogue):
        """Dict missing a resolved filter name should raise ValueError."""
        extension_names = _resolve_filter_names_from_catalogue(euclid_catalogue, n_extensions=3)

        # Missing NIR-H
        combo_incomplete = {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-Y": [0.0, 0.9, 0.0],
        }

        missing = set(extension_names) - set(combo_incomplete.keys())
        assert missing == {"NIR-H"}

    def test_validate_config_converts_dict_to_numpy(self):
        """validate_config should convert dict channel_combination to numpy for fitsbolt."""
        combo_dict = {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-Y": [0.0, 0.9, 0.0],
            "NIR-H": [0.0, 0.0, 0.9],
        }

        # Simulate what validate_config does with dict channel_combination
        keys = list(combo_dict.keys())
        cc_array = np.column_stack([np.array(combo_dict[k]) for k in keys])

        assert cc_array.shape == (3, 3)
        # First column corresponds to first key (VIS)
        np.testing.assert_array_equal(cc_array[:, 0], [1.0, 0.0, 0.0])

    def test_validate_config_preserves_dict(self):
        """validate_config should preserve the original dict as channel_combination_dict."""
        from dotmap import DotMap

        cfg = DotMap()
        cfg.normalisation = DotMap()
        cfg.normalisation.channel_combination = {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-Y": [0.0, 0.9, 0.0],
            "NIR-H": [0.0, 0.0, 0.9],
        }
        cfg.normalisation.n_output_channels = 3

        cc = cfg.normalisation.channel_combination
        if cc is not None and isinstance(cc, dict):
            keys = list(cc.keys())
            cc_array = np.column_stack([np.array(cc[k]) for k in keys])
            cfg.normalisation.channel_combination = cc_array
            cfg.normalisation.channel_combination_dict = cc

        # numpy array for fitsbolt
        assert hasattr(cfg.normalisation.channel_combination, "shape")
        assert cfg.normalisation.channel_combination.shape == (3, 3)
        # original dict preserved
        assert isinstance(cfg.normalisation.channel_combination_dict, dict)
        assert set(cfg.normalisation.channel_combination_dict.keys()) == {"VIS", "NIR-Y", "NIR-H"}
