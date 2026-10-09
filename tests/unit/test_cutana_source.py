#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for cutana_source module-level helpers and edge cases.

Integration tests that stream real FITS cutouts live in
``tests/integration/test_training_data_source_integration.py``.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

import anomaly_match as am
import anomaly_match.datasets.cutana_source as cutana_source
from anomaly_match.data_io.checkpoint_io import sync_normalisation_from_checkpoint
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.datasets.cutana_source import (
    _count_catalogue_rows,
    _is_cutana_catalogue,
    build_cutana_orchestrator_config,
    extract_cutouts_direct,
)


class TestOrchestratorResolutionFollowsModel:
    """Regression for the prediction normalisation bug: the Cutana orchestrator
    resolution + normalisation must come from the model checkpoint, not from a
    stale ``cfg.normalisation`` left over from training or a UI default."""

    def test_target_resolution_uses_synced_model_size(self):
        # A stale cfg pointing at the wrong resolution/method (what a pickled UI
        # default or a prior training run could leave behind).
        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [999, 999]
        cfg.normalisation.normalisation_method = NormalisationMethod.ZSCALE
        # String fits_extension => no catalogue read needed to resolve bands.
        cfg.normalisation.fits_extension = ["VIS"]

        # The model was trained at 150 px with ASINH.
        model_cfg = am.get_default_cfg()
        model_cfg.normalisation.image_size = [150, 150]
        model_cfg.normalisation.n_output_channels = 1
        model_cfg.normalisation.normalisation_method = NormalisationMethod.ASINH
        model_fb = get_fitsbolt_config(model_cfg).fitsbolt_cfg

        # Syncing from the checkpoint (as load_model / the session now do) must
        # make the orchestrator config follow the model, not the stale cfg.
        assert sync_normalisation_from_checkpoint(cfg, model_fb, None)
        orchestrator_cfg = build_cutana_orchestrator_config("/fake/cat.parquet", cfg)

        assert orchestrator_cfg.target_resolution == 150
        # The per-band normalisation handed to Cutana matches the model's method.
        assert (
            orchestrator_cfg.external_fitsbolt_cfg.normalisation_method
            == NormalisationMethod.ASINH.value
        )


class TestNativeCutoutExtraction:
    """The detail view requests native (unresized) cutouts via
    ``do_only_cutout_extraction`` so it can show the source's real pixels."""

    def _cfg(self):
        cfg = am.get_default_cfg()
        cfg.normalisation.fits_extension = ["VIS"]  # avoid a catalogue read
        return get_fitsbolt_config(cfg)

    def test_flag_off_by_default(self):
        ccfg = build_cutana_orchestrator_config("/fake/cat.parquet", self._cfg())
        assert ccfg.do_only_cutout_extraction is False

    def test_flag_set_when_requested(self):
        ccfg = build_cutana_orchestrator_config(
            "/fake/cat.parquet", self._cfg(), do_only_cutout_extraction=True
        )
        assert ccfg.do_only_cutout_extraction is True


class TestExtractCutoutsDirectLogFlag:
    """``extract_cutouts_direct`` opts out of Cutana's per-set heartbeat only
    when the installed Cutana exposes the ``log_set_progress`` keyword."""

    def _call_with_support(self, supports):
        df = pd.DataFrame({"SourceID": ["a"], "fits_file_paths": ["['x.fits']"]})
        mock_create = MagicMock(return_value=[{"cutouts": [], "metadata": []}])
        with (
            patch.object(cutana_source, "_CREATE_CUTOUTS_SUPPORTS_LOG_FLAG", supports),
            patch.object(
                cutana_source, "build_cutana_orchestrator_config", return_value=MagicMock()
            ),
            patch("cutana.create_cutouts_direct", mock_create),
        ):
            extract_cutouts_direct(df, MagicMock())
        return mock_create

    def test_passes_flag_when_supported(self):
        mock_create = self._call_with_support(supports=True)
        _args, kwargs = mock_create.call_args
        assert kwargs.get("log_set_progress") is False

    def test_omits_flag_when_unsupported(self):
        """Older Cutana (e.g. the develop build CI installs) lacks the kwarg —
        we must call without it rather than raising a TypeError."""
        mock_create = self._call_with_support(supports=False)
        _args, kwargs = mock_create.call_args
        assert "log_set_progress" not in kwargs


# ---------------------------------------------------------------------------
# Tests: _is_cutana_catalogue
# ---------------------------------------------------------------------------


class TestIsCutanaCatalogue:
    """Detect whether a file is a valid Cutana catalogue."""

    def test_csv_with_required_columns(self, tmp_path):
        cat = tmp_path / "valid.csv"
        df = pd.DataFrame(
            {
                "SourceID": ["s1"],
                "fits_file_paths": ["f.fits"],
                "extra_col": [1],
            }
        )
        df.to_csv(cat, index=False)
        assert _is_cutana_catalogue(cat) is True

    def test_csv_missing_columns(self, tmp_path):
        cat = tmp_path / "invalid.csv"
        pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(cat, index=False)
        assert _is_cutana_catalogue(cat) is False

    def test_parquet_with_required_columns(self, tmp_path):
        cat = tmp_path / "valid.parquet"
        pd.DataFrame(
            {
                "SourceID": ["s1"],
                "fits_file_paths": ["f.fits"],
            }
        ).to_parquet(cat, index=False)
        assert _is_cutana_catalogue(cat) is True

    def test_parquet_missing_columns(self, tmp_path):
        cat = tmp_path / "bad.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(cat, index=False)
        assert _is_cutana_catalogue(cat) is False

    def test_corrupted_file_returns_false(self, tmp_path):
        """Corrupted files don't crash, just return False."""
        bad = tmp_path / "corrupt.csv"
        bad.write_bytes(b"\x00\x01\x02\x03")
        assert _is_cutana_catalogue(bad) is False


# ---------------------------------------------------------------------------
# Tests: _count_catalogue_rows
# ---------------------------------------------------------------------------


class TestCountCatalogueRows:
    """Row counting for CSV and parquet files."""

    def test_csv_row_count(self, tmp_path):
        cat = tmp_path / "data.csv"
        pd.DataFrame({"SourceID": ["a", "b", "c"]}).to_csv(cat, index=False)
        assert _count_catalogue_rows(cat) == 3

    def test_parquet_row_count(self, tmp_path):
        cat = tmp_path / "data.parquet"
        pd.DataFrame({"SourceID": ["a", "b", "c", "d"]}).to_parquet(cat, index=False)
        assert _count_catalogue_rows(cat) == 4


# ---------------------------------------------------------------------------
# Tests: detect_cutana_filter_names agrees with extraction
# ---------------------------------------------------------------------------

_Q1_BAND_PATHS = str(
    [
        f"/tiles/102044822/EUC_MER_BGSUB-MOSAIC-{band}_TILE102044822-E89E05_00.00.fits"
        for band in ("VIS", "NIR-H", "NIR-Y", "NIR-J")
    ]
)


def _write_q1_catalogue(folder):
    """Write a 4-band catalogue laid out like the Euclid Q1 search catalogues."""
    pd.DataFrame(
        {
            "SourceID": ["s1", "s2"],
            "RA": [150.0, 150.1],
            "Dec": [2.3, 2.4],
            "diameter_pixel": [64, 64],
            "fits_file_paths": [_Q1_BAND_PATHS, _Q1_BAND_PATHS],
        }
    ).to_parquet(folder / "q1_source_cat_search_0.parquet", index=False)


class TestDetectCutanaFilterNames:
    """The channel editor must be sized to the bands extraction will deliver."""

    def test_reads_all_catalogue_bands_by_default(self, tmp_path):
        _write_q1_catalogue(tmp_path)
        cfg = am.get_default_cfg()

        names = cutana_source.detect_cutana_filter_names(str(tmp_path), cfg)

        assert names == ["VIS", "NIR-H", "NIR-Y", "NIR-J"]

    def test_explicit_fits_extension_matches_extraction(self, tmp_path):
        """A notebook-set band subset is what extraction loads, so the editor
        must show it too — not the catalogue's full four bands (the
        "4 columns ... but the batch has 3 band(s)" failure).
        """
        _write_q1_catalogue(tmp_path)
        cfg = am.get_default_cfg()
        cfg.normalisation.fits_extension = ["VIS", "NIR-Y", "NIR-J"]

        names = cutana_source.detect_cutana_filter_names(str(tmp_path), cfg)
        extracted, _ = cutana_source._resolve_extension_names(
            cfg, catalogue_path=str(tmp_path / "q1_source_cat_search_0.parquet")
        )

        assert names == extracted == ["VIS", "NIR-Y", "NIR-J"]

    def test_skips_a_label_csv_that_sorts_first(self, tmp_path):
        _write_q1_catalogue(tmp_path)
        pd.DataFrame({"id": ["s1"], "label": ["normal"]}).to_csv(
            tmp_path / "a_labels.csv", index=False
        )

        names = cutana_source.detect_cutana_filter_names(str(tmp_path), am.get_default_cfg())

        assert names == ["VIS", "NIR-H", "NIR-Y", "NIR-J"]

    def test_folder_without_catalogue_warns(self, tmp_path):
        with patch.object(cutana_source.logger, "warning") as mock_warning:
            names = cutana_source.detect_cutana_filter_names(str(tmp_path), am.get_default_cfg())

        assert names == []
        mock_warning.assert_called_once()
