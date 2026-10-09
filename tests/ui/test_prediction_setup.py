#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for prediction setup screen — edge cases and detection logic.

Happy-path tests (rendering, auto-validation, navigation) are covered
by browser tests in tests/browser/test_prediction_setup.py.
"""

import os
import time
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import zarr
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from PIL import Image

from anomaly_match.data_io.source_scanning import (
    detect_and_count_prediction_sources as _detect_and_count,
)
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match_ui.app import AnomalyMatchApp
from anomaly_match_ui.utils import ui_state
from anomaly_match_ui.utils.backend_interface import BackendInterface

pytestmark = pytest.mark.ui

# Permanent test data directory
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "test_data", "grayscale")


# ── Shared fixtures ──────────────────────────────────────────────


@pytest.fixture()
def mock_session():
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "/fake/data"
    session.cfg.net = "test-cnn"
    session.cfg.N_to_load = 1000
    session.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    session.cfg.normalisation.image_size = [64, 64]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.num_channels = 3
    session.cfg.test_ratio = 0.5
    session.cfg.top_N = 10
    session.cfg.num_eval_iter = 10
    session.cfg.metadata_file = None
    session.cfg.model_path = None
    session.cfg.prediction_search_dir = None
    session.cfg.output_dir = "/fake/output"
    return session


def _wait_until_count_idle(screen, timeout: float = 5.0) -> None:
    """Block until the screen's background source-count thread settles.

    On init the screen auto-validates its initial chooser path (cwd) on a
    daemon thread.  Tests that then call ``_count_sources`` explicitly must
    let that init scan finish first, otherwise it can complete late and
    clobber the test's result (flaky ``_search_ok``).
    """
    deadline = time.monotonic() + timeout
    while screen._counting and time.monotonic() < deadline:
        time.sleep(0.01)


# ── Edge case tests (not covered by browser tests) ───────────────


@patch("IPython.display.display")
def test_start_button_initially_disabled(mock_display, mock_session):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    assert screen._start_btn.disabled is True


@patch("IPython.display.display")
def test_persisted_output_dir_validates_without_cfg(mock_display, mock_session, tmp_path):
    """The output row must judge the folder the chooser actually shows.

    ``initial_dir`` lets a persisted last-browsed dir outrank cfg (#429), so
    deriving the flag from ``cfg.output_dir`` left a valid, visibly selected
    folder reported as "Select an output folder" with Start disabled.
    """
    persisted = tmp_path / "persisted_out"
    persisted.mkdir()
    ui_state.set_last_browsed_dir("output_chooser", str(persisted))
    mock_session.cfg.output_dir = None

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    # Assert the identity, not just "something is selected", so the test
    # fails if the persistence lookup is removed.
    assert os.path.samefile(screen._output_chooser.selected, persisted)
    assert screen._output_ok is True
    assert "Select an output folder" not in screen._render_validation()


@patch("IPython.display.display")
def test_model_chooser_shows_the_configured_model(mock_display, mock_session, tmp_path):
    """Screen-level: the model chooser selects the checkpoint _on_start will use.

    The first round of this bug wrote a non-existent ``.safetensors`` path to
    ``cfg.model_path``: a remembered folder without the configured file was
    composed with its basename.
    """
    model_dir = tmp_path / "session" / "iteration_0"
    model_dir.mkdir(parents=True)
    model = model_dir / "model.safetensors"
    model.write_bytes(b"")
    remembered = tmp_path / "other_session"
    remembered.mkdir()
    ui_state.set_last_browsed_dir("model_chooser", str(remembered))
    mock_session.cfg.model_path = str(model)

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    assert screen._model_chooser.selected == str(model)


@patch("IPython.display.display")
def test_unresolved_output_dir_does_not_adopt_cwd(
    mock_display, mock_session, tmp_path, monkeypatch
):
    """The cwd fallback is where the chooser opens, never a selection.

    On a fresh install ``cfg.output_dir`` is the relative default that does
    not exist yet and nothing is persisted.  Pre-selecting the cwd made
    ``_on_start`` write it over the configured folder, so results landed in
    the notebook's working directory.
    """
    monkeypatch.chdir(tmp_path)
    mock_session.cfg.output_dir = "anomaly_match_results/sessions/"

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    assert screen._output_chooser.selected is None
    # The configured folder is still a valid destination: the run creates it.
    assert screen._output_ok is True
    assert "Output: sessions" in screen._render_validation()


@patch("IPython.display.display")
def test_no_output_dir_anywhere_requires_selection(
    mock_display, mock_session, tmp_path, monkeypatch
):
    """With neither a persisted dir nor a cfg value, the user must pick one."""
    monkeypatch.chdir(tmp_path)
    mock_session.cfg.output_dir = None

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    assert screen._output_chooser.selected is None
    assert screen._output_ok is False
    assert "Select an output folder" in screen._render_validation()


@patch("IPython.display.display")
def test_search_folder_validation_with_catalogue(mock_display, mock_session, tmp_path):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    _wait_until_count_idle(screen)

    with patch.object(
        BackendInterface,
        "detect_and_count_prediction_sources",
        return_value=(DataSourceType.CUTANA, 12500),
    ):
        screen._count_sources(str(tmp_path))

    assert screen._search_ok is True
    assert screen._source_count == 12500
    assert screen._search_type == DataSourceType.CUTANA


@patch("IPython.display.display")
def test_search_folder_validation_empty(mock_display, mock_session, tmp_path):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with patch.object(
        BackendInterface,
        "detect_and_count_prediction_sources",
        return_value=(DataSourceType.IMAGE_FOLDER, 0),
    ):
        screen._count_sources(str(empty_dir))

    assert screen._search_ok is False
    assert screen._source_count == 0


@patch("IPython.display.display")
def test_auto_validate_fires_when_cfg_has_output_dir(mock_display, mock_session, tmp_path):
    """Auto-validate sets output_ok when cfg already has output_dir."""
    mock_session.cfg.output_dir = str(tmp_path)

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    assert screen._output_ok is True


# ── Resume detection ─────────────────────────────────────────────


@patch("IPython.display.display")
def test_resume_shows_count_from_existing_db(mock_display, mock_session, tmp_path):
    """Resume detection shows count from existing predictions.db."""
    from anomaly_match.prediction import AnomalyScoreDB

    mock_session.cfg.output_dir = str(tmp_path)

    # PR-420+ DBs always carry a full compat metadata block — seed one so
    # the schema-strict ``_apply_db_settings`` has the keys it expects.
    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    db.set_metadata_batch(
        {
            "model_path": "/seeded/model.safetensors",
            "model_sha256": "a" * 64,
            "image_size": [64, 64],
            "normalisation_method": str(NormalisationMethod.CONVERSION_ONLY),
            "n_output_channels": 3,
            "net": "efficientnet-lite0",
        }
    )
    db.store_results([("a.png", 0.9), ("b.png", 0.8)])
    db.close()

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    with patch.object(type(screen._output_chooser), "selected", new_callable=PropertyMock) as m:
        m.return_value = str(tmp_path)
        screen._check_resume()

    assert screen._resume_count == 2
    assert "Resume" in screen._render_validation() or "2" in screen._render_validation()


@patch("IPython.display.display")
def test_resume_no_db_no_resume_message(mock_display, mock_session, tmp_path):
    """No predictions.db → no resume info shown."""
    mock_session.cfg.output_dir = str(tmp_path)

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    with patch.object(type(screen._output_chooser), "selected", new_callable=PropertyMock) as m:
        m.return_value = str(tmp_path)
        screen._check_resume()

    assert screen._resume_count is None


@patch("IPython.display.display")
def test_resume_applies_db_settings(mock_display, mock_session, tmp_path):
    """Opening a folder whose DB differs from the current widget must
    auto-apply the DB's normalisation onto the widget + cfg — the DB is
    the source of truth for the scores it already contains."""
    from anomaly_match.prediction import AnomalyScoreDB

    # Seed a DB with settings deliberately different from mock_session's
    # defaults (image_size=[64,64], method=CONVERSION_ONLY, channels=3).
    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    db.set_metadata_batch(
        {
            "model_path": "/seeded/model.safetensors",
            "model_sha256": "a" * 64,
            "image_size": [224, 224],
            "normalisation_method": str(NormalisationMethod.ZSCALE),
            "n_output_channels": 4,
            "net": "efficientnet-lite0",
        }
    )
    db.store_results([("a.png", 0.9)])
    db.close()

    mock_session.cfg.output_dir = str(tmp_path)
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    with patch.object(type(screen._output_chooser), "selected", new_callable=PropertyMock) as m:
        m.return_value = str(tmp_path)
        screen._check_resume()

    # Resume accepted, no mismatch error.
    assert screen._resume_count == 1
    assert screen._resume_compatible is True
    # Settings mirrored onto the cfg.
    assert screen._cfg.normalisation.image_size == [224, 224]
    assert screen._cfg.normalisation.normalisation_method == NormalisationMethod.ZSCALE
    assert screen._cfg.normalisation.n_output_channels == 4
    # Summary made it into the resume banner.
    rendered = screen._render_validation()
    assert "settings loaded from DB" in rendered
    assert "[224, 224]" in rendered
    assert "ZSCALE" in rendered


@patch("IPython.display.display")
def test_resume_model_hash_mismatch_blocks(mock_display, mock_session, tmp_path):
    """A DB created with a different model (different sha256) must not
    auto-reconcile — the user has to pick the right model."""
    from anomaly_match.prediction import AnomalyScoreDB

    # Create an actual model file so compute_model_sha256 can run.
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"current-model-bytes")

    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    db.set_metadata_batch(
        {
            "model_path": "/other/model.safetensors",
            "model_sha256": "b" * 64,  # deliberately not the current file's hash
            "image_size": [64, 64],
            "normalisation_method": str(NormalisationMethod.CONVERSION_ONLY),
            "n_output_channels": 3,
            "net": "efficientnet-lite0",
        }
    )
    db.store_results([("a.png", 0.9)])
    db.close()

    # Rely on cfg-fallback path — _check_resume reads selected chooser
    # values first, falls back to cfg when they're None. Both choosers
    # share the same class, so patching `type(...).selected` once would
    # override both instances.
    mock_session.cfg.model_path = str(model_path)
    mock_session.cfg.output_dir = str(tmp_path)
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen

    screen._check_resume()

    assert screen._resume_compatible is False
    assert "different model" in screen._resume_message


# ── E2E tests for _detect_and_count with real data ───────────────


# ── Model-driven normalisation overview ──────────────────────────


def _save_model(tmp_path, *, method, image_size, n_channels, with_metadata=True):
    """Write a real .safetensors checkpoint for the model chooser to read."""
    import anomaly_match as am
    from anomaly_match.data_io.checkpoint_io import save_checkpoint
    from anomaly_match.data_io.load_images import get_fitsbolt_config

    fb_cfg = None
    if with_metadata:
        cfg = am.get_default_cfg()
        cfg.normalisation.normalisation_method = method
        cfg.normalisation.image_size = image_size
        cfg.normalisation.n_output_channels = n_channels
        fb_cfg = get_fitsbolt_config(cfg).fitsbolt_cfg

    import torch

    path = tmp_path / "model.safetensors"
    save_checkpoint(
        {
            "eval_model": {"w": torch.zeros(2)},
            "net": "efficientnet-lite0",
            "num_channels": n_channels,
            "normalisation_method": method,
            "fitsbolt_cfg": fb_cfg,
        },
        path,
    )
    return str(path)


@patch("IPython.display.display")
def test_overview_prompts_for_model_when_none(mock_display, mock_session):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    assert "Select a model" in screen._norm_overview.value


@patch("IPython.display.display")
def test_model_normalisation_mirrored_onto_cfg_and_overview(mock_display, mock_session, tmp_path):
    """Selecting a model mirrors its embedded normalisation onto cfg and the
    read-only overview — the user no longer sets these by hand."""
    model_path = _save_model(
        tmp_path, method=NormalisationMethod.ASINH, image_size=[96, 96], n_channels=2
    )
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    _wait_until_count_idle(screen)

    # A usable model + ready search/output must not be blocked by the
    # normalisation gate.
    screen._search_ok = True
    screen._output_ok = True
    with patch.object(type(screen._model_chooser), "selected", new_callable=PropertyMock) as m:
        m.return_value = model_path
        screen._on_model_change(screen._model_chooser)

    assert screen._cfg.normalisation.image_size == [96, 96]
    assert screen._cfg.normalisation.normalisation_method == NormalisationMethod.ASINH
    assert screen._cfg.normalisation.n_output_channels == 2

    overview = screen._norm_overview.value
    assert "96×96" in overview
    assert "ASINH" in overview
    assert "Output channels: 2" in overview
    assert screen._start_btn.disabled is False


@patch("IPython.display.display")
def test_model_selection_syncs_fitsbolt_cfg_in_lockstep(mock_display, mock_session, tmp_path):
    """Selecting a model must sync cfg.fitsbolt_cfg alongside cfg.normalisation.

    Regression for the resume crash: mirroring only the cfg.normalisation
    fields left cfg.fitsbolt_cfg stale at the pickled default
    (CONVERSION_ONLY), so the Cutana preview's fitsbolt/normalisation
    agreement guard (build_cutana_orchestrator_config) raised on any model
    trained with a non-linear stretch.  The two surfaces must agree.
    """
    model_path = _save_model(
        tmp_path, method=NormalisationMethod.ASINH, image_size=[96, 96], n_channels=2
    )
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    _wait_until_count_idle(screen)

    with patch.object(type(screen._model_chooser), "selected", new_callable=PropertyMock) as m:
        m.return_value = model_path
        screen._on_model_change(screen._model_chooser)

    # fitsbolt_cfg was populated from the checkpoint (not left at the default)
    # and its method matches cfg.normalisation — the exact invariant the Cutana
    # orchestrator builder asserts before extracting a preview cutout.
    assert screen._cfg.fitsbolt_cfg is not None
    synced_method = NormalisationMethod(screen._cfg.fitsbolt_cfg["normalisation_method"])
    assert synced_method == NormalisationMethod.ASINH
    assert synced_method == screen._cfg.normalisation.normalisation_method


@patch("IPython.display.display")
def test_model_without_metadata_warns_and_blocks_start(mock_display, mock_session, tmp_path):
    """A checkpoint without embedded normalisation surfaces a warning, leaves cfg
    untouched, and blocks Start (load_model would reject it after spawn)."""
    # image_size here is never written (with_metadata=False) and differs from the
    # mock_session default [64, 64], so the assert below genuinely proves cfg
    # was left alone rather than coincidentally matching the model's value.
    model_path = _save_model(
        tmp_path,
        method=NormalisationMethod.CONVERSION_ONLY,
        image_size=[32, 32],
        n_channels=3,
        with_metadata=False,
    )
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    _wait_until_count_idle(screen)

    # Search + output ready, so only the model-normalisation gate can block.
    screen._search_ok = True
    screen._output_ok = True
    with patch.object(type(screen._model_chooser), "selected", new_callable=PropertyMock) as m:
        m.return_value = model_path
        screen._on_model_change(screen._model_chooser)

    assert "no embedded normalisation" in screen._norm_overview.value
    # cfg untouched (kept the mock_session default, not the model's [32, 32]).
    assert screen._cfg.normalisation.image_size == [64, 64]
    assert screen._start_btn.disabled is True


@patch("IPython.display.display")
def test_model_read_error_shows_distinct_state(mock_display, mock_session, tmp_path):
    """A model whose metadata can't be read shows a read-error line (not the
    'select a model' prompt) and blocks Start."""
    model_path = _save_model(
        tmp_path, method=NormalisationMethod.ASINH, image_size=[96, 96], n_channels=2
    )
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    _wait_until_count_idle(screen)
    screen._search_ok = True
    screen._output_ok = True

    with (
        patch.object(type(screen._model_chooser), "selected", new_callable=PropertyMock) as m,
        patch.object(
            BackendInterface, "read_model_normalisation", side_effect=OSError("corrupt header")
        ),
    ):
        m.return_value = model_path
        screen._on_model_change(screen._model_chooser)

    assert "Could not read model normalisation" in screen._norm_overview.value
    assert screen._start_btn.disabled is True


class TestDetectAndCountImages:
    """Validate detection with the permanent JPEG test data."""

    def test_permanent_test_data(self):
        """tests/test_data/ contains 19+ JPEG files — should detect as image."""
        file_type, count = _detect_and_count(os.path.normpath(_TEST_DATA_DIR))
        assert file_type == DataSourceType.IMAGE_FOLDER
        assert count >= 19

    def test_generated_mixed_formats(self, tmp_path):
        """JPG + PNG + TIFF in one folder → image type, all found."""
        for ext, fmt in [("jpg", "JPEG"), ("png", "PNG"), ("tiff", "TIFF")]:
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(tmp_path / f"test.{ext}", format=fmt)

        file_type, count = _detect_and_count(str(tmp_path))
        assert file_type == DataSourceType.IMAGE_FOLDER
        assert count >= 3


class TestDetectAndCountCatalogue:
    """Validate detection with Cutana-style catalogue files."""

    @staticmethod
    def _make_valid_df(n=50):
        return pd.DataFrame(
            {
                "SourceID": range(n),
                "RA": [180.0] * n,
                "Dec": [-30.0] * n,
                "fits_file_paths": ["dummy.fits"] * n,
                "diameter_pixel": [10] * n,
            }
        )

    def test_parquet_catalogue(self, tmp_path):
        self._make_valid_df(200).to_parquet(tmp_path / "cat_a.parquet", index=False)
        self._make_valid_df(300).to_parquet(tmp_path / "cat_b.parquet", index=False)

        file_type, count = _detect_and_count(str(tmp_path))
        assert file_type == DataSourceType.CUTANA
        assert count == 500

    def test_csv_catalogue(self, tmp_path):
        self._make_valid_df(75).to_csv(tmp_path / "sources.csv", index=False)

        file_type, count = _detect_and_count(str(tmp_path))
        assert file_type == DataSourceType.CUTANA
        assert count == 75

    def test_normalisation_consistency_dir(self):
        nc_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, "test_data", "normalisation_consistency"
        )
        if not os.path.isdir(nc_dir):
            pytest.skip("normalisation_consistency test data not present")
        file_type, count = _detect_and_count(nc_dir)
        assert file_type in (DataSourceType.IMAGE_FOLDER, DataSourceType.CUTANA)
        assert count > 0


@patch("IPython.display.display")
def test_zarr_preview_applies_channel_combination(mock_display, mock_session, tmp_path):
    """The Zarr preview decodes like inference does, not raw store pixels (#621)."""
    rgb = np.random.default_rng(0).integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
    store = tmp_path / "store.zarr"
    zarr.open_group(str(store), mode="w").create_array("images", shape=rgb.shape, dtype=np.uint8)[
        :
    ] = rgb
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [16, 16]
    cfg.normalisation.channel_combination = np.array([[0.0, 0.0, 1.0]])
    cfg.normalisation.n_output_channels = 1
    mock_session.cfg = cfg

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction_setup")
    screen = app._current_screen
    # Keep the decoded arrays instead of PNG bytes so the pixels can be checked.
    with (
        patch(
            "anomaly_match_ui.screens.prediction_setup_screen.numpy_array_to_byte_stream",
            side_effect=lambda image: image,
        ),
        patch.object(screen._preview_grid, "update") as mock_update,
    ):
        screen._load_preview(str(store), DataSourceType.ZARR)

    samples = mock_update.call_args.args[0]
    assert sorted(name for name, _ in samples) == ["store__image_000000", "store__image_000001"]
    for name, image in samples:
        # A single output channel: this index's blue band, not the colour image.
        idx = int(name[-6:])
        assert image.shape == (16, 16, 1)
        assert np.array_equal(image[..., 0], rgb[idx, ..., 2])


class TestDetectAndCountZarr:
    def test_zarr_directories(self, tmp_path):
        for i in range(2):
            zarr_path = tmp_path / f"store_{i}.zarr"
            root = zarr.open_group(str(zarr_path), mode="w")
            root.create_dataset("images", shape=(5, 32, 32, 3), dtype=np.uint8)

        file_type, count = _detect_and_count(str(tmp_path))
        assert file_type == DataSourceType.ZARR
        assert count == 10

    def test_batch_folders_with_images_zarr(self, tmp_path):
        for i in range(3):
            batch_dir = tmp_path / f"batch_{i:03d}"
            batch_dir.mkdir()
            zarr_path = batch_dir / "images.zarr"
            root = zarr.open_group(str(zarr_path), mode="w")
            root.create_dataset("images", shape=(4, 32, 32, 3), dtype=np.uint8)

        file_type, count = _detect_and_count(str(tmp_path))
        assert file_type == DataSourceType.ZARR
        assert count == 12

    def test_labeled_data_cache_is_excluded(self, tmp_path):
        """A LabeledDataCache directory (built by training as a sibling of
        the label CSV) must never be scored as a prediction batch, even
        though it has the same <batch>/images.zarr layout.
        """
        from anomaly_match.data_io.labeled_data_cache import LabeledDataCache

        zarr_path = tmp_path / "test_images.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("images", shape=(5, 32, 32, 3), dtype=np.uint8)

        cache_dir = tmp_path / "labeled_data_cache"
        cache_dir.mkdir()
        cache_zarr = zarr.open_group(str(cache_dir / "images.zarr"), mode="w")
        cache_zarr.create_dataset("images", shape=(2, 32, 32, 3), dtype=np.uint8)
        (cache_dir / LabeledDataCache.CACHE_INFO_JSON).write_text("{}")

        file_type, count = _detect_and_count(str(tmp_path))
        assert file_type == DataSourceType.ZARR
        assert count == 5

    def test_search_folder_is_the_zarr_store_itself(self, tmp_path):
        """Selecting the .zarr store directly (not its parent folder) must
        still be detected as ZARR, not scanned as an empty image folder.
        """
        store = tmp_path / "test_images.zarr"
        root = zarr.open_group(str(store), mode="w")
        root.create_dataset("images", shape=(7, 32, 32, 3), dtype=np.uint8)

        file_type, count = _detect_and_count(str(store))
        assert file_type == DataSourceType.ZARR
        assert count == 7


class TestDetectAndCountEmpty:
    def test_empty_folder(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        file_type, count = _detect_and_count(str(empty))
        assert file_type == DataSourceType.IMAGE_FOLDER
        assert count == 0

    def test_unsupported_files_only(self, tmp_path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "notes.docx").touch()
        file_type, count = _detect_and_count(str(tmp_path))
        assert file_type == DataSourceType.IMAGE_FOLDER
        assert count == 0
