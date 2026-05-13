#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests verifying that cutana's prediction pipeline normalises images identically to AM's
training pipeline. A mismatch would cause silent model failure in production (data drift).

The cutana prediction path (prediction_process_cutana.py) passes AM's fitsbolt_cfg to cutana
via external_fitsbolt_cfg so that cutana handles normalisation. This test verifies that
the output matches what AM's load_and_process_wrapper produces for the same raw data.

Test approach:
1. Extract raw (unnormalised) cutouts from a test FITS tile using cutana's
   do_only_cutout_extraction mode, then save as clean FITS files.
2. Prediction path: run cutana with external_fitsbolt_cfg + channel_weights
   (expanding 1->n_output_channels before normalisation) -> convert_cutana_cutout
3. Training path: load_and_process_wrapper on the same raw FITS files
4. Compare pixel-for-pixel (+/-1 tolerance for WCS reprojection float rounding).
"""

import csv
import os

import cutana
import numpy as np
import pytest
from astropy.io import fits
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.data_io.load_images import (
    get_fitsbolt_config,
    load_and_process_wrapper,
)
from anomaly_match.utils.get_default_cfg import get_default_cfg
from prediction_utils import convert_cutana_cutout, create_cutana_format_cfg

# Paths to pre-generated test data
_TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "test_data", "normalisation_consistency"
)
_FITS_TILE = os.path.join(
    _TEST_DATA_DIR,
    "EUC_MER_BGSUB-MOSAIC-VIS_TILE102018211-ACBD03_20251124T100053.096Z_00.00.fits",
)
_CSV_CATALOGUE = os.path.join(_TEST_DATA_DIR, "mock_sources.csv")

_TARGET_RESOLUTION = 150
_MAX_SOURCES = 2


def _rewrite_csv_with_absolute_paths(csv_in, csv_out, fits_tile_abs):
    """Rewrite the catalogue CSV with absolute paths, limited to _MAX_SOURCES."""
    with open(csv_in) as f_in, open(csv_out, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            if i >= _MAX_SOURCES:
                break
            row["fits_file_paths"] = str([fits_tile_abs])
            writer.writerow(row)


def _make_cutana_config(csv_path, n_output_channels=3):
    """Create a base cutana config for the test tile.

    Sets channel_weights so cutana expands single-extension data to
    n_output_channels before normalisation — matching the training path's
    channel_combine -> normalise order in fitsbolt's _process_image.
    """
    config = cutana.get_default_config()
    config.target_resolution = _TARGET_RESOLUTION
    config.source_catalogue = csv_path
    config.fits_extensions = ["PRIMARY"]
    config.selected_extensions = [{"name": "PRIMARY", "ext": "PrimaryHDU"}]
    config.apply_flux_conversion = False
    config.channel_weights = {"PRIMARY": [1.0] * n_output_channels}
    return config


def _run_cutana_normalised(csv_path, fitsbolt_cfg, n_output_channels=3):
    """Run cutana with external_fitsbolt_cfg and return normalised cutout arrays."""
    config = _make_cutana_config(csv_path, n_output_channels)
    config.external_fitsbolt_cfg = fitsbolt_cfg

    orchestrator = cutana.StreamingOrchestrator(config)
    orchestrator.init_streaming(batch_size=10, write_to_disk=False)

    all_cutouts = []
    for _ in range(orchestrator.get_batch_count()):
        batch = orchestrator.next_batch()
        cutouts = batch["cutouts"]
        if isinstance(cutouts, list) and len(cutouts) == 0:
            continue
        if isinstance(cutouts, list):
            cutouts = np.array(cutouts)
        for i in range(cutouts.shape[0]):
            all_cutouts.append(np.array(cutouts[i]))

    orchestrator.cleanup()
    return all_cutouts


def _extract_raw_cutouts(csv_path, output_dir):
    """Extract raw (unnormalised) cutouts as FITS files using cutana."""
    config = _make_cutana_config(csv_path, n_output_channels=1)
    config.do_only_cutout_extraction = True
    config.output_format = "fits"
    config.write_to_disk = True
    config.output_dir = output_dir

    orchestrator = cutana.StreamingOrchestrator(config)
    orchestrator.init_streaming(batch_size=10, write_to_disk=True)
    for _ in range(orchestrator.get_batch_count()):
        orchestrator.next_batch()
    orchestrator.cleanup()

    # Raw cutout data is in HDU[1] of each file
    return sorted(
        os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".fits")
    )


@pytest.fixture(scope="session")
def cutana_test_data(tmp_path_factory):
    """Extract raw cutouts from the test FITS tile using cutana.

    Session-scoped: raw cutout extraction is expensive and identical
    across all normalisation methods.

    Returns:
        tuple: (clean_fits_paths, rewritten_csv_path)
            - clean_fits_paths: list of FITS file paths with raw cutout data in HDU[0]
            - rewritten_csv_path: path to CSV with absolute FITS tile paths
    """
    if not os.path.exists(_CSV_CATALOGUE) or not os.path.exists(_FITS_TILE):
        pytest.skip("Normalisation consistency test data not found")

    tmp_path = tmp_path_factory.mktemp("normalisation_consistency")

    # Rewrite CSV with absolute paths for this environment
    rewritten_csv = str(tmp_path / "sources.csv")
    _rewrite_csv_with_absolute_paths(_CSV_CATALOGUE, rewritten_csv, os.path.abspath(_FITS_TILE))

    # Extract raw cutouts (do_only_cutout_extraction writes FITS with data in HDU[1])
    raw_dir = str(tmp_path / "raw_cutouts")
    os.makedirs(raw_dir, exist_ok=True)
    raw_fits_paths = _extract_raw_cutouts(rewritten_csv, raw_dir)

    if not raw_fits_paths:
        pytest.skip("Cutana did not produce any cutouts from the test tile")

    # Save as clean FITS files with data in HDU[0] for the training path
    clean_fits_paths = []
    for i, raw_path in enumerate(raw_fits_paths):
        with fits.open(raw_path) as hdul:
            raw_data = hdul[1].data
        clean_path = str(tmp_path / f"cutout_{i}.fits")
        fits.PrimaryHDU(raw_data.astype(np.float32)).writeto(clean_path, overwrite=True)
        clean_fits_paths.append(clean_path)

    return clean_fits_paths, rewritten_csv


def _make_cfg(normalisation_method):
    """Create an AM config with fitsbolt_cfg for the given normalisation method."""
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [_TARGET_RESOLUTION, _TARGET_RESOLUTION]
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.normalisation_method = normalisation_method
    cfg.num_channels = 3
    cfg.num_workers = 0
    cfg = get_fitsbolt_config(cfg)
    return cfg


_ALL_METHODS = [
    NormalisationMethod.CONVERSION_ONLY,
    NormalisationMethod.LOG,
    NormalisationMethod.ZSCALE,
    NormalisationMethod.ASINH,
]


def test_cutana_vs_training_normalisation(cutana_test_data):
    """Verify that cutana's normalisation matches AM's training normalisation.

    Tests all normalisation methods in a single function to avoid repeated
    cutana orchestrator initialisation overhead (~3s per method).

    For each normalisation method:
    1. Training path: raw FITS file -> load_and_process_wrapper() -> normalised image
    2. Prediction path: FITS tile -> cutana with external_fitsbolt_cfg ->
       convert_cutana_cutout -> normalised image

    Both paths must produce matching output for the same source data.
    A tolerance of +/-1 (uint8) is allowed because the two separate cutana runs
    (raw extraction vs normalised) may produce tiny float32 differences in WCS
    reprojection, which non-linear stretches (ASINH) can amplify to +/-1/255.
    """
    clean_fits_paths, rewritten_csv = cutana_test_data
    failures = []

    for method in _ALL_METHODS:
        cfg = _make_cfg(method)

        # --- Prediction path: cutana normalisation + format conversion ---
        cutana_normalised = _run_cutana_normalised(
            rewritten_csv, cfg.fitsbolt_cfg, n_output_channels=3
        )
        if len(cutana_normalised) != len(clean_fits_paths):
            failures.append(
                f"{method.name}: cutana returned {len(cutana_normalised)} cutouts "
                f"but {len(clean_fits_paths)} raw cutouts were extracted"
            )
            continue

        format_cfg = create_cutana_format_cfg(cfg)
        prediction_images = [convert_cutana_cutout(c, format_cfg) for c in cutana_normalised]

        # --- Training path: load raw FITS via fitsbolt ---
        training_pairs = load_and_process_wrapper(clean_fits_paths, cfg, show_progress=False)

        # --- Compare ---
        for i, (pred_img, (_, train_img)) in enumerate(zip(prediction_images, training_pairs)):
            if pred_img.shape != train_img.shape:
                failures.append(
                    f"{method.name} cutout {i}: shape mismatch — "
                    f"prediction {pred_img.shape} vs training {train_img.shape}"
                )
                continue
            if pred_img.dtype != train_img.dtype:
                failures.append(
                    f"{method.name} cutout {i}: dtype mismatch — "
                    f"prediction {pred_img.dtype} vs training {train_img.dtype}"
                )
                continue

            diff = np.abs(pred_img.astype(np.int16) - train_img.astype(np.int16))
            max_diff = int(diff.max())
            if max_diff > 1:
                failures.append(
                    f"{method.name} cutout {i}: max abs diff = {max_diff} (tolerance: 1)"
                )

    assert not failures, "Normalisation mismatches:\n" + "\n".join(failures)
