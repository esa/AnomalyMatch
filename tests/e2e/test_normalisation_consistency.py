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
   (expanding 1->n_output_channels before normalisation) -> process_single_wrapper
3. Training path: load_and_process_wrapper on the same raw FITS files
4. Compare pixel-for-pixel (+/-1 tolerance for WCS reprojection float rounding).
"""

import csv
import os

import cutana
import numpy as np
import pytest
from astropy.io import fits
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.data_io.container_loaders import apply_channel_combination_to_cutana_image
from anomaly_match.data_io.load_images import (
    get_fitsbolt_config,
    load_and_process_wrapper,
)
from anomaly_match.datasets.cutana_source import build_cutana_orchestrator_config
from anomaly_match.utils.get_default_cfg import get_default_cfg

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


def _make_cutana_config(csv_path):
    """Create a base cutana config for the test tile.

    Uses identity channel_weights — the orchestrator produces single-band
    output and any broadcast/combine to ``n_output_channels`` is applied
    post-Cutana via :func:`apply_channel_combination_to_cutana_image`,
    mirroring the production path.
    """
    config = cutana.get_default_config()
    config.target_resolution = _TARGET_RESOLUTION
    config.source_catalogue = csv_path
    config.fits_extensions = ["PRIMARY"]
    config.selected_extensions = [{"name": "PRIMARY", "ext": "PrimaryHDU"}]
    config.apply_flux_conversion = False
    config.channel_weights = {"PRIMARY": [1.0]}
    # Match the training-path output dtype so shape comparisons line up.
    config.data_type = "uint8"
    return config


def _run_cutana_normalised(csv_path, fitsbolt_cfg):
    """Run cutana with external_fitsbolt_cfg and return normalised per-band cutouts.

    Returns:
        dict: Source id -> cutout.  Keyed rather than ordered because Cutana's
        workers may return batches out of catalogue order.
    """
    config = _make_cutana_config(csv_path)
    # Fitsbolt config must advertise single-channel output to match the
    # identity channel_weights; copy and override so the caller's cfg is
    # untouched.
    fitsbolt_single = DotMap(fitsbolt_cfg.toDict(), _dynamic=False)
    fitsbolt_single.n_output_channels = 1
    fitsbolt_single.n_expected_channels = 1
    fitsbolt_single.channel_combination = None
    config.external_fitsbolt_cfg = fitsbolt_single

    orchestrator = cutana.StreamingOrchestrator(config)
    orchestrator.init_streaming(batch_size=10, write_to_disk=False)

    all_cutouts = {}
    for _ in range(orchestrator.get_batch_count()):
        batch = orchestrator.next_batch()
        cutouts = batch["cutouts"]
        if isinstance(cutouts, list) and len(cutouts) == 0:
            continue
        if isinstance(cutouts, list):
            cutouts = np.array(cutouts)
        for i, source in enumerate(batch["metadata"]):
            all_cutouts[str(source["source_id"])] = np.array(cutouts[i])

    orchestrator.cleanup()
    return all_cutouts


def _extract_raw_cutouts(csv_path, output_dir):
    """Extract raw (unnormalised) cutouts as FITS files using cutana."""
    config = _make_cutana_config(csv_path)
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
            - clean_fits_paths: dict of source id -> FITS path with raw cutout data in HDU[0]
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
    # Cutana embeds the SourceID in each cutout filename; key by it so the
    # comparison pairs the same source on both paths.
    with open(rewritten_csv, newline="") as handle:
        source_ids = [row["SourceID"] for row in csv.DictReader(handle)]
    clean_fits_paths = {}
    for i, raw_path in enumerate(raw_fits_paths):
        # Cutana writes ``..._<SourceID>_<RA>_<Dec>_cutout.fits``: require the
        # trailing "_" so ``123`` can't match ``1234``, and take the longest hit
        # so a SourceID that ends another one can't match it either.
        name = os.path.basename(raw_path)
        matches = [sid for sid in source_ids if f"{sid}_" in name]
        longest = [sid for sid in matches if len(sid) == max(map(len, matches), default=0)]
        assert len(longest) == 1, f"Cannot tell which source {raw_path} holds: {matches}"
        with fits.open(raw_path) as hdul:
            raw_data = hdul[1].data
        clean_path = str(tmp_path / f"cutout_{i}.fits")
        fits.PrimaryHDU(raw_data.astype(np.float32)).writeto(clean_path, overwrite=True)
        clean_fits_paths[longest[0]] = clean_path

    return clean_fits_paths, rewritten_csv


def _make_cfg(normalisation_method):
    """Create an AM config with fitsbolt_cfg for the given normalisation method.

    Uses ``n_output_channels=1`` so both paths produce single-band output and
    the comparison isolates normalisation behaviour from broadcast ordering.
    """
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [_TARGET_RESOLUTION, _TARGET_RESOLUTION]
    cfg.normalisation.n_output_channels = 1
    cfg.normalisation.normalisation_method = normalisation_method
    # Test FITS fixtures don't carry a MAGZERO header.  The default flipped
    # to True in #411; opt back out so the training path mirrors the cutana
    # config below (which also disables flux conversion).
    cfg.normalisation.apply_flux_conversion = False
    cfg.num_channels = 1
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
       process_single_wrapper -> normalised image

    Both paths must produce matching output for the same source data.
    A tolerance of +/-1 (uint8) is allowed because the two separate cutana runs
    (raw extraction vs normalised) may produce tiny float32 differences in WCS
    reprojection, which non-linear stretches (ASINH) can amplify to +/-1/255.
    """
    clean_fits_paths, rewritten_csv = cutana_test_data
    failures = []

    for method in _ALL_METHODS:
        cfg = _make_cfg(method)

        # --- Prediction path: cutana per-band normalisation + channel broadcast ---
        cutana_normalised = _run_cutana_normalised(rewritten_csv, cfg.fitsbolt_cfg)
        if len(cutana_normalised) != len(clean_fits_paths):
            failures.append(
                f"{method.name}: cutana returned {len(cutana_normalised)} cutouts "
                f"but {len(clean_fits_paths)} raw cutouts were extracted"
            )
            continue

        if cutana_normalised.keys() != clean_fits_paths.keys():
            failures.append(
                f"{method.name}: cutana returned sources {sorted(cutana_normalised)} "
                f"but raw cutouts exist for {sorted(clean_fits_paths)}"
            )
            continue

        # --- Training path: load raw FITS via fitsbolt ---
        training_images = dict(
            load_and_process_wrapper(list(clean_fits_paths.values()), cfg, show_progress=False)
        )

        # --- Compare, pairing by source id ---
        for i in sorted(clean_fits_paths):
            pred_img = apply_channel_combination_to_cutana_image(cutana_normalised[i], cfg)
            train_img = training_images[clean_fits_paths[i]]
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


def test_nonlinear_method_actually_changes_pixels(cutana_test_data):
    """Verify ASINH produces different pixels than linear through Cutana.

    ``test_cutana_vs_training_normalisation`` only checks that the Cutana path
    and the training path *agree*; it would still pass if both silently applied
    a linear (CONVERSION_ONLY) stretch and the configured method were dropped on
    the way into Cutana (the exact regression this module guards against). This
    test closes that gap: it confirms the non-linear stretch actually reaches the
    cutout pixels by requiring the Cutana output to diverge materially from the
    CONVERSION_ONLY output for the same source.

    Only ASINH is exercised (not also LOG/ZSCALE) to keep the extra Cutana
    orchestrator runs — and so the CI time — to a single non-linear method; the
    parametrised consistency test above already covers all four end to end.
    """
    _, rewritten_csv = cutana_test_data

    linear_cutouts = _run_cutana_normalised(
        rewritten_csv, _make_cfg(NormalisationMethod.CONVERSION_ONLY).fitsbolt_cfg
    )
    assert linear_cutouts, "Cutana produced no linear cutouts for the difference check"

    asinh_cutouts = _run_cutana_normalised(
        rewritten_csv, _make_cfg(NormalisationMethod.ASINH).fitsbolt_cfg
    )
    assert len(asinh_cutouts) == len(linear_cutouts), (
        f"ASINH returned {len(asinh_cutouts)} cutouts but linear returned {len(linear_cutouts)}"
    )

    failures = []
    for i, lin in linear_cutouts.items():
        stretched = asinh_cutouts[i]
        diff = np.abs(lin.astype(np.int16) - stretched.astype(np.int16))
        fraction_changed = float((diff > 0).mean())
        # A genuine non-linear stretch on real source data alters the vast
        # majority of pixels; near-zero divergence means the method was
        # silently downgraded to linear before reaching Cutana.
        if fraction_changed < 0.10:
            failures.append(
                f"ASINH cutout {i}: only {fraction_changed:.1%} of pixels differ from "
                f"linear (mean abs diff {diff.mean():.2f}) — the non-linear stretch did "
                "not reach the Cutana cutouts"
            )

    assert not failures, "Non-linear normalisation not applied:\n" + "\n".join(failures)


def test_orchestrator_config_fails_hard_without_fitsbolt_cfg():
    """``build_cutana_orchestrator_config`` must reject a cfg with no fitsbolt_cfg.

    A missing ``cfg.fitsbolt_cfg`` would let Cutana fall back to its own default
    (linear) normalisation, silently discarding the user's configured method.
    The builder must fail hard instead of producing a linear-normalised run.
    The guard runs before any catalogue/disk work, so this needs no real data.
    """
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [_TARGET_RESOLUTION, _TARGET_RESOLUTION]
    cfg.normalisation.n_output_channels = 1
    cfg.normalisation.normalisation_method = NormalisationMethod.ASINH
    cfg.fitsbolt_cfg = None

    with pytest.raises(ValueError, match="cfg.fitsbolt_cfg is None"):
        build_cutana_orchestrator_config("/nonexistent/catalogue.parquet", cfg)


def test_orchestrator_config_fails_hard_on_method_mismatch():
    """``build_cutana_orchestrator_config`` must reject a method mismatch.

    A non-None ``cfg.fitsbolt_cfg`` whose ``normalisation_method`` differs from
    ``cfg.normalisation`` would stretch the cutouts with the fitsbolt method
    (the only one that reaches Cutana) — silently discarding the configured
    method and feeding the model a stretch it may not have been trained on.
    The None guard alone does not catch this. Like that guard, the check runs
    before any catalogue/disk work, so this needs no real data.
    """
    # fitsbolt_cfg is built for ASINH, then cfg.normalisation is switched to
    # LOG without rebuilding it — the exact divergence the guard must catch.
    cfg = _make_cfg(NormalisationMethod.ASINH)
    cfg.normalisation.normalisation_method = NormalisationMethod.LOG

    with pytest.raises(ValueError, match="disagrees"):
        build_cutana_orchestrator_config("/nonexistent/catalogue.parquet", cfg)
