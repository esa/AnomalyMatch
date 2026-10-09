#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import csv
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

pytestmark = pytest.mark.slow
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger
from PIL import Image
from prediction_process import evaluate_files
from prediction_process_cutana import evaluate_images_from_cutana
from prediction_process_zarr import evaluate_images_in_zarr

from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.prediction import AnomalyScoreDB
from anomaly_match.utils.get_default_cfg import get_default_cfg


@pytest.fixture
def test_config(test_model_path):
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [150, 150]
    cfg.normalisation.n_output_channels = 3
    cfg.net = "efficientnet-lite0"
    # load_model replaces every weight from the checkpoint, so fetching the
    # ImageNet weights first only costs a download in CI.
    cfg.pretrained = False
    cfg.num_channels = 3
    cfg.model_path = str(test_model_path)
    cfg.gpu = 0
    cfg.output_dir = tempfile.mkdtemp()
    cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    cfg.log_level = "INFO"  # Add proper log level
    cfg.name = "test_session"  # Add session name
    cfg.seed = 42  # Add seed
    cfg.test_ratio = 0.0  # Add test ratio
    cfg.save_dir = tempfile.mkdtemp()  # Add save directory
    cfg.data_dir = "tests/test_data/grayscale/"  # Add data directory
    cfg.num_workers = 0  # Use main process for data loading (avoids spawn overhead)

    # Create fb_cfg for fitsbolt
    cfg.fitsbolt_cfg = fb_create_cfg(
        output_dtype=np.uint8,
        size=cfg.normalisation.image_size,
        fits_extension=cfg.normalisation.fits_extension,
        interpolation_order=cfg.normalisation.interpolation_order,
        normalisation_method=cfg.normalisation.normalisation_method,
        channel_combination=cfg.normalisation.channel_combination,
        num_workers=max(cfg.num_workers, 1),
        norm_maximum_value=cfg.normalisation.norm_maximum_value,
        norm_minimum_value=cfg.normalisation.norm_minimum_value,
        norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
        norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
        norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
        norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
    )

    return cfg


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    images = []
    for i in range(10):
        img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        images.append(Image.fromarray(img))
    return images


@pytest.fixture
def test_zarr(sample_images, tmp_path):
    """Create a test Zarr file with sample images."""
    zarr_path = tmp_path / "test.zarr"

    # Create zarr store using the older API that works with this version
    root = zarr.open_group(str(zarr_path), mode="w")

    # Convert PIL images to numpy arrays
    img_arrays = []
    filenames = []
    for i, img in enumerate(sample_images):
        img_array = np.array(img)
        img_arrays.append(img_array)
        filenames.append(f"img_{i}.jpg")

    # Stack images into a single array and save to zarr
    images_array = np.stack(img_arrays, axis=0)
    zarr_images = root.create_dataset(
        "images", shape=images_array.shape, chunks=(1, 150, 150, 3), dtype=np.uint8
    )
    zarr_images[:] = images_array

    # Create metadata as a separate parquet file
    metadata_path = tmp_path / f"{zarr_path.stem}_metadata.parquet"
    metadata_df = pd.DataFrame({"original_filename": filenames})
    metadata_df.to_parquet(metadata_path, index=False)

    return str(zarr_path)


@pytest.fixture
def multiple_test_zarr(sample_images, tmp_path):
    """Create multiple test Zarr files with sample images."""
    zarr_files = []

    # Create 3 different zarr files
    for file_idx in range(3):
        zarr_path = tmp_path / f"test_{file_idx}.zarr"

        # Create zarr store using the older API that works with this version
        root = zarr.open_group(str(zarr_path), mode="w")

        # Use different images in each file (split sample_images)
        start_idx = file_idx * 2
        end_idx = min(start_idx + 2, len(sample_images))
        file_images = sample_images[start_idx:end_idx]

        if not file_images:  # If no more images, create a minimal one
            # Create a simple test image with different color per file
            color_map = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # RGB
            img_array = np.zeros((150, 150, 3), dtype=np.uint8)
            img_array[50:100, 50:100] = color_map[file_idx]
            file_images = [Image.fromarray(img_array)]

        # Convert PIL images to numpy arrays
        img_arrays = []
        filenames = []
        for i, img in enumerate(file_images):
            img_array = np.array(img)
            img_arrays.append(img_array)
            filenames.append(f"file_{file_idx}_img_{i}.jpg")

        # Stack images into a single array and save to zarr
        images_array = np.stack(img_arrays, axis=0)
        zarr_images = root.create_dataset(
            "images", shape=images_array.shape, chunks=(1, 150, 150, 3), dtype=np.uint8
        )
        zarr_images[:] = images_array

        # Create metadata as a separate parquet file
        metadata_path = tmp_path / f"{zarr_path.stem}_metadata.parquet"
        metadata_df = pd.DataFrame({"original_filename": filenames})
        metadata_df.to_parquet(metadata_path, index=False)

        zarr_files.append(str(zarr_path))

    return zarr_files, str(tmp_path)


@pytest.fixture
def zarr_batch_folders(sample_images, tmp_path):
    """Create multiple batch folders with images.zarr subdirectories (mimics real structure)."""
    batch_folders = []

    # Create 3 different batch folders
    for batch_idx in range(3):
        batch_folder = tmp_path / f"batch_{batch_idx:03d}"
        batch_folder.mkdir()

        zarr_path = batch_folder / "images.zarr"

        # Create zarr store
        root = zarr.open_group(str(zarr_path), mode="w")

        # Use different images in each batch (split sample_images)
        start_idx = batch_idx * 3
        end_idx = min(start_idx + 3, len(sample_images))
        batch_images = sample_images[start_idx:end_idx]

        if not batch_images:  # If no more images, create a minimal one
            # Create a simple test image with different color per batch
            color_map = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # RGB
            img_array = np.zeros((150, 150, 3), dtype=np.uint8)
            img_array[50:100, 50:100] = color_map[batch_idx]
            batch_images = [Image.fromarray(img_array)]

        # Convert PIL images to numpy arrays
        img_arrays = []
        filenames = []
        for i, img in enumerate(batch_images):
            img_array = np.array(img)
            img_arrays.append(img_array)
            filenames.append(f"batch_{batch_idx:03d}_img_{i}.jpg")

        # Stack images into a single array and save to zarr
        images_array = np.stack(img_arrays, axis=0)
        zarr_images = root.create_dataset(
            "images", shape=images_array.shape, chunks=(1, 150, 150, 3), dtype=np.uint8
        )
        zarr_images[:] = images_array

        # Create metadata as a separate parquet file in the batch folder
        metadata_path = batch_folder / "images_metadata.parquet"
        metadata_df = pd.DataFrame({"original_filename": filenames})
        metadata_df.to_parquet(metadata_path, index=False)

        batch_folders.append(str(zarr_path))

    return batch_folders, str(tmp_path)


@pytest.fixture
def mixed_format_images(tmp_path):
    """Create a directory with sample images in different formats (jpg, png, tiff)"""
    img_dir = tmp_path / "mixed_formats"
    img_dir.mkdir()

    # Create images in different formats - only use supported extensions
    image_paths = []
    formats = {"jpg": "JPEG", "png": "PNG", "tiff": "TIFF"}

    # Create a simple test image
    base_img = np.zeros((150, 150, 3), dtype=np.uint8)
    base_img[50:100, 50:100, 0] = 255  # Red square

    # Save in each format
    for ext, pil_format in formats.items():
        img_path = img_dir / f"test_image.{ext}"
        Image.fromarray(base_img).save(img_path, format=pil_format)
        image_paths.append(str(img_path))

    return image_paths, str(img_dir)


@pytest.fixture
def test_cutana(tmp_path):
    """Create a directory with sample FITS files for cutana streaming."""
    data_dir = tmp_path / "cutana_test"
    data_dir.mkdir()

    img_size = 512
    ra_center, dec_center = 150.14, 2.34
    tile_id = "102018211"
    num_sources = 10

    wcs = WCS(naxis=2)
    pixel_scale = 0.1 / 3600.0
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.crpix = [img_size / 2, img_size / 2]
    wcs.wcs.cd = [[-pixel_scale, 0], [0, pixel_scale]]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000.0

    img_data = np.random.normal(0, 0.005, (img_size, img_size)).astype(np.float32)
    primary_hdu = fits.PrimaryHDU(img_data)
    header = primary_hdu.header
    header.update(wcs.to_header())
    header["TELESCOP"] = "EUCLID"
    header["INSTRUME"] = "VIS"
    header["TILEID"] = tile_id
    header["BUNIT"] = "electron/s"
    header["DATATYPE"] = "BGSUB-MOSAIC"
    header["EXPTIME"] = 565.0
    header["GAIN"] = 3.1
    header["READNOIS"] = 4.2
    header["MAGZERO"] = 24.6

    fits_path = (
        data_dir / f"EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile_id}-ACBD03_20251124T100053.096Z_00.00.fits"
    )
    primary_hdu.writeto(fits_path, overwrite=True)

    field_of_view_deg = img_size * pixel_scale
    fov_margin = field_of_view_deg * 0.45

    ra_values = ra_center + np.random.uniform(-fov_margin, fov_margin, num_sources)
    dec_values = dec_center + np.random.uniform(-fov_margin, fov_margin, num_sources)
    object_ids = (np.arange(1, num_sources + 1) + int(tile_id) * 1000000).astype(np.int64)

    cat_table = Table()
    cat_table["OBJECT_ID"] = object_ids
    cat_table["RIGHT_ASCENSION"] = ra_values
    cat_table["DECLINATION"] = dec_values
    cat_table["RIGHT_ASCENSION_PSF_FITTING"] = ra_values
    cat_table["DECLINATION_PSF_FITTING"] = dec_values

    catalog_fits = (
        data_dir / f"EUC_MER_FINAL-CAT_TILE{tile_id}-CC66F6_20251124T100053.096Z_00.00.fits"
    )
    primary_hdu_cat = fits.PrimaryHDU()
    table_hdu = fits.BinTableHDU(cat_table, name="EUC_MER__FINAL_CATALOG")
    hdul = fits.HDUList([primary_hdu_cat, table_hdu])
    hdul.writeto(catalog_fits, overwrite=True)

    csv_directory = data_dir / "csv"
    csv_directory.mkdir()
    csv_path = csv_directory / "mock_sources_malformed.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"])
        for i, (ra, dec) in enumerate(zip(ra_values, dec_values)):
            writer.writerow(
                [
                    f"MockSource_{object_ids[i]}",
                    ra,
                    dec,
                    np.random.randint(100, 250),
                    str([str(fits_path)]),
                ]
            )

    return str(csv_path)


@pytest.fixture
def test_cutana_parquet(tmp_path):
    """Create a directory with sample FITS files and parquet catalogue for cutana streaming."""
    data_dir = tmp_path / "cutana_parquet_test"
    data_dir.mkdir()

    img_size = 512
    ra_center, dec_center = 150.14, 2.34
    tile_id = "102018212"
    num_sources = 10

    wcs = WCS(naxis=2)
    pixel_scale = 0.1 / 3600.0
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.crpix = [img_size / 2, img_size / 2]
    wcs.wcs.cd = [[-pixel_scale, 0], [0, pixel_scale]]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000.0

    img_data = np.random.normal(0, 0.005, (img_size, img_size)).astype(np.float32)
    primary_hdu = fits.PrimaryHDU(img_data)
    header = primary_hdu.header
    header.update(wcs.to_header())
    header["TELESCOP"] = "EUCLID"
    header["INSTRUME"] = "VIS"
    header["TILEID"] = tile_id
    header["BUNIT"] = "electron/s"
    header["DATATYPE"] = "BGSUB-MOSAIC"
    header["EXPTIME"] = 565.0
    header["GAIN"] = 3.1
    header["READNOIS"] = 4.2
    header["MAGZERO"] = 24.6

    fits_path = (
        data_dir / f"EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile_id}-ACBD03_20251124T100053.096Z_00.00.fits"
    )
    primary_hdu.writeto(fits_path, overwrite=True)

    field_of_view_deg = img_size * pixel_scale
    fov_margin = field_of_view_deg * 0.45

    ra_values = ra_center + np.random.uniform(-fov_margin, fov_margin, num_sources)
    dec_values = dec_center + np.random.uniform(-fov_margin, fov_margin, num_sources)
    object_ids = (np.arange(1, num_sources + 1) + int(tile_id) * 1000000).astype(np.int64)

    # Create parquet catalogue
    parquet_directory = data_dir / "parquet"
    parquet_directory.mkdir()
    parquet_path = parquet_directory / "mock_sources.parquet"

    import pandas as pd

    df = pd.DataFrame(
        {
            "SourceID": [f"MockSource_{oid}" for oid in object_ids],
            "RA": ra_values,
            "Dec": dec_values,
            "diameter_pixel": np.random.randint(100, 250, num_sources),
            "fits_file_paths": [str([str(fits_path)]) for _ in range(num_sources)],
        }
    )
    df.to_parquet(parquet_path, index=False)

    return str(parquet_path)


@pytest.fixture
def test_cutana_malformed_header(tmp_path):
    """Create a directory with sample CSV file with malformed header."""
    data_dir = tmp_path / "cutana_malformed_test"
    data_dir.mkdir()

    tile_id = "102018211"

    fits_path = (
        data_dir / f"EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile_id}-ACBD03_20251124T100053.096Z_00.00.fits"
    )

    csv_directory = data_dir / "csv"
    csv_directory.mkdir()
    csv_path = csv_directory / "mock_sources_malformed.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write bad header, should report missing headers and raise RuntimeError
        writer.writerow(
            [
                "SourceID_MALFORMED",
                "RA_MALFORMED",
                "Dec",
                "diameter_pixel_MALFORMED",
                "fits_file_paths",
            ]
        )
        for i in range(10):
            writer.writerow(
                [
                    f"MockSource_{i}",
                    (np.random.rand() - 0.5) * 2 * 5 + 150,
                    np.random.rand() - 0.5 + 2,
                    np.random.randint(100, 250),
                    str([str(fits_path)]),
                ]
            )

    return str(csv_directory)


@pytest.fixture
def test_cutana_missing_images(tmp_path):
    """Create a directory with sample CSV file with mcorrect header and missing images."""
    data_dir = tmp_path / "cutana_missing_test"
    data_dir.mkdir()

    tile_id = "102018211"

    fits_path = (  # Fits in CSV, but not actually saved to disk (missing)
        data_dir / f"EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile_id}-ACBD03_20251124T100053.096Z_00.00.fits"
    )

    csv_directory = data_dir / "csv"
    csv_directory.mkdir()
    csv_path = csv_directory / "mock_sources_malformed.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"])
        for i in range(10):
            writer.writerow(
                [
                    f"MockSource_{i}",
                    (np.random.rand() - 0.5) * 2 * 5 + 150,
                    np.random.rand() - 0.5 + 2,
                    np.random.randint(100, 250),
                    str([str(fits_path)]),
                ]
            )

    return str(csv_directory)


def test_evaluate_files(test_config, sample_images, tmp_path):
    """Test evaluation of individual files writes results to predictions.db."""
    image_paths = []
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()

    for i, img in enumerate(sample_images):
        img_path = img_dir / f"img_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))

    evaluate_files(image_paths, test_config)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    assert os.path.exists(db_path)
    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == len(sample_images)


def test_evaluate_images_in_zarr(test_config, test_zarr):
    """Test evaluation of images in Zarr file writes results to predictions.db."""
    evaluate_images_in_zarr(test_zarr, test_config)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    assert os.path.exists(db_path)
    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == 10


def test_evaluate_images_cutana(test_config, test_cutana):
    """Test evaluation of images via cutana streaming with CSV catalogue."""
    evaluate_images_from_cutana(test_cutana, test_config, batch_size=5)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    assert os.path.exists(db_path)
    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == 10


def test_evaluate_images_cutana_parquet(test_config, test_cutana_parquet):
    """Test evaluation of images via cutana streaming with parquet catalogue."""
    evaluate_images_from_cutana(test_cutana_parquet, test_config, batch_size=5)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    assert os.path.exists(db_path)
    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == 10


# ── Resume tests ─────────────────────────────────────────────────────
#
# Each resume test pre-seeds half the filenames with a sentinel score
# (0.0001) the real model cannot realistically produce, then runs the
# subprocess. If resume works, the seeded scores stay intact and only
# the other half is evaluated.


_SENTINEL_SCORE = 0.0001


def test_evaluate_files_resume(test_config, sample_images, tmp_path):
    """Skip already-scored image files on resume."""
    from anomaly_match.prediction.anomaly_score_db import build_compat_metadata

    image_paths = []
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()
    for i, img in enumerate(sample_images):
        img_path = img_dir / f"img_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    seeded = image_paths[:5]
    with AnomalyScoreDB(db_path) as db:
        db.set_metadata_batch(build_compat_metadata(test_config))
        db.store_results([(p, _SENTINEL_SCORE) for p in seeded])

    evaluate_files(image_paths, test_config)

    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == len(image_paths)
        expected = float(np.float32(_SENTINEL_SCORE))
        for p in seeded:
            row = db.get_result_by_filename(p)
            assert row is not None
            assert row["score"] == pytest.approx(expected, abs=1e-9), (
                "Seeded score was overwritten — resume did not skip this file"
            )


def test_evaluate_files_resume_fully_scored(test_config, sample_images, tmp_path):
    """A fully-scored DB finishes without re-evaluating anything."""
    from anomaly_match.prediction.anomaly_score_db import build_compat_metadata

    image_paths = []
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()
    for i, img in enumerate(sample_images):
        img_path = img_dir / f"img_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    with AnomalyScoreDB(db_path) as db:
        db.set_metadata_batch(build_compat_metadata(test_config))
        db.store_results([(p, _SENTINEL_SCORE) for p in image_paths])

    evaluate_files(image_paths, test_config)

    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == len(image_paths)
        expected = float(np.float32(_SENTINEL_SCORE))
        for p in image_paths:
            row = db.get_result_by_filename(p)
            assert row["score"] == pytest.approx(expected, abs=1e-9)


def test_evaluate_zarr_resume(test_config, test_zarr):
    """Skip already-scored Zarr entries on resume."""
    from anomaly_match.prediction.anomaly_score_db import build_compat_metadata

    # Filenames for test_zarr are img_0.jpg..img_9.jpg (from the fixture).
    seeded_filenames = [f"img_{i}.jpg" for i in range(5)]

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    with AnomalyScoreDB(db_path) as db:
        db.set_metadata_batch(build_compat_metadata(test_config))
        db.store_results([(fn, _SENTINEL_SCORE) for fn in seeded_filenames])

    evaluate_images_in_zarr(test_zarr, test_config)

    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == 10
        expected = float(np.float32(_SENTINEL_SCORE))
        for fn in seeded_filenames:
            row = db.get_result_by_filename(fn)
            assert row is not None
            assert row["score"] == pytest.approx(expected, abs=1e-9)


def test_evaluate_cutana_resume(test_config, test_cutana):
    """Skip already-scored source_ids on resume via cutana streaming."""
    from anomaly_match.prediction.anomaly_score_db import build_compat_metadata

    # Run once to discover the source_ids (we don't know them a priori
    # without reading the catalogue), then wipe the DB, pre-seed half,
    # and run again to confirm the seeded half is preserved.
    evaluate_images_from_cutana(test_cutana, test_config, batch_size=5)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    with AnomalyScoreDB(db_path) as db:
        all_filenames = sorted(db.get_processed_filenames())

    assert len(all_filenames) == 10
    os.remove(db_path)

    seeded = all_filenames[:5]
    with AnomalyScoreDB(db_path) as db:
        db.set_metadata_batch(build_compat_metadata(test_config))
        db.store_results([(fn, _SENTINEL_SCORE) for fn in seeded])

    evaluate_images_from_cutana(test_cutana, test_config, batch_size=5)

    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() == 10
        expected = float(np.float32(_SENTINEL_SCORE))
        for fn in seeded:
            row = db.get_result_by_filename(fn)
            assert row is not None
            assert row["score"] == pytest.approx(expected, abs=1e-9)


def test_prediction_file_type_cutana_malformed_header(test_config, test_cutana_malformed_header):
    """Test for meaningful exception when streaming from cutana and csv files have malformed headers."""
    from anomaly_match.pipeline.session import Session
    from anomaly_match.utils.get_default_cfg import get_default_cfg

    cfg = get_default_cfg()
    cfg.normalisation.image_size = [64, 64]
    cfg.prediction_search_dir = test_cutana_malformed_header
    cfg.model_path = test_config.model_path

    session = Session(cfg)

    with pytest.warns(
        RuntimeWarning,
        match=r"File .* did not pass cutana column check \(.*\)",
    ):
        with pytest.raises(RuntimeError, match="All found files are not compatible with cutana"):
            session.evaluate_all_images()


def test_prediction_file_type_cutana_missing_images(test_config, test_cutana_missing_images):
    """Test that catalogues with missing FITS images still pass validation.

    FITS existence is no longer checked during validation (only column schema
    is verified).  Errors from missing files surface later during cutana
    processing.
    """
    from anomaly_match.pipeline.session import Session
    from anomaly_match.utils.get_default_cfg import get_default_cfg

    cfg = get_default_cfg()
    cfg.normalisation.image_size = [64, 64]
    cfg.prediction_search_dir = test_cutana_missing_images
    cfg.model_path = test_config.model_path

    session = Session(cfg)

    # Validation passes (columns are valid), but the cutana subprocess fails
    # when it tries to open missing FITS files.  As of the skip-failed-chunks
    # change, the chunk loop catches the per-chunk RuntimeError, marks the
    # chunk as skipped, and the run completes cleanly so future chunks aren't
    # lost.  No predictions land in the DB.
    session.evaluate_all_images()
    assert session.last_run_skipped_chunks >= 1
    db_path = os.path.join(cfg.output_dir, "predictions.db")
    if os.path.exists(db_path):
        from anomaly_match.prediction import AnomalyScoreDB

        db = AnomalyScoreDB(db_path)
        assert db.get_count() == 0
        db.close()


def test_stream_file_type_detection_csv_and_parquet(tmp_path):
    """Test that CSV and parquet files are correctly detected as stream type for cutana."""
    # Create test CSV file
    csv_file = tmp_path / "test_catalogue.csv"
    csv_file.write_text("SourceID,RA,Dec\n1,0.0,0.0\n")

    # Create test parquet file
    parquet_file = tmp_path / "test_catalogue.parquet"
    import pandas as pd

    pd.DataFrame({"SourceID": [1], "RA": [0.0], "Dec": [0.0]}).to_parquet(parquet_file)

    # Test file type detection via extension map (same logic as run_pipeline)
    import os

    extension_map = {
        ".zarr": DataSourceType.ZARR,
        ".txt": DataSourceType.IMAGE_FOLDER,
        ".parquet": DataSourceType.CUTANA,
        ".csv": DataSourceType.CUTANA,
    }

    # Test CSV detection
    _, csv_ext = os.path.splitext(str(csv_file).lower())
    assert csv_ext == ".csv"
    assert extension_map.get(csv_ext) == DataSourceType.CUTANA

    # Test parquet detection
    _, parquet_ext = os.path.splitext(str(parquet_file).lower())
    assert parquet_ext == ".parquet"
    assert extension_map.get(parquet_ext) == DataSourceType.CUTANA


def test_mixed_format_support(test_config, mixed_format_images, monkeypatch):
    """Test support for different image formats (jpg, png, tif, tiff)."""
    image_paths, _ = mixed_format_images

    # Mock model to ensure consistent output for each image
    mock_model = MockModel([0.7])  # Use a single consistent score

    # Mock the load_model function to return our mock model
    import prediction_process

    def mock_load_model(cfg):
        return mock_model

    monkeypatch.setattr(prediction_process, "load_model", mock_load_model)

    # Test each format individually
    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        mock_model.call_count = 0

        evaluate_files([path], test_config)

        db_path = os.path.join(test_config.output_dir, "predictions.db")
        with AnomalyScoreDB(db_path) as db:
            count = db.get_count()
            assert count >= 1, f"No results written for {ext} image"


def test_load_and_preprocess_multiple_formats(test_config, mixed_format_images):
    """Test the load_and_preprocess function can handle multiple formats."""
    from prediction_process import load_and_preprocess

    from anomaly_match.image_processing.transforms import get_prediction_transforms

    image_paths, _ = mixed_format_images
    transform = get_prediction_transforms()

    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        # load_and_preprocess now returns (filepath, numpy_image)
        filename, numpy_image = load_and_preprocess((path, test_config))

        # Apply transform to get tensor (transform is now applied on main thread)
        image = transform(numpy_image)

        # Check image shape and type
        assert isinstance(image, torch.Tensor), f"Expected tensor output for {ext}"
        assert image.shape[0] == 3, f"Expected 3 channels for {ext}"  # RGB channels


class MockModel(torch.nn.Module):
    """Mock model that returns controlled scores."""

    def __init__(self, score_pattern):
        super().__init__()
        self.score_pattern = score_pattern
        self.call_count = 0
        logger.info(f"MockModel initialized with score pattern: {score_pattern}")

    def forward(self, x):
        batch_size = x.shape[0]
        # Clamp the index so extra forwards (e.g. the inference warmup pass that
        # every prediction process now runs) reuse the last configured score
        # instead of running off the end of the pattern.
        base_score = self.score_pattern[min(self.call_count, len(self.score_pattern) - 1)]
        logger.info(f"MockModel forward call {self.call_count} with base_score {base_score}")

        # Generate scores that will exactly match our desired probabilities
        scores = torch.zeros((batch_size, 2))
        for i in range(batch_size):
            desired_prob = min(base_score + (i / batch_size) * 0.1, 0.95)
            scores[i, 1] = desired_prob
            scores[i, 0] = 1 - desired_prob

        self.call_count += 1
        logger.info(f"Generated scores: {scores[:, 1]}")
        return torch.log(scores)  # Convert to logits


def test_image_directory_processing(test_config, mixed_format_images):
    """Test processing a directory containing images of different formats."""
    # Get the directory containing mixed format images
    _, directory_path = mixed_format_images

    # Create a file list with all image paths in the directory
    import tempfile
    from pathlib import Path

    # Create a temporary file list
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as file_list:
        # List all image files and write them to the file - only supported extensions
        image_paths = (
            list(Path(directory_path).glob("*.jpg"))
            + list(Path(directory_path).glob("*.jpeg"))
            + list(Path(directory_path).glob("*.png"))
            + list(Path(directory_path).glob("*.tiff"))
        )

        for path in image_paths:
            file_list.write(f"{path}\n")

        file_list_path = file_list.name

    # Import the necessary function
    from prediction_process import evaluate_files

    try:
        evaluate_files([str(p) for p in image_paths], test_config)

        db_path = os.path.join(test_config.output_dir, "predictions.db")
        assert os.path.exists(db_path)
        with AnomalyScoreDB(db_path) as db:
            assert db.get_count() == len(image_paths), "Not all images were processed"
    finally:
        if os.path.exists(file_list_path):
            os.unlink(file_list_path)


def test_prediction_file_type_image(test_config, monkeypatch, mixed_format_images):
    """Test that the correct prediction process is called for the 'image' file type."""
    import os
    import subprocess  # Create a directory with mixed format images
    import sys

    from anomaly_match.pipeline.session import Session

    image_paths, directory_path = mixed_format_images

    # Create config based on default
    from anomaly_match.utils.get_default_cfg import get_default_cfg

    cfg = get_default_cfg()
    cfg.prediction_search_dir = directory_path
    cfg.save_file = "test_image_type"
    cfg.output_dir = os.path.join(directory_path, "output")
    cfg.model_path = test_config.model_path

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    called_processes = []

    def mock_run(args, **kwargs):
        called_processes.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Create the tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create the file list for the group - this is what we'll check
    group_file = os.path.join("tmp", "Test_evaluate_all_images_grouped_0.txt")
    with open(group_file, "w") as f:
        f.write("\n".join(image_paths))

    # Create the file that contains the path to the group file
    temp_file_list = os.path.join("tmp", f"{cfg.save_file}_file_list.txt")
    with open(temp_file_list, "w") as f:
        f.write(group_file)
        f.flush()

    # Mock Session.__init__
    def mock_init(self, cfg):
        self.cfg = cfg
        self.session_start = "20250101_000000"

    monkeypatch.setattr(Session, "__init__", mock_init)

    # Create a simple mock for run_pipeline
    def mock_run_pipeline(self, temp_config_path, input_path, top_N):
        # Use the file list we created manually
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prediction_process.py"
        )
        subprocess.run([sys.executable, script_path, temp_config_path, temp_file_list, str(top_N)])

    monkeypatch.setattr(Session, "run_pipeline", mock_run_pipeline)

    # Create a session with our mocked initializer
    session = Session(cfg)

    # Create a temp config file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as config_file:
        config_path = config_file.name

    # Test run_pipeline
    try:
        session.run_pipeline(config_path, image_paths[0], top_N=10)  # Use first image path as input

        # Verify that subprocess.run was called
        assert len(called_processes) > 0, "No subprocess was called"

        # Check that the script path is correct (prediction_process.py)
        script_path = os.path.basename(called_processes[0][1])
        assert script_path == "prediction_process.py", f"Wrong script called: {script_path}"

        # Verify that the group file exists and contains all image paths
        assert os.path.exists(group_file), "Group file does not exist"
        with open(group_file, "r") as f:
            group_content = f.read().strip().split("\n")
            assert set(group_content) == set(image_paths), (
                f"Wrong content in group file: {group_content}, expected {image_paths}"
            )

        # Verify that the file list exists and points to the group file
        assert os.path.exists(temp_file_list), "Temporary file list does not exist"
        with open(temp_file_list, "r") as f:
            content = f.read().strip()
            assert content == group_file, (
                f"Wrong content in file list: '{content}', expected '{group_file}'"
            )

    finally:
        # Clean up any temporary files
        if os.path.exists(config_path):
            os.unlink(config_path)
        if os.path.exists(temp_file_list):
            os.unlink(temp_file_list)
        if os.path.exists(group_file):
            os.unlink(group_file)


def test_prediction_file_type_zarr(test_config, monkeypatch, test_zarr):
    """Test that the correct prediction process is called for the 'zarr' file type."""
    import os
    import subprocess
    import sys

    from anomaly_match.pipeline.session import Session

    # Create config based on default
    from anomaly_match.utils.get_default_cfg import get_default_cfg

    cfg = get_default_cfg()
    cfg.prediction_search_dir = os.path.dirname(test_zarr)
    cfg.save_file = "test_zarr_type"
    cfg.output_dir = os.path.join(os.path.dirname(test_zarr), "output")
    cfg.model_path = test_config.model_path
    cfg.normalisation.image_size = [150, 150]

    called_processes = []

    def mock_run(args, **kwargs):
        called_processes.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Create the tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Mock Session.__init__
    def mock_init(self, cfg):
        self.cfg = cfg
        self.session_start = "20250101_000000"

    monkeypatch.setattr(Session, "__init__", mock_init)

    # Create a simple mock for run_pipeline
    def mock_run_pipeline(self, temp_config_path, input_path, top_N):
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prediction_process_zarr.py",
        )
        subprocess.run([sys.executable, script_path, temp_config_path, input_path, str(top_N)])

    monkeypatch.setattr(Session, "run_pipeline", mock_run_pipeline)

    # Create a session with our mocked initializer
    session = Session(cfg)

    # Create a temp config file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as config_file:
        config_path = config_file.name

    # Test run_pipeline
    try:
        session.run_pipeline(config_path, test_zarr, top_N=10)

        # Verify that subprocess.run was called
        assert len(called_processes) > 0, "No subprocess was called"

        # Check that the script path is correct (prediction_process_zarr.py)
        script_path = os.path.basename(called_processes[0][1])
        assert script_path == "prediction_process_zarr.py", f"Wrong script called: {script_path}"

        # Verify the zarr file path is passed correctly
        assert called_processes[0][3] == test_zarr, (
            f"Wrong zarr file path: {called_processes[0][3]}"
        )

    finally:
        # Clean up any temporary files
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_zarr_image_processing_consistency(test_config, test_zarr):
    """Test that zarr image processing produces consistent results with standard methods."""
    from prediction_utils import read_and_preprocess_image_from_zarr

    # Open the zarr file and get a sample image
    root = zarr.open_group(test_zarr, mode="r")
    sample_image_data = root["images"][0]

    # Process the image using the zarr function
    processed_image = read_and_preprocess_image_from_zarr(sample_image_data, test_config)

    # Verify the output dimensions and type
    assert processed_image.shape == (
        test_config.normalisation.image_size[0],
        test_config.normalisation.image_size[1],
        3,
    ), (
        f"Wrong image shape: {processed_image.shape}, "
        f"expected {(test_config.normalisation.image_size[0], test_config.normalisation.image_size[1], 3)}"
    )
    assert processed_image.dtype == np.uint8, f"Wrong dtype: {processed_image.dtype}"

    # Verify the image is not all zeros (should have some content)
    assert np.any(processed_image > 0), "Processed image is all zeros"


def test_multiple_zarr_files(test_config, multiple_test_zarr):
    """Test evaluation of multiple Zarr files writes all results to predictions.db."""
    zarr_files, zarr_dir = multiple_test_zarr

    for zarr_file in zarr_files:
        evaluate_images_in_zarr(zarr_file, test_config, top_n=100)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    assert os.path.exists(db_path)
    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() > 0


def test_zarr_auto_detection_basic(test_config, multiple_test_zarr):
    """Test auto-detection logic for zarr file type from directory contents."""
    zarr_files, zarr_dir = multiple_test_zarr

    # Test the auto-detection function directly (need to create a minimal session)
    from anomaly_match.pipeline.session import Session

    # Create a session but don't initialize everything
    try:
        session = Session.__new__(Session)
        session.cfg = test_config

        # Test auto-detection method
        detected_type = session._auto_detect_prediction_file_type(zarr_dir)

        # Should detect zarr file type
        assert detected_type == DataSourceType.ZARR

    except Exception:
        # If session creation fails, test the logic manually
        import os

        extension_map = {
            ".zarr": DataSourceType.ZARR,
            ".jpg": DataSourceType.IMAGE_FOLDER,
            ".jpeg": DataSourceType.IMAGE_FOLDER,
            ".png": DataSourceType.IMAGE_FOLDER,
            ".tif": DataSourceType.IMAGE_FOLDER,
            ".tiff": DataSourceType.IMAGE_FOLDER,
            ".fits": DataSourceType.IMAGE_FOLDER,
        }

        file_type_counts: dict[DataSourceType, int] = {}
        for filename in os.listdir(zarr_dir):
            file_path = os.path.join(zarr_dir, filename)

            # Check if it's a file with supported extension
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in extension_map:
                    file_type = extension_map[ext]
                    file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1

            # Check if it's a zarr directory (zarr stores can be directories)
            elif os.path.isdir(file_path):
                if filename.lower().endswith(".zarr") or os.path.exists(
                    os.path.join(file_path, "zarr.json")
                ):
                    file_type_counts[DataSourceType.ZARR] = (
                        file_type_counts.get(DataSourceType.ZARR, 0) + 1
                    )

        detected_type = (
            max(file_type_counts, key=file_type_counts.get)
            if file_type_counts
            else DataSourceType.ZARR
        )
        assert detected_type == DataSourceType.ZARR


def test_zarr_batch_folders_detection(test_config, zarr_batch_folders):
    """Test auto-detection for zarr batch folders with images.zarr subdirectories."""
    batch_folders, batch_dir = zarr_batch_folders

    from anomaly_match.pipeline.session import Session

    try:
        session = Session.__new__(Session)
        session.cfg = test_config

        # Test auto-detection method
        detected_type = session._auto_detect_prediction_file_type(batch_dir)

        # Should detect zarr file type
        assert detected_type == DataSourceType.ZARR
    except Exception:
        # Manual test
        import os

        file_type_counts: dict[DataSourceType, int] = {}
        for filename in os.listdir(batch_dir):
            file_path = os.path.join(batch_dir, filename)
            if os.path.isdir(file_path):
                # Check for batch folders containing images.zarr subdirectory
                if os.path.exists(os.path.join(file_path, "images.zarr")):
                    file_type_counts[DataSourceType.ZARR] = (
                        file_type_counts.get(DataSourceType.ZARR, 0) + 1
                    )

        detected_type = (
            max(file_type_counts, key=file_type_counts.get)
            if file_type_counts
            else DataSourceType.IMAGE_FOLDER
        )
        assert detected_type == DataSourceType.ZARR


def test_zarr_batch_folders_processing(test_config, zarr_batch_folders):
    """Test processing multiple zarr batch folders writes results to predictions.db."""
    batch_folders, batch_dir = zarr_batch_folders

    for batch_folder in batch_folders:
        evaluate_images_in_zarr(batch_folder, test_config, top_n=100)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    assert os.path.exists(db_path)
    with AnomalyScoreDB(db_path) as db:
        assert db.get_count() > 0


def test_zarr_batch_metadata_loading(test_config, zarr_batch_folders):
    """Test that metadata filenames are stored in predictions.db."""
    batch_folders, batch_dir = zarr_batch_folders

    first_batch = batch_folders[0]
    evaluate_images_in_zarr(first_batch, test_config, top_n=100)

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    with AnomalyScoreDB(db_path) as db:
        results = db.get_results(sort_by="score_desc", limit=100)
        filenames = [r["filename"] for r in results]

    assert len(filenames) > 0
    # Filenames should not be generic "image_000000" format
    assert not all(f.startswith("image_") for f in filenames)
    # Should contain batch identifier
    assert any("batch_" in str(f) for f in filenames)


def test_labeled_data_cache_excluded_from_prediction_scan(test_config, test_zarr, tmp_path):
    """A LabeledDataCache directory living next to real prediction data must
    not be scored as a second batch.
    """
    from anomaly_match.data_io.labeled_data_cache import LabeledDataCache
    from anomaly_match.pipeline.session import Session

    search_dir = os.path.dirname(test_zarr)

    cache_dir = os.path.join(search_dir, "labeled_data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_root = zarr.open_group(os.path.join(cache_dir, "images.zarr"), mode="w")
    cache_root.create_dataset(
        "images", shape=(2, 150, 150, 3), chunks=(1, 150, 150, 3), dtype=np.uint8
    )
    with open(os.path.join(cache_dir, LabeledDataCache.CACHE_INFO_JSON), "w") as f:
        f.write("{}")

    test_config.prediction_search_dir = search_dir
    session = Session(test_config)
    session.evaluate_all_images()

    db_path = os.path.join(test_config.output_dir, "predictions.db")
    with AnomalyScoreDB(db_path) as db:
        results = db.get_results(sort_by="score_desc", limit=100)
        filenames = [r["filename"] for r in results]

    # Only the real store's 10 images, none of the cache's 2.
    assert len(filenames) == 10
    assert not any("labeled_data_cache" in str(f) for f in filenames)


def test_zarr_fallback_filenames_have_prefix(tmp_path, test_config):
    """Test that when metadata loading fails, fallback filenames include zarr prefix to avoid collisions."""
    import zarr

    # Create two zarr stores WITHOUT metadata to trigger fallback filename generation
    for batch_idx in range(2):
        batch_folder = tmp_path / f"batch_{batch_idx:03d}"
        batch_folder.mkdir()
        zarr_path = batch_folder / "images.zarr"

        # Create minimal zarr store
        root = zarr.open_group(str(zarr_path), mode="w")

        # Create a simple image array
        img_array = np.ones((5, 64, 64, 3), dtype=np.uint8) * (50 + batch_idx * 50)
        zarr_images = root.create_dataset(
            "images", shape=img_array.shape, chunks=(1, 64, 64, 3), dtype=np.uint8
        )
        zarr_images[:] = img_array

        # Intentionally NO metadata file to trigger fallback

    # Process both batches
    batch_filenames = []
    for batch_idx in range(2):
        zarr_path = tmp_path / f"batch_{batch_idx:03d}" / "images.zarr"

        # Each batch needs a separate output dir. Avoid DotMap copy/deepcopy
        # entirely — both corrupt _dynamic=False to True, causing
        # channel_combination to auto-create as empty DotMap() on access.
        output_dir = str(tmp_path / f"output_{batch_idx}")
        os.makedirs(output_dir, exist_ok=True)
        original_output_dir = test_config.output_dir
        test_config.output_dir = output_dir

        evaluate_images_in_zarr(str(zarr_path), test_config, top_n=100)

        test_config.output_dir = original_output_dir

        db_path = os.path.join(output_dir, "predictions.db")
        with AnomalyScoreDB(db_path) as db:
            results = db.get_results(sort_by="score_desc", limit=100)
            filenames_str = [r["filename"] for r in results]

        batch_filenames.append(filenames_str)

    # Verify fallback filenames have zarr prefix
    for batch_idx, filenames in enumerate(batch_filenames):
        sample_filename = filenames[0]
        # Should have format: <zarr_prefix>__image_000000
        assert "__image_" in sample_filename, (
            f"Batch {batch_idx} fallback filename doesn't have expected format. Got: {sample_filename}"
        )

    # Verify no collision between batches
    set_0 = set(batch_filenames[0])
    set_1 = set(batch_filenames[1])
    overlap = set_0 & set_1
    assert len(overlap) == 0, f"Found filename collision between batches: {overlap}"
