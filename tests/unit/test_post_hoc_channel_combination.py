#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""A channel_combination that changes the channel count must reach every decode path.

For non-FITS sources AnomalyMatch applies the matrix after fitsbolt, so fitsbolt
has to return the matrix's *input* channels. Asking it for ``n_output_channels``
made a 1- or 2-output matrix on RGB data fail to decode, and the training dataset
then raised ``n_output_channels`` back to the source's band count.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import zarr
from PIL import Image

from anomaly_match.data_io.container_loaders import decode_zarr_gallery_images, decode_zarr_image
from anomaly_match.data_io.load_images import (
    fitsbolt_decode_config,
    get_fitsbolt_config,
    load_and_process_single_wrapper,
    load_and_process_wrapper,
)
from anomaly_match.data_io.source_scanning import DataSourceType, load_preview_samples
from anomaly_match.datasets.AnomalyDetectionDataset import AnomalyDetectionDataset
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match_ui.utils.backend_interface import BackendInterface

SIZE = 16
# Random pixels, not constants: an axis swap or wrong transpose must change the output.
RGB = np.random.default_rng(0).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
MATRICES = {
    "1x3": np.array([[0.0, 0.0, 1.0]]),
    "2x3": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


def _expected(matrix):
    """CONVERSION_ONLY on uint8 at matching size is identity, so the output is RGB @ matrix."""
    return np.stack([RGB[..., int(np.argmax(row))] for row in matrix], axis=-1)


def _cfg(matrix, fits_extension=None):
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [SIZE, SIZE]
    cfg.normalisation.fits_extension = fits_extension
    cfg.normalisation.channel_combination = matrix
    cfg.normalisation.n_output_channels = matrix.shape[0]
    return get_fitsbolt_config(cfg)


def _store(path, layout, n=2):
    data = np.stack([RGB if layout == "HWC" else RGB.transpose(2, 0, 1)] * n)
    zarr.open_group(str(path), mode="w").create_array("images", shape=data.shape, dtype=np.uint8)[
        :
    ] = data
    return path


@pytest.fixture()
def rgb_png(tmp_path):
    path = tmp_path / "rgb.png"
    Image.fromarray(RGB).save(path)
    return str(path)


class TestFitsboltDecodeConfig:
    def test_non_fits_requests_the_matrix_inputs(self):
        assert fitsbolt_decode_config(_cfg(MATRICES["1x3"])).n_output_channels == 3

    def test_shared_config_keeps_the_model_channel_count(self):
        """checkpoint_io saves this value as the model's channel count."""
        cfg = _cfg(MATRICES["1x3"])
        fitsbolt_decode_config(cfg)
        assert cfg.fitsbolt_cfg.n_output_channels == 1

    def test_fits_source_unchanged(self):
        """fitsbolt combines FITS bands itself, so it must still produce the output count."""
        cfg = _cfg(MATRICES["1x3"], fits_extension=[0, 1, 2])
        assert fitsbolt_decode_config(cfg).n_output_channels == 1

    def test_per_band_asinh_values_survive(self):
        """get_fitsbolt_config cuts the lists to n_output; the decode copy must restore them."""
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [SIZE, SIZE]
        cfg.normalisation.channel_combination = MATRICES["1x3"]
        cfg.normalisation.n_output_channels = 1
        cfg.normalisation.norm_asinh_scale = [0.1, 0.5, 0.9]
        cfg = get_fitsbolt_config(cfg)
        assert list(fitsbolt_decode_config(cfg).normalisation.asinh_scale) == [0.1, 0.5, 0.9]


@pytest.mark.parametrize("name", MATRICES)
class TestImageFolderDecode:
    def test_single_wrapper(self, rgb_png, name):
        image = load_and_process_single_wrapper(rgb_png, _cfg(MATRICES[name]))
        np.testing.assert_array_equal(image, _expected(MATRICES[name]))

    def test_batch_wrapper(self, rgb_png, name):
        ((_, image),) = load_and_process_wrapper(
            [rgb_png], _cfg(MATRICES[name]), show_progress=False
        )
        np.testing.assert_array_equal(image, _expected(MATRICES[name]))


@pytest.mark.parametrize("layout", ["HWC", "CHW"])
@pytest.mark.parametrize("name", MATRICES)
class TestZarr:
    def test_decode(self, name, layout):
        """Channel-first stores were not transposed once a matrix changed the channel count."""
        raw = RGB if layout == "HWC" else RGB.transpose(2, 0, 1)
        image = decode_zarr_image(raw, _cfg(MATRICES[name]))
        np.testing.assert_array_equal(image, _expected(MATRICES[name]))

    def test_decode_ignores_a_stale_fits_extension(self, name, layout):
        """fits_extension may be left over from a FITS session; Zarr still combines post-hoc."""
        cfg = _cfg(MATRICES[name])
        cfg.normalisation.fits_extension = ["VIS", "NIR-Y", "NIR-H"]
        raw = RGB if layout == "HWC" else RGB.transpose(2, 0, 1)
        np.testing.assert_array_equal(decode_zarr_image(raw, cfg), _expected(MATRICES[name]))

    def test_preview(self, tmp_path, name, layout):
        """The preview used to show raw store pixels, ignoring the matrix."""
        store = _store(tmp_path / "data.zarr", layout)
        samples = load_preview_samples(_cfg(MATRICES[name]), str(store), DataSourceType.ZARR)
        assert len(samples) == 2
        for _name, image in samples:
            np.testing.assert_array_equal(image, _expected(MATRICES[name]))

    def test_gallery(self, tmp_path, name, layout):
        """Gallery thumbnails were raw (and uint8-clipped) store pixels."""
        store = _store(tmp_path / "search" / "data.zarr", layout)
        cfg = _cfg(MATRICES[name])
        cfg.prediction_search_dir = str(store.parent)
        session = MagicMock()
        session.cfg = cfg
        BackendInterface.set_session(session)
        try:
            images = BackendInterface.load_container_images(["data__image_000001"])
        finally:
            BackendInterface.set_session(None)
        np.testing.assert_array_equal(images["data__image_000001"], _expected(MATRICES[name]))


def test_gallery_drops_only_the_image_that_fails():
    """One undecodable entry must not blank the rest of the gallery page."""
    bad = np.zeros((SIZE, SIZE, 5), dtype=np.uint8)  # 5 bands for a 3-column matrix
    decoded = decode_zarr_gallery_images({"good": RGB, "bad": bad}, _cfg(MATRICES["1x3"]))
    assert set(decoded) == {"good"}
    np.testing.assert_array_equal(decoded["good"], _expected(MATRICES["1x3"]))


class _FakeRgbSource:
    """Three-band source: the dataset's auto-detection reports 3 channels."""

    def detect_num_channels(self):
        return 3

    def get_total_count(self):
        return 2

    def get_labeled_images(self, label_df):
        return [("A", np.zeros((8, 8, 1), dtype=np.uint8))]

    def get_unlabeled_batch(self, n, exclude=None):
        return [("B", np.zeros((8, 8, 1), dtype=np.uint8))]


def test_dataset_keeps_the_matrix_output_count(tmp_path):
    """A 1x3 matrix outputs one channel; the detected 3 bands must not override it."""
    cfg = _cfg(MATRICES["1x3"])
    cfg.metadata_file = None
    cfg.N_to_load = 10
    cfg.test_ratio = 0.0
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["A"], "label": ["normal"]}).to_csv(label_csv, index=False)
    cfg.label_file = str(label_csv)

    dset = AnomalyDetectionDataset(cfg=cfg, data_source=_FakeRgbSource())

    assert dset.num_channels == 1
    assert cfg.normalisation.n_output_channels == 1
