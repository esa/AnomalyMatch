#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for blanking an output channel with an all-zero channel_combination row (#588)."""

import io

import numpy as np
import pytest
from astropy.io import fits
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from PIL import Image

import anomaly_match as am
from anomaly_match.data_io.checkpoint_io import sync_normalisation_from_checkpoint
from anomaly_match.data_io.container_loaders import decode_zarr_image
from anomaly_match.data_io.load_images import (
    blank_output_rows,
    fitsbolt_safe_channel_combination,
    get_fitsbolt_config,
    load_and_process_single_wrapper,
    load_and_process_wrapper,
    zero_blank_output_channels,
)
from anomaly_match.utils.validate_config import validate_config
from anomaly_match_ui.utils.image_utils import numpy_array_to_byte_stream

# Blanks output channel 1; rows 0 and 2 are identical to those of CONTROL, so any
# difference between the two decodes is caused by the blanking alone.
BLANKED = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1.0]])
CONTROL = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 1.0]])

# Output channel 1 is the difference of two bands: the row sums to zero but is
# not blank, and must not be mistaken for one.
DIFFERENCE = np.array([[1.0, 0, 0], [1.0, -1.0, 0], [0, 0, 1.0]])

# Cutana catalogues name their bands here rather than giving FITS extension
# indices, which is the second face of #588.
BAND_NAMES = ["VIS", "NIR-H", "NIR-J"]

NORMALISATION_METHODS = [
    NormalisationMethod.CONVERSION_ONLY,
    NormalisationMethod.LOG,
    NormalisationMethod.ZSCALE,
    NormalisationMethod.ASINH,
]


@pytest.fixture(scope="module")
def three_band_data(tmp_path_factory):
    """Three bands as a multi-extension FITS file, an RGB PNG and a CHW array.

    Yields:
        tuple: ``(fits_path, png_path, chw_array)`` over the same pixel data, so
        the three decode paths can be compared against each other.
    """
    directory = tmp_path_factory.mktemp("blanking")
    rng = np.random.default_rng(7)
    bands = [rng.uniform(1.0, 1000.0, (40, 40)).astype(np.float32) for _ in range(3)]

    fits_path = directory / "source.fits"
    hdus = [fits.PrimaryHDU(bands[0])] + [fits.ImageHDU(band) for band in bands[1:]]
    fits.HDUList(hdus).writeto(fits_path)

    png_path = directory / "source.png"
    Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(png_path)

    yield str(fits_path), str(png_path), np.stack(bands, axis=0)


def _cfg(channel_combination, fits_extension, method=NormalisationMethod.CONVERSION_ONLY):
    """Build a validated 3-output-channel config for the given matrix."""
    cfg = am.get_default_cfg()
    cfg.num_workers = 1
    # Flux conversion reads only the primary HDU, so it cannot serve a
    # multi-extension channel_combination at all; keep it out of the way here.
    cfg.normalisation.apply_flux_conversion = False
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.normalisation_method = method
    cfg.normalisation.fits_extension = fits_extension
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.channel_combination = channel_combination
    validate_config(cfg, check_paths=False)
    return cfg


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def test_blank_output_rows_finds_all_zero_rows():
    assert blank_output_rows(BLANKED) == [1]
    assert blank_output_rows(CONTROL) == []
    assert blank_output_rows(None) == []


def test_safe_matrix_returns_input_untouched_when_nothing_is_blank():
    # Identity matrices are short-circuited downstream, so a matrix without blank
    # rows must come back as-is rather than as an equal-but-rebuilt array.
    identity = np.eye(3)
    assert fitsbolt_safe_channel_combination(identity) is identity


def test_safe_matrix_substitutes_a_surviving_row():
    safe = fitsbolt_safe_channel_combination(BLANKED)

    assert safe.shape == BLANKED.shape
    assert blank_output_rows(safe) == []
    # The substituted row duplicates a surviving output channel, which is what
    # keeps the kept channels' normalisation unchanged.
    np.testing.assert_array_equal(safe[1], BLANKED[0])
    np.testing.assert_array_equal(safe[[0, 2]], BLANKED[[0, 2]])


def test_a_row_summing_to_zero_is_not_blank():
    # [1, -1] is a band difference, not an empty channel. Blanking it would
    # replace real data with zeros and, because the substitution drops the
    # negative weight before fitsbolt sees it, would also swallow the error
    # fitsbolt owes a FITS source with negative weights.
    assert blank_output_rows(DIFFERENCE) == []
    assert fitsbolt_safe_channel_combination(DIFFERENCE) is DIFFERENCE


def test_negative_weights_still_fail_loudly_on_a_fits_source():
    # fitsbolt rejects negative weights for FITS inputs; blanking support must
    # not turn that hard failure into a silently emptied channel.
    with pytest.raises(ValueError, match="negative"):
        _cfg(DIFFERENCE, [0, 1, 2])


def test_rows_of_negligible_weights_still_count_as_blank():
    # The tolerance is per weight, so a row too small to matter is blanked here
    # rather than tripping fitsbolt's own zero-sum check.
    negligible = np.array([[1.0, 0, 0], [1e-12, 1e-12, 0], [0, 0, 1.0]])

    assert blank_output_rows(negligible) == [1]


def test_zeroing_rejects_an_image_with_too_few_channels():
    # Unreachable through the decode paths (validate_config pins
    # n_output_channels to the matrix's row count), but the guard earns its keep:
    # indexing a 2-D image by channel would silently zero a *column* instead.
    with pytest.raises(ValueError, match="Decoded shape and channel_combination disagree"):
        zero_blank_output_channels(np.ones((8, 8), dtype=np.uint8), [1])


def test_safe_matrix_rejects_an_entirely_blank_matrix():
    with pytest.raises(ValueError, match="every row has all-zero weights"):
        fitsbolt_safe_channel_combination(np.zeros((3, 3)))


def test_safe_matrix_treats_non_arrays_as_none():
    # DotMap.copy() can turn a None config value into an empty DotMap().
    assert fitsbolt_safe_channel_combination(None) is None
    assert fitsbolt_safe_channel_combination(DotMap()) is None


# ---------------------------------------------------------------------------
# Validation (#588: both faces aborted here)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fits_extension",
    [
        pytest.param([0, 1, 2], id="fits_extension_indices"),
        pytest.param(BAND_NAMES, id="cutana_band_names"),
        pytest.param(None, id="no_fits_extension"),
    ],
)
def test_validate_config_accepts_a_blanked_output_channel(fits_extension):
    cfg = _cfg(BLANKED, fits_extension)

    # Validation must not rewrite the user's matrix: the blank row is the config.
    np.testing.assert_array_equal(cfg.normalisation.channel_combination, BLANKED)


def test_fitsbolt_config_never_carries_a_blank_row():
    # fitsbolt re-validates on every load call, so the matrix stored for it has
    # to stay validator-clean while cfg.normalisation keeps the real one.
    cfg = get_fitsbolt_config(_cfg(BLANKED, [0, 1, 2]))

    assert blank_output_rows(cfg.fitsbolt_cfg.channel_combination) == []
    np.testing.assert_array_equal(cfg.normalisation.channel_combination, BLANKED)
    # The row count must survive, or the checkpoint's channel_combination and the
    # model's output channel count stop agreeing.
    assert cfg.fitsbolt_cfg.n_output_channels == 3
    assert np.asarray(cfg.fitsbolt_cfg.channel_combination).shape == BLANKED.shape


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blanked_channel", [3, 2, None], ids=["alpha_band", "shown", "none"])
def test_four_channel_thumbnail_shows_the_first_three_channels(blanked_channel):
    """A fourth output channel must be dropped, never encoded as an alpha band.

    PIL infers RGBA from a 4-channel array, so blanking output channel 3 made
    every thumbnail fully transparent — black, in the gallery. Channel 3 is the
    case that regressed; the others guard the reduction itself.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(30, 255, (16, 16, 4), dtype=np.uint8)
    if blanked_channel is not None:
        image[..., blanked_channel] = 0

    thumbnail = np.asarray(Image.open(io.BytesIO(numpy_array_to_byte_stream(image))))

    # Assert the exact pixels, not just "something rendered": a slice that took
    # the wrong channels would still be non-empty.
    shown = image[..., :3].astype(np.float64)
    # astype truncates rather than rounds, matching the encoder.
    expected = ((shown - shown.min()) / (shown.max() - shown.min()) * 255).astype(np.uint8)
    assert thumbnail.shape == (16, 16, 3)
    np.testing.assert_array_equal(thumbnail, expected)


# ---------------------------------------------------------------------------
# Decoded pixels
# ---------------------------------------------------------------------------


def _decode(source, channel_combination, method, three_band_data):
    """Decode the shared test data through one of the three source paths."""
    fits_path, png_path, chw_array = three_band_data
    if source == "zarr":
        cfg = get_fitsbolt_config(_cfg(channel_combination, BAND_NAMES, method))
        return decode_zarr_image(chw_array, cfg)
    path, fits_extension = (fits_path, [0, 1, 2]) if source == "fits" else (png_path, None)
    cfg = _cfg(channel_combination, fits_extension, method)
    return load_and_process_wrapper([path], cfg)[0][1]


def test_prediction_mode_blanks_the_channel(three_band_data):
    # Prediction reuses the fitsbolt config saved with the model instead of
    # rebuilding it, so it must not depend on get_fitsbolt_config running first.
    fits_path, _, _ = three_band_data
    cfg = get_fitsbolt_config(_cfg(BLANKED, [0, 1, 2]))

    image = load_and_process_single_wrapper(fits_path, cfg, prediction=True)

    assert not np.any(image[..., 1])


def test_checkpoint_round_trip_survives_a_blanked_channel():
    # The checkpoint stores the substituted matrix inside fitsbolt_cfg but the
    # user's matrix standalone, and prediction rejects a row count that
    # disagrees with the model's output channels — so the substitution must not
    # change the shape or the stored channel count.
    cfg = get_fitsbolt_config(_cfg(BLANKED, [0, 1, 2]))
    restored = am.get_default_cfg()

    assert sync_normalisation_from_checkpoint(restored, cfg.fitsbolt_cfg, BLANKED)
    assert restored.normalisation.n_output_channels == 3
    np.testing.assert_array_equal(restored.normalisation.channel_combination, BLANKED)


@pytest.mark.parametrize("method", NORMALISATION_METHODS, ids=lambda m: m.name)
@pytest.mark.parametrize("source", ["fits", "png", "zarr"])
def test_blanked_channel_is_zero_and_costs_the_other_channels_nothing(
    source, method, three_band_data
):
    blanked = _decode(source, BLANKED, method, three_band_data)
    control = _decode(source, CONTROL, method, three_band_data)

    assert blanked.shape[2] == 3
    assert not np.any(blanked[..., 1]), "blanked output channel is not empty"
    # Rows 0 and 2 are identical in both matrices, so the kept channels must be
    # bit-identical: blanking a channel must not restretch the others.
    np.testing.assert_array_equal(blanked[..., 0], control[..., 0])
    np.testing.assert_array_equal(blanked[..., 2], control[..., 2])
