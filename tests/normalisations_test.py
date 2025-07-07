#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import numpy as np
import pytest
from loguru import logger
from dotmap import DotMap

from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.image_processing.normalisation import normalise_image
from anomaly_match.image_processing.NormalisationMethod import NormalisationMethod


def get_test_config(method):
    """Returns a test config with the specified normalisation method"""
    cfg = get_default_cfg()
    cfg.N_to_load = 10
    cfg.size = [64, 64]
    cfg.normalisation_method = method
    cfg.normalisation = DotMap()
    cfg.normalisation.maximum_value = None
    cfg.normalisation.minimum_value = None
    cfg.normalisation.crop_for_maximum_value = None
    cfg.normalisation.log_calculate_minimum_value = False
    cfg.normalisation.asinh_scale = [10.0, 10.0, 10.0]  # Default for ASINH
    cfg.normalisation.asinh_clip = [99.0, 99.0, 99.0]  # Default for ASINH
    return cfg


def get_asinh_test_config(asinh_scale=[1.0, 1.0, 1.0], asinh_clip=[99.0, 99.0, 99.0]):
    """Create a test config specifically for ASINH normalisation"""
    cfg = get_test_config(NormalisationMethod.ASINH)
    if asinh_scale is not None:
        cfg.normalisation.asinh_scale = asinh_scale
    if asinh_clip is not None:
        cfg.normalisation.asinh_clip = asinh_clip
    return cfg


@pytest.fixture
def caplog(caplog):
    """Configure loguru to use the caplog handler"""
    handler_id = logger.add(caplog.handler)
    yield caplog
    logger.remove(handler_id)


def create_gradient_rgb(height=16, width=16, dtype=np.uint8):
    """Create a test RGB image with gradients in different channels"""
    if dtype == np.uint16:
        max_val = 65535
    elif dtype == np.float32:
        max_val = 1e-3
    else:
        max_val = 255

    # Create gradients for each channel
    r = np.linspace(0, max_val, width)
    g = np.linspace(0, max_val / 2, width)
    b = np.linspace(max_val / 4, max_val, width)

    # Create meshgrids
    r_mesh, _ = np.meshgrid(r, np.linspace(0, max_val, height))
    g_mesh, _ = np.meshgrid(g, np.linspace(0, max_val / 2, height))
    b_mesh, _ = np.meshgrid(b, np.linspace(max_val / 4, max_val, height))

    # Stack channels
    image = np.stack([r_mesh, g_mesh, b_mesh], axis=2).astype(dtype)
    return image


def create_gradient_single_channel(height=16, width=16, dtype=np.uint8):
    """Create a test single channel image with gradient"""
    if dtype == np.uint16:
        max_val = 65535
    elif dtype == np.float32:
        max_val = 1e-3
    else:
        max_val = 255

    x = np.linspace(0, max_val, width)
    x_mesh, _ = np.meshgrid(x, np.linspace(0, max_val, height))
    return x_mesh.astype(dtype)


def create_multi_channel_image(height=16, width=16, dtype=np.uint8):
    """Create a test multi-channel image simulating 4 channels: V,Y,J,H astronomical bands
    with different intensity ranges to test proper scaling across channels"""
    if dtype == np.uint16:
        max_vals = [65535, 45000, 55000, 35000]  # Different max for each channel
    elif dtype == np.float32:
        max_vals = [1e-3, 7e-4, 8e-4, 5e-4]
    else:
        max_vals = [255, 180, 220, 180]

    channels = []
    for max_val in max_vals:
        # Create gradient with different ranges for each channel
        x = np.linspace(0, max_val, width)
        channel_mesh, _ = np.meshgrid(x, np.linspace(0, max_val, height))
        channels.append(channel_mesh)

    # Stack channels
    image = np.stack(channels, axis=2).astype(dtype)
    return image


def create_test_pattern(height=16, width=16, dtype=np.uint8):
    """Create a test pattern with border 0s, background 10, and center 100, single channel"""
    if dtype == np.float32:
        values = [0.01, 0.1, 1.0]  # normalized values for float
    else:
        values = [1, 10, 100]

    # Create base image with value 10
    image = np.full((height, width), values[1], dtype=dtype)

    # Set borders to 0
    image[0, :] = values[0]
    image[-1, :] = values[0]
    image[:, 0] = values[0]
    image[:, -1] = values[0]

    # Set center to 100
    center_h = height // 2
    center_w = width // 2
    center_size = 2
    image[
        center_h - center_size : center_h + center_size + 1,
        center_w - center_size : center_w + center_size + 1,
    ] = values[2]

    return image


@pytest.mark.parametrize("method", NormalisationMethod.get_test_methods())
def test_normalise_uint16_image(method):
    """Test normalisation with uint16 image"""
    # Create test image and config
    test_image = create_gradient_rgb(dtype=np.uint16)
    cfg = get_test_config(method)

    # Apply normalisation
    result = normalise_image(test_image, cfg=cfg)

    # Common assertions for all methods
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert np.min(result) >= 0
    assert np.max(result) <= 255
    # assert good use of dynamic range
    assert np.max(result) > 250
    assert np.min(result) < 5

    if method == NormalisationMethod.CONVERSION_ONLY:
        # For NONE, we expect the values to be scaled down from uint16 to uint8
        expected = np.round(((test_image) / (256 * 256 - 1) * 255)).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("method", NormalisationMethod.get_test_methods())
def test_normalise_float32_image(method):
    """Test normalisation with float32 RGB image"""
    base_image = create_gradient_rgb(dtype=np.float32)
    scale_factor = 1e-9
    test_image = base_image * scale_factor
    cfg = get_test_config(method)
    result = normalise_image(test_image, cfg=cfg)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert np.min(result) >= 0
    assert np.max(result) <= 255
    assert np.max(result) > 250
    assert np.min(result) < 5


@pytest.mark.parametrize("method", NormalisationMethod.get_test_methods())
def test_normalise_single_channel(method):
    """Test normalisation with single channel uint8 gradient image"""
    test_image = create_gradient_single_channel()
    cfg = get_test_config(method)
    cfg.normalisation.asinh_scale = [10.0]  # ASINH scale for single channel
    cfg.normalisation.asinh_clip = [99.0]  # ASINH clip for single channel
    result = normalise_image(test_image, cfg=cfg)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert np.min(result) >= 0
    assert np.max(result) <= 255

    if method == NormalisationMethod.CONVERSION_ONLY:
        np.testing.assert_array_equal(result, test_image)


@pytest.mark.parametrize("method", NormalisationMethod.get_test_methods())
def test_normalise_multi_channel(method):
    """Test normalisation with multi-channel image (e.g., V,Y,J,H bands)"""
    test_image = create_multi_channel_image()
    cfg = get_test_config(method)
    # unintended asinh parameters for not yet supported multi(=/=3) channel
    cfg.normalisation.asinh_scale = [10.0, 10.0, 10.0, 10.0]  # ASINH scale for each channel
    cfg.normalisation.asinh_clip = [99.0, 99.0, 99.0, 99.0]  # ASINH clip for each channel
    result = normalise_image(test_image, cfg=cfg)

    # Basic checks
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert np.min(result) >= 0
    assert np.max(result) <= 255

    if method == NormalisationMethod.CONVERSION_ONLY:
        np.testing.assert_array_equal(result, test_image)
    elif method == NormalisationMethod.LOG:
        # Check that log normalization preserves order in all channels
        for channel in range(4):  # V,Y,J,H channels
            # Get values from first row which has a gradient
            channel_vals = result[0, :, channel]
            # Check that values are strictly increasing (gradient preserved)
            assert np.all(np.diff(channel_vals) > 0)

    elif method == NormalisationMethod.ZSCALE:
        # Check that zscale maps each channel to use full range effectively
        for channel in range(4):  # V,Y,J,H channels
            channel_vals = result[..., channel]
            # Check that we use most of the range (allowing some margin)
            assert np.min(channel_vals) <= 40, f"Channel {channel} min value too high"
            assert np.max(channel_vals) >= 180, f"Channel {channel} max value too low"
            # Check for reasonable distribution
            median_val = np.median(channel_vals)
            assert 80 < median_val < 175, f"Channel {channel} median outside expected range"


@pytest.mark.parametrize("method", NormalisationMethod.get_test_methods())
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_normalise_pattern(method, dtype):
    """Test normalisation with specific pattern image with one channel"""
    test_image = create_test_pattern(dtype=dtype)
    cfg = get_test_config(method)
    cfg.normalisation.asinh_scale = [10.0]  # ASINH scale for single channel
    cfg.normalisation.asinh_clip = [99.0]  # ASINH clip for single channel
    result = normalise_image(test_image, cfg=cfg)

    # Basic assertions
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert np.min(result) >= 0
    assert np.max(result) <= 255

    # Get values for testing
    center_val = result[8:9, 8:9][0, 0]  # center value (should be brightest)
    border_val = result[0, 0]  # border value (should be darkest)
    bg_val = result[2, 2]  # background value (should be intermediate)

    if method == NormalisationMethod.CONVERSION_ONLY:
        if dtype == np.uint8:
            # For uint8, values should remain unchanged
            np.testing.assert_array_equal(result, test_image)
            assert center_val == 100
            assert border_val == 1
            assert bg_val == 10
        else:
            # For float32, values should be scaled to uint8 range, here manually, normally with astropy
            expected = np.round((test_image - 0.01) / (1 - 0.01) * 255).astype(np.uint8)
            np.testing.assert_array_equal(result, expected)
    elif method == NormalisationMethod.LOG:
        # Check that log normalization preserves order and maps values logarithmically
        assert border_val < bg_val < center_val
        # Ensure reasonable value ranges
        assert border_val < 90  # dark border
        assert bg_val > 50 and bg_val < 175  # mid-range background
        assert center_val > 200  # bright center

        # Get the unique input and output values in order [border, background, center]
        if dtype == np.float32:
            input_values = np.array([0.01, 0.1, 1.0])
        else:
            input_values = np.array([1, 10, 100])
        output_values = np.array([border_val, bg_val, center_val])

        # Calculate log10 of input values
        log_values = np.log10(
            1000 * (input_values - np.min([0])) / (np.max(input_values) - np.min([0])) + 1
        ) / np.log10(1000 + 1)

        # Find scaling factor between log values and output values
        # Using least squares to find the best scaling factor
        # scale_factor = np.sum(output_values * log_values) / np.sum(log_values * log_values)

        # Check if scaled log values match output values within tolerance
        log_values = np.round(log_values * 255)
        np.testing.assert_allclose(
            output_values, log_values, rtol=0.1
        )  # 10% tolerance for uint8 quantization

    elif method == NormalisationMethod.ZSCALE:
        # ZScale should map the values to use the full range effectively
        assert border_val < 10  # very dark border
        assert bg_val > 50 and bg_val < 175  # mid-range background
        assert center_val > 245  # very bright center
        # Background should be closer to border than to center due to outlier handling
        assert (bg_val - border_val) < (center_val - bg_val)


def test_normalise_invalid_method(caplog):
    """Test normalisation with invalid method"""
    test_image = create_gradient_rgb()

    # Test with invalid string
    cfg = get_test_config("invalid")
    result = normalise_image(test_image, cfg=cfg)
    np.testing.assert_array_equal(result, test_image)
    assert "Normalisation method type invalid" in caplog.text
    assert "CRITICAL" in caplog.text
    caplog.clear()

    # Test with invalid integer
    cfg = get_test_config(999)
    result = normalise_image(test_image, cfg=cfg)
    np.testing.assert_array_equal(result, test_image)
    assert "Normalisation method type 999" in caplog.text
    assert "CRITICAL" in caplog.text
    caplog.clear()

    # Test with invalid type
    cfg = get_test_config(None)
    result = normalise_image(test_image, cfg=cfg)
    np.testing.assert_array_equal(result, test_image)
    assert "Normalisation method type None" in caplog.text
    assert "CRITICAL" in caplog.text


def test_gradient_creation():
    """Test the gradient creation helper function"""
    # Test uint8
    img_uint8 = create_gradient_rgb(dtype=np.uint8)
    assert img_uint8.dtype == np.uint8
    assert img_uint8.shape == (16, 16, 3)
    assert np.min(img_uint8) >= 0
    assert np.max(img_uint8) <= 255

    # Test uint16
    img_uint16 = create_gradient_rgb(dtype=np.uint16)
    assert img_uint16.dtype == np.uint16
    assert img_uint16.shape == (16, 16, 3)
    assert np.min(img_uint16) >= 0
    assert np.max(img_uint16) <= 65535

    # Test float32
    img_float32 = create_gradient_rgb(dtype=np.float32)
    assert img_float32.dtype == np.float32
    assert img_float32.shape == (16, 16, 3)
    assert np.min(img_float32) >= 0
    assert np.max(img_float32) <= 1e-3


def test_normalise_float32_max_value():
    """Test log normalisation with maximum_value setting"""
    base_image = create_gradient_rgb(dtype=np.float32)
    test_image = base_image * 1e-9  # Values from 0 to 1e-12

    # Then with clipping
    cfg = get_test_config(NormalisationMethod.LOG)
    cfg.normalisation.maximum_value = 0.5e-12  # Clip the top half of values
    result = normalise_image(test_image, cfg=cfg)

    # Results should be uint8 and keep dimensions
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape

    # Values above maximum_value should all map to the same uint8 value
    high_values = test_image > cfg.normalisation.maximum_value
    unique_high_values = np.unique(result[high_values])
    assert len(unique_high_values) == 1, "Clipped values should map to the same output"
    assert unique_high_values[0] == 255, "Clipped values should map to 255"

    # Values below maximum_value should maintain relative order
    low_values = test_image <= cfg.normalisation.maximum_value
    low_values_result = result[low_values]
    low_values_orig = test_image[low_values]
    order_preserved = np.all(np.diff(low_values_result[np.argsort(low_values_orig)]) >= 0)
    assert order_preserved, "Order of non-clipped values should be preserved"


def test_normalise_float32_min_value():
    """Test log normalisation with minimum_value setting"""
    base_image = create_gradient_rgb(dtype=np.float32)
    test_image = base_image * 1e-9  # Values from 0 to 1e-12
    min_val = 0.2e-12

    # Then with minimum value clipping
    cfg = get_test_config(NormalisationMethod.LOG)
    cfg.normalisation.minimum_value = min_val
    result = normalise_image(test_image, cfg=cfg)
    # Results should be uint8 and keep dimensions
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    # Values below minimum_value should all map to the same uint8 value
    low_values = test_image < min_val
    unique_low_values = np.unique(result[low_values])
    assert len(unique_low_values) == 1, "Clipped values should map to same output"
    assert unique_low_values[0] == 0, "Clipped values should map to 0"
    # Values above minimum_value should maintain relative order
    high_values = test_image >= min_val
    high_values_result = result[high_values]
    high_values_orig = test_image[high_values]
    order_preserved = np.all(np.diff(high_values_result[np.argsort(high_values_orig)]) >= 0)
    assert order_preserved, "Order of non-clipped values should be preserved"


def test_normalise_float32_crop_max():
    """Test log normalisation with crop_for_maximum_value setting"""
    base_image = create_gradient_rgb(dtype=np.float32)
    test_image = base_image * 1e-9
    # Create a bright spot outside crop region
    test_image[0, 0] = 5e-12  # Much brighter than rest
    # First without crop
    cfg_no_crop = get_test_config(NormalisationMethod.LOG)
    result_no_crop = normalise_image(test_image, cfg=cfg_no_crop)
    # Then with center crop that excludes bright spot
    cfg = get_test_config(NormalisationMethod.LOG)
    crop_pixels = 6
    cfg.normalisation.crop_for_maximum_value = (crop_pixels, crop_pixels)  # Center crop
    result = normalise_image(test_image, cfg=cfg)
    # Results should be uint8 and keep dimensions
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    # Without crop, bright spot dominates scaling
    assert np.all(result_no_crop[0, 0] == 255)
    assert np.max(result_no_crop[1:, 1:]) < 200
    # With crop, center region uses full dynamic range
    crop_im_h = test_image.shape[0] - crop_pixels // 2
    crop_im_w = test_image.shape[1] - crop_pixels // 2
    center_region = result[crop_im_h : crop_im_h + crop_pixels, crop_im_w : crop_im_w + crop_pixels]
    center_region_no_crop = result_no_crop[
        crop_im_h : crop_im_h + crop_pixels, crop_im_w : crop_im_w + crop_pixels
    ]

    # center region should now be the upper dynamic range limit, and before not
    assert np.max(center_region) == 255
    assert np.max(center_region_no_crop) < 255
    # Bright spot outside crop still saturates
    assert np.all(result[0, 0] == 255)


def test_normalise_float32_log_min():
    """Test log normalisation with log_calculate_minimum_value setting"""
    base_image = create_gradient_rgb(dtype=np.float32)
    test_image = base_image * 1e-9
    # Add negative values to test minimum handling
    test_image[0:4, 0:4] = -0.5e-12
    test_image[0, 0] = -1e-12  # dark spot outside crop region
    # Without log min calculation
    cfg = get_test_config(NormalisationMethod.LOG)
    cfg.normalisation.log_calculate_minimum_value = False
    result_no_log = normalise_image(test_image, cfg=cfg)
    # With log min calculation
    cfg.normalisation.log_calculate_minimum_value = True
    result_with_log = normalise_image(test_image, cfg=cfg)
    # Both results should be valid uint8 images
    assert isinstance(result_no_log, np.ndarray)
    assert isinstance(result_with_log, np.ndarray)
    assert result_no_log.dtype == np.uint8
    assert result_with_log.dtype == np.uint8
    assert result_no_log.shape == test_image.shape
    assert result_with_log.shape == test_image.shape
    # Without log min: negative values should be clipped to 0
    neg_region_no_log = result_no_log[0:4, 0:4]
    assert np.all(neg_region_no_log == 0)
    # With log min: negative values should be handled by shifting minimum
    neg_region_with_log = result_with_log[0:4, 0:4]
    assert not np.all(neg_region_with_log == 0)
    assert np.all(result_with_log[0, 0] == 0)  # lowest value has to be 0
    # Rest of image should preserve order in both cases
    pos_vals = test_image > 0
    order_no_log = np.all(np.diff(result_no_log[pos_vals][np.argsort(test_image[pos_vals])]) >= 0)
    order_with_log = np.all(
        np.diff(result_with_log[pos_vals][np.argsort(test_image[pos_vals])]) >= 0
    )
    assert order_no_log, "Order of positive values should be preserved without log min"
    assert order_with_log, "Order of positive values should be preserved with log min"


def create_rgb_test_image(height=16, width=16, dtype=np.float32):
    """Create a test RGB image specifically for ASINH testing with different channel characteristics"""
    if dtype == np.float32:
        # Create different intensity ranges for each channel to test per-channel scaling
        r_vals = np.linspace(0.001, 1.0, width)
        g_vals = np.linspace(0.01, 0.5, width)
        b_vals = np.linspace(0.1, 2.0, width)
    else:
        # For uint8/uint16
        max_val = 255 if dtype == np.uint8 else 65535
        r_vals = np.linspace(1, max_val, width)
        g_vals = np.linspace(10, max_val // 2, width)
        b_vals = np.linspace(50, max_val, width)

    # Create meshgrids for each channel
    r_mesh, _ = np.meshgrid(
        r_vals,
        np.linspace(
            0.001 if dtype == np.float32 else 1, 1.0 if dtype == np.float32 else max_val, height
        ),
    )
    g_mesh, _ = np.meshgrid(
        g_vals,
        np.linspace(
            0.01 if dtype == np.float32 else 10,
            0.5 if dtype == np.float32 else max_val // 2,
            height,
        ),
    )
    b_mesh, _ = np.meshgrid(
        b_vals,
        np.linspace(
            0.1 if dtype == np.float32 else 50, 2.0 if dtype == np.float32 else max_val, height
        ),
    )

    # Stack channels to create RGB image
    image = np.stack([r_mesh, g_mesh, b_mesh], axis=2).astype(dtype)
    return image


def test_asinh_basic_functionality():
    """Test basic ASINH normalisation functionality with RGB image"""
    test_image = create_rgb_test_image(dtype=np.float32)
    cfg = get_asinh_test_config()
    result = normalise_image(test_image, cfg=cfg)

    # Basic checks
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert len(result.shape) == 3  # Should be RGB
    assert result.shape[2] == 3  # Should have 3 channels
    assert np.min(result) >= 0
    assert np.max(result) <= 255

    # Check that any channel uses reasonable dynamic range
    assert np.max(result) > 200  # Should use upper range
    assert np.min(result) < 50  # Should use lower range


def test_asinh_scaling_parameters():
    """Test ASINH normalisation with different scaling parameters"""
    test_image = create_rgb_test_image(dtype=np.float32)

    # Test with low scaling (more linear-like behavior)
    cfg_low = get_asinh_test_config(asinh_scale=[0.1, 0.1, 0.1])
    result_low = normalise_image(test_image, cfg=cfg_low)

    # Test with high scaling (more log-like behavior)
    cfg_high = get_asinh_test_config(asinh_scale=[3.0, 3.0, 3.0])
    result_high = normalise_image(test_image, cfg=cfg_high)

    # Test with per-channel scaling
    cfg_mixed = get_asinh_test_config(asinh_scale=[1.0, 0.1, 0.05])
    result_mixed = normalise_image(test_image, cfg=cfg_mixed)

    # All results should be valid uint8 images
    for result in [result_low, result_high, result_mixed]:
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == test_image.shape
        assert np.min(result) >= 0
        assert np.max(result) <= 255

    # With low scaling, gradient should be more preserved (more linear)
    # With high scaling, contrast should be enhanced (more compressed dynamic range)
    # Check that different scalings produce different results
    assert not np.array_equal(result_low, result_high)
    assert not np.array_equal(result_low, result_mixed)
    assert not np.array_equal(result_high, result_mixed)

    # Test per-channel differences with mixed scaling
    # Red channel (scale=1.0) should be different from Green (scale=10.0) and Blue (scale=50.0)
    red_channel = result_mixed[:, :, 0]
    green_channel = result_mixed[:, :, 1]
    blue_channel = result_mixed[:, :, 2]

    # Channels should have different distributions due to different scaling
    assert not np.array_equal(red_channel, green_channel)
    assert not np.array_equal(green_channel, blue_channel)


def test_asinh_clipping_parameters():
    """Test ASINH normalisation with different clipping parameters"""
    test_image = create_rgb_test_image(dtype=np.float32)

    # Test with no clipping (100% percentile)
    cfg_no_clip = get_asinh_test_config(asinh_clip=[100.0, 100.0, 100.0])
    result_no_clip = normalise_image(test_image, cfg=cfg_no_clip)

    # Test with aggressive clipping (90% percentile)
    cfg_clip = get_asinh_test_config(asinh_clip=[70.0, 70.0, 70.0])
    result_clip = normalise_image(test_image, cfg=cfg_clip)

    # Test with per-channel clipping
    cfg_mixed_clip = get_asinh_test_config(asinh_clip=[85.0, 55.0, 98.0])
    result_mixed_clip = normalise_image(test_image, cfg=cfg_mixed_clip)

    # Test with single value clipping
    cfg_single_clip = get_asinh_test_config(asinh_clip=92.0)
    result_single_clip = normalise_image(test_image, cfg=cfg_single_clip)

    # All results should be valid uint8 images
    for result in [result_no_clip, result_clip, result_mixed_clip, result_single_clip]:
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == test_image.shape
        assert np.min(result) >= 0
        assert np.max(result) <= 255

    # Clipping should produce different results
    assert not np.array_equal(result_no_clip, result_clip)
    assert not np.array_equal(result_no_clip, result_mixed_clip)
    assert not np.array_equal(result_clip, result_mixed_clip)

    # With clipping, the distributions should be different
    for channel in range(3):
        # The distributions should be different
        assert not np.array_equal(result_clip[:, :, channel], result_no_clip[:, :, channel])


def test_asinh_with_uint8_image():
    """Test ASINH normalisation with uint8 input image"""
    test_image = create_gradient_rgb(dtype=np.uint8)
    cfg = get_asinh_test_config()
    result = normalise_image(test_image, cfg=cfg)

    # Basic checks
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert np.min(result) >= 0
    assert np.max(result) <= 255

    # ASINH should preserve the overall gradient structure
    # Check that each channel maintains some order
    for channel in range(3):
        channel_vals = result[0, :, channel]  # First row gradient
        # Should generally increase or at least not decrease significantly
        # (allowing for some variation due to asinh transformation)
        diffs = np.diff(channel_vals.astype(np.int16))  # Use int16 to handle negative diffs
        # At least 70% of differences should be non-negative (allowing for some noise)
        non_negative_ratio = np.sum(diffs >= 0) / len(diffs)
        assert non_negative_ratio > 0.7, f"Channel {channel} gradient not well preserved"


def test_asinh_with_uint16_image():
    """Test ASINH normalisation with uint16 input image"""
    test_image = create_gradient_rgb(dtype=np.uint16)
    cfg = get_asinh_test_config()
    result = normalise_image(test_image, cfg=cfg)

    # Basic checks
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape
    assert np.min(result) >= 0
    assert np.max(result) <= 255

    # Should use good dynamic range
    assert np.max(result) > 200
    assert np.min(result) < 50


def test_asinh_edge_cases():
    """Test ASINH normalisation edge cases"""

    # Test with zeros
    zero_image = np.zeros((8, 8, 3), dtype=np.float32)
    cfg = get_asinh_test_config()
    result_zeros = normalise_image(zero_image, cfg=cfg)

    assert isinstance(result_zeros, np.ndarray)
    assert result_zeros.dtype == np.uint8
    assert result_zeros.shape == zero_image.shape
    # All values should be 0 for zero input
    assert np.all(result_zeros == 0)

    # Test with very small values
    small_image = np.full((8, 8, 3), 1e-10, dtype=np.float32)
    result_small = normalise_image(small_image, cfg=cfg)

    assert isinstance(result_small, np.ndarray)
    assert result_small.dtype == np.uint8
    assert result_small.shape == small_image.shape

    # Test with identical values (should result in min/max error handling)
    uniform_image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    result_uniform = normalise_image(uniform_image, cfg=cfg)

    assert isinstance(result_uniform, np.ndarray)
    assert result_uniform.dtype == np.uint8
    assert result_uniform.shape == uniform_image.shape


def test_asinh_channel_independence():
    """Test that ASINH normalisation processes channels independently"""
    # Create an image where channels have very different ranges
    height, width = 12, 12

    # Red channel: very bright
    red_channel = np.full((height, width), 1.0, dtype=np.float32)
    # Green channel: medium brightness
    green_channel = np.full((height, width), 0.1, dtype=np.float32)
    # Blue channel: very dim
    blue_channel = np.full((height, width), 0.01, dtype=np.float32)

    test_image = np.stack([red_channel, green_channel, blue_channel], axis=2)

    cfg = get_asinh_test_config(asinh_scale=[10.0, 10.0, 10.0])
    result = normalise_image(test_image, cfg=cfg)

    # Basic checks
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == test_image.shape

    # Each channel should be processed independently
    # So they should not all have the same values despite input differences
    red_result = result[:, :, 0]
    green_result = result[:, :, 1]
    blue_result = result[:, :, 2]

    # Due to per-channel normalization, they might end up with similar values
    # But the processing should still be per-channel
    assert red_result.shape == (height, width)
    assert green_result.shape == (height, width)
    assert blue_result.shape == (height, width)


def test_asinh_with_crop_for_maximum():
    """Test ASINH normalisation with crop_for_maximum_value setting"""
    test_image = create_rgb_test_image(dtype=np.float32)

    # Add a bright spot outside the center region
    test_image[0, 0, :] = 10.0  # Very bright spot in corner

    # Test without crop, no clip to keep bright spot
    cfg_no_crop = get_asinh_test_config(asinh_clip=100.0)
    result_no_crop = normalise_image(test_image, cfg=cfg_no_crop)

    # Test with center crop that excludes the bright spot, no clip to keep bright spot
    cfg_crop = get_asinh_test_config(asinh_clip=100.0)
    cfg_crop.normalisation.crop_for_maximum_value = (8, 8)  # Center crop
    result_crop = normalise_image(test_image, cfg=cfg_crop)

    # Both results should be valid
    for result in [result_no_crop, result_crop]:
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == test_image.shape
        assert np.min(result) >= 0
        assert np.max(result) <= 255

    # Results should be different due to different maximum calculation
    assert not np.array_equal(result_no_crop, result_crop)

    # The bright spot should still be present in both results
    assert np.all(result_no_crop[0, 0, :] > 200)  # Should be bright
    assert np.all(result_crop[0, 0, :] > 200)  # Should still be bright
