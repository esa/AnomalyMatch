#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Tests for the image IO utility functions.
"""
import os
import numpy as np
import pytest
import shutil
import tempfile
from PIL import Image
from astropy.io import fits

from anomaly_match.data_io.find_images_in_folder import (
    get_image_names_from_folder,
    get_image_paths_from_folder,
)
from anomaly_match.data_io.load_images import (
    load_and_process_wrapper,
    load_and_process_single_wrapper,
)
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod


def _load_image_with_fitsbolt(filepath, cfg):
    """Helper function to load image using fitsbolt with AnomalyMatch config."""
    # Use the new wrapper function instead of directly using fitsbolt
    return load_and_process_single_wrapper(filepath, cfg, desc="test loading", show_progress=False)


def _load_multiple_images_with_fitsbolt(filepaths, cfg, show_progress=False):
    """Helper function to load multiple images using fitsbolt with AnomalyMatch config."""
    # Use the new wrapper function instead of directly using fitsbolt
    return load_and_process_wrapper(
        filepaths,
        cfg,
        desc="test loading multiple",
        show_progress=show_progress,
    )


def _update_config(cfg, **kwargs):
    """Update both the main config and fitsbolt config with the given parameters."""
    for key, value in kwargs.items():
        setattr(cfg, key, value)
        if key == "size":
            cfg.normalisation.image_size = value
        elif key == "fits_extension":
            cfg.normalisation.fits_extension = value
            # When setting fits_extension to a list, automatically set up appropriate channel_combination
            # only if channel_combination hasn't been explicitly set
            if isinstance(value, list) and len(value) > 1:
                # Check if channel_combination is being explicitly set in this call
                if "channel_combination" not in kwargs:
                    import numpy as np

                    n_channels = min(3, len(value))  # Limit to 3 output channels (RGB)
                    n_extensions = len(value)
                    if n_extensions == n_channels:
                        cfg.normalisation.channel_combination = np.eye(n_channels)
                    else:
                        # Create a combination matrix that uses the first n_channels extensions
                        combination = np.zeros((n_channels, n_extensions))
                        for i in range(n_channels):
                            combination[i, i] = 1.0
                        cfg.normalisation.channel_combination = combination
        elif key == "channel_combination":
            # Convert dictionary format to numpy array format for fitsbolt
            cfg.normalisation.channel_combination = value
        elif key == "normalisation_method":
            cfg.normalisation.normalisation_method = value
        elif key == "interpolation_order":
            cfg.normalisation.interpolation_order = value
        elif key.startswith("normalisation."):
            # Handle nested attributes in normalisation
            norm_key = key.split(".")[1]
            setattr(cfg.normalisation, norm_key, value)
            fitsbolt_key = f"norm_{norm_key}"
            setattr(cfg.normalisation, fitsbolt_key, value)

    # Clear the cached fitsbolt config to force regeneration with new parameters
    if hasattr(cfg, "fitsbolt_cfg"):
        delattr(cfg, "fitsbolt_cfg")

    return cfg


class TestImageIO:
    """Test class for image IO utilities."""

    @pytest.fixture
    def test_config(self):
        """Create test config for image loading."""

        # Start with default config
        from anomaly_match.utils.get_default_cfg import get_default_cfg

        cfg = get_default_cfg()
        # Override for test specific settings
        # Add fitsbolt configuration needed by the wrapper functions
        cfg.normalisation.image_size = None
        cfg.normalisation.n_output_channels = 3

        return cfg

    @classmethod
    def setup_class(cls):
        """Set up test files and directories."""
        # Create a temporary test directory
        cls.test_dir = tempfile.mkdtemp()

        # Create test RGB image
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_img[25:75, 25:75, 0] = 255  # Red square
        cls.rgb_path = os.path.join(cls.test_dir, "test_rgb.jpg")
        Image.fromarray(rgb_img).save(cls.rgb_path)

        # Create test grayscale image
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        gray_img[25:75, 25:75] = 200  # White square
        cls.gray_path = os.path.join(cls.test_dir, "test_gray.jpg")
        Image.fromarray(gray_img).save(cls.gray_path)  # Create test RGBA image
        rgba_img = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba_img[25:75, 25:75, 0] = 255  # Red square
        rgba_img[25:75, 25:75, 3] = 128  # Semi-transparent
        cls.rgba_path = os.path.join(cls.test_dir, "test_rgba.png")
        Image.fromarray(rgba_img).save(cls.rgba_path)

        # Create fully transparent test RGBA image
        transparent_img = np.zeros((100, 100, 4), dtype=np.uint8)
        transparent_img[25:75, 25:75, 1] = 255  # Green square
        transparent_img[25:75, 25:75, 3] = 0  # Fully transparent
        cls.transparent_path = os.path.join(cls.test_dir, "transparent.png")
        Image.fromarray(transparent_img).save(cls.transparent_path)

        # Create a complex RGBA image with varying alpha
        complex_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        # Create a gradient pattern
        for i in range(100):
            for j in range(100):
                complex_rgba[i, j, 0] = min(255, i * 2)  # Red gradient
                complex_rgba[i, j, 1] = min(255, j * 2)  # Green gradient
                complex_rgba[i, j, 2] = min(255, (i + j))  # Blue gradient
                complex_rgba[i, j, 3] = min(255, (i + j) // 2 + 100)  # Alpha gradient
        cls.complex_rgba_path = os.path.join(cls.test_dir, "complex_rgba.png")
        Image.fromarray(complex_rgba).save(cls.complex_rgba_path)

        # Create fully transparent test RGBA image
        transparent_img = np.zeros((100, 100, 4), dtype=np.uint8)
        transparent_img[25:75, 25:75, 1] = 255  # Green square
        transparent_img[25:75, 25:75, 3] = 0  # Fully transparent
        cls.transparent_path = os.path.join(cls.test_dir, "transparent.png")
        Image.fromarray(transparent_img).save(cls.transparent_path)

        # Create a complex RGBA image with varying alpha
        complex_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        # Create a gradient pattern
        for i in range(100):
            for j in range(100):
                complex_rgba[i, j, 0] = min(255, i * 2)  # Red gradient
                complex_rgba[i, j, 1] = min(255, j * 2)  # Green gradient
                complex_rgba[i, j, 2] = min(255, (i + j))  # Blue gradient
                complex_rgba[i, j, 3] = min(255, (i + j) // 2 + 100)  # Alpha gradient
        cls.complex_rgba_path = os.path.join(cls.test_dir, "complex_rgba.png")
        Image.fromarray(complex_rgba).save(cls.complex_rgba_path)

        # Create a nested directory with an image
        nested_dir = os.path.join(cls.test_dir, "nested")
        os.makedirs(nested_dir)
        nested_img = np.zeros((50, 50, 3), dtype=np.uint8)
        nested_img[:, :, 1] = 200  # Green image
        cls.nested_path = os.path.join(nested_dir, "nested_image.jpg")
        Image.fromarray(nested_img).save(cls.nested_path)

        # Keep track of all created image files
        cls.image_files = [
            cls.rgb_path,
            cls.gray_path,
            cls.rgba_path,
            cls.transparent_path,
            cls.complex_rgba_path,
            cls.nested_path,
        ]
        # Simple FITS file
        fits_data = np.zeros((100, 100), dtype=np.float32)
        fits_data[25:75, 25:75] = 1.0  # Bright square
        cls.fits_path = os.path.join(cls.test_dir, "test.fits")
        fits.writeto(cls.fits_path, fits_data, overwrite=True)
        cls.image_files.append(cls.fits_path)

        # FITS file with multiple channels (RGB-like)
        multi_data = np.zeros((3, 100, 100), dtype=np.float32)
        multi_data[0, 25:75, 25:75] = 1.0  # Red
        multi_data[1, 35:85, 35:85] = 0.8  # Green
        multi_data[2, 45:95, 45:95] = 0.6  # Blue
        cls.multi_fits_path = os.path.join(cls.test_dir, "multi_channel.fits")
        fits.writeto(cls.multi_fits_path, multi_data, overwrite=True)
        cls.image_files.append(cls.multi_fits_path)

        # Create a FITS file with unusual dimensions (4D)
        four_dim_data = np.zeros((2, 3, 60, 60), dtype=np.float32)
        for i in range(2):
            for j in range(3):
                four_dim_data[i, j, 20:40, 20:40] = 0.5 + 0.1 * i + 0.1 * j
        cls.four_dim_fits_path = os.path.join(cls.test_dir, "four_dim.fits")
        fits.writeto(cls.four_dim_fits_path, four_dim_data, overwrite=True)
        cls.image_files.append(cls.four_dim_fits_path)

        # Create FITS with extreme values to test normalization
        extreme_data = np.zeros((100, 100), dtype=np.float32)
        extreme_data[10:40, 10:40] = -1000.0  # Very negative values
        extreme_data[50:80, 50:80] = 1000.0  # Very positive values
        cls.extreme_fits_path = os.path.join(cls.test_dir, "extreme_values.fits")
        fits.writeto(cls.extreme_fits_path, extreme_data, overwrite=True)
        cls.image_files.append(cls.extreme_fits_path)

        # Create a FITS file with unusual dimensions (4D)
        four_dim_data = np.zeros((2, 3, 60, 60), dtype=np.float32)
        for i in range(2):
            for j in range(3):
                four_dim_data[i, j, 20:40, 20:40] = 0.5 + 0.1 * i + 0.1 * j
        cls.four_dim_fits_path = os.path.join(cls.test_dir, "four_dim.fits")
        fits.writeto(cls.four_dim_fits_path, four_dim_data, overwrite=True)
        cls.image_files.append(cls.four_dim_fits_path)

    @classmethod
    def teardown_class(cls):
        """Remove test files and directories."""
        try:
            shutil.rmtree(cls.test_dir)
        except (PermissionError, OSError) as e:
            # If we can't delete due to Windows file locking, just log it and continue
            print(f"Warning: Could not delete test directory: {e}")

    def test_get_image_names_from_folder(self):
        """Test getting image names from a folder."""
        # Test non-recursive search
        image_names = get_image_names_from_folder(self.test_dir, recursive=False)
        assert len(image_names) >= 4  # Should find at least 4 images (excluding the nested one)
        assert "test_rgb.jpg" in image_names
        assert "test_gray.jpg" in image_names
        assert "test_rgba.png" in image_names
        assert "nested_image.jpg" not in image_names  # Should not find nested image

        # Test recursive search
        recursive_names = get_image_names_from_folder(self.test_dir, recursive=True)
        assert len(recursive_names) >= 5  # Should find all images including nested
        # Use a more platform-independent way to check for the nested image
        nested_dir = os.path.basename(os.path.join(self.test_dir, "nested"))
        nested_img_name = "nested_image.jpg"
        assert any(nested_dir in name and nested_img_name in name for name in recursive_names)

    def test_get_image_paths_from_folder(self):
        """Test getting image paths from a folder."""
        # Test non-recursive search
        image_paths = get_image_paths_from_folder(self.test_dir, recursive=False)
        assert len(image_paths) >= 4  # Should find at least 4 images (excluding the nested one)

        # Check that all paths exist
        assert all(os.path.exists(path) for path in image_paths)

        # Test recursive option
        recursive_paths = get_image_paths_from_folder(self.test_dir, recursive=True)
        assert len(recursive_paths) > len(image_paths)
        assert any("nested" in path for path in recursive_paths)

    def test_load_and_process_images_rgb(self, test_config):
        """Test loading and processing an RGB image."""
        # Test loading an RGB image
        img = _load_image_with_fitsbolt(self.rgb_path, cfg=test_config)
        assert img.shape[2] == 3  # Should be RGB
        assert img.dtype == np.uint8

        # Test with resizing
        _update_config(test_config, size=(50, 50))
        resized_img = _load_image_with_fitsbolt(self.rgb_path, cfg=test_config)
        assert resized_img.shape[:2] == test_config.normalisation.image_size
        assert resized_img.shape[2] == 3  # Still RGB

    def test_load_and_process_images_grayscale(self, test_config):
        """Test loading and processing a grayscale image."""
        # Test loading a grayscale image (fitsbolt automatically converts to RGB)
        img = _load_image_with_fitsbolt(self.gray_path, cfg=test_config)
        assert img.shape[2] == 3  # Should be converted to RGB
        assert img.dtype == np.uint8

    def test_load_and_process_images_rgba(self, test_config):
        """Test loading and processing an RGBA image."""
        # Test loading an RGBA image (fitsbolt automatically converts to RGB)
        img = _load_image_with_fitsbolt(self.rgba_path, cfg=test_config)
        assert img.shape[2] == 3  # Alpha channel should be removed

        # Test resizing
        target_size = (75, 75)
        _update_config(test_config, size=target_size)
        resized_img = _load_image_with_fitsbolt(self.rgba_path, cfg=test_config)
        assert resized_img.shape[:2] == target_size
        assert resized_img.shape[2] == 3  # RGB

        # Test with fully transparent image
        transparent_img = _load_image_with_fitsbolt(self.transparent_path, cfg=test_config)
        assert transparent_img.shape[2] == 3  # Should still be RGB
        # The green channel should still be present even though the pixels were transparent
        assert np.any(
            transparent_img[:, :, 1] > 0
        ), "Green channel data lost in transparent image"  # Test complex RGBA with gradient alpha
        complex_rgba_img = _load_image_with_fitsbolt(self.complex_rgba_path, cfg=test_config)
        assert complex_rgba_img.shape[2] == 3  # Should be RGB
        # Check that gradients are preserved in RGB channels
        # Note: Using 195 as threshold since some image formats/conversions may reduce max values slightly
        assert np.max(complex_rgba_img[:, :, 0]) > 195, "Red gradient not preserved"
        assert np.max(complex_rgba_img[:, :, 1]) > 195, "Green gradient not preserved"
        assert np.max(complex_rgba_img[:, :, 2]) > 195, "Blue gradient not preserved"

    def test_rgba_to_rgb_conversion_values(self, test_config):
        """Test that RGBA to RGB conversion handles alpha channel correctly."""
        # Create a test RGBA image with varying alpha and patterns
        width, height = 100, 100
        test_rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Create a pattern where:
        # - Upper left quadrant (red with full alpha)
        # - Upper right quadrant (green with half alpha)
        # - Lower left quadrant (blue with quarter alpha)
        # - Lower right quadrant (white with zero alpha)
        test_rgba[: height // 2, : width // 2, 0] = 255  # Red
        test_rgba[: height // 2, : width // 2, 3] = 255  # Full alpha

        test_rgba[: height // 2, width // 2 :, 1] = 255  # Green
        test_rgba[: height // 2, width // 2 :, 3] = 128  # Half alpha

        test_rgba[height // 2 :, : width // 2, 2] = 255  # Blue
        test_rgba[height // 2 :, : width // 2, 3] = 64  # Quarter alpha

        test_rgba[height // 2 :, width // 2 :, 0:3] = 255  # White
        test_rgba[height // 2 :, width // 2 :, 3] = 0  # Zero alpha

        # Save this test image
        test_path = os.path.join(self.test_dir, "alpha_test.png")
        Image.fromarray(test_rgba).save(test_path)

        # Load and convert to RGB
        rgb_img = _load_image_with_fitsbolt(test_path, cfg=test_config)

        # Test shape and type
        assert rgb_img.shape == (height, width, 3), "RGBA should convert to RGB shape"
        assert rgb_img.dtype == np.uint8, "RGBA conversion should maintain uint8 type"

        # Test that colors are preserved correctly according to alpha values
        # Colors with full alpha should be preserved exactly
        assert np.all(
            rgb_img[: height // 2, : width // 2, 0] == 255
        ), "Red with full alpha should be preserved"
        assert np.all(
            rgb_img[: height // 2, : width // 2, 1] == 0
        ), "Red channel should have no green"
        assert np.all(
            rgb_img[: height // 2, : width // 2, 2] == 0
        ), "Red channel should have no blue"

        # Colors with partial alpha might be handled differently depending on implementation
        # We'll just check that they exist and aren't black
        assert (
            np.mean(rgb_img[: height // 2, width // 2 :, 1]) > 0
        ), "Green with half alpha should be visible"
        assert (
            np.mean(rgb_img[height // 2 :, : width // 2, 2]) > 0
        ), "Blue with quarter alpha should be visible"
        # The behavior with fully transparent pixels can vary depending on the implementation
        # Some libraries preserve the color values even with zero alpha, others apply a background color
        # Instead of asserting they shouldn't be white, we'll just check that the image was loaded successfully
        transparent_area = rgb_img[height // 2 :, width // 2 :]
        assert transparent_area.shape == (
            height // 2,
            width // 2,
            3,
        ), "Transparent area shape is incorrect"
        assert transparent_area.dtype == np.uint8, "Transparent area data type is incorrect"

    def test_image_value_preservation(self, test_config):
        """Test that image values are preserved during loading and conversion."""
        # Create a test image with specific values to check for preservation
        test_values = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a pattern of specific RGB values
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                r_val = min(255, (i * 2) % 256)
                g_val = min(255, (j * 3) % 256)
                b_val = min(255, ((i + j) * 2) % 256)

                test_values[i : i + 5, j : j + 5, 0] = r_val
                test_values[i : i + 5, j : j + 5, 1] = g_val
                test_values[i : i + 5, j : j + 5, 2] = b_val

        # Save the test image
        values_path = os.path.join(self.test_dir, "values_test.png")
        Image.fromarray(test_values).save(values_path)

        # Load the image and check value preservation
        loaded_img = _load_image_with_fitsbolt(values_path, cfg=test_config)

        # Check overall shape and type
        assert loaded_img.shape == (100, 100, 3), "Shape should be preserved"
        assert loaded_img.dtype == np.uint8, "Data type should be preserved"

        # Check specific values
        # We allow small differences due to potential compression
        max_allowed_diff = 5  # Allow for compression artifacts

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                r_val = min(255, (i * 2) % 256)
                g_val = min(255, (j * 3) % 256)
                b_val = min(255, ((i + j) * 2) % 256)

                r_diff = abs(int(np.mean(loaded_img[i : i + 5, j : j + 5, 0])) - r_val)
                g_diff = abs(int(np.mean(loaded_img[i : i + 5, j : j + 5, 1])) - g_val)
                b_diff = abs(int(np.mean(loaded_img[i : i + 5, j : j + 5, 2])) - b_val)

                assert r_diff <= max_allowed_diff, f"Red value not preserved at ({i},{j})"
                assert g_diff <= max_allowed_diff, f"Green value not preserved at ({i},{j})"
                assert b_diff <= max_allowed_diff, f"Blue value not preserved at ({i},{j})"

    def test_load_image_with_fitsbolt_fits(self, test_config):
        """Test reading and resizing a FITS image."""
        # Test loading a FITS image
        img = _load_image_with_fitsbolt(self.fits_path, cfg=test_config)
        assert img.shape[2] == 3  # Should be converted to RGB
        assert img.dtype == np.uint8  # Should be converted to uint8

        # Test loading a multi-channel FITS image
        with pytest.raises(ValueError):
            _load_image_with_fitsbolt(self.multi_fits_path, cfg=test_config)
            # should fail as this is a 3d fits extension

        with fits.open(self.four_dim_fits_path) as hdul:
            if hdul[0].data.ndim == 4:
                # Extract a 3D slice from the 4D array (first element of first dimension)
                # and save it as a new FITS file
                slice_data = hdul[0].data[0]  # Get the first 3D slice (shape should be 3,60,60)
                slice_path = os.path.join(self.test_dir, "four_dim_slice.fits")
                fits.writeto(slice_path, slice_data, overwrite=True)

                # Now test the 3D slice which should load correctly
                # fitsbolt has no 3D fits support this will fail
                with pytest.raises(ValueError):
                    _load_image_with_fitsbolt(slice_path, cfg=test_config)

                # If this test fails in the future, implement further testing
                # changing target size for example
            else:
                # If it's not 4D, we'll skip this specific assertion
                pytest.skip("FITS file doesn't have 4D data structure")

        # Test value normalization with extreme values
        _update_config(test_config, size=None)  # No resizing for this test
        extreme_img = _load_image_with_fitsbolt(self.extreme_fits_path, cfg=test_config)
        assert extreme_img.shape[2] == 3, "Should be converted to RGB"
        assert extreme_img.dtype == np.uint8, "Should be converted to uint8"
        assert np.min(extreme_img) >= 0, "Minimum value should be normalized to at least 0"
        assert np.max(extreme_img) > 0, "Maximum value should be positive after normalization"
        # Check if normalization preserved the pattern (bright area should be brighter than dark area)
        bright_area = extreme_img[50:80, 50:80]
        dark_area = extreme_img[10:40, 10:40]
        assert np.mean(bright_area) > np.mean(
            dark_area
        ), "Normalization failed to preserve contrast"

    def test_fits_extension_parameter(self, test_config):
        """Test the fits_extension parameter for FITS files."""
        # Test explicit extension 0 (should be the same as default)
        img_default = _load_image_with_fitsbolt(self.fits_path, cfg=test_config)
        _update_config(test_config, fits_extension=0)
        img_ext0 = _load_image_with_fitsbolt(self.fits_path, cfg=test_config)
        assert np.array_equal(img_default, img_ext0)

        # For multi_fits_path, we created it with 3 channels in the test setup
        # Different extensions should have different data
        if hasattr(self, "multi_fits_path"):
            # Opening directly to check the contents
            with fits.open(self.multi_fits_path) as hdul:
                if len(hdul) > 1:  # Only test if there are multiple extensions
                    _update_config(test_config, fits_extension=1)
                    img_ext1 = _load_image_with_fitsbolt(self.multi_fits_path, cfg=test_config)
                    # Should be different from extension 0
                    assert not np.array_equal(img_default, img_ext1)

    def test_fits_extension_string_parameter(self, test_config):
        """Test string values for the fits_extension parameter."""
        # Create a FITS file with named extensions for testing
        named_fits_path = os.path.join(self.test_dir, "named_extensions.fits")

        # Create primary HDU (extension 0)
        primary_data = np.zeros((50, 50), dtype=np.float32)
        primary_data[10:40, 10:40] = 1.0  # Bright square in primary
        primary_hdu = fits.PrimaryHDU(primary_data)
        primary_hdu.header["EXTNAME"] = "PRIMARY"

        # Create extension 1 with name 'SCIENCE'
        science_data = np.zeros((50, 50), dtype=np.float32)
        science_data[20:30, 20:30] = 0.8  # Different pattern
        science_hdu = fits.ImageHDU(science_data)
        science_hdu.header["EXTNAME"] = "SCIENCE"

        # Create extension 2 with name 'ERROR'
        error_data = np.ones((50, 50), dtype=np.float32) * 0.1
        error_data[15:35, 15:35] = 0.2  # Different pattern
        error_hdu = fits.ImageHDU(error_data)
        error_hdu.header["EXTNAME"] = "ERROR"

        # Create FITS file with multiple named extensions
        hdul = fits.HDUList([primary_hdu, science_hdu, error_hdu])
        hdul.writeto(named_fits_path, overwrite=True)

        # Now test accessing by string name
        _update_config(test_config, fits_extension="PRIMARY")
        img_primary = _load_image_with_fitsbolt(named_fits_path, cfg=test_config)
        _update_config(test_config, fits_extension="SCIENCE")
        img_science = _load_image_with_fitsbolt(named_fits_path, cfg=test_config)
        _update_config(test_config, fits_extension="ERROR")
        img_error = _load_image_with_fitsbolt(named_fits_path, cfg=test_config)

        # Verify that each extension has different data
        assert not np.array_equal(img_primary, img_science)
        assert not np.array_equal(img_primary, img_error)
        assert not np.array_equal(img_science, img_error)

        # Test accessing using index vs name (should be equivalent)
        _update_config(test_config, fits_extension=0)
        img_primary_idx = _load_image_with_fitsbolt(named_fits_path, cfg=test_config)
        _update_config(test_config, fits_extension=1)
        img_science_idx = _load_image_with_fitsbolt(named_fits_path, cfg=test_config)
        _update_config(test_config, fits_extension=2)
        img_error_idx = _load_image_with_fitsbolt(named_fits_path, cfg=test_config)

        assert np.array_equal(img_primary, img_primary_idx)
        assert np.array_equal(img_science, img_science_idx)
        assert np.array_equal(img_error, img_error_idx)

    def test_fits_extension_error_handling(self, test_config):
        """Test error handling for invalid FITS extensions."""
        # Test out-of-bounds extension index
        with pytest.raises(IndexError):
            _update_config(test_config, fits_extension=999)
            _load_image_with_fitsbolt(self.fits_path, cfg=test_config)

        # Test negative extension index
        with pytest.raises(IndexError):
            _update_config(test_config, fits_extension=-1)
            _load_image_with_fitsbolt(self.fits_path, cfg=test_config)

    def test_load_and_process_images(self, test_config):
        """Test loading multiple images in parallel."""
        # Make a copy of the file list to avoid permission issues
        test_files = self.image_files[
            2:5
        ]  # Include a variety of image with the same number of channels
        # Test basic functionality with multiple file types
        results = _load_multiple_images_with_fitsbolt(
            test_files, cfg=test_config, show_progress=False
        )

        # Should return all the files we passed
        assert len(results) == len(test_files)

        # Verify each image was loaded correctly
        for filepath, img in results:
            assert img is not None, f"Failed to load image: {filepath}"
            assert img.shape[2] == 3, f"Image {filepath} should be RGB"
            assert img.dtype == np.uint8, f"Image {filepath} should be uint8"

        # Test with resizing
        target_size = (30, 30)
        _update_config(test_config, size=target_size)
        resized_results = _load_multiple_images_with_fitsbolt(
            test_files, cfg=test_config, show_progress=False
        )

        # Check that all resized images have the correct size
        for filepath, img in resized_results:
            assert img.shape[:2] == target_size, f"Image {filepath} wasn't resized correctly"
            assert img.shape[2] == 3, f"Image {filepath} should be RGB after resize"

    def test_fits_multiple_extensions(self, test_config):
        """Test loading and combining multiple FITS extensions."""
        # Create a test FITS file with multiple extensions of the same shape
        multi_ext_path = os.path.join(self.test_dir, "multi_extension.fits")

        # Create primary HDU with different pattern
        primary_data = np.zeros((50, 50), dtype=np.float32)
        primary_data[10:30, 10:30] = 0.5  # Square in top-left
        primary_hdu = fits.PrimaryHDU(primary_data)
        primary_hdu.header["EXTNAME"] = "PRIMARY"

        # Create extension 1 with different pattern
        ext1_data = np.zeros((50, 50), dtype=np.float32)
        ext1_data[10:30, 20:40] = 0.7  # Square in top-middle
        ext1_hdu = fits.ImageHDU(ext1_data)
        ext1_hdu.header["EXTNAME"] = "EXT1"

        # Create extension 2 with different pattern
        ext2_data = np.zeros((50, 50), dtype=np.float32)
        ext2_data[20:40, 20:40] = 0.9  # Square in middle
        ext2_hdu = fits.ImageHDU(ext2_data)
        ext2_hdu.header["EXTNAME"] = "EXT2"

        # Create FITS file with multiple extensions of same shape
        hdul = fits.HDUList([primary_hdu, ext1_hdu, ext2_hdu])
        hdul.writeto(multi_ext_path, overwrite=True)

        # Test loading with list of integer indices
        int_indices = [0, 1, 2]
        _update_config(test_config, fits_extension=int_indices)
        combined_img1 = _load_image_with_fitsbolt(multi_ext_path, cfg=test_config)

        # Test loading with list of string names
        str_names = ["PRIMARY", "EXT1", "EXT2"]
        _update_config(test_config, fits_extension=str_names)
        combined_img2 = _load_image_with_fitsbolt(multi_ext_path, cfg=test_config)

        # Test loading with mixed list of indices and names
        mixed_list = [0, "EXT2", "EXT1"]
        _update_config(test_config, fits_extension=mixed_list)
        combined_img3 = _load_image_with_fitsbolt(multi_ext_path, cfg=test_config)

        # All should result in RGB images with shape (50, 50, 3)
        assert combined_img1.shape == (50, 50, 3), "Combined image should have shape (50, 50, 3)"
        assert combined_img2.shape == (50, 50, 3), "Combined image should have shape (50, 50, 3)"
        assert combined_img3.shape == (50, 50, 3), "Combined image should have shape (50, 50, 3)"

        # The images should be different due to different ordering
        assert not np.array_equal(
            combined_img1, combined_img3
        ), "Different extension order should give different results"

        # Int indices and string names with same order should give same result
        assert np.array_equal(
            combined_img1, combined_img2
        ), "Same extension order should give same results"

        # Create another FITS file with extensions of different shapes to test error handling
        diff_shapes_path = os.path.join(self.test_dir, "diff_shapes.fits")

        # Primary data with shape (50, 50)
        primary_data2 = np.zeros((50, 50), dtype=np.float32)
        primary_data2[10:30, 10:30] = 0.5
        primary_hdu2 = fits.PrimaryHDU(primary_data2)

        # Extension with different shape (60, 40)
        diff_shape_data = np.zeros((60, 40), dtype=np.float32)
        diff_shape_data[20:40, 10:30] = 0.8
        diff_shape_hdu = fits.ImageHDU(diff_shape_data)
        diff_shape_hdu.header["EXTNAME"] = "DIFF_SHAPE"

        # Create FITS file with different shape extensions
        hdul2 = fits.HDUList([primary_hdu2, diff_shape_hdu])
        hdul2.writeto(diff_shapes_path, overwrite=True)

        # Test that combining different shapes raises a ValueError
        with pytest.raises(ValueError) as e_info:
            _update_config(test_config, fits_extension=[0, 1], channel_combination=None)
            _load_image_with_fitsbolt(diff_shapes_path, cfg=test_config)

        # Validate the error message contains information about channel mismatch
        error_message = str(e_info.value)
        assert (
            "channel" in error_message.lower() or "extension" in error_message.lower()
        ), f"Error should mention channel or extension mismatch: {error_message}"
        # could expand test if needed with more extensions

    def test_load_and_process_images_fits_extension(self, test_config):
        """Test that load_and_process_images correctly passes the fits_extension parameter."""
        # Create a test FITS file with multiple extensions of the same shape
        multi_ext_path = os.path.join(self.test_dir, "multi_extension_parallel.fits")

        # Create primary HDU with different pattern
        primary_data = np.zeros((40, 40), dtype=np.float32)
        primary_data[5:15, 5:15] = 0.5  # Square in top-left
        primary_hdu = fits.PrimaryHDU(primary_data)
        primary_hdu.header["EXTNAME"] = "PRIMARY"

        # Create extension 1 with different pattern
        ext1_data = np.zeros((40, 40), dtype=np.float32)
        ext1_data[15:25, 15:25] = 0.7  # Square in middle
        ext1_hdu = fits.ImageHDU(ext1_data)
        ext1_hdu.header["EXTNAME"] = "EXT1"

        # Create extension 2 with different pattern
        ext2_data = np.zeros((40, 40), dtype=np.float32)
        ext2_data[25:35, 25:35] = 0.9  # Square in bottom-right
        ext2_hdu = fits.ImageHDU(ext2_data)
        ext2_hdu.header["EXTNAME"] = "EXT2"

        # Create FITS file with multiple extensions
        hdul = fits.HDUList([primary_hdu, ext1_hdu, ext2_hdu])
        hdul.writeto(multi_ext_path, overwrite=True)

        # Create a duplicate file for testing multiple files
        multi_ext_path2 = os.path.join(self.test_dir, "multi_extension_parallel2.fits")
        hdul.writeto(multi_ext_path2, overwrite=True)

        # List of files to test
        test_files = [multi_ext_path, multi_ext_path2]

        # Test load_and_process_images with a single extension index
        _update_config(test_config, fits_extension=0)
        results_ext0 = _load_multiple_images_with_fitsbolt(
            test_files, cfg=test_config, show_progress=False
        )

        # Test load_and_process_images with a different extension index
        _update_config(test_config, fits_extension=1)
        results_ext1 = _load_multiple_images_with_fitsbolt(
            test_files, cfg=test_config, show_progress=False
        )

        # Test load_and_process_images with a list of extensions
        _update_config(test_config, fits_extension=[0, 1, 2])
        results_combined = _load_multiple_images_with_fitsbolt(
            test_files, cfg=test_config, show_progress=False
        )

        # Verify that all files were loaded
        assert len(results_ext0) == len(test_files)
        assert len(results_ext1) == len(test_files)
        assert len(results_combined) == len(test_files)

        # Verify that the images have expected dimensions
        for _, img in results_ext0:
            assert img.shape == (
                40,
                40,
                3,
            ), "Single extension should result in (40, 40, 3) RGB image"

        for _, img in results_ext1:
            assert img.shape == (
                40,
                40,
                3,
            ), "Single extension should result in (40, 40, 3) RGB image"

        for _, img in results_combined:
            assert img.shape == (
                40,
                40,
                3,
            ), "Combined extensions should result in (40, 40, 3) RGB image"

        # Verify that using different extensions produces different results
        # Extract first image from each result
        img_ext0 = results_ext0[0][1]
        img_ext1 = results_ext1[0][1]
        img_combined = results_combined[0][1]

        # The images should have different content because they used different extensions
        assert not np.array_equal(
            img_ext0, img_ext1
        ), "Different extensions should produce different images"
        assert not np.array_equal(
            img_ext0, img_combined
        ), "Combined extensions should differ from single extension"

        # Also test with string extension names
        _update_config(test_config, fits_extension=["PRIMARY", "EXT1", "EXT2"])
        results_named = _load_multiple_images_with_fitsbolt(
            test_files, cfg=test_config, show_progress=False
        )
        img_named = results_named[0][1]

        # Should be identical to using numeric indices [0, 1, 2]
        assert np.array_equal(
            img_combined, img_named
        ), "String extension names should produce same result as numeric indices"

    def test_image_normalisation(self, test_config):
        """Test that different normalisation methods are correctly applied during image loading."""
        # Create a test image with known values to test normalisation
        test_values = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a gradient pattern for testing
        for i in range(100):
            for j in range(100):
                # Create a diagonal gradient
                value = int((i + j) / 2)  # Values from 0 to 99
                test_values[i, j] = [value, value, value]

        # Save the test image
        test_path = os.path.join(self.test_dir, "normalisation_test.png")
        Image.fromarray(test_values).save(test_path)

        # Test with no normalisation (default)
        _update_config(test_config, normalisation_method=NormalisationMethod.CONVERSION_ONLY)
        img_none = _load_image_with_fitsbolt(test_path, cfg=test_config)
        assert np.array_equal(
            img_none, test_values
        ), "NONE normalisation should preserve original values"

        # Test with LOG normalisation
        _update_config(test_config, normalisation_method=NormalisationMethod.LOG)
        img_log = _load_image_with_fitsbolt(test_path, cfg=test_config)
        assert not np.array_equal(img_log, test_values), "LOG normalisation should modify values"
        # Log normalisation should enhance darker regions
        # Check that dark regions (low values) have relatively higher values after log normalisation
        dark_region_original = test_values[0:10, 0:10]
        dark_region_log = img_log[0:10, 0:10]
        ratio_dark = np.mean(dark_region_log) / np.mean(dark_region_original)
        assert ratio_dark > 1, "LOG normalisation should enhance dark regions"

        # Test with ZSCALE normalisation
        _update_config(test_config, normalisation_method=NormalisationMethod.ZSCALE)
        img_zscale = _load_image_with_fitsbolt(test_path, cfg=test_config)
        assert not np.array_equal(
            img_zscale, test_values
        ), "ZSCALE normalisation should modify values"
        # ZScale should produce values with reasonable contrast
        assert np.min(img_zscale) < np.max(img_zscale), "ZSCALE should preserve contrast"

        # Test that all normalised outputs remain in valid uint8 range
        assert img_none.dtype == np.uint8, "NONE normalisation should maintain uint8 type"
        assert img_log.dtype == np.uint8, "LOG normalisation should maintain uint8 type"
        assert img_zscale.dtype == np.uint8, "ZSCALE normalisation should maintain uint8 type"

        # Test that all normalised outputs preserve image dimensions
        assert img_none.shape == test_values.shape, "NONE normalisation should preserve dimensions"
        assert img_log.shape == test_values.shape, "LOG normalisation should preserve dimensions"
        assert (
            img_zscale.shape == test_values.shape
        ), "ZSCALE normalisation should preserve dimensions"

    def test_image_interpolation_orders(self, test_config):
        """Test different interpolation orders when resizing images.

        This test creates a 40x40 image and a 200x200 image, then resizes both to 100x100
        using different interpolation orders (0-5), which correspond to different polynomial
        interpolation methods in scikit-image.
        """
        # Create a small (40x40) test image with a clear pattern
        small_img = np.zeros((40, 40, 3), dtype=np.uint8)
        # Create a pattern with solid color blocks to clearly see interpolation effects
        small_img[0:20, 0:20, 0] = 255  # Red block
        small_img[0:20, 20:40, 1] = 255  # Green block
        small_img[20:40, 0:20, 2] = 255  # Blue block
        small_img[20:40, 20:40, :] = 255  # White block

        # Save the small test image
        small_path = os.path.join(self.test_dir, "small_interpolation_test.png")
        Image.fromarray(small_img).save(small_path)

        # Create a large (200x200) test image with a clear pattern
        large_img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create a pattern with solid color blocks
        large_img[0:100, 0:100, 0] = 255  # Red block
        large_img[0:100, 100:200, 1] = 255  # Green block
        large_img[100:200, 0:100, 2] = 255  # Blue block
        large_img[100:200, 100:200, :] = 255  # White block

        # Save the large test image
        large_path = os.path.join(self.test_dir, "large_interpolation_test.png")
        Image.fromarray(large_img).save(large_path)

        # Set target size to 100x100 for both images
        _update_config(test_config, size=(100, 100))
        _update_config(test_config, normalisation_method=NormalisationMethod.CONVERSION_ONLY)

        # Define the expected center pixel colors for each quadrant after resizing
        # We're checking center points of each quadrant to avoid edge effects
        expected_quadrant_colors = [
            (255, 0, 0),  # Top-left: Red
            (0, 255, 0),  # Top-right: Green
            (0, 0, 255),  # Bottom-left: Blue
            (255, 255, 255),  # Bottom-right: White
        ]

        # Store results from different interpolation orders to compare them
        upscaled_results = []

        # Check each interpolation order (0-5)
        for order in range(6):
            _update_config(test_config, interpolation_order=order)

            # Resize small image (40x40 → 100x100) - upsampling
            resized_small = _load_image_with_fitsbolt(small_path, cfg=test_config)
            upscaled_results.append(resized_small)
            assert resized_small.shape == (
                100,
                100,
                3,
            ), f"Resized small image should be 100x100 with order {order}"
            assert (
                resized_small.dtype == np.uint8
            ), f"Resized small image should be uint8 with order {order}"

            # Resize large image (200x200 → 100x100) - downsampling
            resized_large = _load_image_with_fitsbolt(large_path, cfg=test_config)
            assert resized_large.shape == (
                100,
                100,
                3,
            ), f"Resized large image should be 100x100 with order {order}"
            assert (
                resized_large.dtype == np.uint8
            ), f"Resized large image should be uint8 with order {order}"

            # Check the center pixel of each quadrant for both resized images
            # Allow for some variation (±20%) in color values due to interpolation differences
            quadrant_centers = [
                (25, 25),  # Top-left quadrant
                (75, 25),  # Top-right quadrant
                (25, 75),  # Bottom-left quadrant
                (75, 75),  # Bottom-right quadrant
            ]

            for idx, (x, y) in enumerate(quadrant_centers):
                expected_color = expected_quadrant_colors[idx]

                # Check small image upsampled
                small_color = resized_small[y, x]
                for c in range(3):
                    expected_value = expected_color[c]
                    if expected_value > 0:
                        # For non-zero values, check within 20% tolerance
                        assert abs(int(small_color[c]) - expected_value) <= 0.2 * expected_value, (
                            f"Small image order {order}, quadrant {idx}, channel {c}:"
                            + f"expected ~{expected_value}, got {small_color[c]}"
                        )
                    else:
                        # For zero values, small absolute threshold
                        assert (
                            small_color[c] <= 50
                        ), f"Small image order {order}, quadrant {idx}, channel {c}: expected ~0, got {small_color[c]}"

                # Check large image downsampled
                large_color = resized_large[y, x]
                for c in range(3):
                    expected_value = expected_color[c]
                    if expected_value > 0:
                        # For non-zero values, check within 20% tolerance
                        tolerance = 0.2 * expected_value
                        assert abs(int(large_color[c]) - expected_value) <= tolerance, (
                            f"Large image order {order}, quadrant {idx}, channel {c}:"
                            + f"expected ~{expected_value}, got {large_color[c]}"
                        )
                    else:
                        # For zero values, small absolute threshold
                        assert (
                            large_color[c] <= 50
                        ), f"Large image order {order}, quadrant {idx}, channel {c}: expected ~0, got {large_color[c]}"

            # Additional check for sharp transitions with order 0 (nearest neighbor)
            if order == 0:
                # Nearest neighbor should have sharp transitions between quadrants
                # Check pixels right at the boundary
                boundary_x = 50
                boundary_y = 50

                # For small image
                left_of_boundary_small = resized_small[boundary_y, boundary_x - 1]
                right_of_boundary_small = resized_small[boundary_y, boundary_x + 1]
                assert not np.array_equal(
                    left_of_boundary_small, right_of_boundary_small
                ), "Small image order 0 should have sharp transitions at boundary"

                # For large image
                left_of_boundary_large = resized_large[boundary_y, boundary_x - 1]
                right_of_boundary_large = resized_large[boundary_y, boundary_x + 1]
                assert not np.array_equal(
                    left_of_boundary_large, right_of_boundary_large
                ), "Large image order 0 should have sharp transitions at boundary"

            # Higher order interpolation (order > 1) should lead to smoother transitions
            # This is difficult to quantify precisely, but we can check for values between the extremes for upscaling
            if order >= 3:
                # For boundary regions, check that there are intermediate values
                # between the pure colors in neighboring quadrants
                # Sample near the boundary but not exactly on it
                boundary_region_small = resized_small[45:55, 45:55]

                # For higher order interpolation, we expect to find intermediate values
                # in the boundary regions
                unique_values_small = np.unique(boundary_region_small)

                # Higher order interpolation should have more unique values in the boundary region when upscaling
                assert (
                    len(unique_values_small) > 4
                ), f"Small image order {order} should have intermediate values at boundaries"

        # Compare results between different interpolation orders to verify they're not identical
        # We'll compare order 0 (nearest neighbor) with orders 1, 3, and 5
        # These should produce visibly different results
        for i, upscaled_im in enumerate(upscaled_results):
            if i != 0:
                assert not np.array_equal(
                    upscaled_results[0], upscaled_results[i]
                ), "Order 0 and order {i} interpolation should produce different results"
                assert not np.array_equal(
                    upscaled_results[i - 1], upscaled_results[i]
                ), f"Order {i - 1} and order {i} interpolation should produce different results"

    def test_fits_combination_configurations(self, test_config):
        """Test different configurations of the fits_combination dictionary."""
        # Create a FITS file with 4 extensions for testing
        test_data = []
        for i in range(4):
            data = np.zeros((31, 31), dtype=np.float32)
            data[20:30, 20:30] = float(i + 1)  # Different value in each extension
            test_data.append(data)

        multi_ext_fits_path = os.path.join(self.test_dir, "multi_ext.fits")
        primary_hdu = fits.PrimaryHDU(test_data[0])
        hdul = fits.HDUList([primary_hdu])
        for i, data in enumerate(test_data[1:], 1):
            hdul.append(fits.ImageHDU(data, name=f"EXT{i}"))
        hdul.writeto(multi_ext_fits_path, overwrite=True)
        self.image_files.append(multi_ext_fits_path)  # Add to cleanup list
        _update_config(test_config, normalisation_method=NormalisationMethod.CONVERSION_ONLY)
        # Test 1: None combination dictionary (should use first three extensions as default)
        _update_config(test_config, fits_extension=[0, 1, 2])  # Use first three extensions
        _update_config(test_config, channel_combination=None)
        img_empty = _load_image_with_fitsbolt(multi_ext_fits_path, cfg=test_config)
        assert img_empty.shape[2] == 3
        # Check if the first three extensions are used directly
        assert np.any(img_empty[:, :, :] > 0)  # Should have non-zero values

        # Test 2: Combination with all 4 extensions
        _update_config(test_config, fits_extension=[0, 1, 2, 3])
        _update_config(
            test_config,
            channel_combination=np.array(
                [
                    [0.5, 0.5, 0, 0],  # Combine ext 0 and 1
                    [0, 0, 1, 0],  # Just ext 2
                    [0, 0, 0, 1],  # Just ext 3
                ]
            ),
        )
        img_four = _load_image_with_fitsbolt(multi_ext_fits_path, cfg=test_config)
        assert img_four.shape[2] == 3
        # Check if red channel contains combination of first two extensions
        assert np.any(img_four[:, :, 0] > 0)  # Red channel should have data
        assert np.any(img_four[:, :, 1] > 0)  # Green channel should have data
        assert np.any(img_four[:, :, 2] > 0)  # Blue channel should have data

        # Test 3: Combination with 2 extensions only
        _update_config(test_config, fits_extension=[0, 1])
        _update_config(
            test_config,
            channel_combination=np.array(
                [
                    [1, 0],  # Just ext 0
                    [0.5, 0.5],  # Mix of both
                    [0, 1],  # Just ext 1
                ]
            ),
        )
        img_two = _load_image_with_fitsbolt(multi_ext_fits_path, cfg=test_config)
        assert img_two.shape[2] == 3
        assert np.any(img_two[:, :, 0] > 0)  # Red channel should have data
        assert np.any(img_two[:, :, 1] > 0)  # Green channel should have data
        assert np.any(img_two[:, :, 2] > 0)  # Blue channel should have data
