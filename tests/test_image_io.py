#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import pytest
import tempfile
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torch
import copy

from anomaly_match.data_io.load_images import (
    load_and_process_single_wrapper,
    process_single_wrapper,
    get_fitsbolt_config,
)
from anomaly_match.utils.get_default_cfg import get_default_cfg
from prediction_utils import save_results


def _load_image_with_fitsbolt(filepath, cfg):
    """Helper function to load image using fitsbolt with AnomalyMatch config."""
    # Use the new wrapper function instead of directly using fitsbolt
    return load_and_process_single_wrapper(filepath, cfg, desc="test loading", show_progress=False)


def _update_config(cfg, **kwargs):
    """Update both the main config and fitsbolt config with the given parameters."""
    for key, value in kwargs.items():
        setattr(cfg, key, value)
        if key == "size":
            cfg.normalisation.image_size = value
        elif key == "fits_extension":
            cfg.normalisation.fits_extension = value
        elif key.startswith("normalisation."):
            # Handle nested attributes in normalisation
            norm_key = key.split(".")[1]
            setattr(cfg.normalisation, norm_key, value)
            fitsbolt_key = f"norm_{norm_key}"
            setattr(cfg.normalisation, fitsbolt_key, value)
    return cfg


def _process_image_array_with_fitsbolt(image_array, cfg):
    """Helper function to process image array using fitsbolt."""
    # Use the new wrapper function instead of directly using fitsbolt
    testcfg = copy.deepcopy(cfg)
    testcfg = get_fitsbolt_config(testcfg)

    return process_single_wrapper(image_array, testcfg, desc="array")


def create_fits_file(image_data, filepath):
    """
    Create a FITS file from the provided image data.

    Parameters:
    -----------
    image_data : numpy.ndarray
        Image data to save as FITS
    filepath : str or Path
        Path to save the FITS file
    """
    # Import locally to avoid keeping handles open
    from astropy.io import fits

    # Store RGB data as 3 separate FITS extensions to ensure we get a 3D image when loading
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["OBJECT"] = "Test Object"
    primary_hdu.header["TELESCOP"] = "Test Telescope"
    primary_hdu.header["INSTRUME"] = "Test Instrument"
    primary_hdu.header["DATE-OBS"] = "2023-08-15T00:00:00"
    primary_hdu.header["EXPTIME"] = 100.0
    primary_hdu.header["COMMENT"] = "This is a test FITS file for anomaly detection testing"

    # Create HDU for each RGB channel
    if image_data.ndim == 3 and image_data.shape[2] == 3:
        # Extract R, G, B channels
        r_hdu = fits.ImageHDU(data=image_data[:, :, 0])
        r_hdu.header["EXTNAME"] = "R"

        g_hdu = fits.ImageHDU(data=image_data[:, :, 1])
        g_hdu.header["EXTNAME"] = "G"

        b_hdu = fits.ImageHDU(data=image_data[:, :, 2])
        b_hdu.header["EXTNAME"] = "B"

        # Write all channels to FITS file
        hdul = fits.HDUList([primary_hdu, r_hdu, g_hdu, b_hdu])
        hdul.writeto(filepath, overwrite=True)
        hdul.close()
    else:
        # For non-RGB data, store as is
        data_hdu = fits.ImageHDU(data=image_data)
        data_hdu.header["EXTNAME"] = "DATA"
        hdul = fits.HDUList(
            [primary_hdu, data_hdu, data_hdu, data_hdu]
        )  # Duplicate to get 3 channels
        hdul.writeto(filepath, overwrite=True)
        hdul.close()
        # Explicitly delete objects to release file handles
    del primary_hdu
    if "r_hdu" in locals():
        del r_hdu
        del g_hdu
        del b_hdu
    if "data_hdu" in locals():
        del data_hdu
    del hdul


class TestImageIO:
    """Test image input/output operations across different formats."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with known pattern for verification."""
        height, width, channels = 224, 224, 3
        image = np.zeros((height, width, channels), dtype=np.uint8)

        # Red channel: horizontal gradient
        for i in range(height):
            image[i, :, 0] = int((i / height) * 255)

        # Green channel: checkerboard pattern
        for i in range(height):
            for j in range(width):
                if (i // 16 + j // 16) % 2 == 0:
                    image[i, j, 1] = 255

        # Blue channel: solid color with some noise
        np.random.seed(42)  # For reproducible noise
        image[:, :, 2] = 128 + np.random.randint(-50, 51, (height, width))

        return image

    @pytest.fixture
    def test_config(self):
        """Get test configuration."""
        cfg = get_default_cfg()
        # Set fits_extension to use the proper extensions from our test FITS file

        # Add fitsbolt configuration needed by the wrapper functions
        cfg.normalisation.image_size = [224, 224]
        cfg.normalisation.fits_extension = ["R", "G", "B"]
        cfg.normalisation.n_output_channels = 3

        return cfg

    def test_image_format_consistency(self, test_image, test_config):
        """Test that images are consistent across different file formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save as PNG
            png_path = temp_path / "test_image.png"
            Image.fromarray(test_image).save(png_path)

            # Save as HDF5
            hdf5_path = temp_path / "test_image.h5"
            with h5py.File(hdf5_path, "w") as f:
                f.create_dataset("image", data=test_image)

            # Load and compare PNG vs HDF5
            loaded_png = _load_image_with_fitsbolt(str(png_path), test_config)
            loaded_hdf5 = None

            # Load from HDF5
            with h5py.File(hdf5_path, "r") as f:
                hdf5_data = f["image"][:]
                loaded_hdf5 = _process_image_array_with_fitsbolt(hdf5_data, test_config)

            # Both should have the same shape after processing
            if hasattr(loaded_png, "shape"):
                png_shape = loaded_png.shape
            else:
                png_shape = np.array(loaded_png).shape

            if hasattr(loaded_hdf5, "shape"):
                hdf5_shape = loaded_hdf5.shape
            else:
                hdf5_shape = np.array(loaded_hdf5).shape

            assert png_shape == hdf5_shape

            # Convert to arrays for comparison
            png_array = np.array(loaded_png) if hasattr(loaded_png, "size") else loaded_png
            hdf5_array = np.array(loaded_hdf5) if hasattr(loaded_hdf5, "size") else loaded_hdf5

            # Check that arrays are reasonably similar
            # Using a tolerance for potential format differences
            assert np.allclose(png_array, hdf5_array, atol=1)

    def test_save_load_consistency(self, test_image, test_config):
        """Test that saved images can be loaded back consistently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_config.output_dir = str(temp_path)

            # Create mock data for save_results function
            all_scores = np.array([0.5])  # Convert to numpy array
            all_imgs = test_image[np.newaxis, ...]  # Add batch dimension
            all_filenames = np.array(["test_image.png"])  # Convert to numpy array

            # Use save_results to save the results
            save_results(
                cfg=test_config,
                all_scores=all_scores,
                all_imgs=all_imgs,
                all_filenames=all_filenames,
                top_n=1,
            )

            # Check that output files were created
            output_files = list(temp_path.glob("*"))
            assert len(output_files) > 0

            # Check for specific expected files
            csv_files = list(temp_path.glob("*.csv"))
            npy_files = list(temp_path.glob("*.npy"))
            npz_files = list(temp_path.glob("*.npz"))

            # At least one of these should exist
            assert len(csv_files) > 0 or len(npy_files) > 0 or len(npz_files) > 0

    def test_tensor_to_uint8_conversion(self, test_config):
        """Test conversion from float tensors to uint8 images."""
        # Create float tensor in [0, 1] range
        float_tensor = torch.rand(3, 224, 224)  # CHW format

        # Convert to numpy and then to uint8 format that save_results expects
        # This simulates what happens in the prediction pipeline
        float_numpy = float_tensor.permute(1, 2, 0).numpy()  # Convert to HWC

        # Test the conversion that should happen in save_results
        images_uint8 = (float_numpy * 255).astype(np.uint8)

        # Verify the conversion
        assert images_uint8.dtype == np.uint8
        assert images_uint8.min() >= 0
        assert images_uint8.max() <= 255
        assert images_uint8.shape == (224, 224, 3)

        # Test with a known pattern
        test_tensor = torch.zeros(3, 10, 10)
        test_tensor[0, :, :] = 0.5  # Red channel at 50%
        test_tensor[1, :, :] = 1.0  # Green channel at 100%
        test_tensor[2, :, :] = 0.0  # Blue channel at 0%

        test_numpy = test_tensor.permute(1, 2, 0).numpy()
        test_uint8 = (test_numpy * 255).astype(np.uint8)

        assert np.all(test_uint8[:, :, 0] == 127)  # 50% of 255
        assert np.all(test_uint8[:, :, 1] == 255)  # 100% of 255
        assert np.all(test_uint8[:, :, 2] == 0)  # 0% of 255

    def test_image_pipeline_integration(self, test_image, test_config):
        """Test the full image processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save test image
            input_path = temp_path / "input.png"
            Image.fromarray(test_image).save(input_path)

            # Load through the pipeline
            processed_image = _load_image_with_fitsbolt(str(input_path), test_config)

            # Should be processed correctly
            assert processed_image is not None

            # The function returns a numpy array
            assert isinstance(processed_image, np.ndarray)
            expected_size = tuple(test_config.normalisation.image_size)
            assert (
                processed_image.shape[:2] == expected_size
                or processed_image.shape[:2] == expected_size[::-1]
            )
            assert processed_image.dtype == np.uint8
            assert processed_image.ndim == 3
            assert processed_image.shape[2] == 3  # RGB

    def test_prediction_process_integration(self, test_image, test_config):
        """Test integration with prediction processes for different image formats."""
        import tempfile
        from pathlib import Path
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Setup test data directory
            data_dir = temp_path / "test_data"
            data_dir.mkdir()

            # Create test images in different formats
            formats_to_test = [
                ("test_image.png", lambda p: Image.fromarray(test_image).save(p)),
                ("test_image.jpg", lambda p: Image.fromarray(test_image).save(p, quality=95)),
                ("test_image.tiff", lambda p: Image.fromarray(test_image).save(p)),
                ("test_image.fits", lambda p: create_fits_file(test_image, p)),
            ]

            test_files = []
            for filename, save_func in formats_to_test:
                file_path = data_dir / filename
                save_func(file_path)

                # Small delay and explicit GC to help ensure file handles are released
                time.sleep(0.1)  # Ensure the file exists and is accessible
                assert file_path.exists(), f"File {file_path} was not created"
                assert file_path.is_file(), f"File {file_path} is not a regular file"

                test_files.append(str(file_path))

            # Create a file list for prediction
            file_list_path = temp_path / "file_list.txt"
            with open(file_list_path, "w") as f:
                f.write(str(temp_path / "files_to_process.txt") + "\n")

            files_to_process_path = temp_path / "files_to_process.txt"
            with open(files_to_process_path, "w") as f:
                for test_file in test_files:
                    f.write(test_file + "\n")

            # Update test config for prediction
            test_config.output_dir = str(temp_path / "results")
            test_config.save_file = "test_predictions"
            test_config.model_path = "tests/test_data/dummy_model.pth"  # We'll need a dummy model

            # Create output directory
            Path(test_config.output_dir).mkdir(parents=True, exist_ok=True)

            # Save config to TOML
            config_path = temp_path / "test_config.toml"

            # Save config with pickle
            import pickle
            from dotmap import DotMap

            with open(config_path, "wb") as f:
                pickle.dump(test_config.toDict(), f)

            # Reload config to simulate real usage
            with open(config_path, "rb") as f:
                reloaded_config = pickle.load(f)
            reloaded_config = DotMap(reloaded_config)

            # Check that critical fields are present
            assert hasattr(
                reloaded_config, "normalisation"
            ), "normalisation field missing from reloaded config"
            assert hasattr(
                reloaded_config.normalisation, "fits_extension"
            ), "fits_extension field missing from reloaded config.normalisation"
            assert hasattr(
                reloaded_config.normalisation, "size"
            ), "size field missing from reloaded config.normalisation"
            assert hasattr(
                reloaded_config.normalisation, "normalisation_method"
            ), "normalisation_method field missing from reloaded config.normalisation"

            # Test image loading with reloaded config
            for test_file in test_files:
                try:
                    loaded_image = _load_image_with_fitsbolt(test_file, reloaded_config)
                    assert (
                        loaded_image is not None
                    ), f"Failed to load {test_file} with reloaded config"
                    assert isinstance(
                        loaded_image, np.ndarray
                    ), f"Loaded image from {test_file} is not a numpy array"
                    assert (
                        loaded_image.ndim == 3
                    ), f"Loaded image from {test_file} should be 3D (HWC)"
                    assert (
                        loaded_image.shape[2] == 3
                    ), f"Loaded image from {test_file} should have 3 channels"  # Ensure no NaN/inf values
                    assert np.isfinite(
                        loaded_image
                    ).all(), f"Image from {test_file} contains NaN or inf values"
                except Exception as e:
                    pytest.fail(f"Failed to load {test_file} with reloaded config: {e}")

    def test_image_formats_comprehensive(self, test_image, test_config):
        """Test comprehensive image format support."""
        import tempfile
        from pathlib import Path
        import time
        import gc

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test various image formats
            format_tests = [
                # (filename, save_function, expected_load_success)
                ("test.png", lambda p: Image.fromarray(test_image).save(p), True),
                ("test.jpg", lambda p: Image.fromarray(test_image).save(p, quality=95), True),
                ("test.jpeg", lambda p: Image.fromarray(test_image).save(p, quality=95), True),
                ("test.tiff", lambda p: Image.fromarray(test_image).save(p), True),
                (
                    "test.fits",
                    lambda p: create_fits_file(test_image, p),
                    True,
                ),
            ]

            for filename, save_func, should_succeed in format_tests:
                file_path = temp_path / filename
                save_func(file_path)

                # Give some time for file handles to be properly released
                time.sleep(0.1)
                # Force garbage collection to release any lingering handles
                gc.collect()  # Check file is accessible
                assert file_path.exists(), f"File {file_path} was not created"
                assert file_path.is_file(), f"File {file_path} is not a regular file"

                try:
                    loaded_image = _load_image_with_fitsbolt(str(file_path), test_config)

                    if should_succeed:
                        assert loaded_image is not None, f"Failed to load {filename}"
                        assert isinstance(
                            loaded_image, np.ndarray
                        ), f"Loaded {filename} is not a numpy array"

                        # Check dimensions
                        assert (
                            loaded_image.ndim >= 2
                        ), f"Loaded {filename} has insufficient dimensions"

                        # Check data integrity
                        assert np.isfinite(
                            loaded_image
                        ).all(), f"Loaded {filename} contains NaN or inf values"

                        # Check data type
                        assert (
                            loaded_image.dtype == np.uint8
                        ), f"Loaded {filename} has wrong dtype: {loaded_image.dtype}"

                except Exception as e:
                    if should_succeed:
                        pytest.fail(f"Failed to load {filename}: {e}")
                    else:
                        # Expected to fail
                        pass

    def test_numpy_to_byte_stream_nan_inf_handling(self):
        """Test that numpy_to_byte_stream handles NaN and inf values properly."""
        from anomaly_match.utils.numpy_to_byte_stream import numpy_array_to_byte_stream

        # Test array with NaN values
        array_with_nan = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32)

        # Should not raise any warnings or errors
        byte_stream = numpy_array_to_byte_stream(array_with_nan, normalize=True)
        assert isinstance(byte_stream, bytes)
        assert len(byte_stream) > 0

        # Test array with inf values
        array_with_inf = np.array([[1.0, 2.0, np.inf], [4.0, 5.0, -np.inf]], dtype=np.float32)

        # Should not raise any warnings or errors
        byte_stream = numpy_array_to_byte_stream(array_with_inf, normalize=True)
        assert isinstance(byte_stream, bytes)
        assert len(byte_stream) > 0

        # Test array with all same values (edge case)
        uniform_array = np.full((3, 3), 5.0, dtype=np.float32)
        byte_stream = numpy_array_to_byte_stream(uniform_array, normalize=True)
        assert isinstance(byte_stream, bytes)
        assert len(byte_stream) > 0

        # Test without normalization
        clean_array = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        byte_stream = numpy_array_to_byte_stream(clean_array, normalize=False)
        assert isinstance(byte_stream, bytes)
        assert len(byte_stream) > 0
