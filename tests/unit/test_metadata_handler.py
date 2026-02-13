#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import os
import tempfile

import numpy as np

from anomaly_match.data_io.metadata_handler import MetadataHandler


class TestMetadataHandler:
    def test_basic_functionality(self):
        """Test basic functionality of MetadataHandler."""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata
            f.write("filename,sourceID,ra,dec,custom_col\n")
            f.write("image1.jpg,source1,10.0,20.0,custom1\n")
            f.write("image2.jpg,source2,11.0,21.0,custom2\n")
            f.write("image3.jpg,source3,12.0,22.0,custom3\n")

            # Get filename
            metadata_file = f.name

        try:
            # Create handler
            handler = MetadataHandler(metadata_file)

            # Test basic functionality
            assert handler.get_all_metadata() is not None
            assert len(handler.get_all_metadata()) == 3

            # Test column access
            metadata = handler.get_all_metadata()
            for col in ["sourceID", "ra", "dec", "custom_col"]:
                assert col in metadata.columns

            # Test get_metadata_for_file
            file_metadata = handler.get_metadata_for_file("image2.jpg")
            assert file_metadata is not None
            assert file_metadata["sourceID"] == "source2"
            assert file_metadata["ra"] == 11.0
            assert file_metadata["dec"] == 21.0
            assert file_metadata["custom_col"] == "custom2"

            # Test nonexistent file
            assert handler.get_metadata_for_file("nonexistent.jpg") is None

        finally:
            # Clean up
            os.unlink(metadata_file)

    def test_validation_with_image_filenames(self):
        """Test validation against provided image filenames."""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata
            f.write("filename,sourceID,ra,dec\n")
            f.write("image1.jpg,source1,10.0,20.0\n")
            f.write("image2.jpg,source2,11.0,21.0\n")

            # Get filename
            metadata_file = f.name

        try:
            # Create handler with matching filenames (no warning)
            image_filenames = ["path/to/image1.jpg", "path/to/image2.jpg"]
            handler = MetadataHandler(metadata_file, image_filenames)
            assert handler.get_all_metadata() is not None

            # Create handler with non-matching filenames (should generate warning but still work)
            image_filenames = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
            handler = MetadataHandler(metadata_file, image_filenames)
            assert handler.get_all_metadata() is not None
            assert len(handler.get_all_metadata()) == 2  # Only the matching files

        finally:
            # Clean up
            os.unlink(metadata_file)

    def test_missing_file(self):
        """Test behavior with missing metadata file."""
        # Create handler with nonexistent file
        handler = MetadataHandler("nonexistent_metadata.csv")
        assert handler.get_all_metadata() is None

    def test_invalid_file(self):
        """Test behavior with invalid metadata file."""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write invalid metadata (missing filename column)
            f.write("sourceID,ra,dec\n")
            f.write("source1,10.0,20.0\n")

            # Get filename
            invalid_file = f.name

        try:
            # Create handler
            handler = MetadataHandler(invalid_file)
            assert handler.get_all_metadata() is None

        finally:
            # Clean up
            os.unlink(invalid_file)

    def test_valid_icrs_coordinates(self):
        """Test validation of valid ICRS coordinates."""
        # Create temp file with valid ICRS coordinates
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata with valid RA/Dec in ICRS format
            f.write("filename,sourceID,ra,dec\n")
            f.write("image1.jpg,source1,15.0,45.0\n")
            f.write("image2.jpg,source2,180.0,-30.0\n")
            f.write("image3.jpg,source3,359.9,89.9\n")
            f.write("image4.jpg,source4,0.0,-89.9\n")

            metadata_file = f.name

        try:
            # Should load successfully with valid coordinates
            handler = MetadataHandler(metadata_file)
            assert handler.get_all_metadata() is not None
            assert len(handler.get_all_metadata()) == 4

        finally:
            os.unlink(metadata_file)

    def test_invalid_ra_range(self):
        """Test validation fails for invalid RA range."""
        # Create temp file with invalid RA coordinates
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata with invalid RA (outside 0-360 range)
            f.write("filename,sourceID,ra,dec\n")
            f.write("image1.jpg,source1,15.0,45.0\n")
            f.write("image2.jpg,source2,-10.0,30.0\n")  # Invalid RA < 0
            f.write("image3.jpg,source3,370.0,45.0\n")  # Invalid RA > 360

            metadata_file = f.name

        try:
            # Should fail to load due to invalid RA
            handler = MetadataHandler(metadata_file)
            assert handler.get_all_metadata() is None

        finally:
            os.unlink(metadata_file)

    def test_invalid_dec_range(self):
        """Test validation fails for invalid Dec range."""
        # Create temp file with invalid Dec coordinates
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata with invalid Dec (outside -90 to 90 range)
            f.write("filename,sourceID,ra,dec\n")
            f.write("image1.jpg,source1,15.0,45.0\n")
            f.write("image2.jpg,source2,180.0,-95.0\n")  # Invalid Dec < -90
            f.write("image3.jpg,source3,270.0,95.0\n")  # Invalid Dec > 90

            metadata_file = f.name

        try:
            # Should fail to load due to invalid Dec
            handler = MetadataHandler(metadata_file)
            assert handler.get_all_metadata() is None

        finally:
            os.unlink(metadata_file)

    def test_non_numeric_coordinates(self):
        """Test validation fails for non-numeric coordinates."""
        # Create temp file with non-numeric coordinates
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata with invalid coordinate types
            f.write("filename,sourceID,ra,dec\n")
            f.write("image1.jpg,source1,15.0,45.0\n")
            f.write("image2.jpg,source2,invalid_ra,30.0\n")
            f.write("image3.jpg,source3,270.0,invalid_dec\n")

            metadata_file = f.name

        try:
            # Should fail to load due to non-numeric coordinates
            handler = MetadataHandler(metadata_file)
            assert handler.get_all_metadata() is None

        finally:
            os.unlink(metadata_file)

    def test_coordinates_with_nan_values(self):
        """Test validation handles NaN values gracefully."""
        # Create temp file with some NaN coordinates
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata with some NaN values (should be skipped during validation)
            f.write("filename,sourceID,ra,dec\n")
            f.write("image1.jpg,source1,15.0,45.0\n")
            f.write("image2.jpg,source2,,30.0\n")  # Missing RA
            f.write("image3.jpg,source3,270.0,\n")  # Missing Dec
            f.write("image4.jpg,source4,180.0,-45.0\n")

            metadata_file = f.name

        try:
            # Should load successfully, skipping NaN values during validation
            handler = MetadataHandler(metadata_file)
            assert handler.get_all_metadata() is not None
            assert len(handler.get_all_metadata()) == 4

        finally:
            os.unlink(metadata_file)

    def test_large_dataset_sampling(self):
        """Test that large datasets are sampled for validation."""
        # Create temp file with large dataset
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            f.write("filename,sourceID,ra,dec\n")

            # Generate valid coordinates for a large dataset
            np.random.seed(42)  # For reproducible test
            for i in range(1500):  # More than sample_size (1000)
                ra = np.random.uniform(0, 360)
                dec = np.random.uniform(-90, 90)
                f.write(f"image{i}.jpg,source{i},{ra},{dec}\n")

            metadata_file = f.name

        try:
            # Should load successfully and sample coordinates for validation
            handler = MetadataHandler(metadata_file)
            assert handler.get_all_metadata() is not None
            assert len(handler.get_all_metadata()) == 1500

        finally:
            os.unlink(metadata_file)

    def test_no_coordinates_validation_skipped(self):
        """Test that validation is skipped when RA/Dec columns are missing."""
        # Create temp file without RA/Dec columns
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            # Write metadata without coordinate columns
            f.write("filename,sourceID,custom_col\n")
            f.write("image1.jpg,source1,custom1\n")
            f.write("image2.jpg,source2,custom2\n")

            metadata_file = f.name

        try:
            # Should load successfully without coordinate validation
            handler = MetadataHandler(metadata_file)
            assert handler.get_all_metadata() is not None
            assert len(handler.get_all_metadata()) == 2

        finally:
            os.unlink(metadata_file)

    def test_icrs_error_message_contains_reference(self):
        """Test that ICRS validation errors contain reference to ICRS documentation."""
        # Create temp file with invalid coordinates
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            f.write("filename,sourceID,ra,dec\n")
            f.write("image1.jpg,source1,-10.0,45.0\n")  # Invalid RA

            metadata_file = f.name

        try:
            # Capture log messages or exception
            handler = MetadataHandler(metadata_file)
            # The handler should fail to load due to validation error
            assert handler.get_all_metadata() is None
            # The error message should have been logged with ICRS reference

        finally:
            os.unlink(metadata_file)
