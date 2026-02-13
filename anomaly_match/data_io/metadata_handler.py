#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from loguru import logger


class MetadataHandler:
    """Handler for loading and managing metadata from CSV files.

    This class handles loading, validating, and accessing metadata information
    from CSV files that contain additional information about image files.
    """

    def __init__(self, metadata_file=None, image_filenames=None):
        """Initialize the metadata handler.

        Args:
            metadata_file (str, optional): Path to the metadata CSV file.
            image_filenames (list, optional): List of image filenames to validate against.
        """
        self.metadata_file = metadata_file
        self.image_filenames = image_filenames
        self.metadata_df = None

        # Load metadata if file is provided
        if metadata_file is not None:
            self.load_metadata()

    def _validate_icrs_coordinates(self, ra_values, dec_values, sample_size=1000):
        """Validate that RA/Dec coordinates are in valid ICRS format.

        Args:
            ra_values (pd.Series): Series of RA values
            dec_values (pd.Series): Series of Dec values
            sample_size (int): Number of coordinates to sample for validation if dataset is large

        Returns:
            bool: True if coordinates are valid ICRS, False otherwise

        Raises:
            ValueError: If coordinates are not valid ICRS format
        """

        # Check for non-numeric values
        if not pd.api.types.is_numeric_dtype(ra_values) or not pd.api.types.is_numeric_dtype(
            dec_values
        ):
            raise ValueError(
                "RA and Dec values must be numeric. Please ensure coordinates are in decimal degrees "
                "following the ICRS (International Celestial Reference System) standard. "
                "See: https://docs.astropy.org/en/stable/coordinates/index.html"
            )

        # Remove NaN values for validation
        valid_mask = ~(pd.isna(ra_values) | pd.isna(dec_values))
        valid_ra = ra_values[valid_mask]
        valid_dec = dec_values[valid_mask]

        if len(valid_ra) == 0:
            logger.warning("No valid RA/Dec coordinate pairs found for validation")
            return True

        # Sample coordinates if dataset is large
        if len(valid_ra) > sample_size:
            logger.debug(
                f"Large dataset detected ({len(valid_ra)} coordinates). "
                f"Sampling {sample_size} coordinates for ICRS validation"
            )
            sample_indices = np.random.choice(len(valid_ra), size=sample_size, replace=False)
            valid_ra = valid_ra.iloc[sample_indices]
            valid_dec = valid_dec.iloc[sample_indices]

        try:
            # Validate RA range (0 to 360 degrees)
            if (valid_ra < 0).any() or (valid_ra > 360).any():
                raise ValueError(
                    f"RA values must be in range [0, 360] degrees (ICRS format). "
                    f"Found RA range: [{valid_ra.min():.3f}, {valid_ra.max():.3f}]. "
                    f"Please ensure coordinates follow the ICRS standard. "
                    f"See: https://docs.astropy.org/en/stable/coordinates/index.html"
                )

            # Validate Dec range (-90 to 90 degrees)
            if (valid_dec < -90).any() or (valid_dec > 90).any():
                raise ValueError(
                    f"Dec values must be in range [-90, 90] degrees (ICRS format). "
                    f"Found Dec range: [{valid_dec.min():.3f}, {valid_dec.max():.3f}]. "
                    f"Please ensure coordinates follow the ICRS standard. "
                    f"See: https://docs.astropy.org/en/stable/coordinates/index.html"
                )

            # Try to create SkyCoord objects to validate ICRS compatibility
            SkyCoord(ra=valid_ra.values * u.deg, dec=valid_dec.values * u.deg, frame="icrs")

            # If we get here, coordinates are valid
            logger.debug(f"Successfully validated {len(valid_ra)} coordinates as ICRS-compatible")
            return True

        except Exception as e:
            if "ICRS" in str(e) or "coordinate" in str(e).lower():
                raise ValueError(
                    f"Invalid ICRS coordinates detected: {str(e)}. "
                    f"Please ensure RA/Dec values are in decimal degrees following the ICRS standard. "
                    f"See: https://docs.astropy.org/en/stable/coordinates/index.html"
                )
            else:
                raise e

    def load_metadata(self):
        """Load metadata from CSV file and validate it against the available images."""
        if self.metadata_file is None:
            logger.debug("No metadata file provided, skipping metadata loading")
            return

        logger.debug(f"Loading metadata from {self.metadata_file}")
        if not os.path.exists(self.metadata_file):
            logger.warning(f"Metadata file {self.metadata_file} does not exist")
            return

        try:
            self.metadata_df = pd.read_csv(self.metadata_file)

            # Check for required filename column
            if "filename" not in self.metadata_df.columns:
                logger.error("Metadata file does not contain required 'filename' column")
                self.metadata_df = None
                return

            # Check if the number of entries in metadata_file matches the number of images
            # Only if image_filenames is provided
            if self.image_filenames:
                metadata_files = set(self.metadata_df["filename"])
                all_files = set([os.path.basename(f) for f in self.image_filenames])

                if len(metadata_files) != len(all_files):
                    logger.warning(
                        f"Number of entries in metadata file ({len(metadata_files)}) "
                        f"does not match number of images ({len(all_files)})"
                    )

            # Check for essential columns (optional but recommended)
            for col in ["sourceID", "ra", "dec"]:
                if col not in self.metadata_df.columns:
                    logger.debug(f"Optional column '{col}' not found in metadata file")

            # Validate RA/Dec coordinates if both are present
            if "ra" in self.metadata_df.columns and "dec" in self.metadata_df.columns:
                logger.info("Validating RA/Dec coordinates for ICRS compatibility")
                try:
                    self._validate_icrs_coordinates(self.metadata_df["ra"], self.metadata_df["dec"])
                except ValueError as e:
                    logger.error(f"ICRS coordinate validation failed: {str(e)}")
                    self.metadata_df = None
                    return

            # Set index to filename for easier access
            self.metadata_df.set_index("filename", inplace=True)
            logger.info(f"Successfully loaded metadata for {len(self.metadata_df)} files")

        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self.metadata_df = None

    def get_metadata_for_file(self, filename):
        """Get metadata for a specific file.

        Args:
            filename (str): The filename to get metadata for.

        Returns:
            dict: Metadata for the file, or None if not found.
        """
        if self.metadata_df is None or filename not in self.metadata_df.index:
            return None

        # Return as dictionary
        return self.metadata_df.loc[filename].to_dict()

    def get_all_metadata(self):
        """Get all metadata.

        Returns:
            pd.DataFrame: The full metadata DataFrame, or None if no metadata loaded.
        """
        return self.metadata_df
