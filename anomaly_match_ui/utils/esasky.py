#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""ESASky URL generation from RA/Dec coordinates."""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

if TYPE_CHECKING:
    from dotmap import DotMap

_ESASKY_URL = "https://sky.esa.int/esasky/?target={ra}%20{dec}"


def get_esasky_url(ra: float, dec: float) -> str:
    """Generate an ESASky viewer URL for the given coordinates.

    Args:
        ra: Right Ascension in degrees.
        dec: Declination in degrees.

    Returns:
        ESASky URL string.
    """
    return _ESASKY_URL.format(ra=ra, dec=dec)


def get_coordinates(filename: str, cfg: DotMap) -> tuple[float, float] | None:
    """Try to get RA/Dec from metadata CSV, Cutana catalogue, or FITS WCS headers.

    Args:
        filename: The source filename to look up.
        cfg: Configuration object with metadata and search paths.

    Returns:
        ``(ra, dec)`` tuple if coordinates are available, ``None`` otherwise.
    """
    # 1. Check metadata CSV (cfg.metadata_file)
    coords = _from_metadata_csv(filename, cfg)
    if coords is not None:
        return coords

    # 2. Check Cutana catalogue (parquet files in search dir)
    coords = _from_cutana_catalogue(filename, cfg)
    if coords is not None:
        return coords

    # 3. Try FITS WCS headers
    coords = _from_fits_wcs(filename, cfg)
    if coords is not None:
        return coords

    return None


def _from_metadata_csv(filename: str, cfg: DotMap) -> tuple[float, float] | None:
    """Look up coordinates in the metadata CSV file.

    Returns:
        ``(ra, dec)`` tuple, or ``None`` if not found.
    """
    metadata_file = cfg.metadata_file
    if not metadata_file or not os.path.isfile(metadata_file):
        return None

    try:
        df = pd.read_csv(metadata_file)
        basename = os.path.basename(filename)

        # Try matching on filename column
        for col in ("filename", "original_filename", "source_id", "name"):
            if col not in df.columns:
                continue
            match = df[df[col].astype(str).str.contains(basename, regex=False)]
            if len(match) > 0:
                row = match.iloc[0]
                ra_col = _find_column(df, ("ra", "RA", "ra_deg"))
                dec_col = _find_column(df, ("dec", "Dec", "DEC", "dec_deg"))
                if ra_col and dec_col:
                    return float(row[ra_col]), float(row[dec_col])
    except Exception as exc:
        logger.warning("Metadata CSV lookup failed for {}: {}", filename, exc)

    return None


def _from_cutana_catalogue(filename: str, cfg: DotMap) -> tuple[float, float] | None:
    """Look up coordinates in Cutana parquet catalogues.

    Returns:
        ``(ra, dec)`` tuple, or ``None`` if not found.
    """
    search_dir = cfg.prediction_search_dir
    if not search_dir or not os.path.isdir(search_dir):
        return None

    try:
        parquet_files = glob.glob(os.path.join(search_dir, "*.parquet"))
        basename = os.path.basename(filename)

        for pf in parquet_files[:5]:  # Check at most 5 files
            try:
                df = pd.read_parquet(pf)
                for col in ("source_id", "filename"):
                    if col not in df.columns:
                        continue
                    match = df[df[col].astype(str).str.contains(basename, regex=False)]
                    if len(match) > 0:
                        row = match.iloc[0]
                        ra_col = _find_column(df, ("ra", "RA", "ra_deg"))
                        dec_col = _find_column(df, ("dec", "Dec", "DEC", "dec_deg"))
                        if ra_col and dec_col:
                            return float(row[ra_col]), float(row[dec_col])
            except Exception as exc:
                logger.debug("Failed to read parquet {}: {}", pf, exc)
                continue
    except Exception as exc:
        logger.warning("Cutana catalogue lookup failed for {}: {}", filename, exc)

    return None


def _from_fits_wcs(filename: str, cfg: DotMap) -> tuple[float, float] | None:
    """Try to read RA/Dec from FITS WCS headers.

    Returns:
        ``(ra, dec)`` tuple, or ``None`` if not found.
    """
    # Resolve full path
    filepath = filename
    if not os.path.isfile(filepath):
        search_dir = cfg.prediction_search_dir
        if search_dir:
            filepath = os.path.join(search_dir, filename)
            if not os.path.isfile(filepath):
                filepath = os.path.join(search_dir, os.path.basename(filename))

    if not os.path.isfile(filepath) or not filepath.lower().endswith(".fits"):
        return None

    try:
        with fits.open(filepath) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.header.get("NAXIS", 0) >= 2:
                    wcs = WCS(hdu.header)
                    # Get centre pixel coordinates
                    ny, nx = hdu.data.shape[:2]
                    ra, dec = wcs.pixel_to_world_values(nx / 2, ny / 2)
                    return float(ra), float(dec)
    except Exception as exc:
        logger.warning("FITS WCS lookup failed for {}: {}", filename, exc)

    return None


def _find_column(df: object, candidates: tuple[str, ...]) -> str | None:
    """Find the first matching column name from candidates.

    Returns:
        The first matching column name, or ``None``.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None
