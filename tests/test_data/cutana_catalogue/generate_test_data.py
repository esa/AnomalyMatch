#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Generate self-contained cutana catalogue test data.

Creates a small Euclid-compliant FITS mosaic tile and a catalogue CSV with
25 sources whose coordinates fall within the tile's field of view. The
catalogue uses paths relative to the repository root so tests can run
without any external dependencies (no sibling Cutana checkout required).

Usage:
    python tests/test_data/cutana_catalogue/generate_test_data.py
"""

import csv
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

IMG_SIZE = 512
RA_CENTER = 150.12
DEC_CENTER = 2.35
TILE_ID = "102018211"
PIXEL_SCALE = 0.1 / 3600.0  # 0.1 arcsec/pixel in degrees
NUM_SOURCES = 25
SEED = 42

# Euclid-compliant filename (fixed timestamp for reproducibility)
FITS_FILENAME = f"EUC_MER_BGSUB-MOSAIC-VIS_TILE{TILE_ID}-ACBD03_20260320T072326.476Z_00.00.fits"

# The 25 MockSource IDs must include all 10 referenced by labeled_data.csv.
# Object IDs follow the convention: tile_id * 1_000_000 + sequence_number.
OBJECT_IDS = [
    # First 10: referenced by labeled_data.csv
    102018211000366,
    102018211000155,
    102018211000383,
    102018211000428,
    102018211000077,
    102018211000012,
    102018211000021,
    102018211000357,
    102018211000044,
    102018211000418,
    # Remaining 15: unlabeled pool
    102018211000419,
    102018211000145,
    102018211000257,
    102018211000240,
    102018211000146,
    102018211000258,
    102018211000083,
    102018211000063,
    102018211000001,
    102018211000047,
    102018211000456,
    102018211000109,
    102018211000173,
    102018211000283,
    102018211000274,
]


def generate_fits_tile() -> Path:
    """Create a 512x512 Euclid-compliant FITS mosaic tile.

    Returns:
        Path to the generated FITS file.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [RA_CENTER, DEC_CENTER]
    wcs.wcs.crpix = [IMG_SIZE / 2, IMG_SIZE / 2]
    wcs.wcs.cd = [[-PIXEL_SCALE, 0], [0, PIXEL_SCALE]]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000.0

    rng = np.random.RandomState(SEED)
    img_data = rng.normal(0, 0.005, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    primary_hdu = fits.PrimaryHDU(img_data)
    header = primary_hdu.header
    header.update(wcs.to_header())
    header["TELESCOP"] = "EUCLID"
    header["INSTRUME"] = "VIS"
    header["TILEID"] = TILE_ID
    header["BUNIT"] = "electron/s"
    header["DATATYPE"] = "BGSUB-MOSAIC"
    header["EXPTIME"] = 565.0
    header["GAIN"] = 3.1
    header["READNOIS"] = 4.2
    header["MAGZERO"] = 24.6

    fits_path = SCRIPT_DIR / FITS_FILENAME
    primary_hdu.writeto(fits_path, overwrite=True)
    print(f"  Created {fits_path.name} ({fits_path.stat().st_size / 1024:.0f} KB)")
    return fits_path


def generate_catalogue(fits_path: Path) -> Path:
    """Create test_catalogue.csv with 25 sources inside the FITS tile's FoV.

    Args:
        fits_path: Path to the FITS mosaic tile.

    Returns:
        Path to the generated CSV file.
    """
    rng = np.random.RandomState(SEED + 1)

    field_of_view_deg = IMG_SIZE * PIXEL_SCALE
    fov_margin = field_of_view_deg * 0.40  # stay well within tile

    ra_values = RA_CENTER + rng.uniform(-fov_margin, fov_margin, NUM_SOURCES)
    dec_values = DEC_CENTER + rng.uniform(-fov_margin, fov_margin, NUM_SOURCES)
    diameters = rng.randint(100, 260, NUM_SOURCES)

    # Use relative path from repo root for portability
    relative_path = os.path.relpath(fits_path, REPO_ROOT).replace("\\", "/")

    csv_path = SCRIPT_DIR / "test_catalogue.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"])
        for i in range(NUM_SOURCES):
            writer.writerow(
                [
                    f"MockSource_{OBJECT_IDS[i]}",
                    ra_values[i],
                    dec_values[i],
                    diameters[i],
                    f"['{relative_path}']",
                ]
            )

    print(f"  Created {csv_path.name} ({NUM_SOURCES} sources)")
    return csv_path


def main():
    """Generate cutana catalogue test data."""
    print("Generating cutana catalogue test data...")
    fits_path = generate_fits_tile()
    generate_catalogue(fits_path)
    print("Done!")


if __name__ == "__main__":
    main()
