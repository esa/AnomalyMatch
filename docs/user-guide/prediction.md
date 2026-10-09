[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Prediction

## Overview

After training, AnomalyMatch can run predictions on unlabeled images to identify potential anomalies. The system supports multiple data formats and batch processing.

## Supported File Formats

AnomalyMatch supports the following image file formats:

- **Standard formats**: JPEG (`*.jpg`, `*.jpeg`), PNG (`*.png`), TIFF (`*.tif`, `*.tiff`)
- **Astronomical formats**: FITS (`*.fits`)
- **Container formats**: Zarr (`*.zarr`)

If multiple filetypes are present in a directory, all will be loaded.

## Running Predictions

In the UI, prediction is the **Prediction** workflow on the main menu, and it
also runs automatically after each training cycle. From a script, it is
`session.evaluate_all_images()`, which blocks until the run finishes — see
[Headless Mode](headless.md) for a complete script and for reading the scores
back.

All remaining sources are scored: those listed in the label file are skipped,
and so are those already in the database, so an interrupted run resumes.

Scores are written to `predictions.db` (SQLite) in the session directory, not
to a CSV.

`Session` dispatches to the right subprocess by inspecting the search
directory; the scripts are not normally invoked by hand:

| Source | Script |
|--------|--------|
| Image files | `prediction_process.py` |
| Zarr | `prediction_process_zarr.py` |
| Cutana catalogue | `prediction_process_cutana.py` |

## Zarr File Support

Zarr files are useful for large collections of images, distributed workflows, and efficient chunked access.

### Requirements

Zarr files must contain:

- An `images` dataset with shape `(N, height, width, channels)` where N is the number of images
- Optional metadata file (`.parquet` format) containing filenames

### Creating Zarr Files

You can create compatible Zarr files using [images_to_zarr](https://github.com/gomezzz/images_to_zarr/):

```bash
pip install images_to_zarr
```

```python
import images_to_zarr as i2z

i2z.convert(
    output_dir="path/to/output.zarr",
    folders="path/to/images",
    resize=(150, 150),
    chunk_shape=(1000, 4, 150, 150),  # 1000 images per chunk
)
```

For best performance, use chunks of 1000 images (`chunk_shape=(1000, channels, height, width)`).

### Zarr Configuration

AnomalyMatch automatically detects and processes Zarr files in your prediction directory:

```python
cfg.prediction_search_dir = "/path/to/directory/containing/zarr/files"
```

### Multiple Zarr Files

Two folder structures are supported:

**Option 1: Direct Zarr files**

```
prediction_search_dir/
├── dataset_part1.zarr/
│   └── images/           # Zarr array with shape (N, H, W, C)
├── dataset_part1_metadata.parquet
├── dataset_part2.zarr/
│   └── images/
└── dataset_part2_metadata.parquet
```

**Option 2: Batch folders with images.zarr subdirectory**

```
prediction_search_dir/
├── batch_001/
│   ├── images.zarr/
│   │   └── images/
│   └── images_metadata.parquet
├── batch_002/
│   ├── images.zarr/
│   │   └── images/
│   └── images_metadata.parquet
```

**Metadata requirements:**

- Parquet files should contain a `filename`, `original_filename`, or `source_id` column
- For direct zarr files: `<zarr_name>_metadata.parquet` in the same directory
- For batch folders: `images_metadata.parquet` in the batch folder

**Filename handling:** To prevent collisions across zarr files, filenames are automatically prefixed with the zarr/batch folder name (e.g., `batch_001__image_000042`).

## FITS File Handling

- By default, the first extension (index 0) is used when loading FITS files
- Specify extensions using `cfg.normalisation.fits_extension`:
    - Integer values (e.g., `0`, `1`, `2`) to access by index
    - String values (e.g., `"PRIMARY"`, `"SCIENCE"`) to access by name
    - List of integers or strings (e.g., `[0, 1, 2]`) to combine multiple extensions into a single image
- When combining multiple extensions:
    - 2D extensions are combined as channels
    - All extensions must have identical dimensions

## Multispectral Support

AnomalyMatch supports training and prediction on images with arbitrary channel counts (1 to N channels), not just RGB.

### Configuration

```python
import anomaly_match as am

cfg = am.get_default_cfg()

# For FITS files with multiple extensions as channels
cfg.normalisation.fits_extension = ["VIS", "NIR-H", "NIR-J", "NIR-Y"]

# Asinh normalisation parameters (one per channel)
cfg.normalisation.norm_asinh_scale = [0.7, 0.7, 0.7, 0.7]
cfg.normalisation.norm_asinh_clip = [99.8, 99.8, 99.8, 99.8]
```

`n_output_channels` and `channel_combination` are automatically inferred from `fits_extension`.

### Combining Extensions into Fewer Channels

Use `channel_combination` to define a linear mapping from FITS extensions to output channels:

```python
import numpy as np

# 4 FITS extensions → 3 RGB output channels
cfg.normalisation.fits_extension = ["VIS", "NIR-H", "NIR-J", "NIR-Y"]
cfg.normalisation.channel_combination = np.array(
    [
        [1, 0, 0, 0],  # R = VIS
        [0, 0.5, 0.5, 0],  # G = average of NIR-H and NIR-J
        [0, 0, 0, 1],  # B = NIR-Y
    ]
)
```

Each row defines one output channel as a weighted sum of the input extensions. When `channel_combination` is `None` (default), an identity matrix is created automatically.

### Blanking an Output Channel

Give an output channel a row of zeros to leave it empty:

```python
cfg.normalisation.channel_combination = np.array(
    [
        [1, 0, 0, 0],  # R = VIS
        [0, 0, 0, 0],  # G = blank
        [0, 0, 0, 1],  # B = NIR-Y
    ]
)
```

The blanked channel decodes as all zeros. Use it to drop a band from a model you
do not want to retrain at a different channel count — the model keeps the same
number of input channels, so the checkpoint stays compatible.

!!! note "Blanking does not restretch the other channels"
    A blanked channel takes no part in normalisation, so the remaining channels
    decode exactly as they would if the blank row were not there at all.

!!! warning "Flux conversion reads only the primary HDU"
    `apply_flux_conversion` converts the primary HDU alone, so it cannot be
    combined with a multi-extension `channel_combination`. Set it to `False`
    when mapping several extensions to output channels.

### Supported Formats for N-channel Data

- **NumPy arrays (`.npy`)**: Shape `(H, W, C)` where C is the number of channels
- **FITS files**: Multiple extensions combined as channels
- **Zarr**: Arrays with shape `(N, H, W, C)`

### Model Architecture

When using pretrained models (default), AnomalyMatch automatically adapts the first convolutional layer for N-channel input: the first 3 channels use the pretrained RGB weights, additional channels are initialized with averaged RGB weights.

## Cutana Streaming Integration

AnomalyMatch supports streaming predictions via [Cutana](https://github.com/esa/cutana), which enables on-the-fly cutout extraction from FITS tiles.

**How to use:**

1. Prepare a Cutana-compatible source catalogue (CSV or Parquet) with columns for coordinates and FITS file paths
2. Set `cfg.prediction_search_dir` to a folder containing your catalogue files
3. AnomalyMatch will automatically detect the catalogues and stream cutouts via Cutana

**FITS extension configuration:** Ensure `cfg.normalisation.fits_extension` matches the FITS extensions referenced in your catalogue. Naming bands this way hands `channel_combination` to fitsbolt, which rejects negative weights — see [Weights in `channel_combination`](configuration.md#weights-in-channel_combination).

!!! warning "Use the same catalogue for training and prediction"

    Channels are positional: `fits_file_paths` lists one mosaic path per band, band names are parsed from the filenames (read once, from the first row of the first catalogue), and `channel_combination` maps bands to model inputs by position — never by name.

    | Change to `fits_file_paths` | Detected? |
    | --- | --- |
    | Reordered paths | **No** — the count still matches, so prediction runs on permuted channels |
    | Swapped mosaic (other filter or reduction) | **No** — only the pixel content differs |
    | Added or removed paths | Yes — band-count validation raises |

    If mosaics move, rewrite only the path prefix, then delete the labeled cache (`*_cache/` next to the label CSV): it auto-invalidates on band-count changes only.

**Normalisation consistency:** AnomalyMatch automatically passes the same fitsbolt normalisation configuration to Cutana, so training and streaming prediction produce identically normalised images.

**Flux conversion:** Set `cfg.normalisation.apply_flux_conversion = True` when working with Euclid data to convert pixel values to flux density in Jansky using the AB zeropoint (`MAGZERO`) from FITS headers.

For more details, see the [Cutana documentation](https://github.com/esa/cutana).
