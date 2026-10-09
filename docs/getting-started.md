[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Getting Started

## Installation

```bash
# Clone the repository
git clone https://github.com/esa/AnomalyMatch.git
cd AnomalyMatch

# Create and activate conda environment from the environment.yml file
conda env create -f environment.yml
conda activate am

# Install the package (use -e for development mode)
pip install .
```

## Quick Start

In a Jupyter notebook:

```python
from anomaly_match_ui import start_ui

start_ui()
```

The UI opens on a main menu with **Training** and **Prediction** workflows.
Each starts with a setup screen where you select data folders, normalisation
settings, and model checkpoints interactively.

### Scripted setup (advanced)

For reproducible or automated workflows, pre-configure a session to skip
the interactive setup screens:

```python
import anomaly_match as am
from anomaly_match_ui import start_ui

cfg = am.get_default_cfg()
cfg.data_dir = "path/to/images/"
cfg.label_file = "path/to/labeled_data.csv"
cfg.normalisation.image_size = [128, 128]

session = am.Session(cfg)
start_ui(session)
```

See [`StarterNotebook.ipynb`](https://github.com/esa/AnomalyMatch/blob/main/StarterNotebook.ipynb) for a ready-to-run example.

## Recommended Folder Structure

```
project/
├── labeled_data.csv          # Annotations (filename, label columns)
├── metadata.csv              # RA/Dec coordinates (optional)
└── images/                   # Source images for training and prediction
    ├── image1.png
    ├── image2.fits
    ├── dataset.zarr           # Zarr arrays supported
    └── catalogue.parquet      # Cutana catalogues supported
```

The training and prediction setup screens accept any combination of image
files, Zarr, or Cutana catalogues in a single folder.

### Label File

Minimal `labeled_data.csv`:

```csv
id,label
image1.png,normal
image2.png,anomaly
```

The `id` column is the source identifier (filename for image folders, SourceID for Cutana, for Zarr the name from the store's parquet metadata sidecar, else `<store>__image_NNNNNN`). Extra columns are ignored.

### Metadata File

Optional `metadata.csv`:

```csv
filename,sourceID,ra,dec,custom_col
image1.png,source1,10.5,20.3,custom_value1
image2.png,source2,11.2,21.7,custom_value2
```

The metadata file can include columns for `sourceID`, `ra`, `dec`, and any custom columns. This metadata is automatically merged with the labelled data when saving results. Specify with `cfg.metadata_file = "path/to/metadata.csv"`.

The `ra` and `dec` coordinates must be in degrees in the [ICRS frame](https://en.wikipedia.org/wiki/International_Celestial_Reference_System).

## Session Tracking

AnomalyMatch automatically tracks comprehensive session information including training iterations, model checkpoints, labelled samples, and performance metrics. All session data is saved under `anomaly_match_results/sessions/`:

```
session_name_timestamp/
├── session_metadata.json    # Complete session tracking data
├── labeled_data.csv         # All labelled samples
├── config.toml              # Final configuration
├── session.log              # Session log
├── predictions.db           # Anomaly scores (SQLite)
├── subprocess_logs/         # One log per training / prediction subprocess
├── profiling/               # Per-run prediction performance reports
└── iteration_0/             # One directory per training cycle
    ├── model.safetensors    # Checkpoint for this cycle
    ├── labelled_data.csv    # Labels this cycle trained on
    └── training.log
```

`session_metadata.json` is written by `save_session()`, so it appears once a
session has been saved.

You can view any saved session using:

```python
import anomaly_match as am

am.print_session("/path/to/session/directory")
```

To drive all of this from a script instead of the UI, see
[Headless Mode](user-guide/headless.md).
