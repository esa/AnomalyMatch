[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file ‘LICENCE.txt’.)
[![License: ESA permissive](https://img.shields.io/badge/ESA%20Public%20License-Permissive-blue.svg)](https://github.com/esa/AnomalyMatch/blob/main/LICENSE.txt)


# AnomalyMatch
Semi-supervised anomaly detection with active learning

## Overview
This package uses a FixMatch pipeline built on EfficientNet models and provides a mechanism
for active learning to detect anomalies in images. It also offers a GUI via ipywidgets for labeling and managing the detection process.

![Placeholder for the app demo GIF](demo.gif)

## Requirements
Dependencies are listed in the `environment.yml` file. To leverage the full capabilities of 
this package (especially training on large images or predicting over large image datasets), a GPU is strongly recommended.
Use with Jupyter notebooks is recommended (see StarterNotebook.ipynb) since the UI 
relies on ipywidgets.

## Installation

### For Users

```bash
# Clone the repository
git clone https://github.com/ESA/AnomalyMatch.git
cd AnomalyMatch

# Create and activate conda environment from the environment.yml file
conda env create -f environment.yml
conda activate am

# Install the package
pip install .
```

### For Developers

```bash
# Clone the repository
git clone https://github.com/ESA/AnomalyMatch.git
cd AnomalyMatch

# Create and activate conda environment
conda env create -f environment.yml
conda activate am

# Install in development mode
pip install -e .
```

After installation, you can start using AnomalyMatch in your Jupyter notebooks. See `StarterNotebook.ipynb` for an example.

## Recommended Folder Structure
- project/
  - labeled_data.csv | containing annotations of labeled examples
  - training_images/ | the cfg.data_dir
    - image1.jpeg
    - image2.jpeg
  - data_to_predict/ | the cfg.search_dir
    - unlabeled_file_part1.hdf5
    - unlabeled_file_part2.hdf5

Example of a minimal labeled_data.csv:
```
filename,label,your_custom_source_id
image1.jpeg,normal,123456
image2.jpeg,anomaly,424242
```
Here, the additional columns (like "your_custom_source_id") can store your own identifiers or data.

## Supported File Formats

AnomalyMatch supports the following image file formats:
- **Standard formats**: JPEG (*.jpg, *.jpeg), PNG (*.png) (More to follow soon)

Note: If multiple filetypes are present, all will be loaded.

## Key Config Parameters
- `save_dir`: Path to store the trained model output.
- `data_dir`: Location of the training data (*.jpeg, *.jpg, *.png, *.tif, or *.tiff).
- `label_file`: CSV mapping annotated images to labels.
- `search_dir`: Path where data to be predicted is stored.
- `logLevel`: Controls verbosity of training/session logs.
- `test_ratio`: Proportion of data used for evaluation (0.0 disables test evaluation, > 0 shows AUROC/AUPRC curves).
- `size`: Dimensions to which images are resized (below 96x96 is not recommended).
- `N_to_load`: Number of unlabeled images loaded into the training dataset at once.
- `output_dir`: Folder for storing results (e.g., labeled_data.csv or final logs).

## Advanced CFG Parameters

The following advanced parameters can be configured:

### FixMatch Parameters
- `ema_m`: Exponential moving average momentum (default: 0.99)
- `hard_label`: Whether to use hard labels for unlabeled data (default: True)
- `temperature`: Temperature for softmax in semi-supervised learning (default: 0.5)
- `ulb_loss_ratio`: Weight of the unlabeled loss (default: 1.0)
- `p_cutoff`: Confidence threshold for pseudo-labeling (default: 0.95)
- `uratio`: Ratio of unlabeled to labeled data in each batch (default: 5)

### Training Parameters
- `num_workers`: Number of parallel workers for data loading (default: 4)
- `batch_size`: Training batch size (default: 16)
- `lr`: Learning rate (default: 0.0075)
- `weight_decay`: L2 regularization parameter (default: 7.5e-4)
- `opt`: Optimizer type (default: "SGD")
- `momentum`: SGD momentum (default: 0.9)
- `bn_momentum`: Batch normalization momentum (default: 1.0 - ema_m)
- `num_train_iter`: Number of training iterations (default: 200)
- `eval_batch_size`: Batch size for evaluation (default: 500)
- `num_eval_iter`: Evaluation frequency, -1 means no evaluation (default: -1)
- `pretrained`: Whether to use pretrained backbone (default: True)
- `net`: Backbone network architecture (default: "efficientnet-lite0")

### Additional Parameters
- `prediction_file_type`: Type of files for prediction (default: "hdf5")
