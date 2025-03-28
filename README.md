# AnomalyMatch
Semi-supervised anomaly detection with active learning

## Overview
This package uses a FixMatch pipeline built on EfficientNet models and provides a mechanism
for active learning to detect anomalies in images. It also offers a GUI via ipywidgets for labeling and managing the detection process.

## Requirements
Dependencies are listed in the `environment.yml` file. To leverage the full capabilities of 
this package (especially training on large images or predicting over large image datasets), a GPU is strongly recommended.
Use with Jupyter notebooks is recommended (see StarterNotebook.ipynb) since the UI 
relies on ipywidgets.

## Demo
![Placeholder for the app demo GIF](demo.gif)

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
