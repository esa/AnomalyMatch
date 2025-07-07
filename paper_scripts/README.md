[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
This folder contains the code for reproducing the paper results.

## Overview of Scripts

- **create_results.py**: Main script for running all paper experiments. Configure which experiments to run by setting the flags at the top of the file.
- **paper_benchmark.py**: Core benchmarking script that runs AnomalyMatch with different configurations.
- **paper_plots.py**: Visualization utilities for creating publication-quality plots.
- **paper_utils.py**: Helper functions for data loading, metrics calculation, and other utilities.
- **dataset_plot.py**: Creates visualizations of sample images from GalaxyMNIST and MiniImageNet datasets.
- **get_example_images.py**: Generates examples of weakly and strongly augmented images.
- **prepare_datasets.py**: Downloads and processes the GalaxyMNIST and MiniImageNet datasets.
- **results_analysis.py**: Analyzes benchmark results and generates LaTeX tables.
- **test_plots.py**: Test script for visualizing plot improvements using mock data.

## Reproducing Paper Results

In addition to AnomalyMatch requirements, you will also need to `pip install galaxy-datasets` (**watch out: this may interfere with your pytorch version**) for GalaxyMNIST dataset handling as well as download the MiniImageNet dataset and galaxyzoo dataset manually from the following links:
- [MiniImageNet dataset](https://huggingface.co/datasets/timm/mini-imagenet)
- [GalaxyZoo Kaggle dataset](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data)


Additionally, seaborn has to be installed for plotting:
```bash
pip install seaborn
```

To reproduce all results from the paper:

1. **Prepare the datasets**:
   ```bash
   python prepare_datasets.py --dataset all --output_dir datasets
   ```
   
   This will create a directory structure like:
   ```
   datasets/
   ├── galaxymnist/
   │   ├── images/
   │   ├── labels_galaxymnist.csv
   │   └── galaxymnist_224.hdf5
   ├── miniimagenet/
   │   ├── images/
   │   ├── labels_miniimagenet.csv
   │   └── miniimagenet_224.hdf5
   └── galaxyzoo/
       ├── labels_galaxyzoo.csv
       └── galaxyzoo_images.h5
   ```
You will now need to copy the input image files from the galaxyzoo dataset manually into galaxyzoo/ (where the labels_galaxyzoo.csv is located).
2. **Configure experiments** in `create_results.py` by setting the appropriate flags:
   ```python
   # Toggle which experiment sets to run
   RUN_MINIIMAGENET = True       # MiniImageNet experiments
   RUN_GALAXYMNIST = True        # GalaxyMNIST experiments
   RUN_TRAINING_ITERATIONS_STUDY = True  # Different training iterations
   RUN_ACTIVE_LEARNING_ABLATION = True   # With/without active learning
   RUN_N_SAMPLES_ABLATION = True  # Ablation study with varying sample sizes
   ```

3. **Run the main script**:
   ```bash
   python create_results.py --input-dir datasets
   ```

   You can also specify a custom output directory:
   ```bash
   python create_results.py --input-dir datasets --output-dir my_results
   ```

This will:
- Run all benchmark experiments with the configured settings
- Generate performance metrics (AUROC, AUPRC, etc.)
- Create visualization plots
- Save results to the output directory

## Command Line Options

The `create_results.py` script supports several command line options:

- `--input-dir`: Specify where the prepared datasets are located (default: "datasets")
- `--output-dir`: Specify where to save the results (default: timestamped directory)
- `--seed`: Set the random seed for reproducibility (default: 42)
- `--miniimagenet`, `--galaxymnist`, `--galaxyzoo`: Run only specific dataset experiments
- `--training-study`, `--active-learning`: Run only specific ablation studies
- `--all`: Run all experiments

Example usage:
```bash
# Run only GalaxyMNIST experiments with custom directories
python create_results.py --galaxymnist --input-dir /path/to/datasets --output-dir /path/to/results

# Run all experiments with a different seed
python create_results.py --all --seed 123
```

## Additional Visualizations

- Generate dataset visualizations: `python dataset_plot.py`
- Generate augmentation examples: `python get_example_images.py`
- Analyze and create tables from results: `python results_analysis.py`

## Further External Data
- to make comparison plots to Astronomaly the file `AstronomalyFigure5a.csv` is used:
    - The data in this file was extracted from Figure 5a of https://arxiv.org/abs/2010.11202 using the WebPlotDigitizer from https://web.eecs.utk.edu/~dcostine/personal/PowerDeviceLib/DigiTest/index.html


## Key Configuration Parameters

In `create_results.py`, you can adjust these key parameters:

```python
# Default parameters
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TRAINING_RUNS = 3
DEFAULT_TRAIN_ITERATIONS = 100
DEFAULT_N_MISLABELED = 20
DEFAULT_INPUT_DIR = "datasets"  # Where datasets are located
# Dataset-specific sample sizes
MINIIMAGENET_N_SAMPLES = 500
GALAXYMNIST_N_SAMPLES = 40
```

## Dataset Organization

The scripts expect datasets to be organized as follows after running `prepare_datasets.py`:

```
datasets/  (or your custom input directory)
├── galaxymnist/
│   ├── images/          # Individual image files
│   ├── labels_galaxymnist.csv    # Labels and metadata
│   └── galaxymnist_224.hdf5     # HDF5 file for fast loading
├── miniimagenet/
│   ├── images/          # Individual image files  
│   ├── labels_miniimagenet.csv  # Labels and metadata
│   └── miniimagenet_224.hdf5    # HDF5 file for fast loading
└── galaxyzoo/
    ├── images/          # Individual image files
    ├── labels_galaxyzoo.csv     # Labels and metadata
    └── galaxyzoo_images.h5      # HDF5 file for fast loading
```

Results are saved to a timestamped directory inside `benchmark_results/` by default.

