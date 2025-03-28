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

To reproduce all results from the paper:

1. **Prepare the datasets**:
   ```
   python prepare_datasets.py --dataset all
   ```

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
   ```
   python create_results.py
   ```

This will:
- Run all benchmark experiments with the configured settings
- Generate performance metrics (AUROC, AUPRC, etc.)
- Create visualization plots
- Save results to the output directory

## Additional Visualizations

- Generate dataset visualizations: `python dataset_plot.py`
- Generate augmentation examples: `python get_example_images.py`
- Analyze and create tables from results: `python results_analysis.py`

## Key Configuration Parameters

In `create_results.py`, you can adjust these key parameters:

```python
# Default parameters
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TRAINING_RUNS = 3
DEFAULT_TRAIN_ITERATIONS = 100
DEFAULT_N_MISLABELED = 20
# Dataset-specific sample sizes
MINIIMAGENET_N_SAMPLES = 500
GALAXYMNIST_N_SAMPLES = 40
```

Results are saved to a timestamped directory inside `benchmark_results/` by default.

