[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Configuration

## Overview

AnomalyMatch uses `DotMap` for dot-notation configuration access. The default configuration is defined in [`get_default_cfg`][anomaly_match.utils.get_default_cfg].

## Loading Configuration

```python
import anomaly_match as am

cfg = am.get_default_cfg()
```

!!! note "The bundled-test-data defaults differ"

    Until you set `cfg.data_dir`, it points at the test data shipped with the
    repository, and `get_default_cfg` then overrides `num_train_iter` to 32 and
    `image_size` to `[64, 64]` so `start_ui()` works out of the box. The
    defaults in the tables below are what you get once `data_dir` is your own.

## Paths

| Key | Description |
|-----|-------------|
| `cfg.save_dir` | Path to store the trained model output |
| `cfg.data_dir` | Location of data to search, should be a folder of `*.zarr`, images or Cutana-compatible catalogues (`*.csv , *.parquet`) |
| `cfg.label_file` | CSV mapping annotated images to labels |
| `cfg.metadata_file` | Optional CSV containing metadata for images (automatically merged with labelled data) |
| `cfg.prediction_search_dir` | Path where data to be predicted is stored |

## Training Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `cfg.batch_size` | 16 | Training batch size |
| `cfg.num_train_iter` | 200 | Number of training iterations |
| `cfg.lr` | 0.0075 | Learning rate |
| `cfg.weight_decay` | 7.5e-4 | L2 regularization parameter |
| `cfg.opt` | `"SGD"` | Optimizer type |
| `cfg.momentum` | 0.9 | SGD momentum |
| `cfg.bn_momentum` | `1.0 - ema_m` | Batch normalization momentum |
| `cfg.eval_batch_size` | 500 | Batch size for evaluation |
| `cfg.num_eval_iter` | -1 | Evaluation frequency (-1 = no evaluation) |
| `cfg.pretrained` | `True` | Use pretrained backbone |
| `cfg.net` | `"efficientnet-lite0"` | Backbone network architecture |
| `cfg.N_to_load` | 1000 | Number of unlabeled images loaded into the training dataset at once |
| `cfg.num_workers` | 4 | Parallel workers for data loading |
| `cfg.seed` | 42 | Seed for the unlabelled-pool sample and model init |
| `cfg.gpu` | 0 | GPU index to train and predict on |
| `cfg.test_ratio` | 0.0 | Proportion for evaluation (> 0 shows AUROC/AUPRC curves) |
| `cfg.log_level` | `"INFO"` | Controls verbosity of training/session logs |

## FixMatch Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `cfg.ema_m` | 0.99 | Exponential moving average momentum |
| `cfg.hard_label` | `True` | Use hard labels for unlabelled data |
| `cfg.temperature` | 0.5 | Temperature for softmax |
| `cfg.ulb_loss_ratio` | 1.0 | Weight of the unlabeled loss |
| `cfg.p_cutoff` | 0.95 | Confidence threshold for pseudo-labeling |
| `cfg.uratio` | 5 | Ratio of unlabeled to labeled data in each batch |

## Prediction Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `cfg.N_batch_prediction` | `None` | Batch size for prediction; auto-estimated from GPU memory when `None` |
| `cfg.prediction_search_dir` | `None` | Directory containing data to predict |
| `cfg.prediction_db_dir` | `None` | Pin `predictions.db` to a directory; by default it relocates to local scratch when the session directory is networked |
| `cfg.compile_model` | `False` | `torch.compile` the eval model; worth enabling only for backbones larger than `efficientnet-lite0` |

## Normalisation

Normalisation can be selected in the UI via a dropdown, or set in code:

```python
cfg.normalisation.normalisation_method = am.NormalisationMethod.ZSCALE
```

!!! note "The training setup screen remembers your last choice"

    Starting a training run records that screen's normalisation settings in
    `~/.config/anomalymatch/ui_state.json`, and they become the defaults the
    next time it opens — taking precedence over the values set here, so your
    last run's settings survive a kernel restart. Delete that file (or its
    `normalisation_settings` entry) to go back to the values from your
    notebook. The channel-combination matrix is only restored when the
    selected source exposes the same input channels it was recorded against.

Available methods:

| Method | Description |
|--------|-------------|
| `CONVERSION_ONLY` | No normalisation |
| `LOG` | [Logarithmic normalisation](https://docs.astropy.org/en/stable/api/astropy.visualization.LogStretch.html) |
| `ZSCALE` | Linear normalisation based on [zscale](https://docs.astropy.org/en/stable/api/astropy.visualization.ZScaleInterval.html) min and max |
| `ASINH` | [Asinh normalisation](https://docs.astropy.org/en/stable/api/astropy.visualization.AsinhStretch.html) with configurable scale and percentile clipping |

### Normalisation Config

| Key | Default | Description |
|-----|---------|-------------|
| `cfg.normalisation.image_size` | `[150, 150]` | Target image size (below 96x96 not recommended) |
| `cfg.normalisation.n_output_channels` | 3 | Number of output channels (auto-inferred from `fits_extension`) |
| `cfg.normalisation.output_dtype` | `np.uint8` | Output data type |
| `cfg.normalisation.normalisation_method` | `CONVERSION_ONLY` | Stretch applied while loading; also selectable in the UI dropdown |
| `cfg.normalisation.fits_extension` | `None` | FITS extension(s) to load: an index, a name, or a list of either — see [Prediction](prediction.md#fits-file-handling) |
| `cfg.normalisation.channel_combination` | `None` | `(n_output_channels, n_extensions)` matrix mixing extensions into channels; identity when `None`. An all-zero row blanks that output channel — see [Blanking an output channel](prediction.md#blanking-an-output-channel) |
| `cfg.normalisation.interpolation_order` | 1 (bi-linear) | 0-5, matching [skimage resize orders](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp) |
| `cfg.normalisation.norm_asinh_scale` | `[0.7, …]` | Per-channel asinh scale (`ASINH` only) |
| `cfg.normalisation.norm_asinh_clip` | `[99.8, …]` | Per-channel percentile clip (`ASINH` only) |
| `cfg.normalisation.apply_flux_conversion` | `True` | Convert pixel values to flux density in Jansky using the AB zeropoint (`MAGZERO`) from FITS headers |

For the full stretch semantics see the
[normalisation notes](https://github.com/esa/AnomalyMatch/blob/main/anomaly_match/image_processing/Normalisationreadme.md)
in the repository.

#### Weights in `channel_combination`

Each row of `channel_combination` is one output channel and each column one input band.
For image folders and Zarr stores the columns are the source's own channels, so a
`[[1/3, 1/3, 1/3]]` matrix turns RGB into a single greyscale channel. The training
preview and the scoring gallery show the result, decoded with your normalisation
settings rather than as raw pixels.
An all-zero row blanks that output channel — see
[Blanking an Output Channel](prediction.md#blanking-an-output-channel).

Keep row weights non-negative and summing to at most 1. A row summing to more than 1
is rescaled for you, with a warning — otherwise bright pixels would exceed the output
range and be silently clipped.

!!! warning "Negative weights need `fits_extension` unset"

    With `fits_extension` unset (Cutana catalogues, Zarr arrays, image folders),
    AnomalyMatch applies the matrix itself: negative weights are allowed but warned
    about, because the combined value can fall below 0 and clip to 0 on unsigned
    output. With `fits_extension` set — including Cutana catalogues that name their
    bands — fitsbolt applies the matrix and rejects negative weights, so validation
    fails with `channel_combination must not have negative values`.

### Normalisation Consistency

Normalisation settings are saved in the model checkpoint during training. During prediction, they are loaded automatically — there is no need to re-specify them. Both training and Cutana streaming use fitsbolt for normalisation, guaranteeing identical results.
