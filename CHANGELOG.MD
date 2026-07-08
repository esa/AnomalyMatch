[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

## [v1.3.2] – 2026-07-08

### Fixed
- **efficientnet-lite0 fine-tuning regression** introduced by the timm backbone swap in v1.3.0, which dropped FixMatch AUROC from ~0.96 to ~0.81. Three model-level defaults silently lost in the migration are restored: the classifier head is reset to PyTorch initialisation (timm's TF-EfficientNet scaling over-scaled the 2-class head ~25x, so the overconfident fresh head cleared the FixMatch confidence threshold from step one and poisoned the backbone with random pseudo-labels), `cfg.bn_momentum` is now applied to both models' BatchNorm layers instead of leaving them at timm's 10x-too-fast default, and `set_seeds()` is called after config validation so training is reproducible again (#16)

## [v1.3.1] – 2026-05-11

### Fixed
- **Cutana 0.3.0 compatibility** by removing the obsolete `synchronised_loading` kwarg from `StreamingOrchestrator.init_streaming` calls
- **Pretrained weight loading** when the repository is cloned without `git-lfs`: fall back to a HuggingFace download instead of failing on LFS pointer files (#7)

### Changed
- **Security:** replaced pickle-based `torch.save` / `torch.load` checkpointing with `safetensors`, removing the pickle deserialisation risk on untrusted model files (#9)
- **PyTorch minimum** raised to `>=2.6` in the conda environments to pick up upstream CVE fixes (#11)
- **Cutana minimum** raised to `>=0.3.0` in `environment.yml`, `environment_CI.yml`, and `pyproject.toml`

### Documentation
- **Programmatic Evaluation (Headless Mode)** section added to the README, showing how to run batch predictions through `Session.load_model()` + `Session.evaluate_all_images()` without launching the UI (#10)

## [v1.3.0] – 2026-02-11

### Added
- **Multispectral image support** for arbitrary channel count images with configurable `channel_combination` matrices (#255)
- **UI separation** into standalone `anomaly_match_ui` package for cleaner architecture (#257)
- **Flux conversion configuration** (`apply_flux_conversion`) ensuring training/prediction consistency (#272)
- **timm model backend** replacing efficientnet-specific packages for broader model support (#268)
- **`test-cnn` model** for fast unit/integration testing without heavy model downloads (#265)

### Changed
- **Replaced black and flake8 with ruff** for linting and formatting (#258)
- **Restructured test suite** into unit/integration/e2e/ui directories with pytest markers and CI caching (#265)
- **Deduplicated prediction code** into shared `prediction_utils.py` module (#264)
- **Auto-inference of `n_output_channels`** from `channel_combination` matrix or FITS extension count (#276, #278)
- **PIL resize for CONVERSION_ONLY** normalisation achieving up to 73x faster image loading (#259)
- **Faster catalogue validation** by skipping per-chunk FITS existence checks and using parquet metadata (#277)

### Fixed
- **Double normalisation** in cutana streaming pipeline (#271)
- **Normalisation consistency** between cutana and training paths with channel_weights passthrough (#274)
- **Session logging** with eager directory creation and per-session log files (#260)
- **Gallery not updating** after prediction chunks complete (#262)
- **Cutana source ID handling** for non-string int64 source_ids (#261)
- **Albumentations 2.0 compatibility** renaming deprecated `mode` to `border_mode` (#276)
- **Prediction progress bar** with phase tracking for better user feedback (#276)
- **Identity channel_combination** auto-creation for multi-extension FITS configs (#278)
- **ASinh parameters** missing in cutana format config (#279)
- **Filter name resolution** from catalogue for cutana streaming (#281)
- **Primary HDU validation** raising ValueError when no image data found (#271)

### Documentation
- **Normalisation README** with improved channel_combination and flux conversion documentation (#275)
- **Auto-inference documentation** updating multispectral config examples (#284)

---

## [v1.2.0] – 2025-01-13

### Added
- **Cutana streaming integration** for catalogue-based predictions with parquet and CSV support
- **FitsBolt integration** for consistent FITS normalization across training and prediction
- **Iteration score storage** for tracking unlabeled and test data scores per iteration
- **Automatic batch size estimation** using exponential and binary search for optimal GPU memory usage
- **Full resolution image preview** button in the UI for detailed inspection
- **Dead code detection** CI workflow using Vulture for codebase maintenance

### Changed
- **Refactored Widget architecture** by extracting PreviewWidget for better code organization
- **FitsBolt config persistence** in model checkpoints for reproducible normalization
- **Parquet format** for Cutana buffer instead of CSV for improved performance
- **Black line-length** updated to 100 characters for better readability

### Fixed
- **Gallery filename display** for long filenames with improved shortening (#237)
- **Duplicate result accumulation** in prediction process (#238)
- **Error handling** for iteration score CSV saves (#236)
- **FITS extension handling** in Cutana streaming
- **Tensor handling** improvements throughout the codebase

### Removed
- **Dead code cleanup** removing unused functions and imports identified by Vulture
- **IDE/editor files** from repository with updated .gitignore

---

## [v1.1.0] – 2025-07-04

### Added
- **Zarr file format support** for scalable array storage and processing
- **Session tracking and management** with comprehensive iteration history
- **Metadata handling** for associating metadata with images in labeled_data.csv
- **Configuration validation** to ensure proper setup before training
- **Label caching** for improved performance in active learning loops
- **Unlabeling functionality** allowing users to remove labels from images
- **Interpolation order configuration** for image resizing operations
- **ASinh normalization** with grayscale/multichannel and RGB functionality

### Changed
- **Improved UI responsiveness** with optimized image loading and display
- **Enhanced session saving** to capture training results and configurations
- **Streamlined prediction process** with better file type detection
- **Reduced logging verbosity** to minimize output spam
- **Unified image resizing** to use BILINEAR interpolation consistently
- **Improved error handling** for insufficient labeled data scenarios
- **Better memory management** in prediction processes

### Fixed
- **RGB display reset** when using brightness/contrast sliders (#158)
- **Train iterations slider** usability issues (#155)
- **Test ratio image reading** bugs (#181)
- **Cached image normalization** not updating after training (#151)
- **Channel ordering** in TurboJPEG decoded files
- **CPU fallback** when CUDA is not available
- **NaN/inf value handling** in image processing
- **Top image preservation** across prediction batches
- **Label count display** in UI (#185)

### Removed
- **ZIP file support** (kept for benchmarking, removed from prediction process)
- **Redundant configuration options** and deprecated functions

### Performance
- **Faster label lookups** through intelligent caching mechanisms
- **Optimized batch processing** for HDF5 and Zarr formats
- **Reduced memory usage** in prediction workflows
- **Improved UI responsiveness** in ESA Datalabs environment

---

## [v1.0.1] – 2025-05-23

### Added
- **`.fits` image-format support** for direct ingestion and visualisation.
- **`.tiff` image-format support** for direct ingestion and visualisation.
- **CHANGELOG.md** created (you’re reading it!).

### Changed
- **Refactored file-loading layer** to use a centralised
  set of functions for loading and normalising images.

### Experimental
- **On-the-fly image normalisation**  
  Interactive options for  
  - **ZScale** stretch
  - **Logarithmic** stretch

---

## [v1.0] – 2025-05-07  

### Added
- **Core semi-supervised anomaly-detection pipeline**  
  Combines *FixMatch* consistency training with an **EfficientNet-Lite0** backbone,
  formulated as heavily-imbalanced **binary classification** (“nominal” vs “anomaly”).
- **Active-learning loop & GUI**  
  *ipywidgets*-based interface lets experts inspect top-ranked candidates, relabel,
  and trigger rapid re-training cycles (≈100 iterations)
- **Two-tier data augmentation**  
  - *Weak* augmentations for pseudo-labelling  
  - *RandAugment*-driven *strong* augmentations for consistency training.
- **Imbalance handling** through weighted oversampling during minibatch creation.
- **Scalability**  
  Processes ≈ **100 million images in ~3 days on a single GPU**; data stored in HDF5
  for high-throughput streaming.
- **Evaluation metrics & plots**  
  Built-in calculation of **AUROC, AUPRC**; live display inside the GUI.
- **Extensive CI pipeline** with > 70 unit tests for robustness.
- **ESA Datalabs integration** enabling one-click deployment next to Euclid &
  Hubble archives.
- **Benchmark notebooks & pretrained configs**  
  Reproduce the reported results on **miniImageNet** (1 % anomalies, AUROC ≈ 0.95)
  and **GalaxyMNIST** (AUROC ≈ 0.86) out of the box.

---