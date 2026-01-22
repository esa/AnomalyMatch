[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

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