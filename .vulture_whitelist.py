#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Vulture whitelist file.

Add entries here for code that vulture incorrectly identifies as unused.
Format: function_name  # noqa - comment explaining why it's used
"""

# SessionIOHandler methods - public API used in tests
list_sessions  # noqa - Used in test_session_io_handler.py
save_run  # noqa - Used in test_run_label_migration.py
save_labels_to_output_dir  # noqa - Used in test_run_label_migration.py

# FixMatch class attributes
requires_grad  # noqa - PyTorch tensor property set to disable gradient for EMA model

# TestCNN - nn.Module.forward() called implicitly by PyTorch
TestCNN.forward  # noqa - Called via model(x) in FixMatch training loop

# AnomalyDetectionDataset methods used in tests (tests/dataset_test.py)
_read_and_resize_image  # noqa - Used in test_read_and_resize_different_formats
unlabeled_filepaths  # noqa - Used in test_anomaly_detection_dataset_properties
save_as_hdf5  # noqa - Used in test_anomaly_detection_dataset_hdf5
load_from_hdf5  # noqa - Used in test_anomaly_detection_dataset_hdf5

# Transform functions used in paper_scripts/
get_strong_transforms  # noqa - Used in paper_scripts/get_example_images.py

# File I/O utility functions - public API
get_image_paths_from_folder  # noqa - Companion to get_image_names_from_folder, tested

# Session class public API
start_UI  # noqa - Public API - used in StarterNotebook.ipynb

# Widget methods - public API
update_image_display  # noqa - Public API method for updating image display

# ipywidgets style/layout attributes - used by ipywidgets framework
_.style  # noqa - Widget.py: progress_bar.style for visual feedback
_.button_color  # noqa - ipywidgets button styling
_.font_size  # noqa - ipywidgets widget styling
_.width  # noqa - ipywidgets layout attribute
_.height  # noqa - ipywidgets layout attribute

# Learning rate scheduler utility - tested in tests/utils_test.py
get_cosine_schedule_with_warmup  # noqa - Used in tests and available for external use

# Configuration attributes - validated and documented
bn_momentum  # noqa - Part of default config for batch normalization momentum
N_batch_prediction  # noqa - Used in prediction scripts for batch size

# Seed utility function - used in paper_scripts/paper_benchmark.py and tests
set_seeds  # noqa - Used for reproducibility in benchmarks and testing

# PyTorch CUDA attribute - set in set_seeds.py for deterministic/performance mode
_.benchmark  # noqa - torch.backends.cudnn.benchmark attribute

# Image processing functions used in prediction scripts (root level, excluded from scan)
process_single_wrapper  # noqa - Used in prediction_utils.py, prediction_process_hdf5.py
_.n_expected_channels  # noqa - fitsbolt config attribute set dynamically
