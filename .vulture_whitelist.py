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
get_labeled_data_from_checkpoint  # noqa - Public API for extracting labels from checkpoints
list_sessions  # noqa - Used in test_session_io_handler.py
save_labels_to_output_dir  # noqa - Used in test_run_label_migration.py

# FixMatch class attributes
requires_grad  # noqa - PyTorch tensor property set to disable gradient for EMA model

# TestCNN - nn.Module.forward() called implicitly by PyTorch
TestCNN.forward  # noqa - Called via model(x) in FixMatch training loop

# AnomalyDetectionDataset methods used in tests (tests/dataset_test.py)
unlabeled_filepaths  # noqa - Used in test_anomaly_detection_dataset_properties

# TrainingDataSource public API - used by training_process.py and tests
training_data_source  # noqa - Config attribute for selecting data source type
create_training_data_source  # noqa - Factory used by training_process.py
ImageFolderSource  # noqa - Default data source, used by AnomalyDetectionDataset


# File I/O utility functions - public API
get_image_paths_from_folder  # noqa - Companion to get_image_names_from_folder, tested

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
unlabeled_pool_cap  # noqa - Config for training subprocess pool size cap
unlabeled_pool_cap_hires  # noqa - Config for training subprocess high-res cap
unlabeled_pool_hires_threshold  # noqa - Config for high-res image size threshold

# Seed utility function - used in tests for reproducibility
set_seeds  # noqa - Used in test fixtures and scripts

# PyTorch CUDA attribute - set in set_seeds.py for deterministic/performance mode
_.benchmark  # noqa - torch.backends.cudnn.benchmark attribute

# Image processing functions used in prediction scripts (root level, excluded from scan)
process_single_wrapper  # noqa - Used in prediction_utils.py
_.n_expected_channels  # noqa - fitsbolt config attribute set dynamically

# Profiler configuration - used in prediction_process*.py scripts (root level, excluded from scan)

# ipywidgets attribute assignments - used by the widget framework for display updates
_.children  # noqa - ipywidgets VBox.children assignment updates displayed widgets
_.disabled  # noqa - ipywidgets Button.disabled toggles interactive state
_.description  # noqa - ipywidgets Button.description sets the button label text

# styles.py public API - used by screens that depend on ui_scale
get_ui_scale  # noqa - Public API for reading the current UI scale factor
scale_px  # noqa - Public API for scaling pixel values by ui_scale

# AnomalyScoreDB public API - used by prediction screen (#289) and prediction scripts
store_results  # noqa - Core write API for prediction scripts
set_metadata  # noqa - Store run metadata for compatibility validation
set_metadata_batch  # noqa - Batch metadata storage
get_top_results  # noqa - Convenience wrapper for UI gallery
get_count  # noqa - Used by progress panel
get_all_scores  # noqa - Used by UI for score analysis
get_score_range  # noqa - Used by UI for histogram scaling
get_score_histogram  # noqa - Used by UI for live score distribution
is_processed  # noqa - Resume support for individual filename check
get_unprocessed  # noqa - Resume support for batch filename filtering
get_processed_filenames  # noqa - Resume support for full set retrieval
get_heartbeat  # noqa - Heartbeat reader; production consumer (UI staleness banner) lands with #325
validate_compatibility  # noqa - Config compatibility validation
_.row_factory  # noqa - sqlite3 connection attribute for Row access

# ImageCache public API - used by prediction screen (#289)
prefetch  # noqa - Batch pre-loading for gallery display
ImageCache.size  # noqa - Property for current cache occupancy
ImageCache.max_size  # noqa - Property for maximum cache capacity

# Cutana config attributes — we set these on cutana's third-party DotMap
# config object which is consumed by cutana internals. Vulture only scans
# our code, so it sees the writes but not the reads inside cutana.
# Every new cutana config field we use will need an entry here.
_.target_resolution  # noqa - cutana config: output cutout size
_.source_catalogue  # noqa - cutana config: path to source catalogue
_.fits_extensions  # noqa - cutana config: FITS extension names
_.selected_extensions  # noqa - cutana config: extension selection list
_.external_fitsbolt_cfg  # noqa - cutana config: fitsbolt normalization config
_.padding_factor  # noqa - cutana config: cutout zoom-out multiplier
_.channel_weights  # noqa - cutana config: per-extension weight vectors
_.data_type  # noqa - cutana config: final output dtype ("uint8" or "float32")
_.skip_catalogue_validation  # noqa - cutana config: opt out of per-row re-validation

# ipywidgets layout properties
_.flex  # noqa - ipywidgets Layout.flex CSS flexbox property
_.margin  # noqa - ipywidgets Layout.margin CSS property
_.max_width  # noqa - ipywidgets Layout.max_width, set dynamically by scale slider
_.grid_template_columns  # noqa - ipywidgets GridBox layout, set dynamically by scale slider

# BackendInterface prediction API - used in tests
close_prediction_monitor  # noqa - Cleanup method called in test_prediction_screen.py

# ESASky coordinate utilities - ThumbnailCell.update() accepts esasky_url param
get_esasky_url  # noqa - Generates ESASky viewer URL, wired via thumbnail_cell.py
get_coordinates  # noqa - RA/Dec lookup for esasky_url generation

# PredictionProfiler - public API used by prediction scripts in scripts/
stage  # noqa - Context manager for timing pipeline stages, used in all prediction scripts
end_batch  # noqa - Called at end of each prediction batch in scripts
record_finalization  # noqa - Records finalization timing in prediction scripts
save_partial_report  # noqa - Saves per-process JSON report in prediction scripts
_._output_dir  # noqa - Internal attribute used by Coordinator for merge_reports path

# Config comparison utility - used by TrainingScreen for norm change detection
configs_differ  # noqa - Public API in validate_config.py, called by training_screen.py

# NormalisationConfigWidget public API - used by TrainingSetupScreen (#329)
NormalisationConfigWidget  # noqa - Widget class instantiated by setup screen
set_extensions  # noqa - Called when user selects FITS files to update column headers
apply_cutana_bands  # noqa - Called by setup screens and training screen for Cutana sources
get_normalisation_config  # noqa - Extracts config dict for training/prediction
update_from_config  # noqa - Restores widget state from saved config

# LabelableThumbnailCell public API - used by TrainingScreen 2.0 (#329)
LabelableThumbnailCell  # noqa - Widget class instantiated by training gallery
LabelState  # noqa - Enum used by training gallery for label state
set_label  # noqa - Programmatic label update for restoring label state
_.label_state  # noqa - Property read by training gallery
_.font_weight  # noqa - ipywidgets button style attribute

# GalleryScreenBase public API - used by TrainingScreen 2.0 (#329)
GalleryScreenBase  # noqa - ABC extended by PredictionScreen and TrainingScreen
_detail_back_screen  # noqa - Hook for subclasses to set back-navigation target

# TrainingSetupScreen public API - used by app.py navigation (#329)
TrainingSetupScreen  # noqa - Screen class instantiated by app._create_screen
_validate_label_csv  # noqa - Helper tested in test_training_setup.py

# TrainingScreen 2.0 (#329 phase 3B)
shorten_filename  # noqa - Utility function tested in test_shorten_filename.py
TrainingState  # noqa - Enum used by training screen state machine
cell_factory  # noqa - GalleryWidget parameter for custom cell types
launch_training_subprocess  # noqa - BackendInterface method, called by TrainingScreen
merge_gallery_labels  # noqa - BackendInterface method, called by TrainingScreen
_._temp_dir  # noqa - TrainingScreen attribute, set from BackendInterface return value

# BackendInterface.get_session() — public API for external consumers
get_session  # noqa - BackendInterface method, returns the session instance

# SessionTracker methods — used by test fixtures that set up state for
# testing kept functionality (get_session_info, save_session, etc.)
add_labeled_sample  # noqa - Used in test_session_tracker.py and test_session_io_handler.py fixtures
update_test_performance  # noqa - Used in test_session_tracker.py and test_session_io_handler.py fixtures

# tqdm logging sink (anomaly_match/utils/tqdm_logging.py) — tqdm calls the
# file-like object's flush() implicitly on each refresh, so vulture can't see
# the caller.
flush  # noqa - _NewlineTqdmFile.flush, called implicitly by tqdm(file=...)
