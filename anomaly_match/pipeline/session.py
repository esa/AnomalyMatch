#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import datetime
import os
import pickle
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import zarr
from fitsbolt import SUPPORTED_IMAGE_EXTENSIONS
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger

from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.datasets.data_utils import get_prediction_dataloader
from anomaly_match.datasets.SSL_Dataset import SSL_Dataset
from anomaly_match.models.FixMatch import FixMatch
from anomaly_match.pipeline.SessionTracker import SessionTracker
from anomaly_match.utils.cutana_stream_utils import (
    cutana_buffer_generator,
    cutana_validate_files_and_count_sources,
)
from anomaly_match.utils.get_net_builder import get_net_builder
from anomaly_match.utils.get_optimizer import get_optimizer
from anomaly_match.utils.print_cfg import print_cfg
from anomaly_match.utils.set_log_level import set_log_level
from anomaly_match.utils.set_seeds import set_seeds
from anomaly_match.utils.validate_config import validate_config


class Session:
    """Tracks a session of using anomaly_match and its state."""

    labeled_train_dataset = None
    unlabeled_train_dataset = None
    test_dataset = None

    model: FixMatch = None

    active_learning_df = pd.DataFrame(columns=["filename", "label"])
    # Cache for fast label lookups - maps filename to label
    _label_cache = {}
    # Cache for label distribution to avoid expensive pandas operations
    _label_distribution_cache = None

    filenames = None
    scores = None
    img_catalog = None
    cached_image_normalisation_enum = NormalisationMethod.CONVERSION_ONLY

    def __init__(self, cfg):
        """Initializes the session with the given configuration.

        Args:
            cfg (DotMap): Configuration for the session.
        """
        logger.debug("Initializing session")
        self.session_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if cfg.log_level is not None:
            set_log_level(cfg.log_level, cfg)
        if cfg.log_level in ["TRACE", "DEBUG"]:
            print_cfg(cfg)

        # Initialize session tracking first
        session_name = getattr(cfg, "name", None)
        self.session_tracker = SessionTracker(session_name=session_name)
        self.session_io = SessionIOHandler()

        # Update config paths to use centralized session directory BEFORE validation
        self.session_io.update_config_paths_for_session(cfg, self.session_tracker)

        # Validate the config
        validate_config(cfg)

        # Seed all RNGs before datasets and model are built so that the train/test
        # split, weight initialisation and augmentation sampling are reproducible.
        # cfg.seed was defined and validated but never actually applied.
        set_seeds(int(cfg.seed))

        self.cfg = cfg
        self.cached_image_normalisation_enum = cfg.normalisation.normalisation_method
        self.out = None  # Initialize out attribute to None

        # Initialize label cache and distribution cache
        self._label_cache = {}
        self._label_distribution_cache = None
        self._active_learning_counts_cache = None

        logger.debug("Session initialized, loading datasets")
        self._load_datasets()
        logger.debug("Datasets loaded, initializing model")
        self._init_model()

    def _init_model(self):
        """Initializes the model with the configuration settings."""
        net_builder = get_net_builder(
            self.cfg.net,
            pretrained=self.cfg.pretrained,
            in_channels=self.cfg.num_channels,
        )
        self.model = FixMatch(
            net_builder,
            self.cfg.num_classes,
            self.cfg.num_channels,
            self.cfg.ema_m,
            T=self.cfg.temperature,
            p_cutoff=self.cfg.p_cutoff,
            lambda_u=self.cfg.ulb_loss_ratio,
            logger=logger,
            session_tracker=self.session_tracker,
        )

        # Apply the configured BatchNorm momentum (cfg.bn_momentum = 1 - ema_m, ~0.01).
        # It was computed and validated but never set on the model, so BatchNorm ran at
        # timm's 0.1 default (~10x too fast for our small batch size), destabilising the
        # running statistics during fine-tuning.
        for submodel in (self.model.train_model, self.model.eval_model):
            for module in submodel.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.momentum = self.cfg.bn_momentum

        # get optimizer, ADAM and SGD are supported.
        optimizer = get_optimizer(
            self.model.train_model,
            self.cfg.opt,
            self.cfg.lr,
            self.cfg.momentum,
            self.cfg.weight_decay,
        )
        self.model.set_optimizer(optimizer)

        # If a CUDA capable GPU is used, we move everything to the GPU now
        if torch.cuda.is_available():
            self.cfg.gpu = 0
            torch.cuda.set_device(self.cfg.gpu)
            self.model.train_model = self.model.train_model.cuda(self.cfg.gpu)
            self.model.eval_model = self.model.eval_model.cuda(self.cfg.gpu)

        self.model.set_data_loader(
            self.cfg,
            self.labeled_train_dataset,
            self.unlabeled_train_dataset,
            self.test_dataset,
        )

    def _load_datasets(self):
        """Loads the datasets required for training and evaluation."""
        # Construct Dataset
        self.train_dset = SSL_Dataset(
            cfg=self.cfg,
            train=True,
        )
        self.labeled_train_dataset, self.unlabeled_train_dataset = self.train_dset.get_ssl_dset()

        # Update information about cached dataset
        self.cached_image_normalisation_enum = self.cfg.normalisation.normalisation_method

        self.cfg.num_classes = self.train_dset.num_classes
        self.cfg.num_channels = self.train_dset.num_channels

        if self.cfg.test_ratio > 0:
            self.test_dataset = SSL_Dataset(
                cfg=self.cfg,
                train=False,
            ).get_dset()
        else:
            self.test_dataset = None

        self.prediction_dataloader = get_prediction_dataloader(
            self.train_dset.dset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def update_predictions(self, progress_callback=None):
        """Updates the predictions using the current model and datasets."""
        with self.out if self.out is not None else nullcontext():
            logger.debug("Updating predictions")

            self.prediction_dataloader = get_prediction_dataloader(
                self.train_dset.dset,
                batch_size=self.cfg.eval_batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
            )

            scores, imgs, filenames, _ = self.model.get_scored_binary_unlabeled_samples(
                self.prediction_dataloader,
                target_class=1,
                cfg=self.cfg,
                progress_callback=progress_callback,
            )

            self.img_catalog = imgs.permute(0, 2, 3, 1).cpu().numpy()
            self.scores = scores.cpu().numpy()
            self.filenames = np.array(filenames)

            if self.cfg.test_ratio > 0:
                logger.debug("Predictions updated, evaluating model")
                self.eval_performance = self.model.evaluate(
                    cfg=self.cfg,
                    progress_callback=progress_callback,
                )

    def sort_by_anomalous(self):
        """Sorts the images by their anomalous scores in descending order."""
        indices = np.argsort(-self.scores)  # Descending sort for most anomalous
        self._apply_sort(indices)

    def sort_by_nominal(self):
        """Sorts the images by their nominal scores in ascending order."""
        indices = np.argsort(self.scores)  # Ascending sort for most nominal
        self._apply_sort(indices)

    def sort_by_mean(self):
        """Sorts the images by their distance to the mean score."""
        mean_distance = np.abs(self.scores - np.mean(self.scores))
        indices = np.argsort(mean_distance)  # Sort by distance to mean
        self._apply_sort(indices)

    def sort_by_median(self):
        """Sorts the images by their distance to the median score."""
        median_distance = np.abs(self.scores - np.median(self.scores))
        indices = np.argsort(median_distance)  # Sort by distance to median
        self._apply_sort(indices)

    def _apply_sort(self, indices):
        """Applies the given sort indices to the image catalog, scores, and filenames.

        Args:
            indices (np.ndarray): Indices to sort the data.
        """
        self.img_catalog = self.img_catalog[indices]
        self.scores = self.scores[indices]
        self.filenames = self.filenames[indices]

    def save_labels(self):
        """Saves the current labels to a CSV file."""
        with self.out if self.out is not None else nullcontext():
            # Get combined labeled data from dataset and active learning
            combined_df = self._get_combined_labeled_data()

            # Update the session tracker with combined data, preserving iteration info
            self.session_tracker.update_labeled_data(combined_df)

            # Save the session (includes labeled data)
            session_path = self.session_io.save_session(self.session_tracker, cfg=self.cfg)
            logger.debug(f"Session saved to {session_path}")

    def _get_combined_labeled_data(self):
        """Get combined labeled data from dataset and active learning."""
        # Combine active_learning_df with already labeled data in the dataset
        labeled_data = [
            {"filename": filename, "label": "normal" if target == 0 else "anomaly"}
            for filename, target in zip(
                self.labeled_train_dataset.filenames,
                self.labeled_train_dataset.targets,
            )
        ]

        labeled_df = pd.DataFrame(labeled_data)
        combined_df = pd.concat([labeled_df, self.active_learning_df]).drop_duplicates(
            subset="filename", keep="last"
        )

        # Add metadata if available in the dataset
        metadata_df = self.train_dset.dset.get_all_metadata()
        if metadata_df is not None:
            metadata_df = metadata_df.copy()
            metadata_df.reset_index(inplace=True)  # Convert index back to column
            combined_df = combined_df.merge(metadata_df, on="filename", how="left")

        return combined_df

    def set_terminal_out(self, out):
        """Sets the terminal output context.

        Args:
            out (Output): The output context to set.
        """
        # Clear any existing handlers to prevent duplicate logging
        if hasattr(self, "out") and self.out is not None:
            logger.warning("Removing existing output handler")
            logger.remove()
        self.out = out

    def label_image(self, idx, label):
        """Labels an image with the given index and label.

        Args:
            idx (int): Index of the image to label.
            label (str): Label to assign to the image.
        """
        # Currently we assume that the label is either "normal" or "anomaly"
        assert label in ["normal", "anomaly"], f"Invalid label: {label}"
        with self.out if self.out is not None else nullcontext():
            current_filename = self.filenames[idx]
            # Check if the filename already exists in the DataFrame
            if current_filename in self.active_learning_df["filename"].values:
                logger.debug(f"Overwriting label for {current_filename} to {label}")
                self.active_learning_df.loc[
                    self.active_learning_df["filename"] == current_filename, "label"
                ] = label
            else:
                logger.debug(f"Adding label for {current_filename} as {label}")
                new_row = pd.DataFrame({"filename": [current_filename], "label": [label]})
                self.active_learning_df = pd.concat(
                    [self.active_learning_df, new_row], ignore_index=True
                )

            # Update the cache and invalidate distribution cache
            self._label_cache[current_filename] = label
            self._label_distribution_cache = None
            self._active_learning_counts_cache = None

            # Add to session tracker
            self.session_tracker.add_labeled_sample(current_filename, label)

    def unlabel_image(self, idx):
        """Removes the label for the image at the given index.

        Args:
            idx (int): Index of the image to unlabel.
        """
        with self.out if self.out is not None else nullcontext():
            current_filename = self.filenames[idx]
            # Check if the filename exists in the active learning DataFrame
            if current_filename in self.active_learning_df["filename"].values:
                logger.debug(f"Removing label for {current_filename}")
                # Remove the row with this filename
                self.active_learning_df = self.active_learning_df[
                    self.active_learning_df["filename"] != current_filename
                ]

                # Remove from cache and invalidate distribution cache
                self._label_cache.pop(current_filename, None)
                self._label_distribution_cache = None
                self._active_learning_counts_cache = None

                # Also update the session tracker's labeled data if possible
                if hasattr(self.session_tracker, "labeled_data_df"):
                    # This will maintain any iteration information for analytics
                    # but mark the label as "removed" for tracking purposes
                    self.session_tracker.labeled_data_df.loc[
                        self.session_tracker.labeled_data_df["filename"] == current_filename,
                        "label",
                    ] = "removed"
                    logger.debug(f"Updated session tracker to mark {current_filename} as removed")
            else:
                logger.debug(f"No label found for {current_filename}")

    def set_normalisation_method(self, method: NormalisationMethod):
        """Updates the normalization method in the config.

        Args:
            method (NormalisationMethod): The new normalization method to apply.
        """
        # update norm method in session cfg, should
        self.cfg.normalisation.normalisation_method = method

    def _reload_datasets(self):
        """Reloads the datasets if normalisation changed."""
        self._load_datasets()
        # Reinitialize model with new data
        self.model.set_data_loader(
            self.cfg,
            self.labeled_train_dataset,
            self.unlabeled_train_dataset,
            self.test_dataset,
        )

    def remember_current_file(self, filename):
        """Remembers the current file by appending it to a CSV if not already present."""
        with self.out if self.out is not None else nullcontext():
            # Ensure output directory exists before trying to write file
            os.makedirs(self.cfg.output_dir, exist_ok=True)

            # Use cfg.name instead of cfg.save_file for the output filename
            output_file = os.path.join(
                self.cfg.output_dir,
                f"{self.cfg.name}_{self.session_start}_remembered_files.csv",
            )

            # Read existing files or create empty DataFrame
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                if filename in df["filename"].values:
                    logger.debug(f"File {filename} already in remembered files")
                    return
            else:
                df = pd.DataFrame(columns=["filename", "timestamp"])

            # Append the filename with timestamp
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame({"filename": [filename], "timestamp": [current_time]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(output_file, index=False)

            logger.info(f"Remembered file {filename}")

    def save_model(self):
        """Saves the current model state using SessionIOHandler."""
        with self.out if self.out is not None else nullcontext():
            # Ensure fitsbolt config is set before saving model
            # This creates cfg.fitsbolt_cfg from normalisation settings for prediction consistency
            self.cfg = get_fitsbolt_config(self.cfg)

            # Save model using SessionIOHandler
            model_path = self.session_io.save_model(self.model, self.cfg, self.session_tracker)

            logger.info(f"Model saved to: {model_path}")

    def load_model(self):
        """Loads the model state using SessionIOHandler."""
        with self.out if self.out is not None else nullcontext():
            # Save the current normalisation method before loading
            old_normalisation_method = self.cfg.normalisation.normalisation_method

            success = self.session_io.load_model(self.model, self.cfg)

            if success:
                logger.info("Model loaded successfully")

                # Always inform user about loaded normalisation settings
                # (parameters like asinh_scale may differ even if method is the same)
                new_normalisation_method = self.cfg.normalisation.normalisation_method
                logger.info(
                    f"Loaded model normalisation: {new_normalisation_method.name}. "
                    f"Note: normalisation parameters were also loaded from the model checkpoint."
                )

                # Warn if the method itself changed
                if old_normalisation_method != new_normalisation_method:
                    logger.warning(
                        f"Normalisation method changed from {old_normalisation_method.name} "
                        f"to {new_normalisation_method.name}. Images may need to be refreshed."
                    )

                # Update cached normalisation and reload datasets if method changed
                if (
                    self.cached_image_normalisation_enum
                    != self.cfg.normalisation.normalisation_method
                ):
                    logger.info("Normalisation method changed, reloading datasets...")
                    self._reload_datasets()
            else:
                logger.error("Failed to load model")

    def train(self, cfg, progress_callback=None):
        """Trains the model using the given configuration.

        Args:
            cfg (DotMap): Configuration for training.
            progess_callback (function, optional): Callback function to update progress. Defaults to None.
        """
        self.cfg = cfg
        with self.out if self.out is not None else nullcontext():
            # Start a new session iteration
            self.session_tracker.start_new_session_iteration()

            self.save_labels()
            # Update the datasets
            self.labeled_train_dataset, self.unlabeled_train_dataset = self.train_dset.update_dsets(
                self.active_learning_df, N_to_load=self.cfg.N_to_load
            )

            # Clear active_learning_df after updating datasets to prevent double-counting
            # The newly labeled samples are now in the main training dataset
            self.active_learning_df = pd.DataFrame(columns=["filename", "label"])

            # Invalidate caches since the dataset structure has changed
            self._label_distribution_cache = None
            self._active_learning_counts_cache = None
            # Clear the label cache since active_learning_df is now empty
            self._label_cache = {}

            self.model.set_data_loader(
                self.cfg,
                self.labeled_train_dataset,
                self.unlabeled_train_dataset,
                self.test_dataset,
            )

            # Train the model and get evaluation results
            eval_results = self.model.train(cfg, progress_callback=progress_callback)

            # Update session tracker with training results
            test_scores = None
            if eval_results:
                # Filter out large data fields that shouldn't be saved to session metadata
                filtered_eval_results = {
                    k: v
                    for k, v in eval_results.items()
                    if k
                    not in ["eval/predictions_and_labels", "eval/roc_data", "eval/precision_recall"]
                }
                self.session_tracker.update_test_performance(filtered_eval_results)

                # Extract test scores for saving (filename -> anomaly probability)
                if "eval/predictions_and_labels" in eval_results:
                    predictions_and_labels = eval_results["eval/predictions_and_labels"]
                    test_scores = {
                        filename: float(pred_label[0].item())
                        for filename, pred_label in predictions_and_labels.items()
                    }

            # Update total model iterations
            self.session_tracker.total_model_iterations = self.model.total_it

            logger.info("Training complete.")
            # Update cached image normalisation enum
            self.cached_image_normalisation_enum = self.cfg.normalisation.normalisation_method

            # Update predictions to get unlabelled scores for this iteration
            self.update_predictions()

            # Extract unlabelled scores (filename -> anomaly score)
            unlabelled_scores = None
            if self.scores is not None and self.filenames is not None:
                unlabelled_scores = {
                    filename: float(score) for filename, score in zip(self.filenames, self.scores)
                }

            # Save iteration scores (unlabelled and test set scores)
            self.session_io.save_iteration_scores(
                self.session_tracker,
                unlabelled_scores=unlabelled_scores,
                test_scores=test_scores,
            )

            # Save model to session directory using centralized save_model method
            # Note: save_model() ensures fitsbolt_cfg is set before saving
            self.save_model()

            # Save session again to capture training results (test performance, model path)
            session_path = self.session_io.save_session(self.session_tracker, cfg=self.cfg)
            logger.debug(f"Session saved after training completion to {session_path}")

    def get_label_distribution(self):
        """Gets the distribution of labels in the training dataset, including new labels in active_learning_df.

        Returns:
            tuple: A tuple containing the count of normal and anomalous labels.
        """
        # Return cached result if available
        if self._label_distribution_cache is not None:
            return self._label_distribution_cache

        normal_count = torch.sum(self.labeled_train_dataset.targets == 0)
        anomalous_count = len(self.labeled_train_dataset.targets) - normal_count

        if self.active_learning_df is not None and not self.active_learning_df.empty:
            # More efficient way to count new labels
            # Instead of filtering, just count all labels in active_learning_df
            # This assumes active_learning_df only contains new labels not in labeled_train_dataset
            new_normal = np.sum(self.active_learning_df["label"] == "normal")
            new_anomalous = np.sum(self.active_learning_df["label"] == "anomaly")

            normal_count += new_normal
            anomalous_count += new_anomalous

        # Cache the result
        self._label_distribution_cache = (normal_count, anomalous_count)
        return self._label_distribution_cache

    def get_active_learning_counts(self):
        """Gets the count of newly annotated samples in active_learning_df.

        Returns:
            tuple: (new_normal_count, new_anomalous_count)
        """
        if self.active_learning_df is None or self.active_learning_df.empty:
            return 0, 0

        # Use cached values if available
        if (
            hasattr(self, "_active_learning_counts_cache")
            and self._active_learning_counts_cache is not None
        ):
            return self._active_learning_counts_cache

        new_normal = np.sum(self.active_learning_df["label"] == "normal")
        new_anomalous = np.sum(self.active_learning_df["label"] == "anomaly")

        # Cache the result
        self._active_learning_counts_cache = (new_normal, new_anomalous)
        return self._active_learning_counts_cache

    def get_label(self, idx):
        """Gets the label for the image at the given index.

        Args:
            idx (int): Index of the image.

        Returns:
            str: The label of the image.
        """
        if self.active_learning_df is None:
            return "None"

        # Rebuild cache if it's empty but active_learning_df has data
        if not self._label_cache and not self.active_learning_df.empty:
            self._rebuild_label_cache()

        filename = self.filenames[idx]
        # Use fast cache lookup instead of pandas operations
        return self._label_cache.get(filename, "None")

    def load_next_batch(self):
        """Loads the next batch of data and updates predictions."""
        logger.debug("Loading next batch of data")
        # Note that we are updating also the labeled_dataset since the unlabeled
        # data are going to disappear from the unlabeled dataset once we call this function.
        self.labeled_train_dataset, self.unlabeled_train_dataset = self.train_dset.update_dsets(
            label_update=self.active_learning_df, N_to_load=self.cfg.N_to_load
        )

        # Clear active_learning_df after updating datasets to prevent double-counting
        # The newly labeled samples are now in the main training dataset
        self.active_learning_df = pd.DataFrame(columns=["filename", "label"])

        # Invalidate caches since the dataset structure has changed
        self._label_distribution_cache = None
        self._active_learning_counts_cache = None
        # Clear the label cache since active_learning_df is now empty
        self._label_cache = {}

        self.cached_image_normalisation_enum = self.cfg.normalisation.normalisation_method
        # We don't rebuild the cache here since active_learning_df is empty
        # The get_label method will handle finding labels in the main dataset if needed
        self.update_predictions()

    def reset_model(self):
        """Resets the model and reinitializes the session."""
        logger.debug("Resetting model")

        # Reset session tracker
        session_name = getattr(self.cfg, "name", None)
        self.session_tracker = SessionTracker(session_name=session_name)

        # Update config paths to use centralized session directory
        self.session_io.update_config_paths_for_session(self.cfg, self.session_tracker)

        # Clear label cache and distribution cache on reset
        self._label_cache = {}
        self._label_distribution_cache = None
        self._active_learning_counts_cache = None

        self._init_model()
        self.update_predictions()

    def run_pipeline(self, temp_config_path, input_path, top_N, file_type=None):
        """Run the appropriate pipeline subprocess based on file type."""
        # Auto-detect file type if not provided
        if file_type is None:
            if os.path.isfile(input_path):
                # Single file - detect from extension
                _, ext = os.path.splitext(input_path.lower())
                extension_map = {
                    ".h5": "hdf5",
                    ".hdf5": "hdf5",
                    ".zarr": "zarr",
                    ".txt": "image",  # Grouped image files
                    ".parquet": "stream",
                    ".csv": "stream",
                }
                file_type = extension_map.get(ext, "image")
            else:
                # Directory - auto-detect
                file_type = self._auto_detect_prediction_file_type(input_path)

        script_map = {
            "hdf5": "prediction_process_hdf5.py",
            "image": "prediction_process.py",
            "zarr": "prediction_process_zarr.py",
            "stream": "prediction_process_cutana.py",
        }

        script = script_map.get(file_type)
        if not script:
            raise ValueError(f"Unsupported prediction file type: {file_type}")

        # Get the directory two levels up from this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get session.py directory
        root_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
        script_path = os.path.join(root_dir, script)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found at expected path: {script_path}")

        # For image directories, we need to create a temporary file list
        if file_type == "image":
            # Create a temporary file containing the list of image paths
            temp_file_list = os.path.join("tmp", f"{self.cfg.save_file}_file_list.txt")
            with open(temp_file_list, "w") as f:
                f.write(input_path)

            cmd = [sys.executable, script_path, temp_config_path, temp_file_list, str(top_N)]
        else:
            cmd = [sys.executable, script_path, temp_config_path, input_path, str(top_N)]

        logger.info(f"Launching prediction subprocess: {script}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            logger.error(
                f"Prediction subprocess failed (exit code {result.returncode}). "
                f"Check prediction.log in {self.cfg.output_dir} for details."
            )

        # Reset logger to old level
        set_log_level(self.cfg.log_level, self.cfg)

    def evaluate_all_images(self, top_N=1000, progress_callback=None):
        """Evaluates all images and updates the session's img_catalog with the top N images."""
        logger.info("Evaluating all images")
        # check if normalisation changed and reload if necessary
        if self.cfg.normalisation.normalisation_method != self.cached_image_normalisation_enum:
            self._reload_datasets()

        # Check if model exists before proceeding
        if not os.path.exists(self.cfg.model_path):
            error_msg = (
                f"Model not found at {self.cfg.model_path}. "
                "Please train and save a model before running predictions."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Auto-detect file type based on prediction_search_dir
        if not self.cfg.prediction_search_dir:
            error_msg = (
                "No prediction_search_dir configured. "
                "Please set cfg.prediction_search_dir to a directory containing "
                "images, HDF5 files, Zarr files, or Cutana buffer files."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        detected_file_type = self._auto_detect_prediction_file_type(self.cfg.prediction_search_dir)

        # Check for Cutana + MIDTONES incompatibility
        if detected_file_type == "stream":
            if self.cfg.normalisation.normalisation_method == NormalisationMethod.MIDTONES:
                error_msg = (
                    "MIDTONES normalisation is not supported for Cutana streaming predictions. "
                    "Please use CONVERSION_ONLY, LOG, ZSCALE, or ASINH."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Define supported file extensions
        supported_extensions = {
            "hdf5": [".h5", ".hdf5"],
            "image": SUPPORTED_IMAGE_EXTENSIONS,
            "zarr": [".zarr"],
            "stream": [".csv", ".parquet"],
        }

        pattern = supported_extensions.get(detected_file_type)
        if not pattern:
            raise ValueError(f"Unsupported prediction file type: {detected_file_type}")

        # Get all matching files from the cfg.prediction_search_dir
        input_files = []
        for f in os.listdir(self.cfg.prediction_search_dir):
            file_path = os.path.join(self.cfg.prediction_search_dir, f)
            file_ext = os.path.splitext(f.lower())[1]

            if detected_file_type == "zarr":
                # For zarr, check for direct .zarr files/directories
                if file_ext in pattern and (os.path.isfile(file_path) or os.path.isdir(file_path)):
                    input_files.append(file_path)
                # Also check for batch folders containing images.zarr subdirectory
                elif os.path.isdir(file_path) and os.path.exists(
                    os.path.join(file_path, "images.zarr")
                ):
                    # Add the path to the images.zarr subdirectory
                    input_files.append(os.path.join(file_path, "images.zarr"))
            elif file_ext in pattern:
                input_files.append(file_path)

        total_images = 0
        processed_images = 0
        start_time = time.time()

        # First count total images
        logger.debug("Counting total images to process...")
        if detected_file_type != "stream":
            for input_file in input_files:
                try:
                    if detected_file_type == "hdf5":
                        with h5py.File(input_file, "r") as h5f:
                            total_images += len(h5f["images"])
                    elif detected_file_type == "zarr":
                        root = zarr.open_group(input_file, mode="r")
                        if "images" in root:
                            total_images += root["images"].shape[0]
                        else:
                            logger.warning(f"No 'images' array found in Zarr file {input_file}")
                    else:  # jpeg/image files - single file
                        total_images += 1
                except Exception as e:
                    logger.warning(f"Error counting images in {input_file}: {str(e)}")

        else:  # Validates files against cutana and counts sources in valid files
            logger.info("Validating files against cutana")
            input_files, total_images, total_chunks = cutana_validate_files_and_count_sources(
                input_files, chunk_size=self.cfg.subprocess_buffer_size
            )

            if not input_files:
                msg = "All found files are not compatible with cutana"
                logger.error(msg)
                raise RuntimeError(msg)

        num_files = len(input_files)

        logger.info(f"Found total of {total_images:,} images to process in {num_files} files")

        # Group image files if the prediction type is 'image'
        if detected_file_type == "image":
            # Save original total images count
            total_input_images = total_images
            group_size = 10000
            grouped_files = (
                [input_files]
                if len(input_files) <= group_size
                else [
                    input_files[i : i + group_size] for i in range(0, len(input_files), group_size)
                ]
            )
            # Create new input_files list with group file paths
            tmp_dir = Path("tmp")
            tmp_dir.mkdir(exist_ok=True)

            input_files = []
            for idx, group in enumerate(grouped_files):
                path = tmp_dir / f"evaluate_all_images_grouped_{idx}.txt"
                path.write_text("\n".join(group))
                input_files.append(str(path))
            num_files = len(input_files)
            logger.debug(
                f"Created {len(input_files)} group{'s' if len(input_files) != 1 else ''} "
                f"for {total_input_images} images"
            )

        # Creating a generator that loads the csv/parquet in chunks and saves to a temporary file
        elif detected_file_type == "stream":
            # Files are read in chunks and saved into this intermediate buffer
            cutana_buffer_path = Path("tmp") / ".cutana_buffer.parquet"
            input_files = cutana_buffer_generator(
                files=input_files,
                buffer_path=cutana_buffer_path,
                chunk_size=self.cfg.subprocess_buffer_size,
            )
            num_files = total_chunks

        for file_idx, input_file in enumerate(input_files):  # Get number of images in current file
            logger.debug(f"Processing file {file_idx + 1}/{num_files}: {input_file}")
            if detected_file_type == "hdf5":
                with h5py.File(input_file, "r") as h5f:
                    num_items = len(h5f["images"])
            elif detected_file_type == "zarr":
                try:
                    root = zarr.open_group(input_file, mode="r")
                    if "images" in root:
                        num_items = root["images"].shape[0]
                    else:
                        logger.warning(f"No 'images' array found in Zarr file {input_file}")
                        num_items = 0
                except Exception as e:
                    logger.error(f"Error reading Zarr file {input_file}: {e}")
                    num_items = 0
            elif detected_file_type == "stream":
                # Cutana input buffer file (CSV or parquet)
                if str(input_file).endswith(".parquet"):
                    num_items = len(pd.read_parquet(input_file))
                else:
                    num_items = len(pd.read_csv(input_file))
            else:  # image files
                if str(input_file).endswith(".txt"):  # This is a group file
                    with open(input_file, "r") as f:
                        num_items = len(f.readlines())
                else:
                    num_items = 1

            # Calculate timing and progress
            elapsed_time = time.time() - start_time
            if processed_images > 0:  # Only estimate after processing at least one file
                images_per_second = processed_images / elapsed_time
                remaining_images = total_images - processed_images
                eta_seconds = remaining_images / images_per_second
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                progress_percent = processed_images / total_images * 100

                status_message = (
                    f"Processing {num_items:,} images in {input_file}"
                    f"\nProgress: {processed_images:,}/{total_images:,} images"
                    f" ({progress_percent:.1f}%)"
                    f"\nSpeed: {images_per_second:.1f} images/sec"
                    f"\nETA: {eta_str}"
                )

                logger.info(status_message)

                # Update UI with ETA information if callback is provided
                if progress_callback:
                    progress_callback(
                        file_idx + 1,
                        num_files,
                        batch_update=True,
                        eta_str=eta_str,
                        progress_percent=progress_percent,
                        images_per_second=images_per_second,
                    )
            else:
                logger.info(f"Processing {num_items:,} images in {input_file}")
                if progress_callback:
                    progress_callback(file_idx + 1, num_files, batch_update=True)

            # Save config to a toml file in tmp folder, but ensure model_path is correct
            temp_config = self.cfg.toDict()
            if not os.path.exists(self.cfg.model_path):
                raise FileNotFoundError(
                    f"Model file not found at {self.cfg.model_path}. "
                    "Please ensure you have saved the model before running predictions."
                )

            temp_config_path = os.path.join("tmp", f"{self.cfg.save_file}_config.pkl")
            # Make tmp folder if it doesn't exist
            os.makedirs("tmp", exist_ok=True)

            # Save the config to a temporary file as pickle
            with open(temp_config_path, "wb") as f:
                pickle.dump(temp_config, f)
            logger.debug(f"Temporary config saved to {temp_config_path}")

            # Create output directory if it doesn't exist
            os.makedirs(self.cfg.output_dir, exist_ok=True)

            # Run the prediction process script
            self.run_pipeline(temp_config_path, input_file, top_N, detected_file_type)

            # Load results and update UI
            output_csv_path = os.path.join(
                self.cfg.output_dir, f"{self.cfg.save_file}_top{top_N}.csv"
            )
            output_npy_path = os.path.join(
                self.cfg.output_dir, f"{self.cfg.save_file}_top{top_N}.npy"
            )

            if os.path.exists(output_csv_path) and os.path.exists(output_npy_path):
                logger.info("Loading updated results from output files")
                df = pd.read_csv(output_csv_path)
                filenames = df["Filename"].values
                self.filenames = np.array([os.path.basename(str(f)) for f in filenames])
                self.scores = df["Score"].values

                # Load images using consistent format handling (same as load_top_files)
                imgs_data = np.load(output_npy_path)

                # Check if images are already in HWC format or need transpose from CHW
                if len(imgs_data.shape) == 4:
                    if imgs_data.shape[1] <= 4 and imgs_data.shape[3] > 4:  # Likely CHW format
                        logger.debug("Converting loaded images from CHW to HWC format")
                        self.img_catalog = imgs_data.transpose(0, 2, 3, 1)
                    else:  # Already in HWC format
                        logger.debug("Images already in HWC format")
                        self.img_catalog = imgs_data
                else:
                    # Handle unexpected formats gracefully
                    logger.warning(f"Unexpected image shape: {imgs_data.shape}, using as-is")
                    self.img_catalog = imgs_data

                # Ensure images are uint8 for consistency with load_images.py processing
                if self.img_catalog.dtype != np.uint8:
                    logger.debug(f"Converting images from {self.img_catalog.dtype} to uint8")
                    if self.img_catalog.max() <= 1.0:
                        self.img_catalog = (self.img_catalog * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        self.img_catalog = self.img_catalog.clip(0, 255).astype(np.uint8)

                # Notify UI that results are available for display
                if progress_callback:
                    progress_callback(
                        file_idx + 1,
                        num_files,
                        results_updated=True,
                    )

            else:
                logger.error(
                    "Output files not found. Prediction process might have failed. On Datalabs, the process may have exceeded the RAM allocation. Please check logs in the folder <anomaly_match/logs>."  # noqa: E501
                )

            # Log statistics
            if self.scores is not None and len(self.scores) > 0:
                logger.debug(
                    f"File {file_idx} processed, scores mean={np.mean(self.scores):.4f}, "
                    f"std={np.std(self.scores):.4f}, min={np.min(self.scores):.4f}, "
                    f"max={np.max(self.scores):.4f}"
                )

            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            processed_images += num_items

        # Calculate final statistics for entire run
        total_time = time.time() - start_time
        if total_time > 0 and processed_images > 0:
            final_speed = processed_images / total_time
            time_str = str(datetime.timedelta(seconds=int(total_time)))
            final_message = (
                f"Completed processing {processed_images:,} images in {time_str}"
                f"\nFinal average speed: {final_speed:.1f} images/sec"
            )
            logger.success(final_message)

            # Provide final time information to UI if callback exists
            if progress_callback:
                progress_callback(
                    num_files,
                    num_files,
                    batch_update=True,
                    completed=True,
                    total_time_str=time_str,
                    final_speed=final_speed,
                )
        else:
            logger.warning("No images were processed or processing time was too short")

        logger.info(f"Processed {num_files} files with {detected_file_type} format")
        if self.scores is not None:
            logger.debug(f"Total images scored: {len(self.scores)}")
        else:
            logger.warning("No scores were loaded - all prediction subprocesses may have failed")

    def load_top_files(self, top_N):
        """Loads the top files from the output directory using consistent image processing."""
        output_csv_path = os.path.join(self.cfg.output_dir, f"{self.cfg.save_file}_top{top_N}.csv")
        output_npy_path = os.path.join(self.cfg.output_dir, f"{self.cfg.save_file}_top{top_N}.npy")

        if os.path.exists(output_csv_path) and os.path.exists(output_npy_path):
            logger.info("Loading updated results from output files")
            df = pd.read_csv(output_csv_path)
            filenames = df["Filename"].values
            # Convert to basename (str() handles cutana int64 source_ids)
            self.filenames = np.array([os.path.basename(str(f)) for f in filenames])
            self.scores = df["Score"].values

            # Load images using consistent format handling
            imgs_data = np.load(output_npy_path)

            # Check if images are already in HWC format or need transpose from CHW
            if len(imgs_data.shape) == 4:
                if imgs_data.shape[1] <= 4 and imgs_data.shape[3] > 4:  # Likely CHW format
                    logger.debug("Converting loaded images from CHW to HWC format")
                    self.img_catalog = imgs_data.transpose(0, 2, 3, 1)
                else:  # Already in HWC format
                    logger.debug("Images already in HWC format")
                    self.img_catalog = imgs_data
            else:
                # Handle unexpected formats gracefully
                logger.warning(f"Unexpected image shape: {imgs_data.shape}, using as-is")
                self.img_catalog = imgs_data

            # Ensure images are uint8 for consistency with load_images.py processing
            if self.img_catalog.dtype != np.uint8:
                logger.debug(f"Converting images from {self.img_catalog.dtype} to uint8")
                if self.img_catalog.max() <= 1.0:
                    self.img_catalog = (self.img_catalog * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    self.img_catalog = self.img_catalog.clip(0, 255).astype(np.uint8)

            logger.info(
                f"Top {len(self.scores)} filenames and scores collected with mean,std"
                + f" = {np.mean(self.scores)}, {np.std(self.scores)}"
            )
            logger.debug(f"In total scored {len(self.scores)} images")
            logger.debug(
                f"Image catalog shape: {self.img_catalog.shape}, dtype: {self.img_catalog.dtype}"
            )
        else:
            logger.error(
                f"Output files not found at {output_csv_path} and {output_npy_path}. \n Note that you may need to rename the"
                + "output files from the folder anomaly_match_results to top1000.csv and top1000.npy."
                + " (This is to avoid accidental overwriting of results)"
            )

    def get_session_info(self):
        """Get session information from the session tracker."""
        return self.session_tracker.get_session_info()

    def get_iteration_info(self, iteration_number=None):
        """Get iteration information from the session tracker."""
        return self.session_tracker.get_iteration_info(iteration_number)

    def save_session(self):
        """Save the complete session using SessionIOHandler."""
        return self.session_io.save_session(self.session_tracker, cfg=self.cfg)

    def _auto_detect_prediction_file_type(self, search_dir):
        """Auto-detect prediction file type based on files in the directory."""
        if not search_dir or not os.path.exists(search_dir):
            logger.warning(
                f"Search directory {search_dir} does not exist, defaulting to 'image' file type"
            )
            return "image"

        # Define supported file extensions
        extension_map = {
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".zarr": "zarr",
            ".jpg": "image",
            ".jpeg": "image",
            ".png": "image",
            ".tif": "image",
            ".tiff": "image",
            ".fits": "image",
            ".csv": "stream",
            ".parquet": "stream",
        }

        tracked_extenstions = {key: 0 for key in extension_map.keys()}

        # Count files by type
        file_type_counts = {}
        for filename in os.listdir(search_dir):
            file_path = os.path.join(search_dir, filename)

            # Check if it's a file with supported extension
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in extension_map:
                    tracked_extenstions[ext] += 1
                    file_type = extension_map[ext]
                    file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1

            # Check if it's a zarr directory (zarr stores can be directories)
            elif os.path.isdir(file_path):
                # Check for direct zarr store (ends with .zarr or has zarr.json)
                if filename.lower().endswith(".zarr") or os.path.exists(
                    os.path.join(file_path, "zarr.json")
                ):
                    file_type_counts["zarr"] = file_type_counts.get("zarr", 0) + 1
                # Check for batch folders containing images.zarr subdirectory
                elif os.path.exists(os.path.join(file_path, "images.zarr")):
                    file_type_counts["zarr"] = file_type_counts.get("zarr", 0) + 1

        if not file_type_counts:
            logger.warning(
                f"No supported files found in {search_dir}, defaulting to 'image' file type"
            )
            return "image"

        # Return the most common file type
        detected_type = max(file_type_counts, key=file_type_counts.get)

        logger.debug(
            f"Auto-detected prediction file type: {detected_type} (found {file_type_counts[detected_type]} files)"
        )

        return detected_type

    def _rebuild_label_cache(self):
        """Rebuilds the label cache from active_learning_df for fast lookups."""
        self._label_cache = {}
        self._label_distribution_cache = None  # Invalidate distribution cache
        self._active_learning_counts_cache = None  # Invalidate counts cache
        if self.active_learning_df is not None and not self.active_learning_df.empty:
            for _, row in self.active_learning_df.iterrows():
                self._label_cache[row["filename"]] = row["label"]
