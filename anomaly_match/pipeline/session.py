#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import os
import datetime
import time

from loguru import logger
import torch
import pandas as pd
import numpy as np
from contextlib import nullcontext
import subprocess
import sys
import zipfile
import toml
import h5py

from anomaly_match.datasets.SSL_Dataset import SSL_Dataset
from anomaly_match.datasets.data_utils import get_prediction_dataloader
from anomaly_match.models.FixMatch import FixMatch
from anomaly_match.utils.print_cfg import print_cfg
from anomaly_match.utils.set_log_level import set_log_level
from anomaly_match.utils.get_net_builder import get_net_builder
from anomaly_match.utils.get_optimizer import get_optimizer


class Session:
    """Tracks a session of using anomaly_match and its state."""

    labeled_train_dataset = None
    unlabeled_train_dataset = None
    test_dataset = None
    prediction_dataset = None

    widget = None
    model: FixMatch = None

    active_learning_df = pd.DataFrame(columns=["filename", "label"])

    filenames = None
    scores = None
    img_catalog = None

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
        self.cfg = cfg
        self.out = None  # Initialize out attribute to None
        logger.debug("Session initialized, loading datasets")
        self._load_datasets()
        logger.debug("Datasets loaded, initializing model")
        self._init_model()
        self.top_N_filenames_scores = []
        self.eval_predictions = {}  # Initialize empty dict for eval predictions

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
            hard_label=True,
            logger=logger,
        )

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
            self.cfg, self.labeled_train_dataset, self.unlabeled_train_dataset, self.test_dataset
        )

    def _load_datasets(self):
        """Loads the datasets required for training and evaluation."""
        # Construct Dataset
        self.train_dset = SSL_Dataset(
            test_ratio=self.cfg.test_ratio,
            N_to_load=self.cfg.N_to_load,
            train=True,
            data_dir=self.cfg.data_dir,
            seed=self.cfg.seed,
            size=self.cfg.size,
            label_file=self.cfg.label_file,
        )
        self.labeled_train_dataset, self.unlabeled_train_dataset = self.train_dset.get_ssl_dset()

        self.cfg.num_classes = self.train_dset.num_classes
        self.cfg.num_channels = self.train_dset.num_channels

        if self.cfg.test_ratio > 0:
            self.test_dataset = SSL_Dataset(
                test_ratio=self.cfg.test_ratio,
                N_to_load=self.cfg.N_to_load,
                train=False,
                data_dir=self.cfg.data_dir,
                seed=self.cfg.seed,
                size=self.cfg.size,
                label_file=self.cfg.label_file,
            ).get_dset()
        else:
            self.test_dataset = None

        self.prediction_dataloader = get_prediction_dataloader(
            self.train_dset.dset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def update_predictions(self):
        """Updates the predictions using the current model and datasets."""
        with self.out if self.out is not None else nullcontext():
            logger.debug("Updating predictions")
            self.prediction_dataloader = get_prediction_dataloader(
                self.train_dset.dset,
                batch_size=self.cfg.eval_batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
            )

            def progress_callback(current, total):
                if hasattr(self.cfg, "progress_bar") and self.cfg.progress_bar is not None:
                    self.cfg.progress_bar.value = current / total

            if self.widget is not None:
                self.widget.ui["train_label"].value = "Updating predictions..."
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
                if self.widget is not None:
                    self.widget.ui["train_label"].value = "Evaluating model..."
                self.eval_performance = self.model.evaluate(
                    cfg=self.cfg,
                    progress_callback=lambda current, total: progress_callback(current, total),
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
            filepath = os.path.join(self.cfg.output_dir, "labeled_data.csv")
            logger.info(f"Saving labels to {filepath}")

            # Make outfolder if it doesn't exist
            if not os.path.exists(self.cfg.output_dir):
                os.makedirs(self.cfg.output_dir)

            # Combine active_learning_df with already labeled data in the dataset
            labeled_data = [
                {"filename": filename, "label": "normal" if target == 0 else "anomaly"}
                for filename, target in zip(
                    self.labeled_train_dataset.filenames, self.labeled_train_dataset.targets
                )
            ]

            labeled_df = pd.DataFrame(labeled_data)

            combined_df = pd.concat([labeled_df, self.active_learning_df]).drop_duplicates(
                subset="filename", keep="last"
            )

            combined_df.to_csv(filepath, index=False)

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

    def remember_current_file(self, filename):
        """Remembers the current file by appending it to a CSV if not already present."""
        with self.out if self.out is not None else nullcontext():
            # Ensure output directory exists before trying to write file
            os.makedirs(self.cfg.output_dir, exist_ok=True)

            # Use cfg.name instead of cfg.save_file for the output filename
            output_file = os.path.join(
                self.cfg.output_dir, f"{self.cfg.name}_{self.session_start}_remembered_files.csv"
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
        """Saves the current model state."""
        with self.out if self.out is not None else nullcontext():
            self.model.save_model(self.cfg)

    def load_model(self):
        """Loads the model state from the configuration."""
        with self.out if self.out is not None else nullcontext():
            self.model.load_model(self.cfg)

    def train(self, cfg, progress_callback=None):
        """Trains the model using the given configuration.

        Args:
            cfg (DotMap): Configuration for training.
            progess_callback (function, optional): Callback function to update progress. Defaults to None.
        """
        self.cfg = cfg
        self.top_N_filenames_scores = []  # Clear top N filenames and scores
        with self.out if self.out is not None else nullcontext():
            self.save_labels()
            # Update the datasets
            self.labeled_train_dataset, self.unlabeled_train_dataset = self.train_dset.update_dsets(
                self.active_learning_df, N_to_load=self.cfg.N_to_load
            )
            self.model.set_data_loader(
                self.cfg,
                self.labeled_train_dataset,
                self.unlabeled_train_dataset,
                self.test_dataset,
            )
            self.model.train(cfg, progress_callback=progress_callback)
            logger.info("Training complete.")
            self.model.save_run(cfg.save_file, cfg.save_path, cfg=None)

    def get_label_distribution(self):
        """Gets the distribution of labels in the training dataset, including new labels in active_learning_df.

        Returns:
            tuple: A tuple containing the count of normal and anomalous labels.
        """
        normal_count = torch.sum(self.labeled_train_dataset.targets == 0)
        anomalous_count = len(self.labeled_train_dataset.targets) - normal_count
        if self.active_learning_df is not None:
            new_labels = self.active_learning_df[
                ~self.active_learning_df["filename"].isin(self.labeled_train_dataset.data)
            ]
            normal_count += np.sum(new_labels["label"] == "normal")
            anomalous_count += np.sum(new_labels["label"] == "anomaly")

        return normal_count, anomalous_count

    def start_UI(self):
        """Starts the user interface for the session."""
        from anomaly_match.ui.Widget import Widget

        if self.widget is None:
            logger.info("Starting new UI... (this may compute furiously for a few seconds)")
            self.widget = Widget(self)
            self.widget.start()
        else:
            logger.debug("UI already running, restarting")
            self.widget.start()

    def get_label(self, idx):
        """Gets the label for the image at the given index.

        Args:
            idx (int): Index of the image.

        Returns:
            str: The label of the image.
        """
        if self.active_learning_df is None:
            return "None"
        elif self.filenames[idx] not in self.active_learning_df["filename"].values:
            return "None"
        return self.active_learning_df.loc[
            self.active_learning_df["filename"] == self.filenames[idx], "label"
        ].values[0]

    def load_next_batch(self):
        """Loads the next batch of data and updates predictions."""
        logger.debug("Loading next batch of data")
        self.top_N_filenames_scores = []  # Clear top N filenames and scores
        # Note that we are updating also the labeled_dataset since the unlabeled
        # data are going to disappear from the unlabeled dataset once we call this function.
        self.train_dset.update_dsets(
            label_update=self.active_learning_df, N_to_load=self.cfg.N_to_load
        )
        self.update_predictions()

    def reset_model(self):
        """Resets the model and reinitializes the session."""
        logger.debug("Resetting model")
        self._init_model()
        self.update_predictions()

    def run_pipeline(self, temp_config_path, input_path, top_N):
        """Run the appropriate pipeline subprocess based on file type."""
        script_map = {
            "zip": "prediction_process_zip.py",
            "hdf5": "prediction_process_hdf5.py",
            "image": "prediction_process.py",
        }

        script = script_map.get(self.cfg.prediction_file_type)
        if not script:
            raise ValueError(f"Unsupported prediction file type: {self.cfg.prediction_file_type}")

        # Get the directory two levels up from this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get session.py directory
        root_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
        script_path = os.path.join(root_dir, script)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found at expected path: {script_path}")

        # For image directories, we need to create a temporary file list
        if self.cfg.prediction_file_type == "image":
            # Create a temporary file containing the list of image paths
            temp_file_list = os.path.join("tmp", f"{self.cfg.save_file}_file_list.txt")
            with open(temp_file_list, "w") as f:
                f.write(input_path)

            # Call prediction_process.py with the file list
            subprocess.run(
                [sys.executable, script_path, temp_config_path, temp_file_list, str(top_N)]
            )
        else:
            # For zip and hdf5 files, pass the file path directly
            subprocess.run([sys.executable, script_path, temp_config_path, input_path, str(top_N)])

        # Reset logger to old level
        set_log_level(self.cfg.log_level, self.cfg)

    def evaluate_all_images(self, top_N=1000, progress_callback=None):
        """Evaluates all images and updates the session's img_catalog with the top N images."""
        logger.info("Evaluating all images")

        # Check if model exists before proceeding
        if not os.path.exists(self.cfg.model_path):
            error_msg = (
                f"Model not found at {self.cfg.model_path}. "
                "Please train and save a model before running predictions."
            )
            logger.error(error_msg)
            if self.widget is not None:
                self.widget.ui["train_label"].value = "Error: Model not found!"
            raise FileNotFoundError(error_msg)

        # Define supported file extensions
        supported_extensions = {
            "zip": [".zip"],
            "hdf5": [".h5", ".hdf5"],
            "image": [".jpg", ".jpeg", ".png", ".tif", ".tiff"],
        }

        pattern = supported_extensions.get(self.cfg.prediction_file_type)
        if not pattern:
            raise ValueError(f"Unsupported prediction file type: {self.cfg.prediction_file_type}")

        # Get all matching files from the cfg.search_dir
        input_files = []
        for f in os.listdir(self.cfg.search_dir):
            file_ext = os.path.splitext(f.lower())[1]
            if file_ext in pattern:
                input_files.append(os.path.join(self.cfg.search_dir, f))

        num_files = len(input_files)
        total_images = 0
        processed_images = 0
        start_time = time.time()

        # First count total images
        logger.info("Counting total images to process...")
        for input_file in input_files:
            try:
                if self.cfg.prediction_file_type == "zip":
                    with zipfile.ZipFile(input_file, "r") as zip_file:
                        total_images += len(
                            [
                                f
                                for f in zip_file.namelist()
                                if any(
                                    f.lower().endswith(ext)
                                    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
                                )
                            ]
                        )
                elif self.cfg.prediction_file_type == "hdf5":
                    with h5py.File(input_file, "r") as h5f:
                        total_images += len(h5f["images"])
                else:  # jpeg/image files - single file
                    total_images += 1
            except Exception as e:
                logger.warning(f"Error counting images in {input_file}: {str(e)}")

        logger.info(f"Found total of {total_images:,} images to process in {num_files} files")

        for file_idx, input_file in enumerate(input_files):
            # Get number of images in current file
            if self.cfg.prediction_file_type == "zip":
                with zipfile.ZipFile(input_file, "r") as zip_file:
                    num_items = len(
                        [
                            f
                            for f in zip_file.namelist()
                            if any(
                                f.lower().endswith(ext)
                                for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
                            )
                        ]
                    )
            elif self.cfg.prediction_file_type == "hdf5":
                with h5py.File(input_file, "r") as h5f:
                    num_items = len(h5f["images"])
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

            temp_config_path = os.path.join("tmp", f"{self.cfg.save_file}_config.toml")
            # Make tmp folder if it doesn't exist
            os.makedirs("tmp", exist_ok=True)
            with open(temp_config_path, "w") as f:
                toml.dump(temp_config, f)

            # Run the prediction process script
            self.run_pipeline(temp_config_path, input_file, top_N)

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
                self.filenames = np.array([os.path.basename(f) for f in filenames])
                self.scores = df["Score"].values
                self.img_catalog = np.load(output_npy_path).transpose(0, 2, 3, 1)

                # Update UI if available
                if self.widget is not None:
                    self.widget.display_top_files_scores()
            else:
                logger.error(
                    "Output files not found. Prediction process might have failed. Please check logs in the folder <anomaly_match/logs>."  # noqa: E501
                )

            # Log statistics
            if len(self.scores) > 0:
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

        logger.info(f"Processed {num_files} files with {self.cfg.prediction_file_type} format")
        logger.debug(f"Total images scored: {len(self.scores)}")

    def load_top_files(self):
        """Loads the top files from the output directory."""
        output_csv_path = os.path.join(self.cfg.output_dir, "top1000.csv")
        output_npy_path = os.path.join(self.cfg.output_dir, "top1000.npy")

        if os.path.exists(output_csv_path) and os.path.exists(output_npy_path):
            logger.info("Loading updated results from output files")
            df = pd.read_csv(output_csv_path)
            filenames = df["Filename"].values
            # Convert to basename
            self.filenames = np.array([os.path.basename(f) for f in filenames])
            self.scores = df["Score"].values

            imgs_data = np.load(output_npy_path)

            self.img_catalog = imgs_data.transpose(0, 2, 3, 1)

            logger.info(
                f"Top {len(self.scores)} filenames and scores collected with mean,std"
                + f" = {np.mean(self.scores)}, {np.std(self.scores)}"
            )
            logger.debug(f"In total scored {len(self.scores)} images")

            # Call Widget's display_top_files_scores to update the UI
            if self.widget is not None:
                logger.debug("Displaying top files and scores")
                self.widget.display_top_files_scores()
        else:
            logger.error(
                f"Output files not found at {output_csv_path} and {output_npy_path}. \n Note that you may need to rename the"
                + "output files from the folder anomaly_match_results to top1000.csv and top1000.npy."
                + " (This is to avoid accidental overwriting of results)"
            )
