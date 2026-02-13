#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
BackendInterface: Single interface for UI-backend communication.

This module provides a static interface for the UI components to interact with
the backend Session without direct imports, enabling clean separation between
the UI and backend packages.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import anomaly_match


class BackendInterface:
    """Single interface for UI-backend communication.

    This static class wraps all Session methods needed by the UI components,
    providing a clean interface that decouples the UI from the backend implementation.
    """

    _session = None

    # ========== Session Lifecycle ==========

    @staticmethod
    def set_session(session) -> None:
        """Set the session instance to use.

        Args:
            session: The Session instance to interact with.
        """
        BackendInterface._session = session

    @staticmethod
    def get_session():
        """Get the current session instance.

        Returns:
            The current Session instance, or None if not set.
        """
        return BackendInterface._session

    @staticmethod
    def _check_session() -> None:
        """Check that a session is set, raise RuntimeError if not."""
        if BackendInterface._session is None:
            raise RuntimeError("No session set. Call BackendInterface.set_session() first.")

    # ========== Configuration ==========

    @staticmethod
    def get_config():
        """Get the session configuration.

        Returns:
            The session configuration object (DotMap).
        """
        BackendInterface._check_session()
        return BackendInterface._session.cfg

    @staticmethod
    def get_num_channels() -> int:
        """Get the number of channels in the images.

        Returns:
            int: Number of image channels.
        """
        BackendInterface._check_session()
        return getattr(BackendInterface._session.cfg, "num_channels", 3)

    @staticmethod
    def set_normalisation_method(method) -> None:
        """Set the normalisation method.

        Args:
            method (NormalisationMethod): The new normalisation method to apply.
        """
        BackendInterface._check_session()
        BackendInterface._session.set_normalisation_method(method)

    @staticmethod
    def get_cached_normalisation_method():
        """Get the cached normalisation method (what images are currently loaded with).

        Returns:
            NormalisationMethod: The cached normalisation method enum value.
        """
        BackendInterface._check_session()
        return BackendInterface._session.cached_image_normalisation_enum

    # ========== Image Data ==========

    @staticmethod
    def get_image_at_index(index: int) -> np.ndarray:
        """Get the image at the given index from the catalog.

        Args:
            index (int): The index of the image.

        Returns:
            np.ndarray: The image array.
        """
        BackendInterface._check_session()
        return BackendInterface._session.img_catalog[index]

    @staticmethod
    def get_image_count() -> int:
        """Get the total number of images in the catalog.

        Returns:
            int: Number of images.
        """
        BackendInterface._check_session()
        if BackendInterface._session.img_catalog is None:
            return 0
        return len(BackendInterface._session.img_catalog)

    @staticmethod
    def get_scores() -> np.ndarray:
        """Get the anomaly scores for all images.

        Returns:
            np.ndarray: Array of anomaly scores.
        """
        BackendInterface._check_session()
        return BackendInterface._session.scores

    @staticmethod
    def get_filenames() -> np.ndarray:
        """Get the filenames for all images.

        Returns:
            np.ndarray: Array of filenames.
        """
        BackendInterface._check_session()
        return BackendInterface._session.filenames

    # ========== Labeling ==========

    @staticmethod
    def label_image(index: int, label: str) -> None:
        """Label an image at the given index.

        Args:
            index (int): Index of the image to label.
            label (str): Label to assign ("normal" or "anomaly").
        """
        BackendInterface._check_session()
        BackendInterface._session.label_image(index, label)

    @staticmethod
    def unlabel_image(index: int) -> None:
        """Remove the label from an image at the given index.

        Args:
            index (int): Index of the image to unlabel.
        """
        BackendInterface._check_session()
        BackendInterface._session.unlabel_image(index)

    @staticmethod
    def get_label(index: int) -> str:
        """Get the label for an image at the given index.

        Args:
            index (int): Index of the image.

        Returns:
            str: The label ("normal", "anomaly", or "None").
        """
        BackendInterface._check_session()
        return BackendInterface._session.get_label(index)

    @staticmethod
    def get_label_distribution() -> Tuple[int, int]:
        """Get the distribution of labels.

        Returns:
            tuple: (normal_count, anomalous_count)
        """
        BackendInterface._check_session()
        return BackendInterface._session.get_label_distribution()

    @staticmethod
    def get_active_learning_counts() -> Tuple[int, int]:
        """Get the count of newly annotated samples in active learning.

        Returns:
            tuple: (new_normal_count, new_anomalous_count)
        """
        BackendInterface._check_session()
        return BackendInterface._session.get_active_learning_counts()

    @staticmethod
    def save_labels() -> None:
        """Save the current labels to a file."""
        BackendInterface._check_session()
        BackendInterface._session.save_labels()

    # ========== Sorting ==========

    @staticmethod
    def sort_by_anomalous() -> None:
        """Sort images by anomalous scores (most anomalous first)."""
        BackendInterface._check_session()
        BackendInterface._session.sort_by_anomalous()

    @staticmethod
    def sort_by_nominal() -> None:
        """Sort images by nominal scores (most nominal first)."""
        BackendInterface._check_session()
        BackendInterface._session.sort_by_nominal()

    @staticmethod
    def sort_by_mean() -> None:
        """Sort images by distance to mean score."""
        BackendInterface._check_session()
        BackendInterface._session.sort_by_mean()

    @staticmethod
    def sort_by_median() -> None:
        """Sort images by distance to median score."""
        BackendInterface._check_session()
        BackendInterface._session.sort_by_median()

    # ========== Model Operations ==========

    @staticmethod
    def train(cfg, progress_callback: Optional[Callable] = None) -> None:
        """Train the model.

        Args:
            cfg: Configuration for training.
            progress_callback: Optional callback for progress updates.
        """
        BackendInterface._check_session()
        BackendInterface._session.train(cfg, progress_callback=progress_callback)

    @staticmethod
    def save_model() -> None:
        """Save the current model state."""
        BackendInterface._check_session()
        BackendInterface._session.save_model()

    @staticmethod
    def load_model() -> None:
        """Load the model from the saved state."""
        BackendInterface._check_session()
        BackendInterface._session.load_model()

    @staticmethod
    def reset_model() -> None:
        """Reset the model and reinitialize."""
        BackendInterface._check_session()
        BackendInterface._session.reset_model()

    @staticmethod
    def get_model():
        """Get the current model instance.

        Returns:
            The FixMatch model instance.
        """
        BackendInterface._check_session()
        return BackendInterface._session.model

    # ========== Prediction/Evaluation ==========

    @staticmethod
    def update_predictions(progress_callback: Optional[Callable] = None) -> None:
        """Update predictions using the current model.

        Args:
            progress_callback: Optional callback for progress updates.
        """
        BackendInterface._check_session()
        BackendInterface._session.update_predictions(progress_callback=progress_callback)

    @staticmethod
    def evaluate_all_images(top_n: int, progress_callback: Optional[Callable] = None) -> None:
        """Evaluate all images in the prediction search directory.

        Args:
            top_n (int): Number of top images to keep.
            progress_callback: Optional callback for progress updates.
        """
        BackendInterface._check_session()
        BackendInterface._session.evaluate_all_images(
            top_N=top_n, progress_callback=progress_callback
        )

    @staticmethod
    def load_next_batch() -> None:
        """Load the next batch of data and update predictions."""
        BackendInterface._check_session()
        BackendInterface._session.load_next_batch()

    @staticmethod
    def load_top_files(top_n: int) -> None:
        """Load the top files from the output directory.

        Args:
            top_n (int): Number of top files to load.
        """
        BackendInterface._check_session()
        BackendInterface._session.load_top_files(top_n)

    @staticmethod
    def get_eval_performance() -> Optional[Dict[str, Any]]:
        """Get the evaluation performance metrics.

        Returns:
            dict: Evaluation performance metrics, or None if not available.
        """
        BackendInterface._check_session()
        return getattr(BackendInterface._session, "eval_performance", None)

    # ========== Utilities ==========

    @staticmethod
    def set_terminal_output(output_widget) -> None:
        """Set the terminal output widget for logging.

        Args:
            output_widget: The output widget for terminal logging.
        """
        BackendInterface._check_session()
        BackendInterface._session.set_terminal_out(output_widget)

    @staticmethod
    def remember_current_file(filename: str) -> None:
        """Remember the current file by appending it to a CSV.

        Args:
            filename (str): The filename to remember.
        """
        BackendInterface._check_session()
        BackendInterface._session.remember_current_file(filename)

    @staticmethod
    def get_version() -> str:
        """Get the anomaly_match version.

        Returns:
            str: The version string.
        """
        return anomaly_match.__version__
