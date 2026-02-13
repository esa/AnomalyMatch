#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class IterationInfo:
    """Data class to store information about a single training iteration."""

    iteration_number: int
    timestamp: str
    model_loss: Optional[float] = None
    test_performance: Optional[Dict[str, float]] = None
    model_state_path: Optional[str] = None
    num_newly_labeled_anomalous: int = 0
    num_newly_labeled_nominal: int = 0
    # Per-sample scores for this iteration (stored separately as CSV files)
    unlabelled_scores_file: Optional[str] = None
    test_scores_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert IterationInfo to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp,
            "model_loss": self.model_loss,
            "test_performance": self.test_performance,
            "model_state_path": self.model_state_path,
            "num_newly_labeled_anomalous": self.num_newly_labeled_anomalous,
            "num_newly_labeled_nominal": self.num_newly_labeled_nominal,
            "unlabelled_scores_file": self.unlabelled_scores_file,
            "test_scores_file": self.test_scores_file,
        }


class SessionTracker:
    """
    Tracks important session metrics during AnomalyMatch training and evaluation.

    This class maintains a record of:
    - Session iterations and their timestamps
    - Model training iterations and losses
    - Labeled samples (anomalous/nominal) per session iteration
    - Test performance metrics
    - Model states and configurations
    """

    def __init__(self, session_name: str = None):
        """
        Initialize the SessionTracker.

        Args:
            session_name: Optional name for the session. If None, uses timestamp.
        """
        self.session_start_time = datetime.datetime.now()
        self.session_name = (
            session_name or f"session_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}"
        )

        # Session-level tracking
        self.session_iterations: List[IterationInfo] = []
        self.current_session_iteration = 0

        # Model-level tracking
        self.total_model_iterations = 0

        # Configuration and data tracking
        self.labeled_data_df: pd.DataFrame = pd.DataFrame(
            columns=["filename", "label", "iteration"]
        )

        logger.debug(f"Initialized SessionTracker for session: {self.session_name}")

    def start_new_session_iteration(self) -> int:
        """
        Start a new session iteration.

        Returns:
            int: The session iteration number that was started.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        iteration_info = IterationInfo(
            iteration_number=self.current_session_iteration, timestamp=timestamp
        )
        self.session_iterations.append(iteration_info)

        logger.debug(f"Started session iteration {self.current_session_iteration}")
        result = self.current_session_iteration
        self.current_session_iteration += 1
        return result

    def update_model_iteration(self, loss: float = None, num_iterations: int = 1) -> None:
        """
        Update model iteration count and optionally record loss.

        Args:
            loss: Optional loss value for this iteration.
            num_iterations: Number of iterations to add (default 1).
        """
        self.total_model_iterations += num_iterations

        # Update current session iteration if it exists and loss is provided
        if self.session_iterations and loss is not None:
            self.session_iterations[-1].model_loss = loss

    def add_labeled_sample(self, filename: str, label: str, iteration_number: int = None) -> None:
        """
        Add a labeled sample to the current session iteration.

        Args:
            filename: Name of the labeled file.
            label: Label assigned ('anomaly' or 'normal').
            iteration_number: Optional specific iteration number. If None, uses current iteration.
        """
        # Determine current iteration number
        if iteration_number is not None:
            current_iter_num = iteration_number
        elif self.session_iterations:
            # Use the most recent active iteration (last one in the list)
            current_iter_num = self.session_iterations[-1].iteration_number
        else:
            # No session iterations yet, this is initial/pre-training data
            current_iter_num = -1

        # Add to overall labeled data with iteration info
        new_row = pd.DataFrame(
            {"filename": [filename], "label": [label], "iteration": [current_iter_num]}
        )
        self.labeled_data_df = pd.concat([self.labeled_data_df, new_row], ignore_index=True)

        # Update counts for current session iteration (only for active iterations, not initial data)
        if self.session_iterations and current_iter_num >= 0:
            # Find the correct iteration to update
            for iteration in self.session_iterations:
                if iteration.iteration_number == current_iter_num:
                    if label == "anomaly":
                        iteration.num_newly_labeled_anomalous += 1
                    elif label == "normal":
                        iteration.num_newly_labeled_nominal += 1
                    break

        logger.debug(f"Added labeled sample: {filename} -> {label} (iteration {current_iter_num})")

    def update_test_performance(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update test performance for the current session iteration.

        Args:
            performance_metrics: Dictionary of performance metrics (e.g., accuracy, AUC).
        """
        if self.session_iterations:
            self.session_iterations[-1].test_performance = performance_metrics
            logger.debug(f"Updated test performance: {performance_metrics}")

    def update_model_state_path(self, model_path: str) -> None:
        """
        Update the model state path for the current session iteration.

        Args:
            model_path: Path to the saved model state.
        """
        if self.session_iterations:
            self.session_iterations[-1].model_state_path = model_path
            logger.debug(f"Updated model state path: {model_path}")

    def update_unlabelled_scores_path(self, scores_path: str) -> None:
        """
        Update the unlabelled scores file path for the current session iteration.

        Args:
            scores_path: Path to the saved unlabelled scores CSV file.
        """
        if self.session_iterations:
            self.session_iterations[-1].unlabelled_scores_file = scores_path
            logger.debug(f"Updated unlabelled scores path: {scores_path}")

    def update_test_scores_path(self, scores_path: str) -> None:
        """
        Update the test scores file path for the current session iteration.

        Args:
            scores_path: Path to the saved test scores CSV file.
        """
        if self.session_iterations:
            self.session_iterations[-1].test_scores_file = scores_path
            logger.debug(f"Updated test scores path: {scores_path}")

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get comprehensive session information.

        Returns:
            Dict containing session-level information.
        """
        # Count all labeled samples including initial data (iteration = -1)
        total_anomalous = len(self.labeled_data_df[self.labeled_data_df["label"] == "anomaly"])
        total_nominal = len(self.labeled_data_df[self.labeled_data_df["label"] == "normal"])
        total_labeled = len(self.labeled_data_df[self.labeled_data_df["label"] != "removed"])
        total_removed = len(self.labeled_data_df[self.labeled_data_df["label"] == "removed"])

        # Handle case where iteration column might not exist (legacy data)
        if "iteration" in self.labeled_data_df.columns:
            # Count initial samples (iteration = -1)
            initial_samples = len(self.labeled_data_df[self.labeled_data_df["iteration"] == -1])
            # Count samples labeled during iterations (iteration >= 0)
            iteration_samples = len(self.labeled_data_df[self.labeled_data_df["iteration"] >= 0])
        else:
            # If no iteration column, treat all existing data as initial
            initial_samples = total_labeled
            iteration_samples = 0

        return {
            "session_name": self.session_name,
            "session_start_time": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_session_iterations": len(self.session_iterations),
            "total_model_iterations": self.total_model_iterations,
            "total_anomalous_samples": total_anomalous,
            "total_nominal_samples": total_nominal,
            "total_labeled_samples": total_labeled,
            "total_removed_samples": total_removed,
            "initial_labeled_samples": initial_samples,
            "iteration_labeled_samples": iteration_samples,
            "session_duration_minutes": (
                datetime.datetime.now() - self.session_start_time
            ).total_seconds()
            / 60,
        }

    def get_iteration_info(self, iteration_number: int = None) -> Dict[str, Any]:
        """
        Get information about a specific iteration or the latest one.

        Args:
            iteration_number: Specific iteration to get info for. If None, returns latest.

        Returns:
            Dict containing iteration information.
        """
        if not self.session_iterations:
            return {}

        if iteration_number is None:
            iteration = self.session_iterations[-1]
        else:
            matching_iterations = [
                it for it in self.session_iterations if it.iteration_number == iteration_number
            ]
            if not matching_iterations:
                logger.warning(f"No iteration found with number {iteration_number}")
                return {}
            iteration = matching_iterations[0]

        return iteration.to_dict()

    def get_all_iterations_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all session iterations.

        Returns:
            List of dictionaries containing iteration information, ordered by iteration number (newest first).
        """
        # Return iterations in reverse order (newest first)
        return [iteration.to_dict() for iteration in reversed(self.session_iterations)]

    def get_labeled_data_df(self) -> pd.DataFrame:
        """
        Get the current labeled data DataFrame.

        Returns:
            DataFrame with labeled samples.
        """
        return self.labeled_data_df.copy()

    def update_labeled_data(self, labeled_data_df: pd.DataFrame) -> None:
        """
        Update the complete labeled dataset, preserving existing iteration information.
        This method merges new labeled data with existing tracked samples.

        Args:
            labeled_data_df: DataFrame containing labeled data (may or may not have iteration column)
        """
        labeled_df_copy = labeled_data_df.copy()

        # Ensure the new data has an iteration column
        if "iteration" not in labeled_df_copy.columns:
            labeled_df_copy["iteration"] = -1  # Default for data without iteration info

        # If we have no existing data, just use the new data
        if self.labeled_data_df.empty:
            self.labeled_data_df = labeled_df_copy
            logger.debug(f"Initialized labeled data with {len(labeled_df_copy)} samples")
            return

        # Merge new data with existing data, preserving iteration information
        existing_df = self.labeled_data_df.copy()

        # For each sample in the new data, check if it already exists
        for idx, new_row in labeled_df_copy.iterrows():
            filename = new_row["filename"]

            # Check if this filename exists in our current data
            existing_mask = existing_df["filename"] == filename

            if existing_mask.any():
                # File exists, preserve the existing iteration number
                existing_iteration = existing_df.loc[existing_mask, "iteration"].iloc[0]
                labeled_df_copy.loc[idx, "iteration"] = existing_iteration
                logger.debug(
                    f"Preserved iteration {existing_iteration} for existing file: {filename}"
                )
            else:
                # New file, keep the iteration from new data (or -1 if not specified)
                logger.debug(
                    f"Added new file {filename} with iteration {labeled_df_copy.loc[idx, 'iteration']}"
                )

        # Now merge the dataframes, using the new data to update existing entries
        # and adding any new entries that weren't in the existing data
        combined_df = pd.concat([existing_df, labeled_df_copy], ignore_index=True)

        # Remove duplicates, keeping the last occurrence (which prioritizes new data for updates)
        combined_df = combined_df.drop_duplicates(subset="filename", keep="last")

        # Replace the labeled data
        self.labeled_data_df = combined_df
        logger.debug(f"Updated labeled data: {len(combined_df)} samples total")

        # Debug: Show iteration distribution
        if "iteration" in self.labeled_data_df.columns:
            iter_counts = self.labeled_data_df["iteration"].value_counts().sort_index()
            logger.debug(f"Iteration distribution after merge: {dict(iter_counts)}")
