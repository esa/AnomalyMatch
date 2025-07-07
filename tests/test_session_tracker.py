#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import pytest
import datetime
from unittest.mock import patch
import pandas as pd

from anomaly_match.pipeline.SessionTracker import SessionTracker, IterationInfo


class TestIterationInfo:
    """Test the IterationInfo dataclass."""

    def test_iteration_info_creation(self):
        """Test creating an IterationInfo instance."""
        info = IterationInfo(
            iteration_number=1,
            timestamp="2025-05-30 12:00:00",
            model_loss=0.5,
            test_performance={"accuracy": 0.85},
            model_state_path="/path/to/model.pkl",
        )

        assert info.iteration_number == 1
        assert info.timestamp == "2025-05-30 12:00:00"
        assert info.model_loss == 0.5
        assert info.test_performance == {"accuracy": 0.85}
        assert info.model_state_path == "/path/to/model.pkl"

    def test_iteration_info_to_dict(self):
        """Test converting IterationInfo to dictionary."""
        info = IterationInfo(
            iteration_number=0,
            timestamp="2025-05-30 12:00:00",
        )

        result = info.to_dict()

        assert result["iteration_number"] == 0
        assert result["timestamp"] == "2025-05-30 12:00:00"
        assert result["num_newly_labeled_anomalous"] == 0
        assert result["num_newly_labeled_nominal"] == 0
        assert result["model_loss"] is None
        assert result["test_performance"] is None
        assert result["model_state_path"] is None


class TestSessionTracker:
    """Test the SessionTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a SessionTracker instance for testing."""
        return SessionTracker("test_session")

    @pytest.fixture
    def tracker_with_data(self):
        """Create a SessionTracker with some test data."""
        tracker = SessionTracker("test_session")
        tracker.start_new_session_iteration()
        tracker.add_labeled_sample("img1.jpg", "anomaly")
        tracker.add_labeled_sample("img2.jpg", "normal")
        tracker.update_model_iteration(0.5)
        tracker.update_test_performance({"accuracy": 0.85, "auc": 0.92})
        return tracker

    def test_session_tracker_initialization(self, tracker):
        """Test SessionTracker initialization."""
        assert tracker.session_name == "test_session"
        assert tracker.current_session_iteration == 0
        assert tracker.total_model_iterations == 0
        assert len(tracker.session_iterations) == 0
        assert len(tracker.labeled_data_df) == 0
        assert isinstance(tracker.labeled_data_df, pd.DataFrame)
        assert tracker.labeled_data_df.empty

    def test_session_tracker_default_name(self):
        """Test SessionTracker with default name generation."""
        # Mock the datetime.now() method directly on the datetime class in the SessionTracker module
        with patch("anomaly_match.pipeline.SessionTracker.datetime") as mock_datetime:
            mock_now = datetime.datetime(2025, 5, 30, 12, 0, 0)
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.datetime.strftime = datetime.datetime.strftime

            tracker = SessionTracker()
            assert tracker.session_name.startswith("session_")
            assert "20250530_120000" in tracker.session_name

    def test_start_new_session_iteration(self, tracker):
        """Test starting a new session iteration."""
        # Start first iteration
        iter_num = tracker.start_new_session_iteration()
        assert iter_num == 0
        assert tracker.current_session_iteration == 1
        assert len(tracker.session_iterations) == 1

        # Start second iteration
        iter_num = tracker.start_new_session_iteration()
        assert iter_num == 1
        assert tracker.current_session_iteration == 2
        assert len(tracker.session_iterations) == 2

    def test_update_model_iteration(self, tracker):
        """Test updating model iterations."""
        # Update without loss
        tracker.update_model_iteration()
        assert tracker.total_model_iterations == 1

        # Update with loss
        tracker.update_model_iteration(loss=0.5)
        assert tracker.total_model_iterations == 2

        # Update current session iteration if it exists
        tracker.start_new_session_iteration()
        tracker.update_model_iteration(loss=0.3)
        assert tracker.session_iterations[-1].model_loss == 0.3

    def test_add_labeled_sample(self, tracker):
        """Test adding labeled samples."""
        # Add anomalous sample
        tracker.add_labeled_sample("img1.jpg", "anomaly")
        assert len(tracker.labeled_data_df) == 1
        assert tracker.labeled_data_df.iloc[0]["filename"] == "img1.jpg"
        assert tracker.labeled_data_df.iloc[0]["label"] == "anomaly"

        # Add normal sample
        tracker.add_labeled_sample("img2.jpg", "normal")
        assert len(tracker.labeled_data_df) == 2

        # Test with session iteration
        tracker.start_new_session_iteration()
        tracker.add_labeled_sample("img3.jpg", "anomaly")
        tracker.add_labeled_sample("img4.jpg", "normal")

        current_iter = tracker.session_iterations[-1]
        assert current_iter.num_newly_labeled_anomalous == 1
        assert current_iter.num_newly_labeled_nominal == 1

    def test_update_test_performance(self, tracker):
        """Test updating test performance."""
        performance_metrics = {"accuracy": 0.85, "auc": 0.92, "f1": 0.78}

        # Should do nothing if no session iteration exists
        tracker.update_test_performance(performance_metrics)

        # Should update current session iteration
        tracker.start_new_session_iteration()
        tracker.update_test_performance(performance_metrics)

        current_iter = tracker.session_iterations[-1]
        assert current_iter.test_performance == performance_metrics

    def test_update_model_state_path(self, tracker):
        """Test updating model state path."""
        model_path = "/path/to/model.pkl"

        # Should do nothing if no session iteration exists
        tracker.update_model_state_path(model_path)

        # Should update current session iteration
        tracker.start_new_session_iteration()
        tracker.update_model_state_path(model_path)

        current_iter = tracker.session_iterations[-1]
        assert current_iter.model_state_path == model_path

    def test_get_session_info(self, tracker_with_data):
        """Test getting session information."""
        info = tracker_with_data.get_session_info()

        assert info["session_name"] == "test_session"
        assert "session_start_time" in info
        assert info["total_session_iterations"] == 1
        assert info["total_model_iterations"] == 1
        assert info["total_anomalous_samples"] == 1
        assert info["total_nominal_samples"] == 1
        assert info["total_labeled_samples"] == 2
        assert "session_duration_minutes" in info

    def test_get_session_info_empty(self, tracker):
        """Test getting session info when no data exists."""
        info = tracker.get_session_info()

        assert info["total_session_iterations"] == 0
        assert info["total_model_iterations"] == 0
        assert info["total_anomalous_samples"] == 0
        assert info["total_nominal_samples"] == 0
        assert info["total_labeled_samples"] == 0

    def test_get_iteration_info(self, tracker_with_data):
        """Test getting iteration information."""
        # Get latest iteration info
        info = tracker_with_data.get_iteration_info()

        assert info["iteration_number"] == 0
        assert "timestamp" in info
        assert info["model_loss"] == 0.5
        assert info["test_performance"] == {"accuracy": 0.85, "auc": 0.92}
        assert info["num_newly_labeled_anomalous"] == 1
        assert info["num_newly_labeled_nominal"] == 1

        # Get specific iteration info
        info = tracker_with_data.get_iteration_info(0)
        assert info["iteration_number"] == 0

    def test_get_iteration_info_empty(self, tracker):
        """Test getting iteration info when no iterations exist."""
        info = tracker.get_iteration_info()
        assert info == {}

        info = tracker.get_iteration_info(0)
        assert info == {}

    def test_get_iteration_info_nonexistent(self, tracker_with_data):
        """Test getting info for non-existent iteration."""
        info = tracker_with_data.get_iteration_info(999)
        assert info == {}

    def test_get_all_iterations_info(self, tracker_with_data):
        """Test getting all iterations information."""
        # Add another iteration
        tracker_with_data.start_new_session_iteration()
        tracker_with_data.add_labeled_sample("img3.jpg", "anomaly")
        tracker_with_data.update_model_iteration(0.3)

        all_info = tracker_with_data.get_all_iterations_info()

        assert len(all_info) == 2
        # Iterations are returned in reverse order (newest first)
        assert all_info[0]["iteration_number"] == 1  # Newest iteration first
        assert all_info[1]["iteration_number"] == 0  # Older iteration second
        assert all_info[0]["model_loss"] == 0.3  # Loss from newest iteration
        assert all_info[1]["model_loss"] == 0.5  # Loss from older iteration

    def test_get_labeled_data_df(self, tracker_with_data):
        """Test getting labeled data DataFrame."""
        df = tracker_with_data.get_labeled_data_df()

        assert len(df) == 2
        assert df.iloc[0]["filename"] == "img1.jpg"
        assert df.iloc[0]["label"] == "anomaly"
        assert df.iloc[1]["filename"] == "img2.jpg"
        assert df.iloc[1]["label"] == "normal"

        # Should return a copy, not the original
        df.loc[0, "label"] = "modified"
        original_df = tracker_with_data.get_labeled_data_df()
        assert original_df.iloc[0]["label"] == "anomaly"

    def test_multiple_iterations_complex(self, tracker):
        """Test complex scenario with multiple iterations."""
        # Iteration 0
        tracker.start_new_session_iteration()
        tracker.add_labeled_sample("img1.jpg", "anomaly")
        tracker.add_labeled_sample("img2.jpg", "normal")
        tracker.update_model_iteration(0.8)
        tracker.update_model_iteration(0.7)
        tracker.update_test_performance({"accuracy": 0.70})
        tracker.update_model_state_path("/path/to/model1.pkl")

        # Iteration 1
        tracker.start_new_session_iteration()
        tracker.add_labeled_sample("img3.jpg", "anomaly")
        tracker.update_model_iteration(0.6)
        tracker.update_test_performance({"accuracy": 0.80, "auc": 0.85})
        tracker.update_model_state_path("/path/to/model2.pkl")

        # Iteration 2
        tracker.start_new_session_iteration()
        tracker.add_labeled_sample("img4.jpg", "normal")
        tracker.add_labeled_sample("img5.jpg", "normal")
        tracker.update_model_iteration(0.5)
        tracker.update_test_performance({"accuracy": 0.90})

        # Verify session info
        session_info = tracker.get_session_info()
        assert session_info["total_session_iterations"] == 3
        assert session_info["total_model_iterations"] == 4
        assert session_info["total_anomalous_samples"] == 2
        assert session_info["total_nominal_samples"] == 3
        assert session_info["total_labeled_samples"] == 5

        # Verify iteration details
        all_iterations = tracker.get_all_iterations_info()
        assert len(all_iterations) == 3

        # Check iteration 2 (newest, index 0)
        assert all_iterations[0]["num_newly_labeled_anomalous"] == 0
        assert all_iterations[0]["num_newly_labeled_nominal"] == 2
        assert all_iterations[0]["test_performance"]["accuracy"] == 0.90
        assert all_iterations[0]["model_state_path"] is None

        # Check iteration 1 (middle, index 1)
        assert all_iterations[1]["num_newly_labeled_anomalous"] == 1
        assert all_iterations[1]["num_newly_labeled_nominal"] == 0
        assert all_iterations[1]["test_performance"]["accuracy"] == 0.80
        assert all_iterations[1]["model_state_path"] == "/path/to/model2.pkl"

        # Check iteration 0 (oldest, index 2)
        assert all_iterations[2]["num_newly_labeled_anomalous"] == 1
        assert all_iterations[2]["num_newly_labeled_nominal"] == 1
        assert all_iterations[2]["test_performance"]["accuracy"] == 0.70
        assert all_iterations[2]["model_state_path"] == "/path/to/model1.pkl"
