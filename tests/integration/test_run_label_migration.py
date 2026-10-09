#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import os
import tempfile

import pandas as pd
import pytest

from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.pipeline.SessionTracker import SessionTracker


class TestLabelSavingMigration:
    """Test the migration of save_labels functionality from FixMatch to SessionIOHandler."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def session_io(self, temp_dir):
        """Create a SessionIOHandler instance."""
        return SessionIOHandler(base_save_path=temp_dir)

    @pytest.fixture
    def session_tracker(self):
        """Create a SessionTracker instance."""
        return SessionTracker("test_session")

    def test_save_labels_to_output_dir(self, session_io, temp_dir):
        """Test save_labels_to_output_dir functionality."""
        # Create test labeled data
        labeled_data = pd.DataFrame(
            {
                "filename": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "label": ["normal", "anomaly", "normal"],
            }
        )

        output_dir = os.path.join(temp_dir, "output")

        result_path = session_io.save_labels_to_output_dir(labeled_data, output_dir)

        # Check that the labels were saved
        expected_path = os.path.join(output_dir, "labeled_data.csv")
        assert result_path == expected_path
        assert os.path.exists(expected_path)

        # Verify the saved data
        loaded_data = pd.read_csv(expected_path)
        pd.testing.assert_frame_equal(loaded_data, labeled_data)

    def test_save_labels_with_session_tracker(self, session_io, session_tracker, temp_dir):
        """Test save_labels_to_output_dir with session tracker integration."""
        # Create test labeled data
        labeled_data = pd.DataFrame(
            {"id": ["img1.jpg", "img2.jpg"], "label": ["normal", "anomaly"]}
        )

        output_dir = os.path.join(temp_dir, "output")

        session_io.save_labels_to_output_dir(
            labeled_data, output_dir, session_tracker=session_tracker
        )

        result = session_tracker.labeled_data_df
        assert "id" in result.columns
        assert "iteration" in result.columns
        assert set(result["id"]) == {"img1.jpg", "img2.jpg"}

    def test_session_tracker_update_labeled_data(self, session_tracker):
        """Test SessionTracker.update_labeled_data method."""
        labeled_data = pd.DataFrame(
            {"id": ["img1.jpg", "img2.jpg"], "label": ["normal", "anomaly"]}
        )

        session_tracker.update_labeled_data(labeled_data)

        result = session_tracker.labeled_data_df
        assert "id" in result.columns
        assert "iteration" in result.columns
        assert len(result) == 2
        assert (result["iteration"] == -1).all()

    def test_integration_label_saving_flow(self, session_io, session_tracker, temp_dir):
        """Test the complete integration flow for label saving."""
        # Create labeled data
        labeled_data = pd.DataFrame(
            {
                "id": ["sample1.jpg", "sample2.jpg", "sample3.jpg"],
                "label": ["normal", "anomaly", "normal"],
            }
        )

        output_dir = os.path.join(temp_dir, "labels_output")

        # Save labels through SessionIOHandler
        csv_path = session_io.save_labels_to_output_dir(
            labeled_data, output_dir, session_tracker=session_tracker
        )

        # Verify session tracker was updated with id and iteration columns
        result = session_tracker.labeled_data_df
        assert "id" in result.columns
        assert "iteration" in result.columns
        assert len(result) == 3
        assert set(result["id"]) == {"sample1.jpg", "sample2.jpg", "sample3.jpg"}

        # Verify CSV file content
        loaded_data = pd.read_csv(csv_path)
        assert set(loaded_data["id"]) == {"sample1.jpg", "sample2.jpg", "sample3.jpg"}
