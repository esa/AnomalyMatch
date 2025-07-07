#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import os
import shutil
import tempfile
import pandas as pd
import pytest
from dotmap import DotMap

from anomaly_match.datasets.AnomalyDetectionDataset import AnomalyDetectionDataset
from anomaly_match.pipeline.session import Session
from anomaly_match.utils.get_default_cfg import get_default_cfg


class TestMetadata:
    @pytest.fixture(scope="function")
    def setup_test_files(self):
        """Create temporary test files and directories."""
        # Create a temporary directory
        test_dir = tempfile.mkdtemp()

        # Create subdirectories
        data_dir = os.path.join(test_dir, "data")
        output_dir = os.path.join(test_dir, "output")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create dummy image files
        for i in range(3):
            # Create an empty file
            img_path = os.path.join(data_dir, f"test_image_{i}.jpg")
            with open(img_path, "w") as f:
                f.write("")

        # Create labeled_data.csv - label only the first two images
        label_file = os.path.join(test_dir, "labeled_data.csv")
        labels = pd.DataFrame(
            {"filename": [f"test_image_{i}.jpg" for i in range(2)], "label": ["normal", "anomaly"]}
        )
        labels.to_csv(label_file, index=False)

        # Create metadata.csv
        metadata_file = os.path.join(test_dir, "metadata.csv")
        metadata = pd.DataFrame(
            {
                "filename": [f"test_image_{i}.jpg" for i in range(3)],
                "sourceID": [f"source_{i}" for i in range(3)],
                "ra": [10.0 + i for i in range(3)],
                "dec": [20.0 + i for i in range(3)],
                "custom_col": [f"custom_{i}" for i in range(3)],
            }
        )
        metadata.to_csv(metadata_file, index=False)

        # Return paths for test
        yield {
            "test_dir": test_dir,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "label_file": label_file,
            "metadata_file": metadata_file,
        }

        # Cleanup
        shutil.rmtree(test_dir)

    def test_metadata_loading(self, setup_test_files, monkeypatch):
        """Test that metadata is correctly loaded in AnomalyDetectionDataset."""

        # Mock image reading functions
        def mock_read_and_resize(*args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.uint8)

        def mock_get_image_names(dir_path, recursive=False):
            return [os.path.join(dir_path, f"test_image_{i}.jpg") for i in range(3)]

        monkeypatch.setattr(
            "anomaly_match.data_io.load_images.read_and_resize_image", mock_read_and_resize
        )
        monkeypatch.setattr(
            "anomaly_match.data_io.find_images_in_folder.get_image_names_from_folder",
            mock_get_image_names,
        )

        # Set up configuration
        paths = setup_test_files
        cfg = get_default_cfg()
        cfg.data_dir = paths["data_dir"]
        cfg.label_file = paths["label_file"]
        cfg.metadata_file = paths["metadata_file"]

        # Create dataset
        dataset = AnomalyDetectionDataset(cfg, use_hdf5=False)

        # Check that metadata was loaded
        metadata_df = dataset.get_all_metadata()
        assert metadata_df is not None
        assert len(metadata_df) == 3

        # Check that expected columns are present
        for col in ["sourceID", "ra", "dec", "custom_col"]:
            assert col in metadata_df.columns

        # Check some values
        assert metadata_df.loc["test_image_0.jpg", "sourceID"] == "source_0"
        assert metadata_df.loc["test_image_1.jpg", "ra"] == 11.0

    def test_metadata_saving_in_session(self, setup_test_files, monkeypatch):
        """Test that metadata is included when saving labels in Session."""

        # Mock required functions
        def mock_read_and_resize(*args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.uint8)

        def mock_get_image_names(dir_path, recursive=False):
            return [os.path.join(dir_path, f"test_image_{i}.jpg") for i in range(3)]

        monkeypatch.setattr(
            "anomaly_match.data_io.load_images.read_and_resize_image", mock_read_and_resize
        )
        monkeypatch.setattr(
            "anomaly_match.data_io.find_images_in_folder.get_image_names_from_folder",
            mock_get_image_names,
        )

        # Patch model initialization to avoid issues
        def mock_init_model(self):
            self.model = DotMap()
            self.model.train_model = {}

        monkeypatch.setattr(Session, "_init_model", mock_init_model)

        # Set up configuration
        paths = setup_test_files
        cfg = get_default_cfg()
        cfg.data_dir = paths["data_dir"]
        cfg.label_file = paths["label_file"]
        cfg.metadata_file = paths["metadata_file"]
        cfg.output_dir = paths["output_dir"]

        # Create session
        session = Session(cfg)

        # Save labels
        session.save_labels()

        # Check saved file in session directory (not output_dir)
        session_path = session.session_io.get_session_save_path(session.session_tracker)
        output_file = os.path.join(session_path, "labeled_data.csv")
        assert os.path.exists(output_file)

        saved_data = pd.read_csv(output_file)

        # Check that metadata columns are included
        for col in ["sourceID", "ra", "dec", "custom_col"]:
            assert col in saved_data.columns

        # Check that values were preserved
        assert (
            saved_data[saved_data["filename"] == "test_image_0.jpg"]["sourceID"].values[0]
            == "source_0"
        )
        assert saved_data[saved_data["filename"] == "test_image_1.jpg"]["ra"].values[0] == 11.0

    def test_missing_metadata_file(self, setup_test_files, monkeypatch):
        """Test behavior when metadata file is specified but doesn't exist."""

        # Mock required functions
        def mock_read_and_resize(*args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.uint8)

        def mock_get_image_names(dir_path, recursive=False):
            return [os.path.join(dir_path, f"test_image_{i}.jpg") for i in range(3)]

        monkeypatch.setattr(
            "anomaly_match.data_io.load_images.read_and_resize_image", mock_read_and_resize
        )
        monkeypatch.setattr(
            "anomaly_match.data_io.find_images_in_folder.get_image_names_from_folder",
            mock_get_image_names,
        )

        # Set up configuration
        paths = setup_test_files
        cfg = get_default_cfg()
        cfg.data_dir = paths["data_dir"]
        cfg.label_file = paths["label_file"]
        cfg.metadata_file = os.path.join(paths["test_dir"], "nonexistent_metadata.csv")

        # Create dataset - should not raise an exception
        dataset = AnomalyDetectionDataset(cfg, use_hdf5=False)

        # Metadata should be None
        assert dataset.get_all_metadata() is None
