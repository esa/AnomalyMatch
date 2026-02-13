#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import os
from unittest.mock import patch

import ipywidgets as widgets
import matplotlib
import numpy as np
import pytest
from PIL import Image

import anomaly_match as am
from anomaly_match.pipeline.session import Session
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.widget import Widget

pytestmark = [pytest.mark.ui, pytest.mark.slow]

matplotlib.use("Agg")  # Prevent matplotlib windows from opening


@pytest.fixture(scope="session")
def base_config():
    out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid white", height="400px", background_color="black", overflow="auto"
        ),
    )

    cfg = am.get_default_cfg()
    am.set_log_level("debug", cfg)
    cfg.data_dir = "tests/test_data/"
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.net = "test-cnn"
    cfg.pretrained = False
    cfg.num_train_iter = 2
    cfg.num_workers = 0
    cfg.test_ratio = 0.5
    cfg.output_dir = "tests/test_output"
    cfg.prediction_search_dir = "tests/test_data/"  # Set a default search directory
    cfg.top_N = 10
    return cfg, out


@pytest.fixture(scope="session")
def session_fixture(base_config):
    cfg, out = base_config
    session = Session(cfg)
    session.out = out
    session.train(cfg)
    session.update_predictions()
    return session


@pytest.fixture(scope="session")
def ui_widget(session_fixture):
    with patch("IPython.display.display"):  # Prevent actual display calls
        # Set up the backend interface
        BackendInterface.set_session(session_fixture)
        widget = Widget()
        yield widget
        # Only close widgets in teardown if they still exist
        try:
            widget.ui["image_widget"].close()
            widget.ui["filename_text"].close()
            widget.ui["brightness_slider"].close()
            widget.ui["contrast_slider"].close()
        except AttributeError:
            pass


@pytest.fixture(autouse=True)
def setup_display_mocks():
    """Automatically mock display-related functions for all tests"""
    with patch("IPython.display.display"), patch("IPython.display.Image"):
        yield


# Test classes to organize related tests
class TestUIInitialization:
    def test_ui_initialization(self, ui_widget):
        assert BackendInterface.get_session() is not None
        assert isinstance(ui_widget.ui["image_widget"], widgets.Image)
        assert isinstance(ui_widget.ui["filename_text"], widgets.HTML)

    def test_normalization_dropdown(self, ui_widget):
        # Get the initial normalization method
        initial_method = BackendInterface.get_config().normalisation.normalisation_method

        # Get the dropdown options and find a different method
        dropdown_options = ui_widget.ui["normalisation_dropdown"].options
        # Find a method different from the initial one, looking at the enum value in the tuple
        new_method = next(
            method[1]  # Take the enum value from the tuple
            for method in dropdown_options
            if isinstance(method, tuple) and method[1] != initial_method
        )

        # Set the dropdown value to the enum value directly
        ui_widget.ui["normalisation_dropdown"].value = new_method

        # Assert that the session config was updated
        assert BackendInterface.get_config().normalisation.normalisation_method == new_method


class TestUINavigation:
    def test_next_image(self, ui_widget):
        initial_index = ui_widget.preview.current_index
        ui_widget.next_image()
        assert ui_widget.preview.current_index == initial_index + 1

    def test_previous_image(self, ui_widget):
        ui_widget.next_image()  # Move to next image first
        initial_index = ui_widget.preview.current_index
        ui_widget.previous_image()
        assert ui_widget.preview.current_index == initial_index - 1


class TestUISorting:
    def test_sort_by_anomalous(self, ui_widget):
        ui_widget.sort_by_anomalous()
        scores = BackendInterface.get_scores()
        assert scores[0] >= scores[-1]

    def test_sort_by_nominal(self, ui_widget):
        ui_widget.sort_by_nominal()
        scores = BackendInterface.get_scores()
        assert scores[0] <= scores[-1]

    def test_sort_by_mean(self, ui_widget):
        ui_widget.sort_by_mean()
        scores = BackendInterface.get_scores()
        mean_score = scores.mean()
        assert abs(scores[0] - mean_score) <= abs(scores[-1] - mean_score)

    def test_sort_by_median(self, ui_widget):
        ui_widget.sort_by_median()
        scores = BackendInterface.get_scores()
        median_score = np.median(scores)
        assert abs(scores[0] - median_score) <= abs(scores[-1] - median_score)


class TestUIImageProcessing:
    def test_toggle_invert_image(self, ui_widget):
        initial_invert_state = ui_widget.preview.invert
        ui_widget.toggle_invert_image()
        assert ui_widget.preview.invert != initial_invert_state

    def test_toggle_unsharp_mask(self, ui_widget):
        initial_unsharp_mask_state = ui_widget.preview.unsharp_mask_applied
        ui_widget.toggle_unsharp_mask()
        assert ui_widget.preview.unsharp_mask_applied != initial_unsharp_mask_state

    def test_adjust_brightness_contrast(self, ui_widget):
        initial_brightness = ui_widget.ui["brightness_slider"].value
        initial_contrast = ui_widget.ui["contrast_slider"].value
        ui_widget.ui["brightness_slider"].value = initial_brightness + 0.1
        ui_widget.ui["contrast_slider"].value = initial_contrast + 0.1
        assert ui_widget.preview.brightness == initial_brightness + 0.1
        assert ui_widget.preview.contrast == initial_contrast + 0.1


class TestUIModelOperations:
    def test_save_load_model(self, ui_widget):
        ui_widget.save_model()
        cfg = BackendInterface.get_config()
        assert os.path.exists(cfg.model_path)
        ui_widget.load_model()
        assert BackendInterface.get_model() is not None

    def test_train(self, ui_widget):
        ui_widget.train()
        assert BackendInterface.get_eval_performance() is not None

    def test_reset_model(self, ui_widget):
        ui_widget.reset_model()
        assert BackendInterface.get_model() is not None


class TestUIBatchOperations:
    def test_update_batch_size(self, ui_widget):
        initial_batch_size = ui_widget.ui["batch_size_slider"].value
        ui_widget.ui["batch_size_slider"].value = initial_batch_size + 500
        cfg = BackendInterface.get_config()
        assert cfg.N_to_load == initial_batch_size + 500

    def test_next_batch(self, ui_widget):
        ui_widget.next_batch()
        session = BackendInterface.get_session()
        assert session.img_catalog is not None

    def test_search_all_files(self, ui_widget):
        with patch("anomaly_match_ui.widget.display"):
            # Save model first so evaluate_all_images can find it
            ui_widget.save_model()

            cfg = BackendInterface.get_config()
            # Ensure test data directory exists and has files
            os.makedirs(cfg.prediction_search_dir, exist_ok=True)
            # Create a test image if directory is empty
            if not os.listdir(cfg.prediction_search_dir):
                test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                test_img_path = os.path.join(cfg.prediction_search_dir, "test.jpg")
                Image.fromarray(test_img).save(test_img_path)

            ui_widget.search_all_files()
            session = BackendInterface.get_session()
            assert len(session.img_catalog) > 0


def test_cleanup(ui_widget):
    # Verify that widgets can be properly closed
    ui_widget.ui["image_widget"].close()
    ui_widget.ui["filename_text"].close()
    assert not ui_widget.ui["image_widget"]._dom_classes
    assert not ui_widget.ui["filename_text"]._dom_classes
