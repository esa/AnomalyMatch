#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import pytest
import os
import numpy as np
import ipywidgets as widgets
import anomaly_match as am
from anomaly_match.pipeline.session import Session


@pytest.fixture(scope="module")
def base_config():
    out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid white", height="400px", background_color="black", overflow="auto"
        ),
    )
    progress_bar = widgets.FloatProgress(
        value=0.0,
        min=0.0,
        max=1.0,
    )

    cfg = am.get_default_cfg()
    am.set_log_level("debug", cfg)
    cfg.data_dir = "tests/test_data/"
    cfg.size = [64, 64]
    cfg.num_train_iter = 2
    cfg.test_ratio = 0.5
    cfg.output_dir = "tests/test_output"
    cfg.progress_bar = progress_bar
    return cfg, out


@pytest.fixture(scope="module")
def trained_session(base_config):
    cfg, out = base_config
    session = Session(cfg)
    session.out = out
    session.train(cfg)
    session.update_predictions()
    return session, cfg


def test_session_initialization(trained_session):
    session, _ = trained_session
    assert session.model is not None
    assert session.labeled_train_dataset is not None
    assert session.unlabeled_train_dataset is not None
    assert session.test_dataset is not None
    assert session.active_learning_df.empty


def test_session_training(trained_session):
    session, _ = trained_session
    assert session.eval_performance is not None
    assert (
        session.eval_performance["eval/top-1-acc"] >= 0.0
        and session.eval_performance["eval/top-1-acc"] <= 1.0
    )
    assert session.eval_performance["eval/loss"] >= 0.0
    assert (
        session.eval_performance["eval/auroc"] >= 0.0
        and session.eval_performance["eval/auroc"] <= 1.0
    )
    assert (
        session.eval_performance["eval/auprc"] >= 0.0
        and session.eval_performance["eval/auprc"] <= 1.0
    )
    # Also check confusion_matrix, predictions_and_labels, roc_data and pr_data
    assert session.eval_performance["eval/confusion_matrix"] is not None
    assert session.eval_performance["eval/predictions_and_labels"] is not None
    assert session.eval_performance["eval/roc_data"] is not None
    assert session.eval_performance["eval/precision_recall"] is not None


def test_predictions_update(trained_session):
    session, _ = trained_session
    assert session.scores is not None
    assert session.img_catalog is not None
    assert session.filenames is not None
    assert len(session.scores) == len(session.filenames)


def test_sorting_methods(trained_session):
    session, _ = trained_session
    original_order = session.filenames.copy()

    # Test anomalous sorting
    session.sort_by_anomalous()
    assert np.all(np.diff(session.scores) <= 0)  # Verify descending order

    # Test nominal sorting
    session.sort_by_nominal()
    assert np.all(np.diff(session.scores) >= 0)  # Verify ascending order

    # Verify that all files are still present after sorting
    assert set(session.filenames) == set(original_order)


def test_labeling(trained_session):
    session, _ = trained_session

    # Test adding a label
    session.label_image(0, "normal")
    assert len(session.active_learning_df) == 1
    assert session.active_learning_df.iloc[0]["label"] == "normal"

    # Test overwriting a label
    session.label_image(0, "anomaly")
    assert len(session.active_learning_df) == 1
    assert session.active_learning_df.iloc[0]["label"] == "anomaly"

    # Test invalid label
    with pytest.raises(AssertionError):
        session.label_image(0, "invalid")


def test_save_load_operations(trained_session):
    session, cfg = trained_session

    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set name for test
    cfg.name = "MyRun"  # Ensure we use a consistent name

    # Test save labels
    session.label_image(0, "normal")
    session.save_labels()
    assert os.path.exists(os.path.join(cfg.output_dir, "labeled_data.csv"))

    # Test remember current file
    session.remember_current_file(session.filenames[0])
    remembered_file = os.path.join(
        cfg.output_dir, f"MyRun_{session.session_start}_remembered_files.csv"
    )
    assert os.path.exists(remembered_file)


def test_model_save_load(trained_session):
    session, cfg = trained_session

    # Save model state
    session.save_model()
    assert os.path.exists(cfg.model_path)


def test_get_label_distribution(trained_session):
    session, _ = trained_session

    # Initial distribution
    normal, anomalous = session.get_label_distribution()
    initial_total = normal + anomalous

    # Add some labels
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")

    # Check updated distribution
    new_normal, new_anomalous = session.get_label_distribution()
    assert new_normal + new_anomalous > initial_total


def test_get_label(trained_session):
    session, _ = trained_session

    # Test existing label
    session.label_image(0, "normal")
    assert session.get_label(0) == "normal"
