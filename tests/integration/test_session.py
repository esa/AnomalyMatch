#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import os

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.slow

import anomaly_match as am
from anomaly_match.pipeline.session import Session


@pytest.fixture(scope="module")
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


def test_load_top_files(trained_session):
    """Test that load_top_files correctly loads images with consistent format handling."""
    session, cfg = trained_session

    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Create test data files (simulate saved prediction results)
    top_N = 10
    output_csv_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{top_N}.csv")
    output_npy_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{top_N}.npy")

    # Create test CSV data
    test_filenames = [f"test_image_{i}.jpg" for i in range(top_N)]
    test_scores = np.random.random(top_N)

    df = pd.DataFrame({"Filename": test_filenames, "Score": test_scores})
    df.to_csv(output_csv_path, index=False)

    # Create test image data in CHW format (simulate how they are currently saved)
    test_images_chw = np.random.randint(0, 255, (top_N, 3, 64, 64), dtype=np.uint8)
    np.save(output_npy_path, test_images_chw)

    # Test loading
    session.load_top_files(top_N)

    # Verify that images were loaded correctly
    assert session.img_catalog is not None
    assert len(session.img_catalog) == top_N
    assert session.img_catalog.shape == (top_N, 64, 64, 3), "Images should be in HWC format"
    assert session.img_catalog.dtype == np.uint8, "Images should be uint8"

    # Verify that filenames and scores were loaded correctly
    assert len(session.filenames) == top_N
    assert len(session.scores) == top_N
    # Note: scores come from the CSV file, so they should match what we saved
    np.testing.assert_allclose(session.scores, test_scores, rtol=1e-12, atol=1e-12)

    # Verify that the transpose from CHW to HWC was applied correctly
    # Convert the original CHW back to HWC for comparison
    expected_images_hwc = test_images_chw.transpose(0, 2, 3, 1)
    assert np.array_equal(session.img_catalog, expected_images_hwc), (
        "CHW to HWC conversion should be correct"
    )

    # Test loading with images already in HWC format
    test_images_hwc = np.random.randint(0, 255, (top_N, 64, 64, 3), dtype=np.uint8)
    np.save(output_npy_path, test_images_hwc)

    session.load_top_files(top_N)

    # Verify that images in HWC format are loaded without transpose
    assert session.img_catalog.shape == (top_N, 64, 64, 3), "Images should remain in HWC format"
    assert np.array_equal(session.img_catalog, test_images_hwc), "HWC images should be loaded as-is"

    # Test with float32 images that need uint8 conversion
    test_images_float = np.random.random((top_N, 64, 64, 3)).astype(np.float32)
    np.save(output_npy_path, test_images_float)

    session.load_top_files(top_N)

    # Verify that float images were converted to uint8
    assert session.img_catalog.dtype == np.uint8, "Float images should be converted to uint8"
    expected_uint8 = (test_images_float * 255.0).clip(0, 255).astype(np.uint8)
    assert np.array_equal(session.img_catalog, expected_uint8), (
        "Float to uint8 conversion should be correct"
    )

    # Clean up test files
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)
    if os.path.exists(output_npy_path):
        os.remove(output_npy_path)


def test_unlabeling(trained_session):
    session, _ = trained_session

    # First, add labels to two different images
    session.label_image(0, "anomaly")
    session.label_image(1, "normal")

    # Verify both images are labeled
    assert session.get_label(0) == "anomaly"
    assert session.get_label(1) == "normal"
    initial_df_length = len(session.active_learning_df)
    assert initial_df_length >= 2

    # Test unlabeling one of them
    session.unlabel_image(0)

    # Verify image 0 is no longer labeled
    assert session.get_label(0) == "None"

    # Verify image 1 is still labeled
    assert session.get_label(1) == "normal"

    # Verify DataFrame length has decreased by 1
    assert len(session.active_learning_df) == initial_df_length - 1

    # Test unlabeling a non-labeled image (should not change anything)
    orig_df_length = len(session.active_learning_df)
    session.unlabel_image(2)  # Assuming this index wasn't labeled
    assert len(session.active_learning_df) == orig_df_length

    # Test unlabeling all images
    session.unlabel_image(1)
    assert session.get_label(1) == "None"
    assert len(session.active_learning_df) == orig_df_length - 1


def clear_session_cache(session):
    """Helper function to clear all caches and active learning data for testing."""
    session.active_learning_df = pd.DataFrame(columns=["filename", "label"])
    session._label_cache = {}
    session._label_distribution_cache = None
    session._active_learning_counts_cache = None


def test_label_cache_functionality(trained_session):
    """Test that label caching works correctly and improves performance."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Initially cache should be empty
    assert session._label_cache == {}

    # Add some labels
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")
    session.label_image(2, "normal")

    # Cache should be populated
    assert len(session._label_cache) == 3
    assert session._label_cache[session.filenames[0]] == "normal"
    assert session._label_cache[session.filenames[1]] == "anomaly"
    assert session._label_cache[session.filenames[2]] == "normal"

    # Test that get_label uses cache
    assert session.get_label(0) == "normal"
    assert session.get_label(1) == "anomaly"
    assert session.get_label(2) == "normal"
    assert session.get_label(3) == "None"  # Not in cache

    # Test cache consistency when updating labels
    session.label_image(0, "anomaly")  # Change label
    assert session._label_cache[session.filenames[0]] == "anomaly"
    assert session.get_label(0) == "anomaly"


def test_label_cache_invalidation(trained_session):
    """Test that label cache is properly invalidated when labels are modified."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Add labels to populate cache
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")

    # Unlabel should remove from cache
    session.unlabel_image(0)
    assert session.filenames[0] not in session._label_cache
    assert session.filenames[1] in session._label_cache
    assert session.get_label(0) == "None"
    assert session.get_label(1) == "anomaly"

    # Reset should clear cache
    session.reset_model()
    assert session._label_cache == {}


def test_distribution_cache_functionality(trained_session):
    """Test that label distribution caching works correctly."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Initially distribution cache should be None
    assert session._label_distribution_cache is None

    # First call should compute and cache result
    normal1, anomalous1 = session.get_label_distribution()
    assert session._label_distribution_cache is not None
    assert session._label_distribution_cache == (normal1, anomalous1)

    # Second call should return cached result
    normal2, anomalous2 = session.get_label_distribution()
    assert normal2 == normal1
    assert anomalous2 == anomalous1

    # Adding labels should invalidate cache
    session.label_image(0, "normal")
    assert session._label_distribution_cache is None

    # Next call should recompute
    normal3, anomalous3 = session.get_label_distribution()
    assert normal3 == normal1 + 1  # Should have one more normal
    assert anomalous3 == anomalous1
    assert session._label_distribution_cache == (normal3, anomalous3)


def test_active_learning_counts_cache(trained_session):
    """Test that active learning counts caching works correctly."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Initially should return (0, 0)
    normal, anomalous = session.get_active_learning_counts()
    assert normal == 0
    assert anomalous == 0

    # Add some labels
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")
    session.label_image(2, "normal")

    # Should return correct counts
    normal, anomalous = session.get_active_learning_counts()
    assert normal == 2
    assert anomalous == 1

    # Test caching - result should be cached
    assert hasattr(session, "_active_learning_counts_cache")
    assert session._active_learning_counts_cache == (2, 1)

    # Adding another label should invalidate cache
    session.label_image(3, "anomaly")
    assert session._active_learning_counts_cache is None

    # Next call should recompute
    normal, anomalous = session.get_active_learning_counts()
    assert normal == 2
    assert anomalous == 2
    assert session._active_learning_counts_cache == (2, 2)


def test_cache_rebuild_functionality(trained_session):
    """Test that cache rebuilding works correctly."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Add labels
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")

    # Clear cache manually to simulate corruption
    session._label_cache = {}
    session._label_distribution_cache = None
    session._active_learning_counts_cache = None

    # get_label should rebuild cache automatically
    assert session.get_label(0) == "normal"
    assert len(session._label_cache) == 2
    assert session._label_cache[session.filenames[0]] == "normal"
    assert session._label_cache[session.filenames[1]] == "anomaly"

    # Manual rebuild should work
    session._label_cache = {}
    session._rebuild_label_cache()
    assert len(session._label_cache) == 2
    assert session._label_cache[session.filenames[0]] == "normal"
    assert session._label_cache[session.filenames[1]] == "anomaly"


def test_cache_performance_consistency(trained_session):
    """Test that cache provides consistent results compared to non-cached operations."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Create a scenario with multiple labels
    test_labels = [
        (0, "normal"),
        (1, "anomaly"),
        (2, "normal"),
        (3, "anomaly"),
        (4, "normal"),
    ]

    for idx, label in test_labels:
        session.label_image(idx, label)

    # Test that cached get_label returns same results as direct DataFrame lookup
    for idx, expected_label in test_labels:
        cached_result = session.get_label(idx)
        # Direct lookup without cache
        filename = session.filenames[idx]
        if filename in session.active_learning_df["filename"].values:
            direct_result = session.active_learning_df.loc[
                session.active_learning_df["filename"] == filename, "label"
            ].values[0]
        else:
            direct_result = "None"

        assert cached_result == direct_result == expected_label

    # Test distribution consistency
    cached_normal, cached_anomalous = session.get_label_distribution()

    # Count manually
    base_normal = torch.sum(session.labeled_train_dataset.targets == 0)
    base_anomalous = len(session.labeled_train_dataset.targets) - base_normal
    manual_normal = base_normal + np.sum(session.active_learning_df["label"] == "normal")
    manual_anomalous = base_anomalous + np.sum(session.active_learning_df["label"] == "anomaly")

    assert cached_normal == manual_normal
    assert cached_anomalous == manual_anomalous


def test_cache_edge_cases(trained_session):
    """Test cache behavior in edge cases."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Test with empty active_learning_df
    session.active_learning_df = pd.DataFrame(columns=["filename", "label"])
    session._rebuild_label_cache()
    assert session._label_cache == {}

    # Test get_label with no labels
    assert session.get_label(0) == "None"

    # Test distribution with no active learning labels
    normal, anomalous = session.get_label_distribution()
    base_normal = torch.sum(session.labeled_train_dataset.targets == 0)
    base_anomalous = len(session.labeled_train_dataset.targets) - base_normal
    assert normal == base_normal
    assert anomalous == base_anomalous

    # Test active learning counts with no labels
    normal_al, anomalous_al = session.get_active_learning_counts()
    assert normal_al == 0
    assert anomalous_al == 0


def test_cache_consistency_across_operations(trained_session):
    """Test that cache remains consistent across various session operations."""
    session, _ = trained_session

    # Clear any existing cache state from previous tests
    clear_session_cache(session)

    # Add some initial labels
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")

    # Store initial state
    initial_cache_keys = set(session._label_cache.keys())
    initial_distribution = session.get_label_distribution()
    initial_counts = session.get_active_learning_counts()

    # Perform sorting operations (should not affect cache)
    session.sort_by_anomalous()
    assert set(session._label_cache.keys()) == initial_cache_keys
    assert session.get_label_distribution() == initial_distribution
    assert session.get_active_learning_counts() == initial_counts

    session.sort_by_nominal()
    assert set(session._label_cache.keys()) == initial_cache_keys
    assert session.get_label_distribution() == initial_distribution
    assert session.get_active_learning_counts() == initial_counts

    # Update predictions (should not affect cache)
    session.update_predictions()
    assert set(session._label_cache.keys()) == initial_cache_keys
    assert session.get_label_distribution() == initial_distribution
    assert session.get_active_learning_counts() == initial_counts

    # Load next batch should clear active_learning_df and move samples to main dataset
    session.load_next_batch()
    # Cache should be empty since active_learning_df is now empty
    assert len(session._label_cache) == 0
    # Active learning df should be empty
    assert len(session.active_learning_df) == 0
    # Distribution and counts should be recalculated (cache invalidated)
    assert session._label_distribution_cache is None
    assert session._active_learning_counts_cache is None


def test_no_double_counting_after_training(base_config):
    """Test that labels are not double-counted after training."""
    cfg, out = base_config

    # Create a fresh session for this test
    session = Session(cfg)
    session.out = out
    session.update_predictions()

    # Get initial distribution
    initial_normal, initial_anomalous = session.get_label_distribution()
    initial_total = initial_normal + initial_anomalous

    # Add some new labels
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")
    session.label_image(2, "normal")

    # Check distribution before training
    pre_train_normal, pre_train_anomalous = session.get_label_distribution()
    pre_train_total = pre_train_normal + pre_train_anomalous

    # Should have increased by 3 (2 normal + 1 anomaly)
    assert pre_train_total == initial_total + 3
    assert pre_train_normal == initial_normal + 2
    assert pre_train_anomalous == initial_anomalous + 1

    # Verify that active learning df has the new labels
    assert len(session.active_learning_df) == 3
    active_normal, active_anomalous = session.get_active_learning_counts()
    assert active_normal == 2
    assert active_anomalous == 1

    # Train the model (this should move active_learning_df to main dataset)
    session.train(cfg)

    # Check distribution after training - should be the same as before training
    post_train_normal, post_train_anomalous = session.get_label_distribution()
    post_train_total = post_train_normal + post_train_anomalous

    # The total should remain the same (no double-counting)
    assert post_train_total == pre_train_total
    assert post_train_normal == pre_train_normal
    assert post_train_anomalous == pre_train_anomalous

    # Active learning df should be empty after training
    assert len(session.active_learning_df) == 0

    # Active learning counts should be zero
    active_normal_after, active_anomalous_after = session.get_active_learning_counts()
    assert active_normal_after == 0
    assert active_anomalous_after == 0


def test_no_double_counting_after_load_next_batch(base_config):
    """Test that labels are not double-counted after load_next_batch."""
    cfg, out = base_config

    # Create a fresh session for this test
    session = Session(cfg)
    session.out = out
    session.update_predictions()

    # Get initial distribution
    initial_normal, initial_anomalous = session.get_label_distribution()
    initial_total = initial_normal + initial_anomalous

    # Add some new labels
    session.label_image(0, "normal")
    session.label_image(1, "anomaly")

    # Check distribution before load_next_batch
    pre_load_normal, pre_load_anomalous = session.get_label_distribution()
    pre_load_total = pre_load_normal + pre_load_anomalous

    # Should have increased by 2 (1 normal + 1 anomaly)
    assert pre_load_total == initial_total + 2
    assert pre_load_normal == initial_normal + 1
    assert pre_load_anomalous == initial_anomalous + 1

    # Verify that active learning df has the new labels
    assert len(session.active_learning_df) == 2

    # Load next batch (this should move active_learning_df to main dataset)
    session.load_next_batch()

    # Check distribution after load_next_batch - should be the same as before
    post_load_normal, post_load_anomalous = session.get_label_distribution()
    post_load_total = post_load_normal + post_load_anomalous

    # The total should remain the same (no double-counting)
    assert post_load_total == pre_load_total
    assert post_load_normal == pre_load_normal
    assert post_load_anomalous == pre_load_anomalous

    # Active learning df should be empty after load_next_batch
    assert len(session.active_learning_df) == 0

    # Active learning counts should be zero
    active_normal_after, active_anomalous_after = session.get_active_learning_counts()
    assert active_normal_after == 0
    assert active_anomalous_after == 0


def test_iteration_scores_saved_after_training(base_config):
    """Test that unlabelled and test scores are saved after training with correct mappings."""
    cfg, out = base_config

    # Create a fresh session for this test
    session = Session(cfg)
    session.out = out

    # Train the model
    session.train(cfg)

    # Check that session tracker has the iteration
    assert len(session.session_tracker.session_iterations) >= 1

    # Get the latest iteration
    latest_iteration = session.session_tracker.session_iterations[-1]

    # Verify unlabelled scores file was created and path stored
    assert latest_iteration.unlabelled_scores_file is not None
    assert os.path.exists(latest_iteration.unlabelled_scores_file)

    # Load and verify unlabelled scores
    unlabelled_df = pd.read_csv(latest_iteration.unlabelled_scores_file)
    assert "filename" in unlabelled_df.columns
    assert "score" in unlabelled_df.columns
    assert len(unlabelled_df) > 0

    # Verify score values are valid probabilities
    assert unlabelled_df["score"].min() >= 0.0
    assert unlabelled_df["score"].max() <= 1.0

    # Verify that filenames in the CSV match the session's filenames
    csv_filenames = set(unlabelled_df["filename"].tolist())
    session_filenames = set(session.filenames.tolist())
    assert csv_filenames == session_filenames, "Saved filenames don't match session filenames"

    # Verify score mapping: check a few samples match between CSV and session
    for idx, (filename, score) in enumerate(zip(session.filenames[:5], session.scores[:5])):
        csv_score = unlabelled_df[unlabelled_df["filename"] == filename]["score"].values[0]
        assert abs(csv_score - score) < 1e-6, (
            f"Score mismatch for {filename}: {csv_score} vs {score}"
        )

    # If test_ratio > 0, verify test scores were also saved
    if cfg.test_ratio > 0:
        assert latest_iteration.test_scores_file is not None
        assert os.path.exists(latest_iteration.test_scores_file)

        # Load and verify test scores
        test_df = pd.read_csv(latest_iteration.test_scores_file)
        assert "filename" in test_df.columns
        assert "score" in test_df.columns
        assert len(test_df) > 0

        # Verify test score values are valid probabilities
        assert test_df["score"].min() >= 0.0
        assert test_df["score"].max() <= 1.0


def test_iteration_scores_no_test_set(base_config):
    """Test that only unlabelled scores are saved when test_ratio is 0."""
    cfg, out = base_config

    # Modify config to have no test set
    cfg_no_test = cfg.copy()
    cfg_no_test.test_ratio = 0.0

    # Create a fresh session
    session = Session(cfg_no_test)
    session.out = out

    # Train the model
    session.train(cfg_no_test)

    # Get the latest iteration
    latest_iteration = session.session_tracker.session_iterations[-1]

    # Unlabelled scores should still be saved
    assert latest_iteration.unlabelled_scores_file is not None
    assert os.path.exists(latest_iteration.unlabelled_scores_file)

    # Test scores should not be saved (no test set)
    assert latest_iteration.test_scores_file is None
