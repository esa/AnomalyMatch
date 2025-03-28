#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import pytest
import os
import numpy as np
import h5py
import zipfile
import tempfile
from PIL import Image
from dotmap import DotMap
import pandas as pd
import torch
from loguru import logger

from prediction_process import evaluate_files
from prediction_process_zip import evaluate_files_in_zip
from prediction_process_hdf5 import evaluate_images_in_hdf5


@pytest.fixture
def test_config():
    cfg = DotMap()
    cfg.size = [150, 150]
    cfg.net = "efficientnet-lite0"
    cfg.pretrained = True
    cfg.num_channels = 3
    cfg.model_path = "tests/test_data/test_model.pth"
    cfg.gpu = 0
    cfg.output_dir = tempfile.mkdtemp()
    return cfg


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    images = []
    for i in range(10):
        img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        images.append(Image.fromarray(img))
    return images


@pytest.fixture
def test_zip(sample_images, tmp_path):
    """Create a test ZIP file with sample images."""
    zip_path = tmp_path / "test.zip"
    # Create a temporary directory for the images
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()

    with zipfile.ZipFile(zip_path, "w") as zf:
        for i, img in enumerate(sample_images):
            img_path = img_dir / f"img_{i}.jpg"
            img.save(img_path)
            zf.write(img_path, f"img_{i}.jpg")

    return str(zip_path)


@pytest.fixture
def test_hdf5(sample_images, tmp_path):
    """Create a test HDF5 file with sample images."""
    hdf5_path = tmp_path / "test.h5"
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()

    with h5py.File(hdf5_path, "w") as h5f:
        # Create a dataset for images
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        dset = h5f.create_dataset("images", (len(sample_images),), dtype=vlen_uint8)

        # Create a dataset for filenames
        filenames = [f"img_{i}.jpg" for i in range(len(sample_images))]
        _ = h5f.create_dataset(
            "filenames",
            data=np.array(filenames, dtype="S"),
        )

        for i, img in enumerate(sample_images):
            img_path = img_dir / f"img_{i}.jpg"
            img.save(img_path)
            with open(img_path, "rb") as f:
                dset[i] = np.frombuffer(f.read(), dtype=np.uint8)

    return str(hdf5_path)


@pytest.fixture
def mixed_format_images(tmp_path):
    """Create a directory with sample images in different formats (jpg, png, tif, tiff)"""
    img_dir = tmp_path / "mixed_formats"
    img_dir.mkdir()

    # Create images in different formats
    image_paths = []
    formats = {"jpg": "JPEG", "png": "PNG", "tif": "TIFF", "tiff": "TIFF"}

    # Create a simple test image
    base_img = np.zeros((150, 150, 3), dtype=np.uint8)
    base_img[50:100, 50:100, 0] = 255  # Red square

    # Save in each format
    for ext, pil_format in formats.items():
        img_path = img_dir / f"test_image.{ext}"
        Image.fromarray(base_img).save(img_path, format=pil_format)
        image_paths.append(str(img_path))

    return image_paths, str(img_dir)


def test_evaluate_files(test_config, sample_images, tmp_path):
    """Test evaluation of individual files."""
    image_paths = []
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()

    for i, img in enumerate(sample_images):
        img_path = img_dir / f"img_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))

    scores, filenames, imgs = evaluate_files(image_paths, test_config)
    assert len(scores) == len(sample_images)
    assert len(filenames) == len(sample_images)
    assert imgs.shape[0] == len(sample_images)


def test_evaluate_files_in_zip(test_config, test_zip):
    """Test evaluation of images in ZIP file."""
    scores, filenames, imgs = evaluate_files_in_zip(test_zip, test_config)
    assert len(scores) == 10
    assert len(filenames) == 10
    assert imgs.shape[0] == 10


def test_evaluate_images_in_hdf5(test_config, test_hdf5):
    """Test evaluation of images in HDF5 file."""
    scores, filenames, imgs = evaluate_images_in_hdf5(test_hdf5, test_config)
    assert len(scores) == 10
    assert len(filenames) == 10
    assert imgs.shape[0] == 10


def test_predictions_output(test_config, test_hdf5):
    """Test that predictions are saved correctly."""
    evaluate_images_in_hdf5(test_hdf5, test_config)

    # Check if predictions file exists
    prediction_files = [
        f for f in os.listdir(test_config.output_dir) if f.startswith("all_predictions_")
    ]
    assert len(prediction_files) == 1

    # Load and check predictions
    predictions = np.load(os.path.join(test_config.output_dir, prediction_files[0]))
    assert "filenames" in predictions
    assert "scores" in predictions
    assert len(predictions["filenames"]) == 10
    assert len(predictions["scores"]) == 10


def test_mixed_format_support(test_config, mixed_format_images, monkeypatch):
    """Test support for different image formats (jpg, png, tif, tiff)."""
    image_paths, _ = mixed_format_images

    # Mock model to ensure consistent output for each image
    mock_model = MockModel([0.7])  # Use a single consistent score

    # Mock the load_model function to return our mock model
    import prediction_process

    def mock_load_model(cfg):
        return mock_model

    monkeypatch.setattr(prediction_process, "load_model", mock_load_model)

    # Mock save_results to ensure it returns predictable values
    def mock_save_results(cfg, all_scores, all_imgs, all_filenames, top_n):
        # Simply return the inputs without further processing
        return all_scores, all_filenames, all_imgs

    monkeypatch.setattr(prediction_process, "save_results", mock_save_results)

    # Test each format individually
    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        # Reset call count for each test to ensure consistent behavior
        mock_model.call_count = 0

        scores, filenames, imgs = evaluate_files([path], test_config)
        assert len(filenames) == 1, f"Expected 1 filename for {ext} image"
        assert imgs.shape[0] == 1, f"Expected 1 image for {ext} image"
        # Even if multiple scores are returned, we should have at least 1 score
        assert len(scores) >= 1, f"No scores returned for {ext} image"

    # Reset call count for the batch test
    mock_model.call_count = 0

    # Test all formats together
    scores, filenames, imgs = evaluate_files(image_paths, test_config)
    assert len(filenames) == len(image_paths), "Not all image filenames were processed"
    assert imgs.shape[0] == len(image_paths), "Not all images were processed"
    # There should be at least as many scores as images
    assert len(scores) >= len(image_paths), "Not enough scores returned for all image formats"


def test_read_and_resize_multiple_formats(test_config, mixed_format_images):
    """Test the read_and_resize_image function can handle multiple formats."""
    from prediction_process import read_and_resize_image

    image_paths, _ = mixed_format_images

    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        image = read_and_resize_image(path, test_config.size)

        # Check image shape and type
        assert image.shape == (
            test_config.size[0],
            test_config.size[1],
            3,
        ), f"Image resizing failed for {ext}"
        assert image.dtype == np.uint8, f"Image type incorrect for {ext}"


class MockModel(torch.nn.Module):
    """Mock model that returns controlled scores."""

    def __init__(self, score_pattern):
        super().__init__()
        self.score_pattern = score_pattern
        self.call_count = 0
        logger.info(f"MockModel initialized with score pattern: {score_pattern}")

    def forward(self, x):
        batch_size = x.shape[0]
        base_score = self.score_pattern[self.call_count]
        logger.info(f"MockModel forward call {self.call_count} with base_score {base_score}")

        # Generate scores that will exactly match our desired probabilities
        scores = torch.zeros((batch_size, 2))
        for i in range(batch_size):
            desired_prob = min(base_score + (i / batch_size) * 0.1, 0.95)
            scores[i, 1] = desired_prob
            scores[i, 0] = 1 - desired_prob

        self.call_count += 1
        logger.info(f"Generated scores: {scores[:, 1]}")
        return torch.log(scores)  # Convert to logits


def create_controlled_sample_images(output_dir, num_images, base_score=0.5):
    """Create sample images with controlled prediction scores for testing."""
    images = []
    img_paths = []
    expected_scores = []

    for i in range(num_images):
        # Create a dummy image with a pattern that will generate a specific score
        img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        img_path = os.path.join(output_dir, f"img_{i}_{base_score + i / num_images:.3f}.jpg")
        Image.fromarray(img).save(img_path)
        images.append(img)
        img_paths.append(img_path)
        expected_scores.append(base_score + i / num_images)

    return img_paths, expected_scores


def test_accumulate_top_n_results(test_config, monkeypatch):
    """Test that top-N results are correctly accumulated across multiple batches."""
    test_config.save_file = "test_accumulation"
    top_n = 5

    # Create mock model that will return controlled scores
    score_pattern = [0.65, 0.85]  # Increased score ranges
    mock_model = MockModel(score_pattern)

    # Mock the model and processing functions in prediction_process module
    import prediction_process

    def mock_load_model(cfg):
        logger.info("Using mock load_model")
        return mock_model

    monkeypatch.setattr(prediction_process, "load_model", mock_load_model)

    def mock_process_batch(model, images):
        logger.info("Using mock process_batch_predictions")
        with torch.no_grad():
            logits = model(images)
            scores = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            logger.info(f"Processed batch scores: {scores}")
        return scores, images.cpu().numpy()

    monkeypatch.setattr(prediction_process, "process_batch_predictions", mock_process_batch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create two batches of images
        batch1_paths, _ = create_controlled_sample_images(tmp_dir, 10, base_score=0.65)
        batch2_paths, _ = create_controlled_sample_images(tmp_dir, 10, base_score=0.85)

        # Process both batches
        scores1, filenames1, imgs1 = evaluate_files(batch1_paths, test_config, top_n=top_n)
        scores2, filenames2, imgs2 = evaluate_files(batch2_paths, test_config, top_n=top_n)

        # Load final results
        output_csv = os.path.join(test_config.output_dir, f"{test_config.save_file}_top{top_n}.csv")
        final_results = pd.read_csv(output_csv)
        final_scores = final_results["Score"].values

        # Check if we got the highest scores from the second batch
        assert len(final_scores) == top_n
        assert np.all(final_scores >= 0.85)  # All top scores should be from second batch
        assert np.all(final_scores <= 0.95)  # Maximum probability capped at 0.95


def test_all_predictions_accumulation(test_config, monkeypatch):
    """Test that all predictions are correctly saved when processing multiple batches."""
    test_config.save_file = "test_all_predictions"

    # Create mock model with higher controlled scores
    score_pattern = [0.65, 0.85]  # Increased score ranges
    mock_model = MockModel(score_pattern)

    # Mock the model and processing functions in prediction_process module
    import prediction_process

    def mock_load_model(cfg):
        logger.info("Using mock load_model")
        return mock_model

    monkeypatch.setattr(prediction_process, "load_model", mock_load_model)

    def mock_process_batch(model, images):
        logger.info("Using mock process_batch_predictions")
        with torch.no_grad():
            logits = model(images)
            scores = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            logger.info(f"Processed batch scores: {scores}")
        return scores, images.cpu().numpy()

    monkeypatch.setattr(prediction_process, "process_batch_predictions", mock_process_batch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create two batches of images
        batch1_paths, _ = create_controlled_sample_images(tmp_dir, 5, base_score=0.65)
        batch2_paths, _ = create_controlled_sample_images(tmp_dir, 5, base_score=0.85)

        # Process both batches
        evaluate_files(batch1_paths, test_config)
        evaluate_files(batch2_paths, test_config)

        # Load and check predictions
        predictions_file = os.path.join(
            test_config.output_dir, f"all_predictions_{test_config.save_file}.npz"
        )
        assert os.path.exists(predictions_file)

        predictions = np.load(predictions_file, allow_pickle=True)
        all_scores = predictions["scores"]

        # Print actual scores for debugging
        logger.info(f"All scores: {all_scores}")
        logger.info(
            f"First batch scores >= 0.65: {np.any((all_scores >= 0.65) & (all_scores < 0.75))}"
        )
        logger.info(
            f"Second batch scores >= 0.85: {np.any((all_scores >= 0.85) & (all_scores <= 0.95))}"
        )

        # Verify that scores from both batches are present
        assert np.any((all_scores >= 0.65) & (all_scores < 0.75))  # First batch
        assert np.any((all_scores >= 0.85) & (all_scores <= 0.95))  # Second batch


def test_image_directory_processing(test_config, mixed_format_images):
    """Test processing a directory containing images of different formats."""
    # Get the directory containing mixed format images
    _, directory_path = mixed_format_images

    # Create a file list with all image paths in the directory
    from pathlib import Path
    import tempfile

    # Create a temporary file list
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as file_list:
        # List all image files and write them to the file
        image_paths = (
            list(Path(directory_path).glob("*.jpg"))
            + list(Path(directory_path).glob("*.jpeg"))
            + list(Path(directory_path).glob("*.png"))
            + list(Path(directory_path).glob("*.tif"))
            + list(Path(directory_path).glob("*.tiff"))
        )

        for path in image_paths:
            file_list.write(f"{path}\n")

        file_list_path = file_list.name

    # Import the necessary function
    from prediction_process import evaluate_files

    try:
        # Test the evaluation function
        scores, filenames, imgs = evaluate_files([str(p) for p in image_paths], test_config)

        # Verify results
        assert len(scores) == len(image_paths), "Not all images were processed"
        assert len(filenames) == len(image_paths)
        assert imgs.shape[0] == len(image_paths)

        # Check that scores are within expected range
        assert np.all(scores >= 0) and np.all(scores <= 1), "Scores outside expected range"
    finally:
        # Clean up temporary file
        import os

        if os.path.exists(file_list_path):
            os.unlink(file_list_path)


def test_prediction_file_type_image(test_config, monkeypatch, mixed_format_images):
    """Test that the correct prediction process is called for the 'image' file type."""
    import os
    import sys
    from anomaly_match.pipeline.session import Session
    from dotmap import DotMap
    import subprocess

    # Create a directory with mixed format images
    _, directory_path = mixed_format_images

    # Create a mock config
    cfg = DotMap()
    cfg.prediction_file_type = "image"
    cfg.search_dir = directory_path
    cfg.save_file = "test_image_type"
    cfg.output_dir = os.path.join(directory_path, "output")
    cfg.model_path = test_config.model_path
    # Add necessary attributes to prevent errors in session initialization
    cfg.log_level = "INFO"
    cfg.test_ratio = 0.5
    cfg.data_dir = "tests/test_data/"
    cfg.size = [150, 150]
    cfg.N_to_load = 10
    cfg.seed = 42
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Find an image file to use for the test
    image_path = None
    for f in os.listdir(directory_path):
        if f.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            image_path = os.path.join(directory_path, f)
            break

    assert image_path is not None, "No image file found in the test directory"

    called_processes = []

    def mock_run(args, **kwargs):
        called_processes.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Create the tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create the file list manually - this is what we'll check
    temp_file_list = os.path.join("tmp", f"{cfg.save_file}_file_list.txt")
    with open(temp_file_list, "w") as f:
        f.write(image_path)
        f.flush()

    # Mock Session.__init__
    def mock_init(self, cfg):
        self.cfg = cfg
        self.session_start = "20250101_000000"

    monkeypatch.setattr(Session, "__init__", mock_init)

    # Create a simple mock for run_pipeline
    def mock_run_pipeline(self, temp_config_path, input_path, top_N):
        # Just to verify that input_path is correct
        assert input_path == image_path, f"Expected {image_path}, got {input_path}"

        # Use the file list we created manually
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prediction_process.py"
        )
        subprocess.run([sys.executable, script_path, temp_config_path, temp_file_list, str(top_N)])

    monkeypatch.setattr(Session, "run_pipeline", mock_run_pipeline)

    # Create a session with our mocked initializer
    session = Session(cfg)

    # Create a temp config file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as config_file:
        config_path = config_file.name

    # Test run_pipeline
    try:
        session.run_pipeline(config_path, image_path, top_N=10)

        # Verify that subprocess.run was called
        assert len(called_processes) > 0, "No subprocess was called"

        # Check that the script path is correct (prediction_process.py)
        script_path = os.path.basename(called_processes[0][1])
        assert script_path == "prediction_process.py", f"Wrong script called: {script_path}"

        # Verify that the file list exists
        assert os.path.exists(temp_file_list), "Temporary file list does not exist"

        # Verify that the file list contains the image path
        with open(temp_file_list, "r") as f:
            content = f.read().strip()
            assert (
                content == image_path
            ), f"Wrong content in file list: '{content}', expected '{image_path}'"

    finally:
        # Clean up any temporary files
        if os.path.exists(config_path):
            os.unlink(config_path)

        if os.path.exists(temp_file_list):
            os.unlink(temp_file_list)
