#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from concurrent.futures import ThreadPoolExecutor
import imageio.v2 as imageio
import numpy as np
import os
import h5py
import pandas as pd
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from loguru import logger

from .Label import Label


class AnomalyDetectionDataset(torch.utils.data.Dataset):
    """AnomalyDetectionDataset for binary classification of normal vs anomaly images."""

    def __init__(
        self,
        test_ratio,
        root_dir="data/",
        transform=None,
        seed=42,
        size=[300, 300],
        use_hdf5=True,
        file_extensions=[".jpeg", ".jpg", ".png", ".tif", ".tiff"],
        N_to_load=10000,
        label_file=None,
    ):
        """
        Args:
            test_ratio (float): Ratio of test data to total data.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            seed (int): seed used for train/test split
            use_hdf5 (bool): If true, load/save the dataset from/to HDF5 file.
            file_extensions (list): List of file extensions to include when loading images.
            N_to_load (int): Number of unlabeled images to load per batch
            label_file (string, optional): Path to the CSV file with labels. If None,
                will look for 'labeled_data.csv' in the root directory.
        """
        logger.debug(f"Loading AnomalyDetectionDataset from {root_dir}")

        # Initialize key variables
        self.classes = ["normal", "anomaly"]
        self.seed = seed
        self.size = size
        self.num_channels = 3
        self.root_dir = root_dir
        self.transform = transform
        self.test_ratio = test_ratio
        self.file_extensions = file_extensions
        self.N_to_load = N_to_load
        self.current_batch_idx = 0
        self.current_file_idx = 0
        self.label_file = label_file if label_file else os.path.join(root_dir, "labeled_data.csv")

        # Initialize paths including size in the name
        self.labeled_hdf5 = os.path.join(root_dir, f"labeled_{size[0]}x{size[1]}.hdf5")
        self.batch_hdf5_template = os.path.join(root_dir, f"batch_{size[0]}x{size[1]}_{{}}.hdf5")

        logger.trace(f"Using HDF5 files: {self.labeled_hdf5}, {self.batch_hdf5_template}")

        # Get all filenames first
        self.all_filenames = []
        for f in os.listdir(self.root_dir):
            if any(f.lower().endswith(ext.lower()) for ext in self.file_extensions):
                self.all_filenames.append(f)

        # If we have fewer than N_to_load images, set N_to_load to the number of images
        self.N_to_load = min(self.N_to_load, len(self.all_filenames))

        # Shuffle them to ensure randomness
        np.random.seed(self.seed)
        np.random.shuffle(self.all_filenames)

        logger.info(f"Found {len(self.all_filenames)} total images in {self.root_dir}")

        # To store the dataset split
        self.split_indices = None

        # Initialize data dictionary
        self.data_dict = {}

        # Load or create the dataset
        if os.path.exists(self.labeled_hdf5) and use_hdf5:
            self._load_labeled_from_hdf5()
            self._load_current_unlabeled_batch()
        else:
            self._load_initial_data()
            self.mean, self.std = self.compute_mean_std()
            # self._save_split_hdf5()

        # Load the labels from the CSV file
        self._load_csv_and_apply_labels()

        # Split the data into training and testing sets
        self._split_data()

    def _load_csv_and_apply_labels(self):
        """Load CSV label file and apply labels to the dataset."""
        logger.debug(f"Reloading CSV files from {self.label_file} and applying labels")
        # Assert file is there
        assert os.path.exists(self.label_file), (
            f"No label file found at {self.label_file}. Please provide a csv file in the format: filename,"
            + "label with labels 'normal' and 'anomaly'."
        )
        labeled_data = pd.read_csv(self.label_file)

        # Assert the required columns exist in the CSV
        required_columns = ["filename", "label"]
        for col in required_columns:
            assert col in labeled_data.columns, f"CSV file must contain column '{col}'"

        # Check that labels are valid
        assert set(labeled_data["label"].unique()) <= set(
            ["normal", "anomaly"]
        ), "Labels should be either 'normal' or 'anomaly' but found" + str(
            set(labeled_data["label"].unique())
        )

        # Label distribution in the new CSV
        normal_count = labeled_data["label"].value_counts().get("normal", 0)
        anomaly_count = labeled_data["label"].value_counts().get("anomaly", 0)
        logger.debug(
            f"Label distribution in CSV file: Normal: {normal_count}, Anomaly: {anomaly_count}"
        )

        # Update the dataset with new labels
        self.update_labels(labeled_data, update_training_data=False)

    def _read_and_resize_image(self, filepath):
        """Read an image file and resize it."""
        try:
            image = imageio.imread(filepath)
            if len(image.shape) == 2:  # Convert grayscale to RGB
                image = np.stack((image,) * 3, axis=-1)
            elif len(image.shape) > 2 and image.shape[2] > 3:  # Handle RGBA or other formats
                image = image[:, :, :3]  # Keep only RGB channels
            if image.shape[:2] != tuple(self.size):
                image = img_as_ubyte(resize(image, self.size, anti_aliasing=True))
            return img_as_ubyte(image)
        except Exception as e:
            logger.error(f"Error reading image {filepath}: {e}")
            # Raise exception to stop execution
            raise e

    def _load_initial_data(self):
        """Load labeled data and first batch of unlabeled data."""
        # Load labeled data from CSV
        labeled_data = pd.read_csv(self.label_file)
        labeled_dict = labeled_data.set_index("filename")["label"].to_dict()
        labeled_files = set(labeled_dict.keys())

        # Filter out non-existent files
        labeled_files = [f for f in labeled_files if f in self.all_filenames]

        def load_labeled_image(filename):
            try:
                sub_f = os.path.join(self.root_dir, filename)
                image = self._read_and_resize_image(sub_f)
                label = Label.NORMAL if labeled_dict[filename] == "normal" else Label.ANOMALY
                return filename, (image, label)
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                return None

        # Load labeled images in parallel
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(load_labeled_image, labeled_files),
                    desc="Loading labeled data",
                    total=len(labeled_files),
                )
            )

        # Update data dictionary with successfully loaded images
        self.data_dict.update({k: v for r in results if r is not None for k, v in [r]})

        logger.debug(f"Loaded {len(self.data_dict)} labeled images")

        # Then load first batch of unlabeled data
        self._load_next_unlabeled_batch()

    def _load_next_unlabeled_batch(self):
        # Remove unlabeled data from previous batch
        self.data_dict = {k: v for k, v in self.data_dict.items() if v[1] != Label.UNKNOWN}

        # Get all unlabeled filenames that have not been processed yet
        unlabeled_files = [f for f in self.all_filenames if f not in self.data_dict]

        logger.trace(f"{len(unlabeled_files)} unlabeled images remain.")

        # Check if there are any unlabeled files left
        if not unlabeled_files:
            logger.debug("No unlabeled images left to load.")
            return

        # Calculate batch indices
        start_idx = self.current_file_idx
        end_idx = min(self.current_file_idx + self.N_to_load, len(unlabeled_files))

        batch_files = unlabeled_files[start_idx:end_idx]

        # Load new batch

        def load_and_resize(filename):
            sub_f = os.path.join(self.root_dir, filename)
            image = self._read_and_resize_image(sub_f)
            return filename, image

        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(load_and_resize, batch_files),
                    desc=f"Loading batch {self.current_batch_idx}",
                    total=len(batch_files),
                )
            )

        for filename, image in results:
            self.data_dict[filename] = (image, Label.UNKNOWN)

        self.current_batch_idx += 1
        self.current_file_idx = end_idx
        if self.current_file_idx == len(unlabeled_files):
            self.current_file_idx = 0
        logger.debug(
            f"Loaded {len(batch_files)} unlabeled images in batch {self.current_batch_idx - 1}"
        )

    def load_next_batch(self, N_to_load=None):
        """Public method to load next batch of unlabeled data."""
        if N_to_load is not None:
            self.N_to_load = np.minimum(N_to_load, self.get_nr_of_unlabeled_images())
        self._load_next_unlabeled_batch()

    def compute_mean_std(self):
        """Compute the mean and standard deviation of the dataset."""
        logger.debug("Computing mean and standard deviation of the dataset")
        # Process in batches to reduce memory usage
        batch_size = 100
        means = []
        stds = []

        images = [img for img, _ in self.data_dict.values()]
        for i in range(0, len(images), batch_size):
            batch = np.array(images[i : i + batch_size]).astype(np.float32) / 255.0
            means.append(batch.mean(axis=(0, 1, 2)))
            stds.append(batch.std(axis=(0, 1, 2)))

        mean = np.array(means).mean(axis=0)
        std = np.array(stds).mean(axis=0)

        logger.debug(f"Dataset Mean: {mean}, Std: {std}")
        return mean, std

    def _split_data(self):
        """Splits the data into training and testing sets, only using labeled images."""
        logger.debug(f"Splitting data with seed={self.seed}")
        # Assert data was not split before
        assert self.split_indices is None, "Data was already split before"

        # Prepare data for splitting only with labeled data
        labeled_data = {
            filename: (img, label)
            for filename, (img, label) in self.data_dict.items()
            if label != Label.UNKNOWN
        }

        filenames = np.array(list(labeled_data.keys()))
        labels = np.array([label for _, label in labeled_data.values()])
        logger.trace(f"Labels to split: {labels}")

        # Handle cases where all labels might be the same by checking for non-unique labels
        stratify = labels if len(set(labels)) > 1 else None

        logger.trace(f"Number of labeled data points: {len(filenames)}")
        if self.test_ratio > 0:
            filenames_train, filenames_test = train_test_split(
                filenames,
                test_size=self.test_ratio,
                random_state=self.seed,
                stratify=stratify,
            )
        else:
            filenames_train, filenames_test = filenames, []

        # Store split indices to maintain consistency
        self.split_indices = {"train": set(filenames_train), "test": set(filenames_test)}

    @property
    def unlabeled_filepaths(self):
        """Return unlabeled filepaths."""
        return [
            os.path.join(self.root_dir, filename)
            for filename in self.all_filenames
            if (filename not in self.data_dict or self.data_dict[filename][1] == Label.UNKNOWN)
        ]

    @property
    def unlabeled_filenames(self):
        """Return unlabeled filenames."""
        return [
            filename
            for filename in self.all_filenames
            if (filename not in self.data_dict or self.data_dict[filename][1] == Label.UNKNOWN)
        ]

    @property
    def unlabeled(self):
        """Return unlabeled data in format [[imgs],[filenames]]."""
        filenames = [
            filename for filename in self.data_dict if self.data_dict[filename][1] == Label.UNKNOWN
        ]
        return [
            [self.data_dict[filename][0] for filename in filenames],
            filenames,
        ]

    @property
    def train_data(self):
        """Return training data in format [[filenames],[imgs],[labels]])."""
        filenames = [filename for filename in self.split_indices["train"]]
        return [
            filenames,
            [self.data_dict[filename][0] for filename in filenames],
            [self.data_dict[filename][1] for filename in filenames],
        ]

    @property
    def test_data(self):
        """Return testing data in format [[filenames],[imgs],[labels]])."""
        filenames = [filename for filename in self.split_indices["test"]]
        return [
            filenames,
            [self.data_dict[filename][0] for filename in filenames],
            [self.data_dict[filename][1] for filename in filenames],
        ]

    def __len__(self):
        return len(self.data_dict)

    def get_nr_of_unlabeled_images(self):
        return len(self.all_filenames) - len(self.train_data[0]) - len(self.test_data[0])

    def get_nr_of_batches(self):
        return int(np.ceil(self.get_nr_of_unlabeled_images() / self.N_to_load))

    def reset_batch_idx(self):
        self.current_batch_idx = 0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]

    def update_labels(self, new_labels_df, update_training_data=True):
        """
        Update the dataset with new labels for previously unlabeled images.

        Args:
            new_labels_df (pd.DataFrame): DataFrame with columns 'filename" and 'label'.
            update_training_data (bool): If True, update the training data with new labels.
        """
        logger.debug(f"Updating labels for {new_labels_df.shape[0]} images")
        existing_values_changed = 0

        for _, row in new_labels_df.iterrows():
            filename = os.path.basename(row["filename"])
            label = row["label"].lower()

            if label == "normal":
                label_enum = Label.NORMAL
            elif label == "anomaly":
                label_enum = Label.ANOMALY
            else:
                raise ValueError(f"Invalid label {label} for {filename}")

            # Check if the filename already exists in the data
            if filename in self.data_dict:
                image, current_label = self.data_dict[filename]
                if current_label != label_enum and current_label != Label.UNKNOWN:
                    self.data_dict[filename] = (image, label_enum)
                    existing_values_changed += 1
                elif current_label == Label.UNKNOWN:
                    self.data_dict[filename] = (image, label_enum)
            else:
                logger.warning(f"Updating labels: Filename {filename} not found in dataset")

        logger.debug(f"Updated {existing_values_changed} existing labels")

        # Update training data if needed
        if update_training_data:
            self._update_training_data()

    def _update_training_data(self):
        """Update the training data with new labels, preserving the test split."""
        logger.info("Starting to update training data.")

        if self.split_indices is None:
            logger.error("Split indices not found. Cannot update training data.")
            raise ValueError("Split indices not found. Cannot update training data.")

        # Prepare data for updating only with labeled data
        labeled_data = {
            filename: (img, label)
            for filename, (img, label) in self.data_dict.items()
            if label != Label.UNKNOWN
        }

        filenames = np.array(list(labeled_data.keys()))
        logger.debug(f"Number of labeled data points: {len(filenames)}")

        # Remove test data from the labeled data
        filenames = [f for f in filenames if f not in self.split_indices["test"]]
        # Update split indices to include new training data
        self.split_indices["train"] = self.split_indices["train"].union(set(filenames))

        logger.debug(f"Number of training data points: {len(self.split_indices['train'])}")

    def save_as_hdf5(self, hdf5_path):
        with h5py.File(hdf5_path, "w") as f:
            # Save all data in a single dataset
            dtype = h5py.special_dtype(vlen=str)
            data_dtype = np.dtype(
                [
                    ("filename", dtype),
                    ("image", h5py.vlen_dtype(np.dtype("uint8"))),
                ]
            )

            data = [(filename, image.flatten()) for filename, (image, _) in self.data_dict.items()]

            dset = f.create_dataset("data", (len(data),), dtype=data_dtype)

            for i, (filename, image) in enumerate(data):
                dset[i] = (filename, image)

            # Save mean and std
            f.create_dataset("mean", data=self.mean)
            f.create_dataset("std", data=self.std)

        logger.info(f"Dataset saved to {hdf5_path}")

    def load_from_hdf5(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            self.data_dict = {}

            # Load data from the compound dataset
            for entry in f["data"]:
                filename = entry["filename"].decode("utf-8")  # Decode bytes to string
                image = np.array(entry["image"]).reshape(self.size + [3])  # Reshape back to image
                self.data_dict[filename] = (image, Label.UNKNOWN)

            # Load mean and std if they exist
            if "mean" in f and "std" in f:
                self.mean = f["mean"][:]
                self.std = f["std"][:]
                logger.debug(f"Loaded Mean: {self.mean}, Std: {self.std}")
            else:
                logger.warning("Mean and Std not found in HDF5 file; computing them now.")
                self.mean, self.std = self.compute_mean_std()

        logger.info(f"Dataset loaded from {hdf5_path}")

    def _save_split_hdf5(self):
        """Save labeled and unlabeled data in separate HDF5 files."""
        # Save labeled data
        with h5py.File(self.labeled_hdf5, "w") as f:
            labeled_data = {k: v for k, v in self.data_dict.items() if v[1] != Label.UNKNOWN}
            dtype = h5py.special_dtype(vlen=str)
            data_dtype = np.dtype(
                [
                    ("filename", dtype),
                    ("image", h5py.vlen_dtype(np.dtype("uint8"))),
                    ("label", np.int8),
                ]
            )

            data = [(k, v[0].flatten(), v[1]) for k, v in labeled_data.items()]
            dset = f.create_dataset("data", (len(data),), dtype=data_dtype)
            for i, (filename, image, label) in enumerate(data):
                dset[i] = (filename, image, label)

            f.create_dataset("mean", data=self.mean)
            f.create_dataset("std", data=self.std)

        # Save current unlabeled batch
        batch_file = self.batch_hdf5_template.format(self.current_batch_idx)
        with h5py.File(batch_file, "w") as f:
            unlabeled_data = {k: v for k, v in self.data_dict.items() if v[1] == Label.UNKNOWN}
            data = [(k, v[0].flatten()) for k, v in unlabeled_data.items()]

            dtype = h5py.special_dtype(vlen=str)
            data_dtype = np.dtype(
                [("filename", dtype), ("image", h5py.vlen_dtype(np.dtype("uint8")))]
            )

            dset = f.create_dataset("data", (len(data),), dtype=data_dtype)
            for i, (filename, image) in enumerate(data):
                dset[i] = (filename, image)

    def _load_labeled_from_hdf5(self):
        """Load labeled data from HDF5."""
        with h5py.File(self.labeled_hdf5, "r") as f:
            for entry in f["data"]:
                filename = entry["filename"].decode("utf-8")
                image = np.array(entry["image"]).reshape(self.size + [3])
                label = entry["label"]
                self.data_dict[filename] = (image, label)

            self.mean = f["mean"][:]
            self.std = f["std"][:]

    def _load_current_unlabeled_batch(self):
        """Load current batch of unlabeled data, checking against labeled data."""
        batch_file = self.batch_hdf5_template.format(self.current_batch_idx)
        if not os.path.exists(batch_file):
            self._load_next_unlabeled_batch()
            return

        with h5py.File(batch_file, "r") as f:
            for entry in f["data"]:
                filename = entry["filename"].decode("utf-8")
                # Skip if file is now labeled
                if filename not in self.data_dict:
                    image = np.array(entry["image"]).reshape(self.size + [3])
                    self.data_dict[filename] = (image, Label.UNKNOWN)
