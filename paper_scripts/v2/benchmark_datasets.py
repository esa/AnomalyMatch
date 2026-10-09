#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Dataset registry and labeled-set construction for the v2 benchmark.

This module is deliberately independent of the AnomalyMatch training/prediction
API. It knows only about the on-disk paper datasets (image folders, label CSVs)
and how to turn a chosen anomaly class + label budget into a ``labeled_data.csv``
that the AnomalyMatch training pipeline can consume.

The v1 paper benchmark validated performance by iteratively refining a model
across active-learning cycles. The v2 pipeline drops iterative refinement, so
here we instead build a *fixed* labeled set for a given label budget and train a
single model on it (see ``run_benchmark.py``).

Ground-truth evaluation always uses the full per-image class index from the
dataset's label CSV, independent of which images ended up in the labeled set.
"""

import os
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

# Root of the paper datasets on the shared team workspace. Kept as a module
# constant rather than a config field because these benchmark scripts are only
# ever run against this fixed data location.
PAPER_DATASETS_ROOT = "/media/team_workspaces/AnomalyMatch/paper_datasets"


@dataclass(frozen=True)
class BenchmarkDataset:
    """Static description of a benchmark dataset.

    Attributes:
        name: Short identifier used on the command line and in output paths.
        labels_csv: Path to the CSV holding one row per image with columns
            ``filename``, ``label_idx`` (and, for galaxymnist, a readable
            ``label`` name column).
        image_dir: Directory containing the individual image files referenced by
            ``filename`` in the labels CSV.
        class_name_to_idx: Mapping from human-readable anomaly class name to the
            integer ``label_idx`` used in the labels CSV.
    """

    name: str
    labels_csv: str
    image_dir: str
    class_name_to_idx: dict = field(default_factory=dict)


# miniImageNet class indices were verified by rendering sample images per class
# (the labels CSV only stores integer labels, no readable names). These five are
# the anomaly classes used in the AnomalyMatch paper; hourglass is the primary
# example. Each class has exactly 650 images (1.0% prevalence in 65k images).
MINIIMAGENET = BenchmarkDataset(
    name="miniimagenet",
    labels_csv=os.path.join(PAPER_DATASETS_ROOT, "labels_miniimagenet.csv"),
    image_dir=os.path.join(PAPER_DATASETS_ROOT, "miniimagenet"),
    class_name_to_idx={
        "hourglass": 57,
        "piano": 85,
        "guitar": 48,
        "printer": 68,
        "orange": 95,
    },
)

# galaxymnist stores readable names directly in the ``label`` column; the mapping
# here mirrors label_idx 0..3 and lets anomaly classes be selected by name.
GALAXYMNIST = BenchmarkDataset(
    name="galaxymnist",
    labels_csv=os.path.join(PAPER_DATASETS_ROOT, "labels_galaxymnist.csv"),
    image_dir=os.path.join(PAPER_DATASETS_ROOT, "galaxymnist"),
    class_name_to_idx={
        "smooth_round": 0,
        "smooth_cigar": 1,
        "edge_on_disk": 2,
        "unbarred_spiral": 3,
    },
)

REGISTRY = {ds.name: ds for ds in (MINIIMAGENET, GALAXYMNIST)}


def get_dataset(name):
    """Return the :class:`BenchmarkDataset` for ``name``.

    Args:
        name: Dataset identifier (``"miniimagenet"`` or ``"galaxymnist"``).

    Returns:
        The registered :class:`BenchmarkDataset` (raises ``KeyError`` for an
        unknown name).
    """
    return REGISTRY[name]


def resolve_anomaly_class(dataset, anomaly_class):
    """Resolve an anomaly class given as a name or an integer index.

    Args:
        dataset: The :class:`BenchmarkDataset` to resolve against.
        anomaly_class: Either a class name (e.g. ``"hourglass"``) or an integer
            ``label_idx``.

    Returns:
        The integer ``label_idx`` for the anomaly class (raises ``KeyError``
        for an unregistered class name).
    """
    if isinstance(anomaly_class, str) and not anomaly_class.lstrip("-").isdigit():
        return dataset.class_name_to_idx[anomaly_class]
    return int(anomaly_class)


def load_ground_truth(dataset):
    """Load the full ground-truth label table for evaluation.

    Args:
        dataset: The :class:`BenchmarkDataset` to load.

    Returns:
        A DataFrame with (at least) columns ``filename`` and ``label_idx`` for
        every image in the dataset.
    """
    df = pd.read_csv(dataset.labels_csv)
    logger.info(
        f"Loaded ground truth for {dataset.name}: {len(df)} images, "
        f"{df['label_idx'].nunique()} classes"
    )
    return df


def build_labeled_csv(
    dataset,
    anomaly_class_idx,
    n_anomaly,
    n_nominal,
    output_path,
    seed,
):
    """Construct a fixed labeled set and write it as ``labeled_data.csv``.

    Anomalies are drawn from images whose ``label_idx`` equals
    ``anomaly_class_idx``; nominal (normal) images are drawn from every other
    class. Sampling is reproducible for a given ``seed``.

    Args:
        dataset: The :class:`BenchmarkDataset` to sample from.
        anomaly_class_idx: Integer ``label_idx`` treated as the anomaly class.
        n_anomaly: Number of anomaly images to label.
        n_nominal: Number of normal images to label.
        output_path: Where to write the CSV. The AnomalyMatch v2 training
            pipeline requires columns ``id`` (image basename) and ``label``
            (``"anomaly"``/``"normal"``), so that is what we write — note this
            differs from the v1 ``filename,label`` header.
        seed: Random seed controlling the sample.

    Returns:
        The written labeled DataFrame with columns ``id`` and ``label``.

    Raises:
        ValueError: If the dataset does not contain enough images of either
            class to satisfy the requested counts. We fail hard rather than
            silently shrinking the labeled set, because a short labeled set
            would quietly change the experiment.
    """
    full = pd.read_csv(dataset.labels_csv)

    anomaly_pool = full[full["label_idx"] == anomaly_class_idx]
    nominal_pool = full[full["label_idx"] != anomaly_class_idx]

    if len(anomaly_pool) < n_anomaly:
        raise ValueError(
            f"Requested {n_anomaly} anomalies for class {anomaly_class_idx} but only "
            f"{len(anomaly_pool)} available in {dataset.name}"
        )
    if len(nominal_pool) < n_nominal:
        raise ValueError(
            f"Requested {n_nominal} nominal samples but only {len(nominal_pool)} "
            f"available in {dataset.name}"
        )

    selected_anomaly = anomaly_pool.sample(n=n_anomaly, random_state=seed)
    selected_nominal = nominal_pool.sample(n=n_nominal, random_state=seed)

    # The v2 image-folder training source matches the CSV ``id`` column against
    # image basenames from the folder, so ``id`` == the CSV ``filename`` value.
    labeled = pd.concat(
        [
            pd.DataFrame({"id": selected_anomaly["filename"], "label": "anomaly"}),
            pd.DataFrame({"id": selected_nominal["filename"], "label": "normal"}),
        ],
        ignore_index=True,
    )
    # Shuffle so the CSV order does not encode the label, matching how a user
    # would have labeled a mixed stream of images.
    labeled = labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    labeled.to_csv(output_path, index=False)

    prevalence = n_anomaly / (n_anomaly + n_nominal)
    logger.info(
        f"Wrote labeled set to {output_path}: {n_anomaly} anomaly + {n_nominal} normal "
        f"({prevalence:.2%} anomaly) for class {anomaly_class_idx} of {dataset.name}"
    )
    return labeled
