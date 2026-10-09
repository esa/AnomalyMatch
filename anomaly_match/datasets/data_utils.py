#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Data loading and splitting utilities for dataset preparation."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, sampler
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler


def get_sampler_by_name(name: str) -> type:
    """Get a PyTorch sampler class by its name.

    Args:
        name: Name of the sampler class (e.g., "RandomSampler")

    Returns:
        The sampler class corresponding to the name

    Raises:
        AttributeError: If the sampler name is not found in torch.utils.data.sampler
    """
    # List all available samplers for error message
    sampler_name_list = sorted(
        name
        for name in torch.utils.data.sampler.__dict__
        if not name.startswith("_") and callable(sampler.__dict__[name])
    )

    try:
        # Handle distributed sampler separately as it's in a different module
        if name == "DistributedSampler":
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except AttributeError as e:
        logger.error(f"Sampler '{name}' not found. Available samplers: {sampler_name_list}")
        raise AttributeError(f"Sampler '{name}' not found") from e


def get_data_loader(
    dset: Dataset,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_sampler: str | type | None = None,
    replacement: bool = True,
    num_epochs: int | None = None,
    num_iters: int | None = None,
    generator: torch.Generator | None = None,
    drop_last: bool = True,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader with various sampling strategies.

    This function provides a flexible way to create DataLoader objects with different
    sampling strategies, including weighted sampling for imbalanced datasets.

    Args:
        dset: Dataset object
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data (only used if data_sampler is None)
        num_workers: Number of subprocesses to use for data loading
        pin_memory: If True, copy tensors into CUDA pinned memory
        data_sampler: Sampler name (string) or sampler class
        replacement: Whether to sample with replacement
        num_epochs: Number of epochs (used to calculate total samples)
        num_iters: Number of iterations (alternative way to calculate total samples)
        generator: Random number generator for reproducibility
        drop_last: Whether to drop the last incomplete batch
        use_weighted_sampler: If True, use WeightedRandomSampler for class imbalance

    Returns:
        DataLoader: PyTorch DataLoader configured with the specified parameters

    Raises:
        RuntimeError: If an unsupported sampler is specified
    """
    assert batch_size is not None, "Batch size must be specified"

    # Option 1: Use weighted sampler for class imbalance
    if use_weighted_sampler:
        # Extract labels from the dataset
        labels = dset.targets
        labels = np.array(labels)

        # Compute class weights
        if np.unique(labels).size == 1:
            # If only one class exists, use uniform weights
            samples_weight = np.ones(len(labels))
            logger.debug("Only one class found in dataset, using uniform weights")
        else:
            # Calculate weights inversely proportional to class frequencies
            class_sample_count = np.array(
                [len(np.where(labels == t)[0]) for t in np.unique(labels)]
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in labels])
            logger.debug(f"Class distribution: {class_sample_count}, weights: {weight}")

        # Convert weights to a tensor
        samples_weight = torch.from_numpy(samples_weight).float()

        # Calculate total number of samples to draw
        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters
        else:
            num_samples = len(dset)

        # Create weighted sampler
        weighted_sampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )

        return DataLoader(
            dset,
            batch_size=batch_size,
            sampler=weighted_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
        )

    # Option 2: Use standard DataLoader with shuffle
    elif data_sampler is None:
        return DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
        )

    # Option 3: Use custom sampler
    else:
        # Convert string sampler name to actual sampler class
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        # Calculate total number of samples
        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters
        else:
            num_samples = len(dset)

        # Currently only RandomSampler is fully supported
        if data_sampler.__name__ == "RandomSampler":
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"Sampler {data_sampler.__name__} is not fully implemented.")

        # Use BatchSampler to handle batching
        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(
            dset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
        )
