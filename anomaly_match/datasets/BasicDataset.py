#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import torch

from anomaly_match.image_processing.transforms import get_strong_transforms


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and returns both weakly and strongly augmented images.
    """

    def __init__(
        self,
        data,
        filenames,
        targets=None,
        num_classes=None,
        transform=None,
        use_strong_transform=False,
        strong_transform=None,
    ):
        """
        Args
            data: x_data as numpy array or torch tensor
            filenames: filenames of x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
        """
        super(BasicDataset, self).__init__()

        # Ensure data and filenames are of the same length
        assert len(data) == len(filenames), "data and filenames must have the same length"
        if targets is not None:
            assert len(data) == len(targets), "data and targets must have the same length"

        # Convert data to shared memory to minimize duplication across workers
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)  # Convert to torch tensor
        elif isinstance(data, list):
            if len(data) > 0:
                self.data = torch.stack([torch.tensor(d) for d in data])
            else:
                self.data = torch.empty(0)  # Handle the empty dataset case
        else:
            self.data = data  # Assume it's a torch tensor already

        # If data is non-empty, place tensor in shared memory to reduce duplication
        if len(self.data) > 0:
            self.data.share_memory_()

        self.filenames = filenames
        if targets is not None:
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            self.targets = targets.clone().detach()

        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform

        self.transform = transform
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = get_strong_transforms()
            else:
                self.strong_transform = strong_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target, filename
        else:
            return weak_augment_image, strong_augment_image, target
        """
        # Set idx-th target
        target = self.targets[idx] if self.targets is not None else None

        # Set augmented images
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target, self.filenames[idx]
        else:
            img = Image.fromarray(img.numpy()) if isinstance(img, torch.Tensor) else img
            img_w = self.transform(img)

        if not self.use_strong_transform:
            return img_w, target, self.filenames[idx]
        else:
            img_s = self.strong_transform(img)
            return img_w, img_s, target

    def __len__(self):
        return len(self.data)
