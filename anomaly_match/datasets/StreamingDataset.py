#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import torch
from torch.utils.data import IterableDataset
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import numpy as np
import time
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from loguru import logger
from PIL import Image
from typing import Optional, Tuple
from anomaly_match.datasets.Label import Label


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        file_list: list,
        size: tuple,
        mean: list,
        std: list,
        transform=None,
        prefetch_size: int = 1000,  # Reduced from 20000
        num_workers: int = 4,
    ):
        self.file_list = file_list
        self.size = size
        self.mean = mean
        self.std = std
        self.transform = transform

        # Smaller queue size to prevent memory buildup
        self.prefetch_size = min(prefetch_size, len(file_list))
        self.num_workers = num_workers

        # Use a bounded queue with block=True
        self.queue = Queue(maxsize=self.prefetch_size)
        self.shutdown_event = threading.Event()

        # Track processed items for cleanup
        self.processed_count = 0
        self.batch_size = 32  # Default batch size for cleanup

        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def _read_and_resize_image(self, filepath: str) -> np.ndarray:
        """Read an image file and resize it."""
        try:
            image = imageio.imread(filepath)
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)
            if image.shape[:2] != tuple(self.size):
                image = resize(image, self.size, anti_aliasing=True)
                image = img_as_ubyte(image)
            else:
                image = img_as_ubyte(image)
            return image
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None

    def _process_image(self, filepath: str) -> Optional[Tuple[torch.Tensor, str]]:
        """Process a single image file."""
        try:
            image = self._read_and_resize_image(filepath)
            if image is None:
                return None

            if self.transform:
                image = Image.fromarray(image)
                image = self.transform(image)

            # Convert to tensor and free numpy array
            if isinstance(image, np.ndarray):
                tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                del image

                # # Normalize
                # mean = torch.tensor(self.mean, device=tensor.device).view(-1, 1, 1)
                # std = torch.tensor(self.std, device=tensor.device).view(-1, 1, 1)
                # tensor = (tensor - mean) / std

                return tensor, Label.UNKNOWN, filepath

            return image, Label.UNKNOWN, filepath

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None

    def _prefetch_worker(self):
        """Worker thread that prefetches and processes images."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            file_idx = 0
            futures = []

            while not self.shutdown_event.is_set():
                # Submit new tasks if queue has space
                while len(futures) < self.prefetch_size and file_idx < len(self.file_list):
                    filepath = self.file_list[file_idx]
                    futures.append(executor.submit(self._process_image, filepath))
                    file_idx += 1

                # Process completed futures
                done_futures = []
                for future in futures:
                    if future.done():
                        try:
                            result = future.result()
                            if result is not None:
                                # Block if queue is full
                                self.queue.put(result, block=True)
                            done_futures.append(future)
                        except Exception as e:
                            logger.error(f"Error in prefetch worker: {e}")
                            done_futures.append(future)

                # Remove processed futures
                for future in done_futures:
                    futures.remove(future)

                # Reset if needed
                if file_idx >= len(self.file_list) and not futures:
                    file_idx = 0

                # Sleep briefly to prevent CPU thrashing
                time.sleep(0.001)

    def __iter__(self):
        return self

    def __next__(self):
        if self.queue.empty() and self.shutdown_event.is_set():
            raise StopIteration

        try:
            item = self.queue.get(timeout=5)
            self.processed_count += 1

            # Periodic cleanup
            if self.processed_count % self.batch_size == 0:
                torch.cuda.empty_cache()

            return item

        except Exception:
            raise StopIteration

    def __del__(self):
        """Cleanup when the dataset is destroyed."""
        self.shutdown_event.set()
        if hasattr(self, "prefetch_thread"):
            self.prefetch_thread.join(timeout=1)
        self.queue = None  # Remove queue reference
