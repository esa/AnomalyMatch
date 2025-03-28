#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import random

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import albumentations as A
import numpy as np


def AutoContrast(img, _):
    """Apply automatic contrast to the image.

    Args:
        img: PIL Image to be processed
        _: Unused parameter

    Returns:
        PIL Image with automatic contrast applied
    """
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    """Adjust the brightness of the image.

    Args:
        img: PIL Image to be processed
        v: Brightness factor (must be >= 0.0)

    Returns:
        PIL Image with adjusted brightness
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    """Adjust the color saturation of the image.

    Args:
        img: PIL Image to be processed
        v: Color saturation factor (must be >= 0.0)

    Returns:
        PIL Image with adjusted color saturation
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    """Adjust the contrast of the image.

    Args:
        img: PIL Image to be processed
        v: Contrast factor (must be >= 0.0)

    Returns:
        PIL Image with adjusted contrast
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    """Equalize the histogram of the image.

    Args:
        img: PIL Image to be processed
        _: Unused parameter

    Returns:
        PIL Image with equalized histogram
    """
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    """Invert the colors of the image.

    Args:
        img: PIL Image to be processed
        _: Unused parameter

    Returns:
        PIL Image with inverted colors
    """
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    """Return the image unchanged.

    Args:
        img: PIL Image to be processed
        v: Unused parameter

    Returns:
        The original PIL Image
    """
    return img


def Posterize(img, v):
    """Reduce the number of bits for each color channel.

    Args:
        img: PIL Image to be processed
        v: Number of bits to keep for each channel (range [4, 8])

    Returns:
        PIL Image with reduced color channels
    """
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):
    """Rotate the image by v degrees.

    Args:
        img: PIL Image to be processed
        v: Rotation angle in degrees (range [-30, 30])

    Returns:
        Rotated PIL Image
    """
    return img.rotate(v)


def Sharpness(img, v):
    """Adjust the sharpness of the image.

    Args:
        img: PIL Image to be processed
        v: Sharpness factor (must be >= 0.0)

    Returns:
        PIL Image with adjusted sharpness
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):
    """Apply horizontal shear to the image.

    Args:
        img: PIL Image to be processed
        v: Shear factor (range [-0.3, 0.3])

    Returns:
        PIL Image with horizontal shear applied
    """
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    """Apply vertical shear to the image.

    Args:
        img: PIL Image to be processed
        v: Shear factor (range [-0.3, 0.3])

    Returns:
        PIL Image with vertical shear applied
    """
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    """Translate the image horizontally by a percentage of its width.

    Args:
        img: PIL Image to be processed
        v: Translation factor as a percentage of image width (range [-0.3, 0.3])

    Returns:
        PIL Image with horizontal translation applied
    """
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):
    """Translate the image horizontally by an absolute amount.

    Args:
        img: PIL Image to be processed
        v: Absolute translation amount in pixels

    Returns:
        PIL Image with horizontal translation applied
    """
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    """Translate the image vertically by a percentage of its height.

    Args:
        img: PIL Image to be processed
        v: Translation factor as a percentage of image height (range [-0.3, 0.3])

    Returns:
        PIL Image with vertical translation applied
    """
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):
    """Translate the image vertically by an absolute amount.

    Args:
        img: PIL Image to be processed
        v: Absolute translation amount in pixels

    Returns:
        PIL Image with vertical translation applied
    """
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):
    """Invert all pixel values above a threshold.

    Args:
        img: PIL Image to be processed
        v: Threshold for solarization (range [0, 256])

    Returns:
        Solarized PIL Image
    """
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):
    """Apply cutout augmentation to the image.

    Creates a square mask at a random location in the image.

    Args:
        img: PIL Image or numpy array to be processed
        v: Size of the cutout as a percentage of image size (range [0.0, 0.5])

    Returns:
        Image with cutout applied
    """
    assert 0.0 <= v <= 0.5
    if v <= 0.0:
        return img

    if isinstance(img, np.ndarray):
        v = int(v * img.shape[0])
        return A.Cutout(1, v, v, always_apply=True, fill_value=(128))(image=img)["image"]
    else:
        v = v * img.size[0]
        return CutoutAbs(img, v)


def CutoutAbs(img, v):
    """Apply cutout augmentation with absolute size to the image.

    Creates a square mask at a random location in the image with specified absolute size.

    Args:
        img: PIL Image to be processed
        v: Absolute size of the cutout in pixels

    Returns:
        PIL Image with cutout applied
    """
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    """Return a list of available augmentation operations with their value ranges.

    Returns:
        List of tuples (operation, min_value, max_value) for each augmentation
    """
    augments = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]
    return augments


class RandAugment:
    """Random augmentation pipeline implementation.

    Randomly applies a series of image transformations.

    Attributes:
        n: Number of augmentation operations to apply
        m: Magnitude parameter (deprecated)
        augment_list: List of available augmentation operations
    """

    def __init__(self, n, m, use_ms_augmentations=False):
        """Initialize the RandAugment pipeline.

        Args:
            n: Number of augmentation operations to apply
            m: Magnitude parameter [0, 30] (deprecated)
            use_ms_augmentations: Whether to use MS-specific augmentations (currently unused)
        """
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        """Apply random augmentations to the input image.

        Args:
            img: PIL Image to be augmented

        Returns:
            Augmented PIL Image
        """
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)  # for fixmatch
        return img


if __name__ == "__main__":
    randaug = RandAugment(3, 5, True)
    test_img = np.zeros([32, 32, 13], dtype="uint8")
    print(randaug)

    for op, min_val, max_val in randaug.augment_list:
        val = min_val + float(max_val - min_val) * random.random()
        print(op)
        img = op(test_img, val)
